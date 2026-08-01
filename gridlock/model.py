"""Pyomo model builder for the gridlock production cost model.

A model is built for one *window*: a contiguous run of hours. A monolithic
annual solve is simply a single window covering every hour. In
rolling-horizon mode the runner builds one model per window and threads an
:class:`InitialState` between them.

Formulation summary (full math in docs/formulation.md):

- Load balance per node and hour: generation + storage discharge + line
  imports (net of losses) + unserved energy = demand + storage charge +
  line exports.
- Generators that need on/off state (non-zero minimum, startup/shutdown
  cost, no-load cost, or min up/down times) get commitment variables
  ``u``/``v``/``w`` (on / started / stopped). With ``config.unit_commitment``
  True these are binary; False relaxes them to [0, 1] leaving the model
  structure otherwise identical. Units that don't need state (typically
  renewables) dispatch freely in [0, availability * max].
- Availability factors derate both maximum and minimum stable output, so a
  partially derated unit may stay committed at reduced output.
- Network is a pipe-and-bubble transport model: each line carries a
  forward and a reverse flow, both capped at the line rating, and the
  receiving node gets ``(1 - loss_factor)`` of the sent power.
- Storage is a bathtub: state of charge is a tracked state variable with
  the round-trip efficiency split evenly between charging and discharging.
  With no initial state the window is cyclic (ending SOC equals the free
  starting SOC variable); rolling windows instead start from the carried
  SOC value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pyomo.environ as pyo

from .config import RunConfig
from .data import SystemData


@dataclass
class InitialState:
    """Dispatch state carried into a window from the hour before it starts.

    Used by the rolling-horizon runner. Fields left as None skip the
    corresponding linkage; ``initial=None`` (monolithic mode) gives a free
    initial commitment state and cyclic storage.

    commitment:       unit -> 0/1 on/off state in the previous hour
    output:           unit -> MW output in the previous hour
    state_hours:      unit -> consecutive hours on (positive) or off
                      (negative) as of the previous hour; used to enforce
                      min up/down times across window boundaries
    soc:              storage unit -> MWh state of charge at the previous
                      hour's end
    min_terminal_soc: storage unit -> minimum MWh at the window's last hour
                      (set on the final window so the year cannot end drained)
    """

    commitment: dict[str, float] | None = None
    output: dict[str, float] | None = None
    state_hours: dict[str, int] | None = None
    soc: dict[str, float] | None = None
    min_terminal_soc: dict[str, float] | None = None


def build_model(
    system: SystemData,
    config: RunConfig,
    hours: Sequence[int],
    initial: InitialState | None = None,
) -> pyo.ConcreteModel:
    """Build the dispatch model for ``hours`` (contiguous, 0-based)."""
    hours = [int(t) for t in hours]
    if not hours:
        raise ValueError("hours must be non-empty")
    if hours != list(range(hours[0], hours[0] + len(hours))):
        raise ValueError("hours must be a contiguous ascending range")
    if hours[-1] >= system.num_hours:
        raise ValueError("hours extend beyond the input data horizon")

    m = pyo.ConcreteModel(name="gridlock")
    _add_sets(m, system, hours)
    _add_generator_variables(m, system, config, hours)
    _add_storage_variables(m, system, initial)
    _add_network_variables(m, system)
    _add_unserved_energy_variables(m, system, hours)

    _add_generator_limits(m, system, hours)
    _add_commitment_logic(m, system, hours, initial)
    _add_min_up_down_times(m, system, hours, initial)
    _add_ramp_limits(m, system, hours, initial)
    _add_storage_constraints(m, system, hours, initial)
    _add_load_balance(m, system, hours)
    _add_objective(m, system, config)
    return m


# --------------------------------------------------------------------------
# Sets and variables
# --------------------------------------------------------------------------


def _add_sets(m: pyo.ConcreteModel, system: SystemData, hours: list[int]) -> None:
    gens = system.generators
    m.T = pyo.Set(initialize=hours, ordered=True, doc="hours in this window")
    m.G = pyo.Set(initialize=list(gens.index), ordered=True, doc="all generators")
    m.G_UC = pyo.Set(
        initialize=list(gens.index[gens["needs_commitment"]]),
        within=m.G,
        ordered=True,
        doc="generators with commitment (on/off) state",
    )
    m.S = pyo.Set(initialize=list(system.storage.index), ordered=True, doc="storage units")
    m.N = pyo.Set(initialize=list(system.nodes.index), ordered=True, doc="nodes")
    m.L = pyo.Set(initialize=list(system.network.index), ordered=True, doc="lines")


def _add_generator_variables(
    m: pyo.ConcreteModel, system: SystemData, config: RunConfig, hours: list[int]
) -> None:
    gens = system.generators
    availability = system.availability

    def output_bounds(m, g, t):
        return (0.0, gens.at[g, "max_mw"] * availability.at[t, g])

    m.p = pyo.Var(m.G, m.T, bounds=output_bounds, doc="generation (MW)")

    # The unit-commitment switch: binary on/off versus its LP relaxation.
    commitment_domain = pyo.Binary if config.unit_commitment else pyo.UnitInterval
    m.u = pyo.Var(m.G_UC, m.T, domain=commitment_domain, doc="committed (on/off)")
    m.v = pyo.Var(m.G_UC, m.T, domain=pyo.UnitInterval, doc="startup indicator")
    m.w = pyo.Var(m.G_UC, m.T, domain=pyo.UnitInterval, doc="shutdown indicator")


def _add_storage_variables(
    m: pyo.ConcreteModel, system: SystemData, initial: InitialState | None
) -> None:
    stor = system.storage

    def power_bounds(m, s, t):
        return (0.0, stor.at[s, "power_mw"])

    def energy_bounds(m, s, t=None):
        return (0.0, stor.at[s, "energy_mwh"])

    m.charge = pyo.Var(m.S, m.T, bounds=power_bounds, doc="grid-side charging (MW)")
    m.discharge = pyo.Var(m.S, m.T, bounds=power_bounds, doc="grid-side discharging (MW)")
    m.soc = pyo.Var(m.S, m.T, bounds=energy_bounds, doc="state of charge (MWh)")

    # Cyclic windows choose their own starting state of charge.
    if _storage_is_cyclic(initial):
        m.soc_start = pyo.Var(m.S, bounds=energy_bounds, doc="free starting SOC (MWh)")


def _add_network_variables(m: pyo.ConcreteModel, system: SystemData) -> None:
    net = system.network

    def flow_bounds(m, l, t):
        return (0.0, net.at[l, "capacity_mw"])

    m.flow_fwd = pyo.Var(m.L, m.T, bounds=flow_bounds, doc="flow from_node -> to_node (MW)")
    m.flow_rev = pyo.Var(m.L, m.T, bounds=flow_bounds, doc="flow to_node -> from_node (MW)")


def _add_unserved_energy_variables(
    m: pyo.ConcreteModel, system: SystemData, hours: list[int]
) -> None:
    demand = system.demand

    def shed_bounds(m, n, t):
        return (0.0, demand.at[t, n])

    m.shed = pyo.Var(m.N, m.T, bounds=shed_bounds, doc="unserved energy (MW)")


# --------------------------------------------------------------------------
# Generator constraints
# --------------------------------------------------------------------------


def _add_generator_limits(m: pyo.ConcreteModel, system: SystemData, hours: list[int]) -> None:
    """Committed units run between derated min and max; others are bound-only."""
    gens = system.generators
    availability = system.availability

    def max_output_rule(m, g, t):
        return m.p[g, t] <= gens.at[g, "max_mw"] * availability.at[t, g] * m.u[g, t]

    m.max_output = pyo.Constraint(m.G_UC, m.T, rule=max_output_rule)

    def min_output_rule(m, g, t):
        if gens.at[g, "min_mw"] <= 0:
            return pyo.Constraint.Skip
        return m.p[g, t] >= gens.at[g, "min_mw"] * availability.at[t, g] * m.u[g, t]

    m.min_output = pyo.Constraint(m.G_UC, m.T, rule=min_output_rule)


def _add_commitment_logic(
    m: pyo.ConcreteModel,
    system: SystemData,
    hours: list[int],
    initial: InitialState | None,
) -> None:
    """Link on/off state to startup/shutdown indicators: u_t - u_{t-1} = v_t - w_t."""
    first = hours[0]
    u0 = initial.commitment if initial is not None and initial.commitment else None

    def logic_rule(m, g, t):
        if t == first:
            if u0 is None or g not in u0:
                # Free initial state: hour one carries no startup/shutdown.
                return pyo.Constraint.Skip
            return m.u[g, t] - u0[g] == m.v[g, t] - m.w[g, t]
        return m.u[g, t] - m.u[g, t - 1] == m.v[g, t] - m.w[g, t]

    m.commitment_logic = pyo.Constraint(m.G_UC, m.T, rule=logic_rule)


def _add_min_up_down_times(
    m: pyo.ConcreteModel,
    system: SystemData,
    hours: list[int],
    initial: InitialState | None,
) -> None:
    gens = system.generators
    first = hours[0]

    m.G_MIN_UP = pyo.Set(
        initialize=[g for g in m.G_UC if gens.at[g, "min_up_time_hr"] > 1],
        within=m.G_UC,
        ordered=True,
    )
    m.G_MIN_DOWN = pyo.Set(
        initialize=[g for g in m.G_UC if gens.at[g, "min_down_time_hr"] > 1],
        within=m.G_UC,
        ordered=True,
    )

    def min_up_rule(m, g, t):
        up_time = int(gens.at[g, "min_up_time_hr"])
        window_start = max(first, t - up_time + 1)
        return (
            pyo.quicksum(m.v[g, tau] for tau in range(window_start, t + 1)) <= m.u[g, t]
        )

    m.min_up_time = pyo.Constraint(m.G_MIN_UP, m.T, rule=min_up_rule)

    def min_down_rule(m, g, t):
        down_time = int(gens.at[g, "min_down_time_hr"])
        window_start = max(first, t - down_time + 1)
        return (
            pyo.quicksum(m.w[g, tau] for tau in range(window_start, t + 1))
            <= 1 - m.u[g, t]
        )

    m.min_down_time = pyo.Constraint(m.G_MIN_DOWN, m.T, rule=min_down_rule)

    # Carry unfinished min up/down obligations across a window boundary by
    # fixing the first hours of this window to the inherited state.
    if initial is not None and initial.state_hours:
        for g, run_hours in initial.state_hours.items():
            if g not in m.G_UC:
                continue
            up_time = int(gens.at[g, "min_up_time_hr"])
            down_time = int(gens.at[g, "min_down_time_hr"])
            if run_hours > 0 and run_hours < up_time:
                hours_to_fix = min(up_time - run_hours, len(hours))
                for t in hours[:hours_to_fix]:
                    m.u[g, t].fix(1.0)
            elif run_hours < 0 and -run_hours < down_time:
                hours_to_fix = min(down_time + run_hours, len(hours))
                for t in hours[:hours_to_fix]:
                    m.u[g, t].fix(0.0)


def _add_ramp_limits(
    m: pyo.ConcreteModel,
    system: SystemData,
    hours: list[int],
    initial: InitialState | None,
) -> None:
    """Hour-to-hour ramp limits, with startup/shutdown ramps of max(min_mw, ramp)."""
    gens = system.generators
    first = hours[0]
    u0 = initial.commitment if initial is not None and initial.commitment else None
    p0 = initial.output if initial is not None and initial.output else None

    # A unit that can traverse its whole range in one hour needs no ramp rows.
    m.G_RAMP = pyo.Set(
        initialize=[
            g for g in m.G if gens.at[g, "ramp_rate_mw_per_hr"] < gens.at[g, "max_mw"]
        ],
        within=m.G,
        ordered=True,
    )

    def start_ramp(g):
        return max(gens.at[g, "min_mw"], gens.at[g, "ramp_rate_mw_per_hr"])

    def ramp_up_rule(m, g, t):
        ramp = gens.at[g, "ramp_rate_mw_per_hr"]
        if g in m.G_UC:
            if t == first:
                if p0 is None or g not in p0 or u0 is None or g not in u0:
                    return pyo.Constraint.Skip
                return m.p[g, t] - p0[g] <= ramp * u0[g] + start_ramp(g) * m.v[g, t]
            return (
                m.p[g, t] - m.p[g, t - 1]
                <= ramp * m.u[g, t - 1] + start_ramp(g) * m.v[g, t]
            )
        if t == first:
            if p0 is None or g not in p0:
                return pyo.Constraint.Skip
            return m.p[g, t] - p0[g] <= ramp
        return m.p[g, t] - m.p[g, t - 1] <= ramp

    m.ramp_up = pyo.Constraint(m.G_RAMP, m.T, rule=ramp_up_rule)

    def ramp_down_rule(m, g, t):
        ramp = gens.at[g, "ramp_rate_mw_per_hr"]
        if g in m.G_UC:
            if t == first:
                if p0 is None or g not in p0:
                    return pyo.Constraint.Skip
                return p0[g] - m.p[g, t] <= ramp * m.u[g, t] + start_ramp(g) * m.w[g, t]
            return (
                m.p[g, t - 1] - m.p[g, t]
                <= ramp * m.u[g, t] + start_ramp(g) * m.w[g, t]
            )
        if t == first:
            if p0 is None or g not in p0:
                return pyo.Constraint.Skip
            return p0[g] - m.p[g, t] <= ramp
        return m.p[g, t - 1] - m.p[g, t] <= ramp

    m.ramp_down = pyo.Constraint(m.G_RAMP, m.T, rule=ramp_down_rule)


# --------------------------------------------------------------------------
# Storage, network and load balance
# --------------------------------------------------------------------------


def _storage_is_cyclic(initial: InitialState | None) -> bool:
    return initial is None or initial.soc is None


def _add_storage_constraints(
    m: pyo.ConcreteModel,
    system: SystemData,
    hours: list[int],
    initial: InitialState | None,
) -> None:
    """Bathtub SOC accounting: soc_t = soc_{t-1} + eta * charge - discharge / eta."""
    stor = system.storage
    first, last = hours[0], hours[-1]
    cyclic = _storage_is_cyclic(initial)

    def soc_balance_rule(m, s, t):
        eta = stor.at[s, "one_way_efficiency"]
        if t == first:
            previous_soc = m.soc_start[s] if cyclic else initial.soc[s]
        else:
            previous_soc = m.soc[s, t - 1]
        return m.soc[s, t] == previous_soc + eta * m.charge[s, t] - m.discharge[s, t] / eta

    m.soc_balance = pyo.Constraint(m.S, m.T, rule=soc_balance_rule)

    if cyclic:
        # End where you started; the start itself is the optimizer's choice.
        def cyclic_rule(m, s):
            return m.soc[s, last] == m.soc_start[s]

        m.soc_cyclic = pyo.Constraint(m.S, rule=cyclic_rule)
    elif initial is not None and initial.min_terminal_soc:

        def terminal_rule(m, s):
            if s not in initial.min_terminal_soc:
                return pyo.Constraint.Skip
            return m.soc[s, last] >= initial.min_terminal_soc[s]

        m.soc_terminal = pyo.Constraint(m.S, rule=terminal_rule)


def _add_load_balance(m: pyo.ConcreteModel, system: SystemData, hours: list[int]) -> None:
    """Supply equals demand at every node and hour.

    Line losses are charged to the receiving end: a flow f delivers
    (1 - loss_factor) * f, so the loss is implicitly extra generation the
    sending side must provide.
    """
    net = system.network
    demand = system.demand

    generators_at = {n: [] for n in m.N}
    for g, row in system.generators.iterrows():
        generators_at[row["node"]].append(g)
    storage_at = {n: [] for n in m.N}
    for s, row in system.storage.iterrows():
        storage_at[row["node"]].append(s)
    lines_from = {n: [] for n in m.N}
    lines_to = {n: [] for n in m.N}
    for l, row in net.iterrows():
        lines_from[row["from_node"]].append(l)
        lines_to[row["to_node"]].append(l)

    def load_balance_rule(m, n, t):
        generation = pyo.quicksum(m.p[g, t] for g in generators_at[n])
        storage_net = pyo.quicksum(
            m.discharge[s, t] - m.charge[s, t] for s in storage_at[n]
        )
        imports = pyo.quicksum(
            (1 - net.at[l, "loss_factor"]) * m.flow_fwd[l, t] for l in lines_to[n]
        ) + pyo.quicksum(
            (1 - net.at[l, "loss_factor"]) * m.flow_rev[l, t] for l in lines_from[n]
        )
        exports = pyo.quicksum(m.flow_fwd[l, t] for l in lines_from[n]) + pyo.quicksum(
            m.flow_rev[l, t] for l in lines_to[n]
        )
        return (
            generation + storage_net + imports - exports + m.shed[n, t]
            == demand.at[t, n]
        )

    m.load_balance = pyo.Constraint(m.N, m.T, rule=load_balance_rule)


# --------------------------------------------------------------------------
# Objective
# --------------------------------------------------------------------------


def _add_objective(m: pyo.ConcreteModel, system: SystemData, config: RunConfig) -> None:
    """Minimize variable + no-load + startup/shutdown costs plus shed penalty."""
    gens = system.generators
    marginal_cost = gens["marginal_cost"].to_dict()
    no_load_cost = gens["no_load_cost"].to_dict()
    startup_cost = gens["startup_cost"].to_dict()
    shutdown_cost = gens["shutdown_cost"].to_dict()

    energy_cost = pyo.quicksum(marginal_cost[g] * m.p[g, t] for g in m.G for t in m.T)
    commitment_cost = pyo.quicksum(
        no_load_cost[g] * m.u[g, t]
        + startup_cost[g] * m.v[g, t]
        + shutdown_cost[g] * m.w[g, t]
        for g in m.G_UC
        for t in m.T
    )
    shed_penalty = config.voll * pyo.quicksum(m.shed[n, t] for n in m.N for t in m.T)

    m.total_cost = pyo.Objective(
        expr=energy_cost + commitment_cost + shed_penalty, sense=pyo.minimize
    )
