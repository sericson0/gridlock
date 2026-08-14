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
  True these are integral; False relaxes them leaving the model structure
  otherwise identical. Units that don't need state (typically renewables)
  dispatch freely in [0, availability * max]. A generator row with
  ``num_units > 1`` is a *cluster* of identical units: ``u`` counts how
  many are committed (0..N) instead of being binary, which removes the
  permutation symmetry between interchangeable units.
- ``config.tight_generation_limits`` and ``config.tight_ramp_limits`` swap
  the output and ramp rows for their convex-hull-tighter equivalents (see
  the respective builders). They change the LP relaxation, never the set
  of integer-feasible schedules.
- Availability factors derate both maximum and minimum stable output, so a
  partially derated unit may stay committed at reduced output. Units with
  no availability profile are fully available every hour.
- Network is a pipe-and-bubble transport model: each line carries a
  forward and a reverse flow, both capped at the line rating, and the
  receiving node gets ``(1 - loss_factor)`` of the sent power.
- Storage is a bathtub: state of charge is a tracked state variable with
  the round-trip efficiency split evenly between charging and discharging.
- Windows with no initial state (monolithic runs) are *cyclic*: every
  time-linked constraint (commitment logic, min up/down, ramps, storage
  SOC) treats the hour before the first hour as the window's last hour.
  Rolling windows instead link their first hour to the carried state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pyomo.environ as pyo

from .config import RunConfig
from .data import SystemData


@dataclass
class InitialState:
    """Dispatch state carried into a window from the hour before it starts.

    Used by the rolling-horizon runner. Fields left as None skip the
    corresponding linkage; ``initial=None`` (monolithic mode) makes the
    window cyclic: the hour before the first hour is the last hour.

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
    _add_storage_variables(m, system)
    _add_network_variables(m, system)
    _add_unserved_energy_variables(m, system, hours)

    _add_generator_limits(m, system, config, hours, initial)
    _add_commitment_logic(m, system, hours, initial)
    _add_min_up_down_times(m, system, hours, initial)
    _add_ramp_limits(m, system, config, hours, initial)
    _add_storage_constraints(m, system, hours, initial)
    _add_load_balance(m, system, hours)
    _add_objective(m, system, config)
    return m


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _previous_hour(m: pyo.ConcreteModel, t: int) -> int:
    """The hour before ``t`` in this window, wrapping to the last hour.

    ``Set.prevw`` is Pyomo's previous-with-wraparound lookup on the ordered
    hour set; the wrap at the first hour is what makes windows without
    carried state cyclic.
    """
    return m.T.prevw(t)


def _next_hour(m: pyo.ConcreteModel, t: int) -> int:
    """The hour after ``t``, wrapping the last hour back to the first."""
    return m.T.nextw(t)


def _availability_arrays(system: SystemData) -> dict[str, np.ndarray]:
    """Hourly availability per unit, indexable by (global) hour.

    Units without a profile column are omitted from the dict and treated
    as fully available (factor 1.0) wherever it is consulted.
    """
    return {g: system.availability[g].to_numpy() for g in system.availability.columns}


def _demand_arrays(system: SystemData) -> dict[str, np.ndarray]:
    """Hourly demand per node, indexable by (global) hour."""
    return {n: system.demand[n].to_numpy() for n in system.demand.columns}


def _unit_counts(system: SystemData) -> dict[str, int]:
    """Units represented by each generator row (1 unless it is a cluster)."""
    return {g: int(n) for g, n in system.generators["num_units"].items()}


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
    max_mw = system.generators["max_mw"].to_dict()
    units = _unit_counts(system)
    availability = _availability_arrays(system)

    def output_bounds(m, g, t):
        profile = availability.get(g)
        upper = max_mw[g] if profile is None else max_mw[g] * profile[t]
        return (0.0, upper * units[g])

    m.p = pyo.Var(m.G, m.T, bounds=output_bounds, doc="generation (MW)")

    # The unit-commitment switch: integral commitment versus its LP
    # relaxation. A cluster of N identical units commits an integer count
    # in [0, N]; the common N == 1 case stays a plain binary so the model
    # HiGHS receives is byte-for-byte what it was before clustering existed.
    def commitment_domain(m, g, t):
        if not config.unit_commitment:
            return pyo.NonNegativeReals
        return pyo.Binary if units[g] == 1 else pyo.NonNegativeIntegers

    def count_bounds(m, g, t):
        return (0.0, float(units[g]))

    m.u = pyo.Var(
        m.G_UC,
        m.T,
        domain=commitment_domain,
        bounds=count_bounds,
        doc="units committed (on/off, or a count for clusters)",
    )
    m.v = pyo.Var(
        m.G_UC, m.T, domain=pyo.NonNegativeReals, bounds=count_bounds, doc="startups"
    )
    m.w = pyo.Var(
        m.G_UC, m.T, domain=pyo.NonNegativeReals, bounds=count_bounds, doc="shutdowns"
    )


def _add_storage_variables(m: pyo.ConcreteModel, system: SystemData) -> None:
    power = system.storage["power_mw"].to_dict()
    energy = system.storage["energy_mwh"].to_dict()

    def power_bounds(m, s, t):
        return (0.0, power[s])

    def energy_bounds(m, s, t):
        return (0.0, energy[s])

    m.charge = pyo.Var(m.S, m.T, bounds=power_bounds, doc="grid-side charging (MW)")
    m.discharge = pyo.Var(m.S, m.T, bounds=power_bounds, doc="grid-side discharging (MW)")
    m.soc = pyo.Var(m.S, m.T, bounds=energy_bounds, doc="state of charge (MWh)")


def _add_network_variables(m: pyo.ConcreteModel, system: SystemData) -> None:
    capacity = system.network["capacity_mw"].to_dict()

    def flow_bounds(m, l, t):
        return (0.0, capacity[l])

    m.flow_fwd = pyo.Var(m.L, m.T, bounds=flow_bounds, doc="flow from_node -> to_node (MW)")
    m.flow_rev = pyo.Var(m.L, m.T, bounds=flow_bounds, doc="flow to_node -> from_node (MW)")


def _add_unserved_energy_variables(
    m: pyo.ConcreteModel, system: SystemData, hours: list[int]
) -> None:
    demand = _demand_arrays(system)

    def shed_bounds(m, n, t):
        return (0.0, demand[n][t])

    m.shed = pyo.Var(m.N, m.T, bounds=shed_bounds, doc="unserved energy (MW)")


# --------------------------------------------------------------------------
# Generator constraints
# --------------------------------------------------------------------------


def _add_generator_limits(
    m: pyo.ConcreteModel,
    system: SystemData,
    config: RunConfig,
    hours: list[int],
    initial: InitialState | None,
) -> None:
    """Committed units run between derated min and max; others are bound-only.

    With ``config.tight_generation_limits`` the upper bound also charges
    for startup and shutdown: a unit that starts in hour t cannot exceed
    its startup capability there, and one that stops in t+1 cannot exceed
    its shutdown capability in t. Same variables, tighter relaxation.
    """
    gens = system.generators
    max_mw = gens["max_mw"].to_dict()
    min_mw = gens["min_mw"].to_dict()
    ramp_rate = gens["ramp_rate_mw_per_hr"].to_dict()
    up_time = {g: int(v) for g, v in gens["min_up_time_hr"].items()}
    availability = _availability_arrays(system)
    last = hours[-1]
    cyclic = initial is None

    def available_mw(g, t):
        profile = availability.get(g)
        return max_mw[g] if profile is None else max_mw[g] * profile[t]

    def ramp_capability(g, t):
        """Output allowed in the hour a unit starts up or shuts down.

        Mirrors the startup/shutdown ramp allowance used by the ramp rows:
        a unit may come up to its minimum stable level even when that
        exceeds its hourly ramp rate. Capped at what is available so a
        derated hour can never permit more than the derated maximum.
        """
        return min(max(min_mw[g], ramp_rate[g]), available_mw(g, t))

    if not config.tight_generation_limits:

        def max_output_rule(m, g, t):
            return m.p[g, t] <= available_mw(g, t) * m.u[g, t]

        m.max_output = pyo.Constraint(m.G_UC, m.T, rule=max_output_rule)
    else:
        # A unit with a one-hour minimum up time can start in t and stop in
        # t+1, so both allowances would apply at once and could bind below
        # its minimum stable level. Those units get the two terms as
        # separate rows; everything else gets the single tighter row.
        m.G_UT1 = pyo.Set(
            initialize=[g for g in m.G_UC if up_time[g] <= 1], within=m.G_UC, ordered=True
        )
        m.G_UT2 = pyo.Set(
            initialize=[g for g in m.G_UC if up_time[g] > 1], within=m.G_UC, ordered=True
        )

        def shutdown_term(m, g, t):
            """(available - SD) * w_{t+1}, or zero where t+1 leaves the window."""
            if t == last and not cyclic:
                return 0.0
            nxt = _next_hour(m, t)
            return (available_mw(g, t) - ramp_capability(g, t)) * m.w[g, nxt]

        def max_output_rule(m, g, t):
            available = available_mw(g, t)
            startup = (available - ramp_capability(g, t)) * m.v[g, t]
            return m.p[g, t] <= available * m.u[g, t] - startup - shutdown_term(m, g, t)

        m.max_output = pyo.Constraint(m.G_UT2, m.T, rule=max_output_rule)

        def max_output_startup_rule(m, g, t):
            available = available_mw(g, t)
            return (
                m.p[g, t]
                <= available * m.u[g, t]
                - (available - ramp_capability(g, t)) * m.v[g, t]
            )

        m.max_output_startup = pyo.Constraint(m.G_UT1, m.T, rule=max_output_startup_rule)

        def max_output_shutdown_rule(m, g, t):
            return m.p[g, t] <= available_mw(g, t) * m.u[g, t] - shutdown_term(m, g, t)

        m.max_output_shutdown = pyo.Constraint(m.G_UT1, m.T, rule=max_output_shutdown_rule)

    def min_output_rule(m, g, t):
        if min_mw[g] <= 0:
            return pyo.Constraint.Skip
        profile = availability.get(g)
        floor = min_mw[g] if profile is None else min_mw[g] * profile[t]
        return m.p[g, t] >= floor * m.u[g, t]

    m.min_output = pyo.Constraint(m.G_UC, m.T, rule=min_output_rule)


def _add_commitment_logic(
    m: pyo.ConcreteModel,
    system: SystemData,
    hours: list[int],
    initial: InitialState | None,
) -> None:
    """Link on/off state to startup/shutdown indicators: u_t - u_{t-1} = v_t - w_t."""
    first = hours[0]
    cyclic = initial is None
    u0 = initial.commitment if initial is not None and initial.commitment else None

    def logic_rule(m, g, t):
        if t == first and not cyclic:
            if u0 is None or g not in u0:
                # Free initial state: hour one carries no startup/shutdown.
                return pyo.Constraint.Skip
            return m.u[g, t] - u0[g] == m.v[g, t] - m.w[g, t]
        return m.u[g, t] - m.u[g, _previous_hour(m, t)] == m.v[g, t] - m.w[g, t]

    m.commitment_logic = pyo.Constraint(m.G_UC, m.T, rule=logic_rule)


def _add_min_up_down_times(
    m: pyo.ConcreteModel,
    system: SystemData,
    hours: list[int],
    initial: InitialState | None,
) -> None:
    gens = system.generators
    first = hours[0]
    cyclic = initial is None
    window_len = len(hours)
    up_time = {g: int(v) for g, v in gens["min_up_time_hr"].items()}
    down_time = {g: int(v) for g, v in gens["min_down_time_hr"].items()}

    m.G_MIN_UP = pyo.Set(
        initialize=[g for g in m.G_UC if up_time[g] > 1],
        within=m.G_UC,
        ordered=True,
    )
    m.G_MIN_DOWN = pyo.Set(
        initialize=[g for g in m.G_UC if down_time[g] > 1],
        within=m.G_UC,
        ordered=True,
    )

    def lookback(t, length):
        """The ``length`` hours ending at ``t``; cyclic windows wrap past the start."""
        if not cyclic:
            return range(max(first, t - length + 1), t + 1)
        return [first + (t - first - k) % window_len for k in range(min(length, window_len))]

    def min_up_rule(m, g, t):
        return (
            pyo.quicksum(m.v[g, tau] for tau in lookback(t, up_time[g])) <= m.u[g, t]
        )

    m.min_up_time = pyo.Constraint(m.G_MIN_UP, m.T, rule=min_up_rule)

    units = _unit_counts(system)

    def min_down_rule(m, g, t):
        # Units shut down in the last DT hours must still be off, so at
        # most (fleet size - committed) of them can be mid-downtime.
        return (
            pyo.quicksum(m.w[g, tau] for tau in lookback(t, down_time[g]))
            <= units[g] - m.u[g, t]
        )

    m.min_down_time = pyo.Constraint(m.G_MIN_DOWN, m.T, rule=min_down_rule)

    # Carry unfinished min up/down obligations across a window boundary by
    # fixing the first hours of this window to the inherited state. Only
    # meaningful for single units: a cluster's members are mid-obligation
    # individually, which one aggregate count cannot express.
    if initial is not None and initial.state_hours:
        for g, run_hours in initial.state_hours.items():
            if g not in m.G_UC or units[g] > 1:
                continue
            if run_hours > 0 and run_hours < up_time[g]:
                hours_to_fix = min(up_time[g] - run_hours, window_len)
                for t in hours[:hours_to_fix]:
                    m.u[g, t].fix(1.0)
            elif run_hours < 0 and -run_hours < down_time[g]:
                hours_to_fix = min(down_time[g] + run_hours, window_len)
                for t in hours[:hours_to_fix]:
                    m.u[g, t].fix(0.0)


def _add_ramp_limits(
    m: pyo.ConcreteModel,
    system: SystemData,
    config: RunConfig,
    hours: list[int],
    initial: InitialState | None,
) -> None:
    """Hour-to-hour ramp limits, with startup/shutdown ramps of max(min_mw, ramp).

    The default rows bound the step by ``ramp * u`` on the committed side
    plus a startup (or shutdown) allowance. With
    ``config.tight_ramp_limits`` they become the two-period convex-hull
    inequalities of Damcı-Kurt et al., which additionally subtract the
    minimum stable level on the other side of the step:

        p_t - p_{t-1} <= (Pmin + RU) u_t - Pmin u_{t-1} - (Pmin + RU - SU) v_t

    Both forms give SU in the startup hour and RU while running, but the
    tight one also states that a unit shutting down must fall by at least
    its minimum stable level — which the loose form leaves to a big-M.
    """
    gens = system.generators
    first = hours[0]
    cyclic = initial is None
    u0 = initial.commitment if initial is not None and initial.commitment else None
    p0 = initial.output if initial is not None and initial.output else None

    max_mw = gens["max_mw"].to_dict()
    min_mw = gens["min_mw"].to_dict()
    ramp_rate = gens["ramp_rate_mw_per_hr"].to_dict()
    availability = _availability_arrays(system)
    committed = set(gens.index[gens["needs_commitment"]])
    start_ramp = {g: max(min_mw[g], ramp_rate[g]) for g in committed}

    def floor_mw(g, t):
        profile = availability.get(g)
        return min_mw[g] if profile is None else min_mw[g] * profile[t]

    def step_floor(g, t, prev):
        """Minimum stable level valid on both sides of a t-1 -> t step.

        Taking the smaller of the two derated floors keeps the inequality
        valid when an availability profile lowers the floor across the
        step; without a profile both are the same number.
        """
        return min(floor_mw(g, prev), floor_mw(g, t))

    # A unit that can traverse its whole range in one hour needs no ramp rows.
    m.G_RAMP = pyo.Set(
        initialize=[g for g in m.G if ramp_rate[g] < max_mw[g]],
        within=m.G,
        ordered=True,
    )

    tight = config.tight_ramp_limits

    def ramp_up_rule(m, g, t):
        ramp = ramp_rate[g]
        if t == first and not cyclic:
            if p0 is None or g not in p0:
                return pyo.Constraint.Skip
            if g not in committed:
                return m.p[g, t] - p0[g] <= ramp
            if u0 is None or g not in u0:
                return pyo.Constraint.Skip
            if tight:
                floor = step_floor(g, t, t)
                return m.p[g, t] - p0[g] <= (floor + ramp) * m.u[g, t] - floor * u0[
                    g
                ] - (floor + ramp - start_ramp[g]) * m.v[g, t]
            return m.p[g, t] - p0[g] <= ramp * u0[g] + start_ramp[g] * m.v[g, t]
        prev = _previous_hour(m, t)
        if g not in committed:
            return m.p[g, t] - m.p[g, prev] <= ramp
        if tight:
            floor = step_floor(g, t, prev)
            return (
                m.p[g, t] - m.p[g, prev]
                <= (floor + ramp) * m.u[g, t]
                - floor * m.u[g, prev]
                - (floor + ramp - start_ramp[g]) * m.v[g, t]
            )
        return (
            m.p[g, t] - m.p[g, prev]
            <= ramp * m.u[g, prev] + start_ramp[g] * m.v[g, t]
        )

    m.ramp_up = pyo.Constraint(m.G_RAMP, m.T, rule=ramp_up_rule)

    def ramp_down_rule(m, g, t):
        ramp = ramp_rate[g]
        if t == first and not cyclic:
            if p0 is None or g not in p0:
                return pyo.Constraint.Skip
            if g not in committed:
                return p0[g] - m.p[g, t] <= ramp
            if tight and u0 is not None and g in u0:
                floor = step_floor(g, t, t)
                return p0[g] - m.p[g, t] <= (floor + ramp) * u0[g] - floor * m.u[
                    g, t
                ] - (floor + ramp - start_ramp[g]) * m.w[g, t]
            return p0[g] - m.p[g, t] <= ramp * m.u[g, t] + start_ramp[g] * m.w[g, t]
        prev = _previous_hour(m, t)
        if g not in committed:
            return m.p[g, prev] - m.p[g, t] <= ramp
        if tight:
            floor = step_floor(g, t, prev)
            return (
                m.p[g, prev] - m.p[g, t]
                <= (floor + ramp) * m.u[g, prev]
                - floor * m.u[g, t]
                - (floor + ramp - start_ramp[g]) * m.w[g, t]
            )
        return (
            m.p[g, prev] - m.p[g, t]
            <= ramp * m.u[g, t] + start_ramp[g] * m.w[g, t]
        )

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
    first, last = hours[0], hours[-1]
    cyclic = _storage_is_cyclic(initial)
    eta = system.storage["one_way_efficiency"].to_dict()

    def soc_balance_rule(m, s, t):
        if t == first and not cyclic:
            previous_soc = initial.soc[s]
        else:
            # Cyclic windows wrap the first hour back to the last, which
            # both sets the start-of-window SOC and forces the window to
            # end where it began — no free starting variable needed.
            previous_soc = m.soc[s, _previous_hour(m, t)]
        return (
            m.soc[s, t]
            == previous_soc + eta[s] * m.charge[s, t] - m.discharge[s, t] / eta[s]
        )

    m.soc_balance = pyo.Constraint(m.S, m.T, rule=soc_balance_rule)

    if not cyclic and initial.min_terminal_soc:

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
    demand = _demand_arrays(system)
    delivered = (1.0 - net["loss_factor"]).to_dict()

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
            delivered[l] * m.flow_fwd[l, t] for l in lines_to[n]
        ) + pyo.quicksum(
            delivered[l] * m.flow_rev[l, t] for l in lines_from[n]
        )
        exports = pyo.quicksum(m.flow_fwd[l, t] for l in lines_from[n]) + pyo.quicksum(
            m.flow_rev[l, t] for l in lines_to[n]
        )
        return (
            generation + storage_net + imports - exports + m.shed[n, t]
            == demand[n][t]
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
