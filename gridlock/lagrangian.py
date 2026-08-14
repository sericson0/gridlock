"""A one-shot Lagrangian dual bound at a given price vector.

Dualising the load-balance rows at prices ``lambda`` makes the model
separable: every generator, storage unit, line and shed variable faces the
prices alone, and the sum of their individually optimal values plus the
price-weighted demand is a valid lower bound on the MIP for *any* price
vector — no iteration required. Evaluated at the LP relaxation's own duals
it answers the question this module exists for: **how much tighter than the
LP bound can an external bound be?** The threshold that governs solve time
is ``bound / (1 - mip_gap)`` (see
:func:`gridlock.heuristics.gap_threshold_objective`), so every dollar of
bound lift raises the threshold a warm start has to clear — and unlike a
better start, a better bound composes with *any* incumbent.

Why there is room above the LP at all: the LP relaxes integrality, so a
unit can be 0.3 committed and pay 0.3 of its no-load cost. The subproblems
here keep commitment integral and min up/down exact (the per-unit dynamic
program in :mod:`gridlock.rounding`), so at the LP's optimal duals the
bound is **guaranteed at or above the LP objective** — provided the
subproblems keep every non-dualised constraint, which brings in the one
deliberate relaxation:

- **Ramp rows are dropped inside the subproblems.** A ramp-aware exact
  single-unit DP needs the previous hour's output in its state; the run
  frontier in :mod:`gridlock.rounding` does not carry it. Dropping rows
  from a minimisation can only lower the value, so the bound stays *valid*,
  merely weaker — and on RTS-GMLC, where face-value ramp rates leave almost
  no unit ramp-limited over an hour (see docs/profiling.md), the loss is
  expected to be small. On a system where ramps bind, the guarantee
  "Lagrangian >= LP" no longer holds and the honest usable bound is
  ``max(lagrangian, lp_bound)``.

The other subproblems are exact: storage faces the prices as a small
arbitrage LP (solved with highspy directly), lines and shed are closed-form
per hour, and a cluster of N identical units is N independent copies of the
single-unit DP.

Only cyclic (monolithic) windows are supported — the probe's use case is
the weekly experiment, which is cyclic. A carried initial state would need
the DP's linear boundary conditions threaded through; nothing here does
that yet.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .config import RunConfig, SolverSettings
from .data import SystemData
from .model import build_model
from .rounding import commitment_value, single_unit_schedule
from .solver import HighsSession


@dataclass
class LagrangianBound:
    """The bound and where it came from, part by part.

    ``total`` is the valid lower bound on the MIP objective. ``parts``
    attributes it: ``price_weighted_demand`` (the constant), ``thermal_uc``
    (the per-unit DPs, always <= 0 — it is minus the fleet's maximum profit
    at these prices), ``free_generation``, ``lines``, ``storage`` and
    ``shed`` (each <= 0: they are minimised values of profit-seeking
    subproblems).
    """

    total: float
    parts: dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0

    def lift_over(self, lp_bound: float) -> float:
        """Bound improvement over the LP, as a fraction of the LP bound."""
        return (self.total - lp_bound) / lp_bound


def schedule_value(
    schedule: np.ndarray,
    value_on: np.ndarray,
    startup_cost: float,
    shutdown_cost: float,
) -> float:
    """What a cyclic 0/1 schedule earns at the DP's own hourly values.

    Recomputed from the schedule rather than read out of the DP's internal
    frontier, so a disagreement between the frontier's value and its
    reconstruction cannot slip a phantom value into the bound.

    The direction of the risk is worth stating precisely: the bound needs
    each subproblem's true **maximum** profit — an *underestimate* there
    (a suboptimal DP schedule) would **overstate** the bound and could
    invalidate it. So exactness of :func:`~gridlock.rounding.single_unit_schedule`
    is load-bearing here; the test suite pins it against brute-force
    enumeration of every legal cyclic schedule on short horizons.
    """
    u = np.asarray(schedule, dtype=float).round()
    previous = np.roll(u, 1)
    starts = float(np.maximum(u - previous, 0.0).sum())
    stops = float(np.maximum(previous - u, 0.0).sum())
    return float((value_on * u).sum() - startup_cost * starts - shutdown_cost * stops)


def _availability(system: SystemData, g: str, hours: Sequence[int]) -> np.ndarray:
    if g in system.availability.columns:
        return system.availability[g].loc[list(hours)].to_numpy(dtype=float)
    return np.ones(len(hours))


def _thermal_uc_value(
    system: SystemData, hours: Sequence[int], prices: pd.DataFrame
) -> float:
    """Minus the committed fleet's maximum profit at these prices.

    One exact single-unit DP per generator row, valued over the unit's
    **full** output range (``ceiling=None``): the "capacity" valuation that
    measurably loses as a *guess* is exactly the right subproblem for a
    *bound*, because the bound must allow every unit the dispatch the model
    would. A cluster row is N independent identical units, so its value is
    N times the single unit's.
    """
    gens = system.generators
    total = 0.0
    for g in gens.index[gens["needs_commitment"]]:
        value_on = commitment_value(system, g, hours, prices, ceiling=None)
        schedule = single_unit_schedule(
            value_on,
            min_up=int(gens.at[g, "min_up_time_hr"]),
            min_down=int(gens.at[g, "min_down_time_hr"]),
            startup_cost=float(gens.at[g, "startup_cost"]),
            shutdown_cost=float(gens.at[g, "shutdown_cost"]),
            cyclic=True,
        )
        profit = schedule_value(
            schedule,
            value_on,
            float(gens.at[g, "startup_cost"]),
            float(gens.at[g, "shutdown_cost"]),
        )
        # A legal schedule with negative value never beats staying off.
        total -= max(0.0, profit) * float(gens.at[g, "num_units"])
    return total


def _free_generation_value(
    system: SystemData, hours: Sequence[int], prices: pd.DataFrame
) -> float:
    """Units without commitment state: run flat out when the price pays.

    ``min over p in [0, cap] of (mc - lambda) * p`` per hour. Their ramp
    rows, where they exist, are dropped — same direction as the thermal
    relaxation, same validity argument.
    """
    gens = system.generators
    total = 0.0
    for g in gens.index[~gens["needs_commitment"]]:
        price = prices[gens.at[g, "node"]].to_numpy(dtype=float)
        capacity = (
            float(gens.at[g, "max_mw"])
            * float(gens.at[g, "num_units"])
            * _availability(system, g, hours)
        )
        margin = float(gens.at[g, "marginal_cost"]) - price
        total += float((capacity * np.minimum(0.0, margin)).sum())
    return total


def _lines_value(
    system: SystemData, hours: Sequence[int], prices: pd.DataFrame
) -> float:
    """Each direction of each line flows at capacity when the spread pays.

    A forward flow appears in the sender's balance at -1 and the receiver's
    at ``(1 - loss)``, so its dualised cost is
    ``lambda_from - (1 - loss) * lambda_to`` per MW; negative means the
    line profits and the subproblem runs it at its rating.
    """
    total = 0.0
    for _, row in system.network.iterrows():
        price_from = prices[row["from_node"]].to_numpy(dtype=float)
        price_to = prices[row["to_node"]].to_numpy(dtype=float)
        delivered = 1.0 - float(row["loss_factor"])
        capacity = float(row["capacity_mw"])
        forward = np.minimum(0.0, price_from - delivered * price_to)
        reverse = np.minimum(0.0, price_to - delivered * price_from)
        total += capacity * float((forward + reverse).sum())
    return total


def _shed_value(
    system: SystemData, hours: Sequence[int], prices: pd.DataFrame, voll: float
) -> float:
    """``min over sigma in [0, D] of (VOLL - lambda) * sigma`` per node-hour."""
    total = 0.0
    for n in system.demand.columns:
        demand = system.demand[n].loc[list(hours)].to_numpy(dtype=float)
        margin = voll - prices[n].to_numpy(dtype=float)
        total += float((demand * np.minimum(0.0, margin)).sum())
    return total


def _storage_value(
    system: SystemData, hours: Sequence[int], prices: pd.DataFrame
) -> float:
    """The storage fleet's best arbitrage against the prices, exactly.

    ``min sum_t lambda * (charge - discharge)`` over the cyclic bathtub —
    a small LP (3 columns per storage-hour), solved directly with highspy
    so the probe never pays a Pyomo build. Units are independent, so one
    block-diagonal LP covers the fleet.
    """
    storage = system.storage
    if not len(storage):
        return 0.0
    import highspy

    T = len(hours)
    num_vars = 3 * T * len(storage)
    costs = np.zeros(num_vars)
    lower = np.zeros(num_vars)
    upper = np.zeros(num_vars)

    rows: list[dict[int, float]] = []
    for k, (s, row) in enumerate(storage.iterrows()):
        base = 3 * T * k
        charge, discharge, soc = base, base + T, base + 2 * T
        price = prices[row["node"]].to_numpy(dtype=float)
        eta = float(row["one_way_efficiency"])
        costs[charge : charge + T] = price
        costs[discharge : discharge + T] = -price
        upper[charge : charge + T] = float(row["power_mw"])
        upper[discharge : discharge + T] = float(row["power_mw"])
        upper[soc : soc + T] = float(row["energy_mwh"])
        for t in range(T):
            # soc_t - soc_{t-1} - eta * charge_t + discharge_t / eta = 0,
            # with the first hour wrapping to the last (cyclic window).
            coefficients: dict[int, float] = {}
            coefficients[soc + t] = coefficients.get(soc + t, 0.0) + 1.0
            previous = soc + (t - 1) % T
            coefficients[previous] = coefficients.get(previous, 0.0) - 1.0
            coefficients[charge + t] = -eta
            coefficients[discharge + t] = 1.0 / eta
            rows.append(coefficients)

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.addCols(num_vars, costs, lower, upper, 0, [], [], [])
    starts, indices, values = [0], [], []
    for coefficients in rows:
        for index, value in sorted(coefficients.items()):
            indices.append(index)
            values.append(value)
        starts.append(len(indices))
    h.addRows(
        len(rows),
        np.zeros(len(rows)),
        np.zeros(len(rows)),
        len(indices),
        np.array(starts[:-1], dtype=np.int32),
        np.array(indices, dtype=np.int32),
        np.array(values, dtype=float),
    )
    h.run()
    status = h.getModelStatus()
    if h.modelStatusToString(status) != "Optimal":
        raise RuntimeError(
            f"storage subproblem did not solve: {h.modelStatusToString(status)}"
        )
    return float(h.getInfo().objective_function_value)


def lagrangian_bound(
    system: SystemData,
    hours: Sequence[int],
    prices: pd.DataFrame,
    voll: float,
) -> LagrangianBound:
    """Evaluate the dual function at ``prices``. Valid for any price vector.

    ``prices`` is an hours x nodes frame — typically
    :func:`gridlock.rounding.nodal_prices` on a solved LP relaxation, which
    is the price vector at which this bound provably dominates the LP's
    (up to the dropped ramp rows; see the module docstring).
    """
    started = time.perf_counter()
    hours = list(hours)
    constant = 0.0
    for n in system.demand.columns:
        demand = system.demand[n].loc[hours].to_numpy(dtype=float)
        constant += float((prices[n].to_numpy(dtype=float) * demand).sum())

    parts = {
        "price_weighted_demand": constant,
        "thermal_uc": _thermal_uc_value(system, hours, prices),
        "free_generation": _free_generation_value(system, hours, prices),
        "lines": _lines_value(system, hours, prices),
        "storage": _storage_value(system, hours, prices),
        "shed": _shed_value(system, hours, prices, voll),
    }
    return LagrangianBound(
        total=sum(parts.values()),
        parts=parts,
        seconds=time.perf_counter() - started,
    )


def probe_at_lp_prices(
    system: SystemData,
    config: RunConfig,
    hours: Sequence[int] | None = None,
) -> dict:
    """Solve the LP relaxation, read its duals, evaluate the bound there.

    Returns ``lp_bound``, ``lagrangian`` (a :class:`LagrangianBound`),
    ``usable_bound`` (their max — the number a termination test may use),
    and the LP's own timings. One LP plus milliseconds of DP: cheap enough
    to run per window.
    """
    from .rounding import nodal_prices

    hours = list(range(system.num_hours)) if hours is None else list(hours)
    lp_config = RunConfig(
        unit_commitment=False,
        cyclic=True,
        tight_generation_limits=config.tight_generation_limits,
        tight_ramp_limits=config.tight_ramp_limits,
        voll=config.voll,
    )
    model = build_model(system, lp_config, hours)
    info, duals = HighsSession(model).solve(SolverSettings(), want_duals=True)
    prices = nodal_prices(model, duals, hours)
    bound = lagrangian_bound(system, hours, prices, config.voll)
    return {
        "lp_bound": info.objective,
        "lp_seconds": info.solve_seconds,
        "lagrangian": bound,
        "usable_bound": max(info.objective, bound.total),
        "prices": prices,
    }
