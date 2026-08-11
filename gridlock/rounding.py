"""Discretise a fractional commitment with a per-unit dynamic program.

``round()`` followed by :func:`~gridlock.heuristics.repair_min_up_down` and
:func:`~gridlock.heuristics.enforce_adequacy` is *monotone*: every pass only
ever commits more. Across the 12-month RTS-GMLC study the schedule that came
out of that pipeline ran 2.7% over the optimum's committed unit-hours while
the raw fractional relaxation it was built from sat within 0.3% of it. The
bias is not in the LP; it is in the way the LP is discretised.

What replaces it here is the price-based single-unit commitment subproblem.
The load-balance rows are the only thing coupling units to each other, so
once their duals are held fixed as nodal prices the problem separates: each
unit faces the schedule-independent value

    value(t) = (lambda[node(g), t] - marginal_cost) * p*(t) - no_load_cost

of being committed in hour t, and pays ``startup_cost`` on every transition
into an on-run. Minimum up and down times make the transitions legal or
illegal. That subproblem is solved *exactly* by dynamic programming, which
is the one question no rounding rule can answer: a rounded relaxation (like
a merit-order stack) knows what capacity an hour *needs*, and only a DP
knows whether a three-hour start pays for itself over the minimum up time it
commits to.

Three assumptions, all deliberate:

- **Ramp limits are ignored.** The subproblem values a committed hour at an
  output the unit's own bounds allow, not at one reachable from the previous
  hour, so it overstates the value of short runs on slow units. It is left
  in because commitment, not dispatch, is what this produces: the completion
  LP re-optimises every MW against the real ramp rows, so the error costs
  accuracy in the *ranking* of schedules and never feasibility.
- **``p*`` is not the unit's capacity.** Textbook Lagrangian UC values a
  committed hour at whichever end of the output range the price favours,
  which is what ``dp_output="capacity"`` does and which is measurably the
  wrong question: at a *marginal* price every unit below the margin is
  profitable flat out, and nothing in a separated subproblem notices that
  they cannot all sell. On RTS-GMLC weeks 00/09/18 that valuation committed
  2,046/2,398/2,262 raw unit-hours against relaxations summing to
  1,907/2,336/2,256, and scored +13.3%/+4.0%/+8.6% against a round-and-repair
  baseline of +2.38%/+1.53%/+5.42%. The default instead caps the profitable
  end at the MW the relaxation actually ran the unit at
  (:func:`dispatch_ceiling`) — the same "LP dispatch surplus" a decommitment
  pass would rank by.
- **A price vector cannot see inside a cluster.** Identical units at one
  node facing one price make identical decisions, so a DP over a
  ``num_units > 1`` row can only answer "all of them" or "none of them" —
  which is exactly the count granularity the cluster exists to express.
  Clustered rows therefore keep the rounded relaxation
  (:func:`dp_commitment` takes it as ``fallback``) and are counted in the
  returned notes rather than silently DP'd.

**What this is currently worth, measured — it loses.** The DP's output is
*not* monotone, which is the point, so it still goes through
:func:`~gridlock.heuristics.enforce_adequacy` before delivery — and on
RTS-GMLC that guarantee is not strong enough to hold it. All six
valuation/scope combinations scored worse than round-and-repair on all three
weeks tested (00/09/18), and the entire loss is unserved energy. The best of
them (``dispatch``/``contested``) delivers +9.15%/+1.84%/+22.12% against the
baseline's +2.38%/+1.53%/+5.42% while shedding 46/0/139 MWh against
3.5/0/26.6 — and at a VOLL of 10,000 those MWh *are* the margin.

*Net of shed* the same schedules are competitive or better: +1.05%/+1.84%/
+1.45% against the baseline's +1.77%/+1.53%/+1.48%, and week00's committed
unit-hours fall from 2,027 to 1,983 against a relaxation summing to 1,907.
So the schedules are not what fails; the static, per-hour,
optimistic-imports adequacy test is. A shed-driven repair that reads the
completion LP's own verdict (plan item 1f) is a prerequisite for this
module, not a complement to it.

Worth knowing before spending more on the idea: the prize is smaller than
the +2.7% unit-hour bias suggests. No-load plus startup is $1.34 M of
week00's $5.68 M threshold and the best variant moves it by $15 k — 0.27
points of a 2.38-point gap. The rest of that gap is dispatch cost, because
what a wrong commitment really costs is the minimum output it forces into
the stack (or, when it is missing, the load that goes unserved). Both ends
of that are dispatch, and neither is reachable by re-deciding units one at a
time against a fixed price.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import pandas as pd

from .data import SystemData
from .model import InitialState

_NEG = -np.inf
# Stands in for "the carried obligation is already met" when a rolling
# window hands over a commitment without saying how long it has held.
_SATISFIED = 1_000_000


def nodal_prices(model, duals: dict | None, hours: Sequence[int]) -> pd.DataFrame:
    """Hourly nodal prices from the load-balance duals of a solved relaxation.

    The dual of ``load_balance[n, t]`` is the cost of one more MW of demand
    at that node and hour, which is precisely the price the single-unit
    subproblem sells into. Congestion and shedding are already in it: a node
    whose local capacity is exhausted prices at VOLL, which is what makes
    the DP commit there rather than leave the completion to shed.
    """
    if duals is None:
        raise RuntimeError(
            "the LP relaxation returned no duals; DP rounding needs nodal "
            "prices (solve with want_duals=True)"
        )
    prices = {}
    for n in model.N:
        column = []
        for t in hours:
            value = duals.get(model.load_balance[n, t])
            if value is None:
                raise RuntimeError(f"no dual for load_balance[{n}, {t}]")
            column.append(float(value))
        prices[n] = column
    return pd.DataFrame(prices, index=list(hours), dtype=float)


def single_unit_schedule(
    value_on: np.ndarray,
    min_up: int,
    min_down: int,
    startup_cost: float = 0.0,
    shutdown_cost: float = 0.0,
    cyclic: bool = True,
    initial_state: tuple[int, int] | None = None,
) -> np.ndarray:
    """The most profitable legal on/off pattern for one unit, exactly.

    Maximises ``sum_t value_on[t] * u[t] - startup * starts - shutdown *
    stops`` over schedules whose on-runs are at least ``min_up`` hours and
    whose off-runs are at least ``min_down``.

    ``cyclic`` matches the model's own wrap: hour 0's predecessor is the
    last hour, so a run may span the seam and is measured across it. That is
    handled by enumerating the seam — see :func:`_cyclic_schedule` — rather
    than by pretending the horizon has ends. ``initial_state`` is the
    ``(on/off, hours already in that state)`` pair a rolling window carries
    in, and applies only when ``cyclic`` is False; ``None`` there means the
    first hours are unconstrained, which is how :func:`~gridlock.model.build_model`
    treats a window with no carried commitment.

    Rather than tabulate ``(hour, state, hours-in-state)`` this walks the
    equivalent frontier over *runs*, which costs O(hours) per boundary
    condition instead of O(hours x min-time) — the same optimum, small
    enough to run for every unit of a 168-hour week in well under a second.
    """
    value_on = np.asarray(value_on, dtype=float)
    horizon = len(value_on)
    if horizon == 0:
        return np.zeros(0)
    min_up = max(1, int(min_up))
    min_down = max(1, int(min_down))
    prefix = np.concatenate(([0.0], np.cumsum(value_on)))
    if cyclic:
        return _cyclic_schedule(prefix, min_up, min_down, startup_cost, shutdown_cost)
    return _linear_schedule(
        prefix, min_up, min_down, startup_cost, shutdown_cost, initial_state
    )


# --------------------------------------------------------------------------
# The frontier recursion
# --------------------------------------------------------------------------


def _frontier(
    prefix: np.ndarray,
    open_on: np.ndarray,
    open_off: np.ndarray,
    min_up: int,
    min_down: int,
    startup_cost: float,
    shutdown_cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Best value of every prefix of hours that ends ready to switch state.

    Two frontiers, both indexed by the hour a run *would start*:

    ``on_ready[b]``   hours [0, b) are legal, end committed, and the on-run
                      covering b-1 has served its minimum up time — so a
                      shutdown at b is allowed.
    ``off_ready[a]``  the same with the states exchanged: a startup at a is
                      allowed.

    ``open_on``/``open_off`` seed them with the boundary conditions: what
    the hours *before* hour 0 permit, which is the only place the cyclic
    seam and a carried initial state enter.

    Reading the recursion through ``carry[a] = off_ready[a] - startup -
    prefix[a]`` is what removes the inner loop: an on-run [a, b) is worth
    ``prefix[b] - prefix[a]``, so the best predecessor of ``on_ready[b]`` is
    the running maximum of ``carry`` lagged by the minimum up time.
    """
    horizon = len(prefix) - 1
    on_ready = np.full(horizon + 1, _NEG)
    off_ready = np.full(horizon + 1, _NEG)
    carry = np.full(horizon + 1, _NEG)
    on_from = np.full(horizon + 1, -1, dtype=int)
    off_from = np.full(horizon + 1, -1, dtype=int)

    best_carry, best_carry_at = _NEG, -1
    best_on, best_on_at = _NEG, -1
    for t in range(horizon + 1):
        lag = t - min_up
        if lag >= 0 and carry[lag] > best_carry:
            best_carry, best_carry_at = carry[lag], lag
        candidate = _NEG if best_carry == _NEG else prefix[t] + best_carry
        if open_on[t] >= candidate:
            on_ready[t], on_from[t] = open_on[t], -1
        else:
            on_ready[t], on_from[t] = candidate, best_carry_at

        lag = t - min_down
        if lag >= 0 and on_ready[lag] > best_on:
            best_on, best_on_at = on_ready[lag], lag
        candidate = _NEG if best_on == _NEG else best_on - shutdown_cost
        if open_off[t] >= candidate:
            off_ready[t], off_from[t] = open_off[t], -1
        else:
            off_ready[t], off_from[t] = candidate, best_on_at

        if off_ready[t] > _NEG:
            carry[t] = off_ready[t] - startup_cost - prefix[t]
    return on_ready, carry, on_from, off_from


def _reconstruct(
    final_state: int,
    final_start: int,
    horizon: int,
    on_from: np.ndarray,
    off_from: np.ndarray,
) -> np.ndarray:
    """Walk the frontier's predecessors back into a 0/1 schedule.

    ``final_start`` is the hour the last run begins; a ``-1`` predecessor
    means the run reaches hour 0 as the boundary condition allowed.
    """
    schedule = np.zeros(horizon, dtype=float)
    if final_state:
        schedule[final_start:] = 1.0
        state, cursor = 0, final_start  # the run before an on-run is off
    else:
        state, cursor = 1, final_start
    while cursor > 0:
        if state:
            previous = on_from[cursor]
            if previous < 0:
                schedule[:cursor] = 1.0
                break
            schedule[previous:cursor] = 1.0
            state, cursor = 0, previous
        else:
            previous = off_from[cursor]
            if previous < 0:
                break  # already zero
            state, cursor = 1, previous
    return schedule


def _best_candidate(
    prefix: np.ndarray,
    on_ready: np.ndarray,
    carry: np.ndarray,
    shutdown_cost: float,
    last_on_start: int,
    last_off_start: int,
) -> tuple[float, int, int]:
    """Best (value, final state, first hour of the final run) of one pass.

    ``last_on_start`` / ``last_off_start`` cap where the final run may
    begin, which is how a terminal condition is imposed: a cyclic pass that
    assumed *k* hours of credit at the seam must deliver a final run of at
    least *k* hours for the assumption to hold.
    """
    horizon = len(prefix) - 1
    best, state, start = _NEG, 0, -1
    if last_on_start >= 0:
        window = carry[: last_on_start + 1]
        index = int(np.argmax(window))
        if window[index] > _NEG:
            best, state, start = prefix[horizon] + window[index], 1, index
    if last_off_start >= 0:
        window = on_ready[: last_off_start + 1]
        index = int(np.argmax(window))
        if window[index] > _NEG and window[index] - shutdown_cost > best:
            best, state, start = window[index] - shutdown_cost, 0, index
    return best, state, start


# --------------------------------------------------------------------------
# Boundary conditions
# --------------------------------------------------------------------------


def _cyclic_schedule(
    prefix: np.ndarray,
    min_up: int,
    min_down: int,
    startup_cost: float,
    shutdown_cost: float,
) -> np.ndarray:
    """Solve the wrapped problem by enumerating what happens at the seam.

    A cyclic schedule is either constant (always on or always off, both
    trivially legal) or it has runs, exactly one of which may straddle hour
    0. That straddling run is the only thing an acyclic DP cannot see, and
    it is fully described by two numbers: which state it is in, and how many
    of its hours fall at the *end* of the horizon. Fixing that credit *k*
    makes the problem linear — the opening run only has to serve its minimum
    time less *k*, and the closing run must be at least *k* hours long for
    the assumption to have been true — so the wrap costs one pass per
    ``k``, at most ``min_up + min_down`` of them.

    Assuming less credit than the schedule turns out to have is safe (the
    pass then enforces *more* than the wrap requires), so every legal
    schedule is found by the pass with ``k = min(closing run, minimum
    time)`` and none is admitted that the wrap would reject.
    """
    horizon = len(prefix) - 1
    # The constant schedules need no pass: neither has a transition to be
    # legal about, and the always-on one is what a must-run unit costs.
    best, best_schedule = 0.0, np.zeros(horizon)
    if prefix[horizon] > best:
        best, best_schedule = prefix[horizon], np.ones(horizon)

    for state, minimum in ((1, min_up), (0, min_down)):
        for credit in range(1, min(minimum, horizon) + 1):
            open_on = np.full(horizon + 1, _NEG)
            open_off = np.full(horizon + 1, _NEG)
            if state:
                # The seam run is on: the horizon opens committed, and may
                # shut down once the two halves of that run add up.
                opens_at = max(0, min_up - credit)
                open_on[opens_at:] = prefix[opens_at:]
            else:
                opens_at = max(0, min_down - credit)
                open_off[opens_at:] = 0.0
            on_ready, carry, on_from, off_from = _frontier(
                prefix, open_on, open_off, min_up, min_down, startup_cost, shutdown_cost
            )
            value, final_state, start = _best_candidate(
                prefix,
                on_ready,
                carry,
                shutdown_cost,
                horizon - credit if state else -1,
                -1 if state else horizon - credit,
            )
            if value > best:
                best = value
                best_schedule = _reconstruct(
                    final_state, start, horizon, on_from, off_from
                )
    return best_schedule


def _linear_schedule(
    prefix: np.ndarray,
    min_up: int,
    min_down: int,
    startup_cost: float,
    shutdown_cost: float,
    initial_state: tuple[int, int] | None,
) -> np.ndarray:
    """Solve a window that starts from a carried state and simply ends.

    The run in progress at the last hour carries no obligation — its
    minimum time runs past the horizon, which is exactly how the model's
    truncated lookback treats it — so there is no terminal condition here,
    only the opening one.
    """
    horizon = len(prefix) - 1
    open_on = np.full(horizon + 1, _NEG)
    open_off = np.full(horizon + 1, _NEG)
    constant = _NEG
    if initial_state is None:
        # A free initial hour: the model skips the commitment-logic row
        # there, so the window may open in either state at no charge.
        open_on[:] = prefix
        open_off[:] = 0.0
        constant = max(0.0, prefix[horizon])
    else:
        state, held = initial_state
        if state:
            opens_at = max(0, min_up - held)
            open_on[opens_at:] = prefix[opens_at:]
            constant = prefix[horizon]
        else:
            opens_at = max(0, min_down - held)
            open_off[opens_at:] = 0.0
            constant = 0.0

    on_ready, carry, on_from, off_from = _frontier(
        prefix, open_on, open_off, min_up, min_down, startup_cost, shutdown_cost
    )
    value, final_state, start = _best_candidate(
        prefix, on_ready, carry, shutdown_cost, horizon - 1, horizon - 1
    )
    if constant >= value:
        if initial_state is None:
            return np.ones(horizon) if prefix[horizon] > 0.0 else np.zeros(horizon)
        return np.ones(horizon) if initial_state[0] else np.zeros(horizon)
    return _reconstruct(final_state, start, horizon, on_from, off_from)


# --------------------------------------------------------------------------
# Applying the DP across a fleet
# --------------------------------------------------------------------------


def _availability(system: SystemData, g: str, hours: Sequence[int]) -> np.ndarray:
    if g in system.availability.columns:
        return system.availability[g].loc[list(hours)].to_numpy(dtype=float)
    return np.ones(len(hours))


def _carried_state(g: str, initial: InitialState | None) -> tuple[int, int] | None:
    if initial is None or not initial.commitment or g not in initial.commitment:
        return None
    state = 1 if float(initial.commitment[g]) > 0.5 else 0
    held = abs(int((initial.state_hours or {}).get(g, 0)))
    return state, held or _SATISFIED


def commitment_value(
    system: SystemData,
    g: str,
    hours: Sequence[int],
    prices: pd.DataFrame,
    ceiling: np.ndarray | None = None,
) -> np.ndarray:
    """What committing unit ``g`` is worth in each hour at the given prices.

    Output is chosen at whichever end of the unit's range the price
    favours — full output when the price beats marginal cost, minimum
    stable level when it does not — because within one hour the subproblem
    is linear in output. Availability derates both ends, so an outage hour
    is worth its no-load cost and nothing else.

    ``ceiling`` caps the *profitable* end hour by hour, which is how the
    caller supplies a reference dispatch. It matters more than it looks: at
    a *marginal* price every unit cheaper than the margin is profitable flat
    out, so valuing each one at its own full capacity values the fleet at
    several times the load it is competing for. See :func:`dispatch_ceiling`.

    The minimum stable level is deliberately *not* capped with it. Capping
    both ends prices the hours a unit sits through at a loss as if it could
    sit through them at zero output, which is the one thing a committed unit
    cannot do — and it made the DP hold runs open through troughs it should
    have shut down for (3,232 committed unit-hours against a relaxation
    summing to 1,907 on RTS-GMLC week00).
    """
    gens = system.generators
    availability = _availability(system, g, hours)
    price = prices[gens.at[g, "node"]].to_numpy(dtype=float)
    marginal = float(gens.at[g, "marginal_cost"])
    floor = float(gens.at[g, "min_mw"]) * availability
    top = float(gens.at[g, "max_mw"]) * availability
    if ceiling is not None:
        top = np.clip(ceiling, floor, top)
    output = np.where(price > marginal, top, floor)
    return (price - marginal) * output - float(gens.at[g, "no_load_cost"])


def dispatch_ceiling(
    relaxation: pd.DataFrame, output: pd.DataFrame, tolerance: float = 1e-6
) -> pd.DataFrame:
    """How much a *committed* unit actually ran in the relaxation, MW per unit.

    ``output / relaxation`` un-scales the LP's dispatch by its fractional
    commitment: a unit half committed and running half its capacity was, as
    far as the relaxation is concerned, one unit running flat out. Where the
    relaxation committed nothing there is nothing to un-scale and the
    ceiling is zero, which :func:`commitment_value` reads as "if it ran here
    it would run at its minimum" — the relaxation found no room for it, so
    it gets no credit for capacity it would not have sold.

    This is the same quantity plan item 1a ranks by ("LP dispatch surplus"),
    and using it as the ceiling turns the DP from an optimistic bidder into
    a pass that asks whether each run the relaxation *actually made use of*
    earned its no-load and startup costs back.
    """
    committed = relaxation.to_numpy(dtype=float)
    ran = output.to_numpy(dtype=float)
    per_unit = np.where(committed > tolerance, ran / np.maximum(committed, tolerance), 0.0)
    return pd.DataFrame(per_unit, index=relaxation.index, columns=relaxation.columns)


def dp_commitment(
    system: SystemData,
    units: Sequence[str],
    hours: Sequence[int],
    prices: pd.DataFrame,
    cyclic: bool,
    initial: InitialState | None = None,
    fallback: pd.DataFrame | None = None,
    ceiling: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Solve every unit's own commitment problem at the given nodal prices.

    ``fallback`` supplies the columns this cannot answer — clustered rows,
    whose integer count no single price vector can resolve (see the module
    docstring). ``ceiling`` caps the output each committed unit is valued at
    (:func:`dispatch_ceiling`). Returns the schedule and notes recording
    which units took which path.
    """
    started = time.perf_counter()
    gens = system.generators
    hours = list(hours)
    schedule = pd.DataFrame(0.0, index=hours, columns=list(units))
    clustered = []
    for g in units:
        if int(gens.at[g, "num_units"]) > 1:
            clustered.append(g)
            if fallback is not None:
                schedule[g] = fallback[g].to_numpy(dtype=float).round()
            continue
        schedule[g] = single_unit_schedule(
            commitment_value(
                system,
                g,
                hours,
                prices,
                None if ceiling is None else ceiling[g].to_numpy(dtype=float),
            ),
            min_up=int(gens.at[g, "min_up_time_hr"]),
            min_down=int(gens.at[g, "min_down_time_hr"]),
            startup_cost=float(gens.at[g, "startup_cost"]),
            shutdown_cost=float(gens.at[g, "shutdown_cost"]),
            cyclic=cyclic,
            initial_state=_carried_state(g, initial),
        )
    notes = {
        "dp_units": len(units) - len(clustered),
        "dp_clustered_units": len(clustered),
        "dp_seconds": time.perf_counter() - started,
    }
    return schedule, notes


def restrict_to_decommitments(
    schedule: pd.DataFrame,
    reference: pd.DataFrame,
    contested: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Keep the DP's answer only where it commits *less* than ``reference``.

    The DP disagrees with a rounded relaxation in both directions, and the
    two directions are not equally trustworthy. Dropping a run is a claim
    about that unit's own economics, which is precisely what the DP knows.
    Adding one is a claim that the fleet has room for it, which a single
    price vector does not know — at a marginal price every unit below the
    margin looks worth starting, and nothing in the subproblem notices that
    they cannot all sell.

    ``contested`` narrows it further to entries the relaxation itself left
    unresolved. Most of what the DP drops is already there (54 of 75, 130 of
    138 and 55 of 74 on RTS-GMLC weeks 00/09/18), so the restriction is
    nearly free — and the handful it excludes are entries the LP resolved at
    1, which is where a decommitment turns into unserved energy.

    The bound is on what this hands to the repair, not on what comes out of
    it: dropping an hour can move a run's start, and the repair extends runs
    *forward*, so a shortened run can end one hour later than the one it
    came from. In practice the repair adds a handful of hours against the
    tens this removes.
    """
    allowed = schedule.to_numpy(dtype=float) < reference.to_numpy(dtype=float)
    if contested is not None:
        allowed = allowed & contested.to_numpy()
    return pd.DataFrame(
        np.where(allowed, schedule.to_numpy(dtype=float), reference.to_numpy(dtype=float)),
        index=schedule.index,
        columns=schedule.columns,
    )
