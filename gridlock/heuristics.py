"""Domain heuristics: guess the commitment schedule before the solver runs.

The variable anatomy of this model (2026-08 profiling) motivates everything
here: the optimal schedule's information content is tiny — on a 52-unit
week, 8 startups/shutdowns hiding in 8,736 binaries, with 50 of 52 units
never changing state — and which units are "hard" is predictable from
merit-order position. A guess that captures the easy 96% turns the MIP's
expensive incumbent search (the sub-MIP heuristics that dominate root
time) into a cheap warm start.

Every heuristic produces a :class:`CommitmentGuess`: a full 0/1 commitment
frame plus a *confidence* mask. The runner delivers it in one of three
modes (``RunConfig.heuristic_fixing``):

- ``off``        warm start only — exact, the solver may overrule anything
- ``screen``     additionally fix the entries the heuristic is confident
                 about — near-exact, bounded by screen quality
- ``aggressive`` fix every guessed value — an upper bound on speed and a
                 lower bound on quality, useful to bracket experiments

Heuristics (``RunConfig.heuristic``):

- ``priority``      dispatch-curve stacking against hourly net load
                    (demand minus renewables, optionally storage-smoothed),
                    with min up/down repair and outage persistence
- ``similar_days``  cluster days by net-load shape, solve one small UC per
                    representative day, transfer schedules to lookalike
                    days and repair the seams
- ``lp``            the LP-relaxation oracle: keep integral values, round
                    the fractional core (99%+ accurate here, but needs a
                    full-horizon LP solve)

Guesses are *completed* into full solutions by solving the model once with
the guessed commitment fixed (every continuous variable then gets a value,
which the appsi warm start requires). An over-committed guess can make
that completion infeasible — committed minimum generation has nowhere to
go — in which case the minimum-output rows are temporarily relaxed rather
than giving up (the warm start's own integer-fixed LP has the final say).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from .config import RunConfig, SolverSettings
from .data import SystemData
from .model import InitialState, build_model
from .solver import HighsSession, SolveInfo

HEURISTICS = ("priority", "similar_days", "lp")
FIXING_MODES = ("off", "screen", "aggressive")

# Availability below this is treated as an outage hour when stacking.
_OUTAGE_TOL = 0.05
_TOL = 1e-6


@dataclass
class CommitmentGuess:
    """A guessed commitment schedule and how sure the heuristic is of it.

    ``commitment``: hours x committed-unit frame of 0/1 values.
    ``certain``: same shape, True where the heuristic would stake a fixing
    on the value (used by the ``screen`` delivery mode).
    ``relaxation``: the continuous values ``commitment`` was rounded from,
    for the heuristics that have any (only ``lp`` does). Kept because the
    fractional values carry the heuristic's own uncertainty, which the
    rounded frame has thrown away.
    """

    name: str
    commitment: pd.DataFrame
    certain: pd.DataFrame
    build_seconds: float = 0.0
    notes: dict = field(default_factory=dict)
    relaxation: pd.DataFrame | None = None


# --------------------------------------------------------------------------
# Shared ingredients
# --------------------------------------------------------------------------


def net_load_mw(system: SystemData, hours: list[int]) -> pd.Series:
    """Hourly load left for committed units: demand minus free generation.

    "Free" is every unit without commitment state (renewables and any
    unconstrained thermal): they run whenever they can, so the committable
    fleet only ever sees what remains. Negative values (renewables exceed
    demand) are meaningful — they mark hours where nothing needs to run —
    and are kept, not clipped.
    """
    demand = system.demand.loc[hours].sum(axis=1)
    free = pd.Series(0.0, index=hours)
    gens = system.generators
    for g in gens.index[~gens["needs_commitment"]]:
        capacity = gens.at[g, "max_mw"] * float(gens.at[g, "num_units"])
        if g in system.availability.columns:
            free = free + system.availability[g].loc[hours] * capacity
        else:
            free = free + capacity
    return (demand - free).rename("net_load_mw")


def smooth_net_load_with_storage(
    net_load: pd.Series, system: SystemData, day_hours: int = 24
) -> pd.Series:
    """Approximate how storage flattens the net load a committed fleet sees.

    Per day: shift each hour toward the daily mean, bounded by total
    storage power, then scale the shift down if it would cycle more energy
    than the fleet holds. A deliberate approximation — it ignores
    efficiency losses and cross-day carryover — good enough to stop the
    stack from committing a peaker for a spike the batteries will absorb.
    """
    power = float((system.storage["power_mw"]).sum())
    energy = float((system.storage["energy_mwh"]).sum())
    if power <= 0 or energy <= 0:
        return net_load

    values = net_load.to_numpy(dtype=float).copy()
    for start in range(0, len(values), day_hours):
        chunk = values[start : start + day_hours]
        shift = np.clip(chunk - chunk.mean(), -power, power)
        cycled = shift[shift > 0].sum()
        if cycled > energy:
            shift *= energy / cycled
        chunk -= shift
        values[start : start + day_hours] = chunk
    return pd.Series(values, index=net_load.index, name=net_load.name)


def merit_order(system: SystemData) -> pd.DataFrame:
    """Committed units ranked by all-in cost at full output.

    All-in = marginal + no-load spread over max output: the $/MWh a unit
    actually costs when run flat out, which is what a dispatch curve ranks
    by. Startup costs are deliberately excluded — they decide *when* to
    switch, not *who* is cheap, and the repair pass owns switching.
    """
    gens = system.generators
    committed = gens[gens["needs_commitment"]].copy()
    spread = committed["no_load_cost"] / committed["max_mw"].replace(0.0, np.nan)
    committed["all_in_cost"] = committed["marginal_cost"] + spread.fillna(0.0)
    return committed.sort_values("all_in_cost")


def _availability(system: SystemData, g: str, hours: list[int]) -> np.ndarray:
    if g in system.availability.columns:
        return system.availability[g].loc[hours].to_numpy(dtype=float)
    return np.ones(len(hours))


def import_capacity_mw(system: SystemData) -> dict[str, float]:
    """Most power each node could receive if every incident line ran inward.

    An optimistic bound (it ignores what the rest of the system can spare),
    which is the right side to err on: it is used to decide how much local
    generation a node *must* carry, so overstating imports understates the
    local floor, and the system-wide pass catches the remainder.
    """
    net = system.network
    limits = {n: 0.0 for n in system.nodes.index}
    for line, row in net.iterrows():
        delivered = row["capacity_mw"] * (1.0 - row["loss_factor"])
        limits[row["to_node"]] += delivered
        limits[row["from_node"]] += delivered
    return limits


def _free_generation_by_node(
    system: SystemData, hours: list[int]
) -> dict[str, np.ndarray]:
    """Hourly output of units that carry no commitment state, per node."""
    gens = system.generators
    free = {n: np.zeros(len(hours)) for n in system.nodes.index}
    for g in gens.index[~gens["needs_commitment"]]:
        capacity = gens.at[g, "max_mw"] * float(gens.at[g, "num_units"])
        free[gens.at[g, "node"]] += _availability(system, g, hours) * capacity
    return free


def local_need_mw(system: SystemData, hours: list[int]) -> dict[str, np.ndarray]:
    """Load each node must serve locally once imports are exhausted.

    ``demand - free local generation - import capacity``, floored at zero.
    Storage is deliberately ignored: treating it as unavailable commits a
    little extra thermal capacity, and against a VOLL of thousands per MWh
    that is the cheap mistake.
    """
    limits = import_capacity_mw(system)
    free = _free_generation_by_node(system, hours)
    need = {}
    for n in system.nodes.index:
        demand = system.demand[n].loc[hours].to_numpy(dtype=float)
        need[n] = np.maximum(0.0, demand - free[n] - limits[n])
    return need


def enforce_adequacy(
    commitment: pd.DataFrame,
    system: SystemData,
    hours: list[int],
    margin: float = 0.0,
) -> tuple[pd.DataFrame, int]:
    """Commit more units until every hour can physically serve its load.

    Two checks per hour, both on derated capacity: each node must cover the
    load it cannot import (:func:`local_need_mw`), and the system must cover
    total net load with ``margin`` to spare. Shortfalls are filled from the
    merit order — cheapest first, locally for a node shortfall.

    This is the invariant that makes fixing safe. Without it, any guess
    that under-commits gets "repaired" by the solver in the only way left
    to it: shedding load at VOLL. Returns the schedule and how many
    unit-hours it had to add.
    """
    ranked = merit_order(system)
    units = [g for g in ranked.index if g in commitment.columns]
    if not units:
        return commitment, 0

    gens = system.generators
    node_of = {g: gens.at[g, "node"] for g in units}
    counts = {g: float(gens.at[g, "num_units"]) for g in units}
    capacity = {g: gens.at[g, "max_mw"] for g in units}
    availability = {g: _availability(system, g, hours) for g in units}
    by_node: dict[str, list[str]] = {}
    for g in units:
        by_node.setdefault(node_of[g], []).append(g)

    local = local_need_mw(system, hours)
    system_need = net_load_mw(system, hours).to_numpy(dtype=float) * (1.0 + margin)

    values = commitment[units].to_numpy(dtype=float).copy()
    column = {g: i for i, g in enumerate(units)}
    added = 0

    def derated(g: str, i: int) -> float:
        return capacity[g] * availability[g][i]

    for i in range(len(hours)):
        # Node-local adequacy.
        for node, members in by_node.items():
            need = local[node][i]
            if need <= 0:
                continue
            covered = sum(
                derated(g, i) * values[i, column[g]] for g in members
            )
            for g in members:
                if covered >= need:
                    break
                if values[i, column[g]] >= counts[g] or availability[g][i] < _OUTAGE_TOL:
                    continue
                values[i, column[g]] = counts[g]
                covered += derated(g, i) * counts[g]
                added += 1

        # System-wide adequacy.
        need = system_need[i]
        if need <= 0:
            continue
        covered = sum(derated(g, i) * values[i, column[g]] for g in units)
        for g in units:
            if covered >= need:
                break
            if values[i, column[g]] >= counts[g] or availability[g][i] < _OUTAGE_TOL:
                continue
            values[i, column[g]] = counts[g]
            covered += derated(g, i) * counts[g]
            added += 1

    result = commitment.copy()
    result[units] = values
    return result, added


def repair_min_up_down(
    commitment: pd.DataFrame, system: SystemData, cyclic: bool
) -> pd.DataFrame:
    """Make a 0/1 schedule respect minimum up/down times, *monotonically*.

    Both repairs only ever commit **more**: off-gaps shorter than the down
    time are filled in, and on-runs shorter than the up time are extended
    forward rather than erased. That direction is not an aesthetic choice —
    it is what makes the repair safe. Erasing a short run removes capacity,
    and when the guess is then fixed into the model the missing capacity
    reappears as unserved energy priced at VOLL, which is how a heuristic
    turns a 135 s solve into a 500%-more-expensive one. Committing too much
    only wastes a little money and is caught by the completion's own LP.

    Because each pass can create new violations of the other kind, they are
    iterated to a fixed point (bounded; the schedule is monotone so it can
    only converge or saturate at always-on).
    """
    gens = system.generators
    repaired = commitment.copy()

    for g in repaired.columns:
        up = int(gens.at[g, "min_up_time_hr"])
        down = int(gens.at[g, "min_down_time_hr"])
        if up <= 1 and down <= 1:
            continue
        series = repaired[g].to_numpy(dtype=float).round().astype(int)
        for _ in range(8):
            filled = _fill_short_gaps(series, min_length=down, cyclic=cyclic)
            extended = _extend_short_runs(filled, min_length=up, cyclic=cyclic)
            if (extended == series).all():
                break
            series = extended
        repaired[g] = series.astype(float)
    return repaired


def _runs_of(series: np.ndarray, value: int, cyclic: bool) -> list[tuple[int, int]]:
    """(start, length) of every maximal run equal to ``value``.

    With ``cyclic`` a run spanning the seam is reported once, as the tail's
    start with the combined length.
    """
    n = len(series)
    runs, start = [], None
    for i in range(n):
        if series[i] == value:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, n - start))

    if cyclic and len(runs) >= 2:
        (first_start, first_len), (last_start, last_len) = runs[0], runs[-1]
        if first_start == 0 and last_start + last_len == n:
            runs = runs[1:-1] + [(last_start, last_len + first_len)]
    return runs


def _fill_short_gaps(series: np.ndarray, min_length: int, cyclic: bool) -> np.ndarray:
    """Commit through off-gaps shorter than the minimum down time.

    In an acyclic window a gap touching either end carries no in-window
    obligation — the model's lookback is truncated at the window edges — so
    those are left alone rather than needlessly committed.
    """
    if min_length <= 1:
        return series
    n = len(series)
    out = series.copy()
    if (series == 0).all():
        return out  # never on: nothing to keep running
    for start, length in _runs_of(series, 0, cyclic):
        if length >= min_length:
            continue
        if not cyclic and (start == 0 or start + length >= n):
            continue
        for k in range(length):
            out[(start + k) % n] = 1
    return out


def _extend_short_runs(series: np.ndarray, min_length: int, cyclic: bool) -> np.ndarray:
    """Stretch on-runs shorter than the minimum up time, forward in time.

    A run cut short by the end of an acyclic window is not a violation
    (the obligation simply runs past the horizon), and must not wrap into
    the window's first hours.
    """
    if min_length <= 1:
        return series
    n = len(series)
    out = series.copy()
    for start, length in _runs_of(series, 1, cyclic):
        if length >= min_length or length >= n:
            continue
        if not cyclic and start + length >= n:
            continue
        for k in range(length, min(min_length, n)):
            index = start + k
            if not cyclic and index >= n:
                break
            out[index % n] = 1
    return out


def derive_startup_shutdown(
    commitment: pd.DataFrame, cyclic: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """v/w frames implied by a commitment frame (first hour free if acyclic)."""
    u = commitment.to_numpy(dtype=float)
    previous = np.roll(u, 1, axis=0)
    if not cyclic:
        previous[0] = u[0]  # no transition charged at a free initial hour
    delta = u - previous
    v = np.where(delta > 0, delta, 0.0)
    w = np.where(delta < 0, -delta, 0.0)
    index, columns = commitment.index, commitment.columns
    return (
        pd.DataFrame(v, index=index, columns=columns),
        pd.DataFrame(w, index=index, columns=columns),
    )


def _require_no_clusters(system: SystemData, name: str) -> None:
    if (system.generators["num_units"] > 1).any():
        raise NotImplementedError(
            f"heuristic '{name}' does not support clustered units yet; "
            "use heuristic='lp' or unclustered data"
        )


# --------------------------------------------------------------------------
# Heuristic 1: priority-list stacking against net load
# --------------------------------------------------------------------------


def priority_list_guess(
    system: SystemData,
    config: RunConfig,
    hours: list[int],
    reserve_margin: float = 0.05,
    screen_margin: float = 0.15,
    smooth_storage: bool = True,
    screen_directions: str = "on",
) -> CommitmentGuess:
    """Stack the merit order against hourly net load.

    An hour's committed capacity must cover net load times
    ``1 + reserve_margin``. Outage hours inherit the previous hour's state
    (staying committed through an outage is free and avoids a restart),
    and the result is repaired to respect minimum up/down times and then
    made capacity-adequate.

    ``screen_directions`` controls what the guess will vouch for, and the
    default of ``"on"`` is a measured choice rather than caution for its
    own sake. A stack knows what capacity is *needed*; it does not know
    what is *economic*. Its "certainly on" verdict — needed even at zero
    margin — is sound, and fixing those cost +0.17% on the 52-unit test
    system. Its "certainly off" verdict is not: committing a unit can be
    optimal purely to displace dearer generation already running, which
    an adequacy test cannot see. Fixing those cost +2.36%, and widening
    the screen margin from 0.15 to 0.40 barely moved it (+2.37%) —
    confirming the criterion is wrong rather than merely loose. Use
    ``"both"`` only to reproduce that result; prefer the ``lp`` heuristic
    when you want trustworthy off-fixings.
    """
    start = time.perf_counter()
    _require_no_clusters(system, "priority")

    net = net_load_mw(system, hours)
    if smooth_storage:
        net = smooth_net_load_with_storage(net, system)
    ranked = merit_order(system)
    units = list(ranked.index)
    gens = system.generators
    capacity = {g: ranked.at[g, "max_mw"] for g in units}
    node_of = {g: gens.at[g, "node"] for g in units}
    availability = {g: _availability(system, g, hours) for g in units}
    by_node: dict[str, list[str]] = {}
    for g in units:
        by_node.setdefault(node_of[g], []).append(g)
    local = local_need_mw(system, hours)
    cyclic = config.cyclic and config.window_hours is None

    def stack(margin: float) -> pd.DataFrame:
        """Cover each node's un-importable load, then top up system-wide.

        Stacking only against system totals is what made an earlier version
        of this heuristic unsafe: a transport model can have ample capacity
        and still strand a node behind its line limits.
        """
        frame = pd.DataFrame(0.0, index=hours, columns=units)
        system_need = net.to_numpy(dtype=float) * (1.0 + margin)
        for i, t in enumerate(hours):
            for node, members in by_node.items():
                need = local[node][i] * (1.0 + margin)
                covered = 0.0
                for g in members:
                    if covered >= need:
                        break
                    if availability[g][i] < _OUTAGE_TOL:
                        continue
                    frame.at[t, g] = 1.0
                    covered += capacity[g] * availability[g][i]

            need = system_need[i]
            if need <= 0:
                continue
            covered = sum(
                capacity[g] * availability[g][i] * frame.at[t, g] for g in units
            )
            for g in units:
                if covered >= need:
                    break
                if availability[g][i] < _OUTAGE_TOL or frame.at[t, g] > 0:
                    continue
                frame.at[t, g] = 1.0
                covered += capacity[g] * availability[g][i]
        return frame

    guess = stack(reserve_margin)
    generous = stack(reserve_margin + screen_margin)
    lean = stack(0.0)

    # Outage persistence: keep the pre-outage state through unavailable hours.
    for g in units:
        available = availability[g] >= _OUTAGE_TOL
        values = guess[g].to_numpy(dtype=float)
        for i in range(len(values)):
            if not available[i]:
                values[i] = values[i - 1] if i > 0 else values[i]
        guess[g] = values

    guess = repair_min_up_down(guess, system, cyclic)
    guess, added = enforce_adequacy(guess, system, hours, margin=reserve_margin)
    if added:
        guess = repair_min_up_down(guess, system, cyclic)

    never_needed = generous.sum(axis=0) == 0.0
    always_needed = pd.Series(
        {
            g: bool(
                (lean[g].to_numpy() + (availability[g] < _OUTAGE_TOL) >= 1.0).all()
            )
            for g in units
        }
    )
    certain = pd.DataFrame(False, index=hours, columns=units)
    screen_off = screen_directions in ("both", "off")
    screen_on = screen_directions in ("both", "on")
    for g in units:
        if screen_off and never_needed[g] and guess[g].sum() == 0.0:
            certain[g] = True  # off everywhere, with margin to spare
        elif screen_on and always_needed[g] and (guess[g] == 1.0).all():
            certain[g] = True  # on everywhere, even with no margin

    return CommitmentGuess(
        name="priority",
        commitment=guess,
        certain=certain,
        build_seconds=time.perf_counter() - start,
        notes={
            "reserve_margin": reserve_margin,
            "screen_margin": screen_margin,
            "adequacy_additions": added,
            "certain_share": float(certain.to_numpy().mean()),
        },
    )


# --------------------------------------------------------------------------
# Heuristic 2: similar days share a commitment
# --------------------------------------------------------------------------


def similar_days_guess(
    system: SystemData,
    config: RunConfig,
    hours: list[int],
    num_representatives: int | None = None,
    representative_gap: float = 0.005,
    representative_time_limit: float = 120.0,
) -> CommitmentGuess:
    """Solve a few representative days, reuse their schedules on lookalikes.

    Days are clustered on their 24-hour net-load shape (k-means on the
    profile, deterministic seed); the day nearest each centroid is solved
    as a small cyclic UC and every member of the cluster inherits that
    schedule. Seams between days are then repaired for min up/down times.
    Confidence: an entry is certain only when *every* representative
    schedule agrees on that unit holding that state all day — the
    cross-cluster version of "this unit never makes a decision".
    """
    start = time.perf_counter()
    _require_no_clusters(system, "similar_days")

    net = net_load_mw(system, hours).to_numpy(dtype=float)
    num_days = len(hours) // 24
    if num_days < 1:
        raise ValueError("similar_days needs at least one full day")
    tail = len(hours) - num_days * 24

    profiles = net[: num_days * 24].reshape(num_days, 24)
    scale = profiles.std() or 1.0
    normalized = profiles / scale

    if num_representatives is None:
        num_representatives = max(2, min(8, 2 + num_days // 7))
    k = min(num_representatives, num_days)

    assignments, centroids = _kmeans(normalized, k)

    # The concrete day nearest each centroid becomes the one we solve.
    medoids = []
    for c in range(k):
        members = np.flatnonzero(assignments == c)
        if len(members) == 0:
            continue
        distances = ((normalized[members] - centroids[c]) ** 2).sum(axis=1)
        medoids.append(int(members[np.argmin(distances)]))

    day_config = RunConfig(
        unit_commitment=True,
        tight_generation_limits=config.tight_generation_limits,
        tight_ramp_limits=config.tight_ramp_limits,
    )
    settings = SolverSettings(
        mip_gap=representative_gap, time_limit=representative_time_limit
    )
    schedules: dict[int, pd.DataFrame] = {}
    for day in medoids:
        day_hours = hours[day * 24 : (day + 1) * 24]
        day_model = build_model(system, day_config, day_hours)
        HighsSession(day_model).solve(settings)
        schedules[day] = pd.DataFrame(
            {g: [round(day_model.u[g, t].value) for t in day_hours] for g in day_model.G_UC},
            index=range(24),
            dtype=float,
        )

    medoid_of_cluster = {assignments[d]: d for d in medoids}
    units = list(next(iter(schedules.values())).columns)
    frames = []
    for day in range(num_days):
        source = schedules[medoid_of_cluster[assignments[day]]]
        frame = source.copy()
        frame.index = hours[day * 24 : (day + 1) * 24]
        frames.append(frame)
    if tail:
        # Partial trailing day: nearest representative by truncated distance.
        stub = net[num_days * 24 :] / scale
        best = min(
            medoids,
            key=lambda d: float(((normalized[d][: len(stub)] - stub) ** 2).sum()),
        )
        frame = schedules[best].iloc[:tail].copy()
        frame.index = hours[num_days * 24 :]
        frames.append(frame)

    guess = pd.concat(frames, axis=0)
    cyclic = config.cyclic and config.window_hours is None
    guess = repair_min_up_down(guess, system, cyclic)
    # A representative day's schedule is optimal for *its* load, not for
    # the peakier days that inherit it, so adequacy is re-checked here.
    guess, added = enforce_adequacy(guess, system, hours)
    if added:
        guess = repair_min_up_down(guess, system, cyclic)

    all_schedules = list(schedules.values())
    certain = pd.DataFrame(False, index=guess.index, columns=units)
    for g in units:
        states = {frame[g].iloc[0] for frame in all_schedules}
        constant = all((frame[g] == frame[g].iloc[0]).all() for frame in all_schedules)
        if constant and len(states) == 1 and (guess[g] == guess[g].iloc[0]).all():
            certain[g] = True

    return CommitmentGuess(
        name="similar_days",
        commitment=guess,
        certain=certain,
        build_seconds=time.perf_counter() - start,
        notes={
            "days": num_days,
            "representatives": len(medoids),
            "adequacy_additions": added,
            "certain_share": float(certain.to_numpy().mean()),
        },
    )


def _kmeans(
    points: np.ndarray, k: int, iterations: int = 25, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centroids = points[rng.choice(len(points), size=k, replace=False)]
    assignments = np.zeros(len(points), dtype=int)
    for _ in range(iterations):
        distances = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_assignments = distances.argmin(axis=1)
        if (new_assignments == assignments).all():
            break
        assignments = new_assignments
        for c in range(k):
            members = points[assignments == c]
            if len(members):
                centroids[c] = members.mean(axis=0)
    return assignments, centroids


# --------------------------------------------------------------------------
# Heuristic 3: the LP relaxation as an oracle
# --------------------------------------------------------------------------


def lp_relaxation_guess(
    system: SystemData,
    config: RunConfig,
    hours: list[int],
    initial: "InitialState | None" = None,
) -> CommitmentGuess:
    """Round the LP relaxation's commitment; trust the values it resolved.

    The strongest oracle available (>99% agreement with the MIP optimum on
    everything it calls integral) but also the dearest: it needs a
    full-horizon LP solve, so at annual scale it costs what the LP costs.
    Confidence is exactly integrality: fractional values — the genuinely
    contested commitments — stay uncertain.
    """
    start = time.perf_counter()
    lp_config = RunConfig(
        unit_commitment=False,
        cyclic=config.cyclic,
        tight_generation_limits=config.tight_generation_limits,
        tight_ramp_limits=config.tight_ramp_limits,
    )
    # ``initial`` must be forwarded: build_model reads cyclic-ness from it
    # (``cyclic = initial is None``), so omitting it would relax the oracle
    # onto a *different* problem than the model this guess has to seed --
    # and a guess that ignores carried min up/down obligations makes the
    # completion infeasible rather than merely inaccurate.
    lp_model = build_model(system, lp_config, hours, initial)
    lp_info, _ = HighsSession(lp_model).solve(SolverSettings())

    units = list(lp_model.G_UC)
    values = pd.DataFrame(
        {g: [lp_model.u[g, t].value for t in hours] for g in units},
        index=hours,
        dtype=float,
    )
    certain = (values - values.round()).abs() < _TOL

    # Rounding a 0.49 down silently deletes that unit's capacity, so the
    # rounded schedule gets the same adequacy guarantee as the others.
    # Cyclic-ness is read the same way build_model reads it, so the repair
    # cannot disagree with the model about whether hour 0 wraps.
    cyclic = initial is None
    rounded = repair_min_up_down(values.round(), system, cyclic)
    rounded, added = enforce_adequacy(rounded, system, hours)
    if added:
        rounded = repair_min_up_down(rounded, system, cyclic)
    # Anything adequacy or repair had to move is no longer an LP verdict.
    certain = certain & (rounded == values.round())

    return CommitmentGuess(
        name="lp",
        commitment=rounded,
        certain=certain,
        relaxation=values,
        build_seconds=time.perf_counter() - start,
        notes={
            "adequacy_additions": added,
            "certain_share": float(certain.to_numpy().mean()),
            # The relaxation's objective is a valid lower bound on the MIP,
            # so this is the per-run integrality gap reference.
            "lp_objective": lp_info.objective,
            "lp_seconds": lp_info.solve_seconds,
        },
    )


# --------------------------------------------------------------------------
# Dispatcher and completion
# --------------------------------------------------------------------------


def build_guess(
    system: SystemData,
    config: RunConfig,
    hours: list[int],
    initial: InitialState | None = None,
) -> CommitmentGuess:
    """Build the guess selected by ``config.heuristic``.

    ``initial`` is the state the seeded model starts from. Only ``lp``
    can honour it (it solves the same model relaxed); the structural
    heuristics reason about net load and merit order and have no way to
    represent a carried min up/down obligation, so pinning them against a
    concrete initial state would produce a guess the completion cannot
    satisfy. They are refused rather than allowed to fail downstream.
    """
    options = dict(config.heuristic_options)
    carries_state = initial is not None and (
        bool(initial.commitment) or bool(initial.state_hours)
    )
    if config.heuristic == "lp":
        return lp_relaxation_guess(system, config, hours, initial=initial, **options)
    if carries_state:
        raise NotImplementedError(
            f"heuristic '{config.heuristic}' cannot honour a carried initial "
            "state; use heuristic='lp' for chained or rolling-state runs"
        )
    if config.heuristic == "priority":
        return priority_list_guess(system, config, hours, **options)
    if config.heuristic == "similar_days":
        return similar_days_guess(system, config, hours, **options)
    raise ValueError(f"unknown heuristic '{config.heuristic}' (available: {HEURISTICS})")


def complete_solution(
    model: pyo.ConcreteModel,
    guess: CommitmentGuess,
    settings: SolverSettings | None = None,
    session: HighsSession | None = None,
) -> tuple[SolveInfo, bool]:
    """Fill every continuous variable by solving with the guess pinned.

    Fixes ``u`` to the guess, solves (all integers fixed, so effectively an
    LP), and lets ``load_vars`` deposit a complete solution on the model —
    which is what the appsi warm start needs. If the guess over-commits
    (fixed minimum generation exceeds what the system can absorb) the
    completion is infeasible; the minimum-output rows are then relaxed for
    the completion only, flagged in the second return value. ``u`` is
    unfixed before returning either way.

    Pass the ``session`` that will run the main solve: the completion then
    shares its translated model, and the follow-up solve pushes only the
    unfixing as an incremental update instead of re-translating.
    """
    settings = settings or SolverSettings(time_limit=600)
    for (g, t), var in model.u.items():
        var.fix(float(guess.commitment.at[t, g]))

    session = session or HighsSession(model)
    used_fallback = False
    try:
        try:
            info, _ = session.solve(settings)
        except RuntimeError:
            used_fallback = True
            model.min_output.deactivate()
            try:
                info, _ = session.solve(settings)
            finally:
                model.min_output.activate()
    finally:
        for var in model.u.values():
            var.unfix()
    return info, used_fallback


def apply_fixing(model: pyo.ConcreteModel, guess: CommitmentGuess, mode: str) -> int:
    """Fix commitment variables per the delivery mode; returns how many."""
    if mode == "off":
        return 0
    fixed = 0
    for (g, t), var in model.u.items():
        if mode == "aggressive" or bool(guess.certain.at[t, g]):
            var.fix(float(guess.commitment.at[t, g]))
            fixed += 1
    return fixed


def match_fraction(model: pyo.ConcreteModel, guess: CommitmentGuess) -> float:
    """Share of commitment variables where the solve agreed with the guess."""
    total = agree = 0
    for (g, t), var in model.u.items():
        if var.value is None:
            continue
        total += 1
        if abs(round(var.value) - round(float(guess.commitment.at[t, g]))) < 0.5:
            agree += 1
    return agree / total if total else float("nan")
