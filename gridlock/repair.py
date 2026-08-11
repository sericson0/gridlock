"""Repair a commitment guess against the completion LP's own verdict.

Everything upstream of this module guesses in the dark. ``round()``,
:func:`~gridlock.heuristics.repair_min_up_down` and
:func:`~gridlock.heuristics.enforce_adequacy` reason about *estimated*
capacity — per hour, per node, bounding imports by total incident line
rating — and all three are monotone, so the schedule they hand over is
systematically over-committed and still occasionally infeasible. The
completion LP has neither problem: it solves the actual dispatch with the
guess pinned, so it knows exactly which node-hours cannot be served and
exactly what every committed run earns. Two passes read that verdict back
into the schedule.

**Shed repair** (plan item 1f). Week44 of the RTS-GMLC baseline sheds
259 MWh across six hours — $2.59 M at VOLL, 55 of its 56 points of hot-start
margin — from a schedule that passed ``enforce_adequacy`` and matches the
MIP optimum on 98.3% of entries. The static test cleared it because it
bounds imports optimistically and ignores ramps, simultaneous network flow
and storage. Here the shed is read off the completion, more capacity is
committed at the offending node-hours (cheapest available locally first),
and the completion is re-run. Costs one LP per round.

**Decommitment** (plan item 1a). Nothing in the pipeline ever removes a
commitment, and across the 12-month study the delivered guess carries 2.7%
more committed unit-hours than the optimum while starting 4.6% *fewer*
units — extended runs, not extra starts, so the waste is no-load plus
min-output fuel. Every committed hour is priced at the completion's own
nodal duals (it costs no-load and earns ``(price - marginal cost) x
output``), and the hours that lose money are proposed for decommitment
worst-first, re-completed, and kept only if the result is feasible and
cheaper.

The granularity is what the measurement dictates. Ranking whole runs and
dropping the worst — the obvious reading of "over-committed" — was tried
first and refused every time on RTS-GMLC: a run that loses money *on
average* over 168 hours is still load-carrying in its peak hours, so
removing it shed 100–5,000 MWh. The excess is in the shoulders, which is
exactly what "extended runs, not extra starts" says, so a cut grows inward
from one end of a run and stops at the first hour that is either profitable
or physically hard to replace.

The two pull in opposite directions, so the order is fixed: shed repair
first, then decommitment, which never accepts a schedule that sheds more
than the repaired one it started from. Both passes are greedy and bounded
by an explicit budget of LP solves, and both are *tested* rather than
trusted — a candidate schedule is accepted only on the completion's
evidence, so a mis-ranked candidate costs one LP and nothing else.

Completions here solve the model with ``unit_commitment=False``. With every
``u`` fixed to an integer that is the same problem the MIP-mode completion
solves (verified equal to 1e-9 on the example system), but HiGHS sees a
pure LP, which is faster and — the reason it matters — carries duals. Those
nodal prices are how a committed hour is valued, with the caveat that
energy prices in a UC never recover no-load cost, so they are a ranking
signal and not a verdict (see :func:`_replaceable`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import RunConfig, SolverSettings
from .data import SystemData
from .heuristics import (
    _OUTAGE_TOL,
    CommitmentGuess,
    _availability,
    _runs_of,
    merit_order,
    repair_min_up_down,
)
from .model import InitialState, build_model
from .scoring import shedding_hours
from .solver import HighsSession

# MWh below this is solver noise, not unserved energy.
_SHED_TOL = 1e-4
# A candidate must beat the incumbent by more than solver noise to be kept.
_IMPROVEMENT_TOL = 1e-9


@dataclass
class _Completion:
    """What one fixed-commitment LP says about a candidate schedule."""

    objective: float
    shed_mwh: float
    shed_rows: list[tuple]
    dispatch: np.ndarray  # hours x committed units, MW
    prices: np.ndarray | None  # hours x nodes, $/MWh, None if duals unavailable


class _Completer:
    """One LP-mode model, re-pinned and re-solved for each candidate schedule.

    The model is built once and bound to one :class:`HighsSession`, so every
    candidate after the first is an incremental bound change on a translated
    model with a warm basis — which is what makes a test-per-candidate loop
    affordable at all.
    """

    def __init__(
        self,
        system: SystemData,
        config: RunConfig,
        hours: list[int],
        initial: InitialState | None,
        settings: SolverSettings | None,
    ) -> None:
        lp_config = RunConfig(
            unit_commitment=False,
            cyclic=config.cyclic,
            tight_generation_limits=config.tight_generation_limits,
            tight_ramp_limits=config.tight_ramp_limits,
            # The completion has to price unserved energy exactly as the MIP
            # does, or the shed it reports is not the shed being repaired.
            voll=config.voll,
        )
        self.model = build_model(system, lp_config, hours, initial)
        self.session = HighsSession(self.model)
        self.settings = settings or SolverSettings(time_limit=600)
        self.hours = hours
        self.units = list(self.model.G_UC)
        self.nodes = list(self.model.N)
        self.solves = 0

    def complete(self, commitment: pd.DataFrame, want_duals: bool = False):
        """Pin ``commitment``, solve, and read the result back. None if infeasible.

        Infeasibility is a verdict, not an error: an over-committed candidate
        whose minimum outputs cannot be absorbed is simply rejected. That is
        why this does not reuse :func:`~gridlock.heuristics.complete_solution`,
        which relaxes the minimum-output rows to keep a warm start alive —
        the right answer when the completion is the last word, the wrong one
        when it is a test.
        """
        values = commitment.loc[self.hours, self.units].to_numpy(dtype=float)
        position = {t: i for i, t in enumerate(self.hours)}
        column = {g: j for j, g in enumerate(self.units)}
        for (g, t), var in self.model.u.items():
            var.fix(float(values[position[t], column[g]]))
        try:
            info, duals = self.session.solve(self.settings, want_duals=want_duals)
        except RuntimeError:
            return None
        finally:
            for var in self.model.u.values():
                var.unfix()
        self.solves += 1

        dispatch = np.array(
            [[self.model.p[g, t].value or 0.0 for g in self.units] for t in self.hours]
        )
        prices = None
        if want_duals and duals:
            prices = np.array(
                [
                    [duals.get(self.model.load_balance[n, t], 0.0) for n in self.nodes]
                    for t in self.hours
                ]
            )
        rows = shedding_hours(self.model)
        return _Completion(
            objective=info.objective,
            shed_mwh=float(sum(mw for _, _, mw in rows)),
            shed_rows=rows,
            dispatch=dispatch,
            prices=prices,
        )


# --------------------------------------------------------------------------
# Pass 1: commit against the shed the completion actually reports
# --------------------------------------------------------------------------


def _neighbours(system: SystemData) -> dict[str, list[str]]:
    """Nodes one line away from each node."""
    adjacency: dict[str, set[str]] = {n: set() for n in system.nodes.index}
    for _, row in system.network.iterrows():
        adjacency[row["from_node"]].add(row["to_node"])
        adjacency[row["to_node"]].add(row["from_node"])
    return {n: sorted(members) for n, members in adjacency.items()}


def _commit_against_shed(
    values: np.ndarray,
    system: SystemData,
    hours: list[int],
    shed_rows: list[tuple],
    span: int,
    cyclic: bool,
    ranked: list[str],
    column: dict[str, int],
) -> int:
    """Add capacity at the node-hours the completion could not serve.

    Cheapest available unit at the shedding node first, then one hop out,
    then anywhere — a node that sheds behind a congested line is helped by
    local capacity, but a node with none left is helped by a neighbour that
    can push power in, and refusing to look is how a repairable schedule
    gets declared unrepairable.

    ``span`` widens the committed block to ``[t - span, t + span]``. Not all
    shed is a capacity shortfall: a unit committed only in the shedding hour
    starts at its startup ramp and may still not reach the output the hour
    needs, so successive rounds commit earlier as well. Returns unit-hours
    added.
    """
    gens = system.generators
    node_of = {g: gens.at[g, "node"] for g in ranked}
    counts = {g: float(gens.at[g, "num_units"]) for g in ranked}
    capacity = {g: float(gens.at[g, "max_mw"]) for g in ranked}
    availability = {g: _availability(system, g, hours) for g in ranked}
    position = {t: i for i, t in enumerate(hours)}
    neighbours = _neighbours(system)
    horizon = len(hours)

    at_node: dict[str, list[str]] = {}
    for g in ranked:
        at_node.setdefault(node_of[g], []).append(g)

    def block(i: int) -> list[int]:
        if cyclic:
            return sorted({(i + k) % horizon for k in range(-span, span + 1)})
        return [k for k in range(i - span, i + span + 1) if 0 <= k < horizon]

    added = 0
    for hour, node, shortfall in shed_rows:
        i = position.get(hour)
        if i is None:
            continue
        tiers = [
            at_node.get(node, []),
            [g for other in neighbours.get(node, []) for g in at_node.get(other, [])],
            ranked,
        ]
        need = shortfall
        for tier in tiers:
            # Merit order within the tier: the tiers are built from a ranked
            # list, but the neighbour tier concatenates several nodes.
            for g in sorted(tier, key=ranked.index):
                if need <= 0:
                    break
                j = column[g]
                if values[i, j] >= counts[g] or availability[g][i] < _OUTAGE_TOL:
                    continue
                for k in block(i):
                    if availability[g][k] < _OUTAGE_TOL:
                        continue
                    if values[k, j] < counts[g]:
                        values[k, j] += 1.0
                        added += 1
                need -= capacity[g] * availability[g][i]
            if need <= 0:
                break
    return added


def _repair_shed(
    system: SystemData,
    commitment: pd.DataFrame,
    completion: _Completion,
    completer: _Completer,
    hours: list[int],
    cyclic: bool,
    rounds: int,
    notes: dict,
) -> tuple[pd.DataFrame, _Completion]:
    """Commit against the reported shed until it is gone or stops falling."""
    ranked = [g for g in merit_order(system).index if g in commitment.columns]
    column = {g: j for j, g in enumerate(commitment.columns)}
    added_total = 0
    used = 0

    for round_index in range(rounds):
        if completion.shed_mwh <= _SHED_TOL:
            break
        values = commitment.to_numpy(dtype=float).copy()
        added = _commit_against_shed(
            values,
            system,
            hours,
            completion.shed_rows,
            span=round_index,
            cyclic=cyclic,
            ranked=ranked,
            column=column,
        )
        if not added:
            break  # nothing left to commit at the failing node-hours
        candidate = pd.DataFrame(values, index=commitment.index, columns=commitment.columns)
        candidate = repair_min_up_down(candidate, system, cyclic)
        # Duals cost nothing extra here and save the decommit pass having to
        # re-solve this very schedule to get the prices it ranks with.
        trial = completer.complete(candidate, want_duals=True)
        used += 1
        # Commitments that do not buy back shed are pure no-load cost, so a
        # round that fails to improve is reverted rather than kept.
        if trial is None or trial.shed_mwh >= completion.shed_mwh - _SHED_TOL:
            break
        commitment, completion = candidate, trial
        added_total += added

    notes["repair_shed_rounds"] = used
    notes["repair_added_unit_hours"] = added_total
    return commitment, completion


# --------------------------------------------------------------------------
# Pass 2: drop the runs that lose money at the completion's own prices
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Cut:
    """Hours proposed for decommitment, and what dropping them looks worth."""

    unit: str
    run: tuple[str, int, int]  # the run it came from: unit, start, length
    positions: tuple[int, ...]
    saving: float

    @property
    def key(self) -> tuple:
        return (self.unit, self.positions)


def _replaceable(
    system: SystemData,
    commitment: pd.DataFrame,
    completion: _Completion,
    hours: list[int],
    units: list[str],
) -> np.ndarray:
    """hours x units: could the rest of the committed fleet cover this output?

    A screen, not a proof. It compares a unit's output against the headroom
    the *other* committed units have in that hour, each capped at its own
    ramp rate because the pick-up has to happen within the hour. It ignores
    the network (optimistic) and storage (pessimistic), so it neither
    guarantees nor forbids anything — the re-completion still decides. What
    it does is stop the ranking proposing cuts that must obviously shed,
    which on RTS-GMLC is otherwise most of them: LP energy prices never
    recover a committed unit's no-load cost (the standard non-convexity), so
    priced at duals alone almost every run in the fleet looks like waste.
    """
    gens = system.generators
    dispatch = completion.dispatch
    schedule = commitment.loc[hours, units].to_numpy(dtype=float)
    ceiling = np.array(
        [
            _availability(system, g, hours) * float(gens.at[g, "max_mw"])
            for g in units
        ]
    ).T
    ramp = np.array([float(gens.at[g, "ramp_rate_mw_per_hr"]) for g in units])
    headroom = np.minimum(np.maximum(ceiling * schedule - dispatch, 0.0), ramp)
    return headroom.sum(axis=1, keepdims=True) - headroom >= dispatch


def _rank_cuts(
    system: SystemData,
    commitment: pd.DataFrame,
    completion: _Completion,
    hours: list[int],
    cyclic: bool,
    units: list[str],
    nodes: list[str],
    rejected: set,
) -> list[_Cut]:
    """Decommitment candidates, best-looking first.

    An hour of commitment costs its no-load charge and earns the value of
    the energy it produced above what that energy cost to make, priced at
    the LP's nodal duals. A cut collects the hours at one end of a run that
    both lose money on that trade and pass :func:`_replaceable`, stopping at
    the first hour that fails either — so a cut is the *shoulder* of a run,
    which is what the measured defect asks for: the guess starts 4.6% fewer
    units than the optimum while carrying 2.7% more committed hours, so the
    excess is extended runs, not spurious ones. A whole run is offered as a
    cut too (it also saves the startup), but only when every one of its
    hours qualifies.

    Growing a cut hour by hour rather than taking the most profitable prefix
    is deliberate and was measured: maximising cumulative saving happily
    eats through a run's valuable peak hours to reach cheap ones beyond
    them, and every such cut was refused on RTS-GMLC — 144 hours of a
    168-hour run, shedding 2,850 MWh.
    """
    gens = system.generators
    node_index = {n: i for i, n in enumerate(nodes)}
    unit_column = {g: j for j, g in enumerate(units)}
    horizon = len(hours)
    prices = completion.prices
    replaceable = _replaceable(system, commitment, completion, hours, units)
    cuts: list[_Cut] = []

    for g in commitment.columns:
        if g not in unit_column:
            continue
        # A cluster commits a count, so "the run" is not one unit's history
        # and cutting it would drop the whole fleet at that row. Left alone.
        if float(gens.at[g, "num_units"]) > 1:
            continue
        series = commitment[g].to_numpy(dtype=float).round().astype(int)
        if not series.any():
            continue
        no_load = float(gens.at[g, "no_load_cost"])
        startup = float(gens.at[g, "startup_cost"])
        shutdown = float(gens.at[g, "shutdown_cost"])
        marginal = float(gens.at[g, "marginal_cost"])
        min_up = max(1, int(gens.at[g, "min_up_time_hr"]))
        min_down = max(1, int(gens.at[g, "min_down_time_hr"]))
        node = node_index[gens.at[g, "node"]]
        column = unit_column[g]

        def hour_saving(i: int, node=node, column=column, no_load=no_load,
                        marginal=marginal) -> float:
            if prices is None:
                return no_load
            return no_load - (prices[i, node] - marginal) * completion.dispatch[i, column]

        def droppable(i: int, column=column) -> bool:
            return bool(replaceable[i, column]) and hour_saving(i) > 0.0

        for start, length in _runs_of(series, 1, cyclic):
            positions = tuple((start + k) % horizon for k in range(length))
            run = (g, start, length)
            candidates = []

            if all(droppable(i) for i in positions):
                whole = sum(hour_saving(i) for i in positions)
                # An always-on unit never starts inside the horizon, and in
                # an acyclic window the first hour is free (no transition is
                # charged there) — neither carries a startup to save.
                if length < horizon and (cyclic or start > 0):
                    whole += startup
                if length < horizon and (cyclic or start + length < horizon):
                    whole += shutdown
                candidates.append(_Cut(g, run, positions, whole))

            # Trimming leaves the run shorter, so it must stay at or above
            # the minimum up time; and a trim that cuts into an always-on
            # unit opens a *new* off-gap, which owes the minimum down time.
            longest = length - min_up
            shortest = min_down if length >= horizon else 1
            for ordered in (positions, positions[::-1]):
                saving = 0.0
                taken = 0
                while taken < longest and droppable(ordered[taken]):
                    saving += hour_saving(ordered[taken])
                    taken += 1
                if taken >= shortest:
                    candidates.append(
                        _Cut(g, run, tuple(sorted(ordered[:taken])), saving)
                    )

            cuts.extend(cut for cut in candidates if cut.key not in rejected)

    cuts.sort(key=lambda cut: -cut.saving)
    return cuts


def _batch(cuts: list[_Cut], size: int) -> list[_Cut]:
    """The best ``size`` cuts, at most one per run.

    Two cuts on one run would compose into a shortening neither of them
    checked against the minimum up time.
    """
    chosen: list[_Cut] = []
    seen: set = set()
    for cut in cuts:
        if cut.run in seen:
            continue
        seen.add(cut.run)
        chosen.append(cut)
        if len(chosen) == size:
            break
    return chosen


def _without(commitment: pd.DataFrame, cuts: list[_Cut]) -> pd.DataFrame:
    candidate = commitment.copy()
    values = candidate.to_numpy(dtype=float)
    column = {g: j for j, g in enumerate(candidate.columns)}
    for cut in cuts:
        for i in cut.positions:
            values[i, column[cut.unit]] = 0.0
    candidate.loc[:, :] = values
    return candidate


def _decommit(
    system: SystemData,
    commitment: pd.DataFrame,
    completion: _Completion,
    completer: _Completer,
    hours: list[int],
    cyclic: bool,
    budget: int,
    max_stall: int,
    notes: dict,
) -> tuple[pd.DataFrame, _Completion]:
    """Take the cuts that test out, batch at a time.

    Each test costs an LP, so the batch size doubles on every acceptance and
    halves on every refusal — a schedule with sixteen droppable shoulders
    clears in five solves, and one whose ranking is wrong degrades to
    one-at-a-time. ``max_stall`` single cuts refused in a row ends the pass:
    without it a guess with nothing left to give spends its whole budget
    proving so, which is exactly the case on a week that needed only the
    shed repair.
    """
    baseline_shed = completion.shed_mwh
    rejected: set = set()
    accepted = dropped_hours = 0
    used = stalled = 0
    batch_size = 1

    while used < budget and stalled < max_stall:
        cuts = [
            cut
            for cut in _rank_cuts(
                system,
                commitment,
                completion,
                hours,
                cyclic,
                completer.units,
                completer.nodes,
                rejected,
            )
            if cut.saving > 0.0
        ]
        batch = _batch(cuts, batch_size)
        if not batch:
            break
        candidate = _without(commitment, batch)
        trial = completer.complete(candidate, want_duals=True)
        used += 1

        cheaper = trial is not None and trial.objective < completion.objective * (
            1.0 - _IMPROVEMENT_TOL
        )
        # Decommitment must never buy its saving with unserved energy: the
        # shed repair that ran first is what the margin is being rescued
        # from, and VOLL would swamp any no-load cost it recovers.
        feasible = trial is not None and trial.shed_mwh <= baseline_shed + _SHED_TOL
        if cheaper and feasible:
            commitment, completion = candidate, trial
            accepted += len(batch)
            dropped_hours += sum(len(cut.positions) for cut in batch)
            batch_size *= 2
            stalled = 0
            continue
        if len(batch) == 1:
            # One cut, tested and refused: remember it, or the next ranking
            # puts it straight back at the top of the list.
            rejected.add(batch[0].key)
            stalled += 1
        batch_size = max(1, len(batch) // 2)

    notes["repair_decommit_solves"] = used
    notes["repair_decommitted_cuts"] = accepted
    notes["repair_decommitted_unit_hours"] = dropped_hours
    return commitment, completion


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def repair_guess(
    system: SystemData,
    config: RunConfig,
    hours: list[int],
    guess: CommitmentGuess,
    initial: InitialState | None = None,
    shed_rounds: int = 3,
    decommit_solves: int = 24,
    decommit_stall: int = 4,
    decommit: bool = True,
    settings: SolverSettings | None = None,
) -> CommitmentGuess:
    """Repair ``guess`` against its own completion: shed first, then waste.

    Returns a new guess. ``certain`` is masked wherever the schedule moved —
    an entry the completion overruled is no longer the heuristic's verdict,
    and a screen that pinned it would hand the MIP a schedule it must repair
    by shedding.

    ``shed_rounds`` and ``decommit_solves`` bound the LP spend: at most
    ``1 + shed_rounds + decommit_solves`` completions, each of which is a
    warm-basis re-solve of one translated model, and the decommit pass gives
    up early after ``decommit_stall`` single cuts are refused in a row. Set
    ``decommit=False`` to run the safety pass alone.

    If the incoming guess cannot be completed at all the guess is returned
    untouched: that is the case
    :func:`~gridlock.heuristics.complete_solution` handles by relaxing the
    minimum-output rows, and second-guessing it here would only hide it.
    """
    start = time.perf_counter()
    hours = list(hours)
    cyclic = initial is None
    completer = _Completer(system, config, hours, initial, settings)
    commitment = guess.commitment.copy()

    completion = completer.complete(commitment, want_duals=True)
    if completion is None:
        return guess

    notes = dict(guess.notes)
    notes["repair_shed_before"] = completion.shed_mwh
    notes["repair_objective_before"] = completion.objective

    commitment, completion = _repair_shed(
        system, commitment, completion, completer, hours, cyclic, shed_rounds, notes
    )
    if decommit:
        if completion.prices is None:
            # The shed pass does not ask for duals, so the ranking would be
            # pure fixed cost with no offsetting value -- it would rank a
            # baseload unit's week-long run above a peaker's idle hour.
            completion = completer.complete(commitment, want_duals=True) or completion
        commitment, completion = _decommit(
            system,
            commitment,
            completion,
            completer,
            hours,
            cyclic,
            decommit_solves,
            decommit_stall,
            notes,
        )

    moved = commitment.round() != guess.commitment.round()
    seconds = time.perf_counter() - start
    notes["repair_shed_after"] = completion.shed_mwh
    notes["repair_objective_after"] = completion.objective
    notes["repair_moved_entries"] = int(moved.to_numpy().sum())
    notes["repair_solves"] = completer.solves
    notes["repair_seconds"] = seconds

    return CommitmentGuess(
        name=guess.name,
        commitment=commitment,
        certain=guess.certain & ~moved,
        build_seconds=guess.build_seconds + seconds,
        notes=notes,
        relaxation=guess.relaxation,
        soft=None if guess.soft is None else (guess.soft & ~moved),
    )
