"""Score a commitment guess without solving the MIP.

The expensive question — "is this guess good?" — has a cheap answer. A
guess plus a completion gives the objective the schedule actually costs,
and comparing that to ``lp_bound / (1 - mip_gap)`` says whether it would
end the solve at the root (see
:func:`gridlock.heuristics.gap_threshold_objective`). That is one LP and
one completion, 45-90 s on RTS-GMLC at 168 h, against 270-1,200 s+ for the
MIP it replaces.

Use this to iterate on guess construction. Use a real solve to confirm the
winner: the margin is a *sufficient* condition for a one-node solve, not a
predictor of solve time, and it ranks nothing across different weeks (a
week can solve at one node from a 5% start if HiGHS's root loop happens to
close the gap unaided). Within one week, comparing variants, it is exactly
the right number.

Two diagnostics ride along because both name a defect the baseline found
and neither is visible in the margin alone:

- ``shed_mwh``: unserved energy in the completion. A guess can be 98%
  correct, pass the adequacy test and still leave the dispatch unable to
  serve load, at which point VOLL swamps the objective and the margin
  reads as a catastrophe rather than as the six bad hours it is.
- ``committed_unit_hours`` / ``startups``: the guess pipeline only ever
  commits *more* (rounding, min up/down repair and the adequacy pass are
  all monotone), and across the 12-month study that bias runs +2.7%
  against the optimum. A variant that claims to fix it should move this.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pyomo.environ as pyo

from .config import RunConfig, SolverSettings
from .data import SystemData
from .heuristics import (
    CommitmentGuess,
    build_guess,
    complete_solution,
    gap_threshold_objective,
    hot_start_margin,
)
from .model import InitialState, build_model
from .polish import polish_guess
from .solver import HighsSession


@dataclass
class GuessScore:
    """What one guess costs and how far that is from a one-node solve."""

    name: str
    completion_objective: float
    lp_bound: float | None
    threshold_objective: float | None
    threshold_margin: float | None
    """``completion / threshold - 1``. **At or below zero is a one-node
    solve.** None when the guess carries no bound (the structural
    heuristics) and none was supplied."""
    shed_mwh: float
    shed_cost: float
    objective_net_of_shed: float
    """The completion objective with the VOLL penalty removed. When a guess
    sheds, this separates "the schedule is expensive" from "the schedule is
    infeasible in six hours", which the raw objective conflates."""
    used_fallback: bool
    committed_unit_hours: float
    startups: float
    guess_seconds: float
    completion_seconds: float
    notes: dict = field(default_factory=dict)

    @property
    def clears(self) -> bool:
        """Whether this start alone would end the solve at the root node."""
        return self.threshold_margin is not None and self.threshold_margin <= 0.0

    def as_row(self) -> dict:
        row = {
            "name": self.name,
            "completion_objective": self.completion_objective,
            "lp_bound": self.lp_bound,
            "threshold_objective": self.threshold_objective,
            "threshold_margin": self.threshold_margin,
            "clears": self.clears,
            "shed_mwh": self.shed_mwh,
            "shed_cost": self.shed_cost,
            "objective_net_of_shed": self.objective_net_of_shed,
            "used_fallback": self.used_fallback,
            "committed_unit_hours": self.committed_unit_hours,
            "startups": self.startups,
            "guess_seconds": self.guess_seconds,
            "completion_seconds": self.completion_seconds,
        }
        row.update({f"note_{k}": v for k, v in self.notes.items()})
        return row


def commitment_census(guess: CommitmentGuess) -> tuple[float, float]:
    """(committed unit-hours, startups) implied by a guess's schedule."""
    values = guess.commitment.to_numpy(dtype=float).round()
    previous = np.roll(values, 1, axis=0)
    return float(values.sum()), float(np.maximum(values - previous, 0.0).sum())


def score_guess(
    system: SystemData,
    config: RunConfig,
    hours: list[int] | None = None,
    initial: InitialState | None = None,
    guess: CommitmentGuess | None = None,
    lp_bound: float | None = None,
    name: str | None = None,
    settings: SolverSettings | None = None,
) -> GuessScore:
    """Build (or accept) a guess, complete it, and report the margin.

    ``guess`` lets a caller score a schedule it built itself, which is how
    a new heuristic gets measured before it is wired into
    :func:`~gridlock.heuristics.build_guess`.

    ``lp_bound`` overrides the bound read from ``guess.notes``. Supply it
    when scoring several variants against one week: the LP relaxation is
    the dearest part of the exercise and its objective does not depend on
    which guess is being scored, so solving it once and passing it here
    turns an N-variant comparison from N LPs into one. It is also the only
    way to get a margin for the structural heuristics, which carry no bound
    of their own.

    With ``config.polish_options`` set, the completed guess is additionally
    handed to :func:`~gridlock.polish.polish_guess` and what is scored is
    the schedule that comes back. The *true* MIP is still never solved —
    the polish solves a restricted one, whose objective is a feasible cost
    for the real model and never a bound on it.

    Confirm a winner with a real run.
    """
    hours = list(range(system.num_hours)) if hours is None else list(hours)

    guess_start = time.perf_counter()
    if guess is None:
        guess = build_guess(system, config, hours, initial)
    guess_seconds = time.perf_counter() - guess_start

    model = build_model(system, config, hours, initial)
    completion_start = time.perf_counter()
    # One session serves the completion and the polish, so the model is
    # translated once and the sub-MIP's pinning arrives as an incremental
    # bound update.
    session = HighsSession(model) if config.polish_options is not None else None
    info, used_fallback = complete_solution(
        model, guess, settings=settings, session=session
    )
    completion_seconds = time.perf_counter() - completion_start

    if config.polish_options is not None:
        polished = polish_guess(
            model,
            guess,
            config,
            session=session,
            warmstart=not used_fallback,
            baseline_objective=info.objective,
            **config.polish_options,
        )
        # Taken either way: a failed polish returns the guess untouched apart
        # from the notes recording the attempt, which is what a sweep needs.
        guess = polished.guess
        if polished.succeeded:
            # The sub-MIP optimises dispatch against the schedule it chose,
            # over the *unrelaxed* model, so its objective is the polished
            # guess's completion objective — and a fallback completion's
            # understated one is superseded rather than merely repeated.
            info, used_fallback = polished.info, False
        completion_seconds += polished.seconds

    shed_mwh = float(sum(var.value or 0.0 for var in model.shed.values()))
    shed_cost = shed_mwh * config.voll

    bound = lp_bound if lp_bound is not None else guess.notes.get("lp_objective")
    threshold = (
        None if bound is None else gap_threshold_objective(bound, config.solver.mip_gap)
    )
    # A fallback completion solved a model with the minimum-output rows
    # switched off, so its objective understates what the schedule really
    # costs -- optimistic in exactly the direction that misleads. Withhold
    # the margin rather than report a flattering one; the objective stays.
    margin = (
        None
        if used_fallback
        else hot_start_margin(info.objective, bound, config.solver.mip_gap)
    )

    committed, startups = commitment_census(guess)
    return GuessScore(
        name=name or guess.name,
        completion_objective=info.objective,
        lp_bound=bound,
        threshold_objective=threshold,
        threshold_margin=margin,
        shed_mwh=shed_mwh,
        shed_cost=shed_cost,
        objective_net_of_shed=info.objective - shed_cost,
        used_fallback=used_fallback,
        committed_unit_hours=committed,
        startups=startups,
        guess_seconds=guess_seconds,
        completion_seconds=completion_seconds,
        notes=dict(guess.notes),
    )


def shedding_hours(model: pyo.ConcreteModel, tolerance: float = 1e-6) -> list[tuple]:
    """(hour, node, MW) for every node-hour the completion could not serve.

    Sorted worst first. This is the handle a shed-driven adequacy repair
    needs: the completion LP knows exactly where the schedule fails, which
    a static per-hour capacity test only estimates.
    """
    rows = [
        (t, n, float(var.value))
        for (n, t), var in model.shed.items()
        if var.value and var.value > tolerance
    ]
    rows.sort(key=lambda row: -row[2])
    return rows
