"""Sub-MIP polish: manufacture a cheap incumbent, then give the freedom back.

A guess is delivered today by *completing* it — solving the model once with
the whole commitment schedule pinned. That produces a feasible start, and a
measured 1.5-5.4% expensive one against the incumbent that would end the
solve at a single node (see
:func:`gridlock.heuristics.gap_threshold_objective`). The completion cannot
do better by construction: it optimises dispatch against a schedule that
rounding, min up/down repair and the adequacy pass have already decided,
and none of those three ever removes a commitment.

The polish buys the missing few percent with a *restricted MIP*: pin only
what the guess is confident about, leave the contested core integral, and
solve that small problem. The core is small because the LP relaxation
resolves most of the schedule — 2.5-5.4% of binaries are fractional on
these instances — so the sub-MIP branches over a few hundred variables
instead of ~12,000, and it is free to *decommit*, which is the one thing
the guess pipeline structurally cannot do.

**Every fixing is released before returning.** The restriction exists only
to manufacture an incumbent; the schedule it produces is handed to the
real, unrestricted model as a MIP start, where the solver may still
overrule any entry. That is what keeps the run exact. It also means a
restricted objective is *not* a bound on the true problem — this module
therefore returns a schedule and the solve info that produced it, and
nothing here may be routed into a bound or a termination decision.

The polish runs *after* the completion and on the same model, which is not
an implementation detail: the completed solution is a feasible point of the
restricted problem, so handing it over as the sub-MIP's own MIP start makes
the polish monotone — it can only return a schedule at least as cheap as
the one it was given.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from .config import RunConfig, SolverSettings
from .heuristics import CommitmentGuess, complete_solution
from .solver import HighsSession, SolveInfo

SCREENS = ("unit", "entry")

# Defaults chosen against what the polish is trying to buy, not for their
# own sake. The threshold sits ``mip_gap/(1-mip_gap)`` above the LP bound —
# 0.5% on these runs — so a sub-MIP allowed to stop 0.5% short of its own
# optimum could spend the entire margin it was called to recover; a tenth of
# that leaves the arithmetic room to work. The time limit is the honest
# constraint: the polish is only worth doing if it costs meaningfully less
# than the solve it replaces (1,200 s+, censored, on the hard RTS-GMLC
# weeks), and a sub-MIP that has not converged in two minutes is one whose
# free core was mis-screened rather than one that needs longer.
_DEFAULT_GAP = 5e-4
_DEFAULT_SECONDS = 120.0


@dataclass
class PolishResult:
    """What the polish produced, and what it cost to produce it.

    ``succeeded`` is False when the restricted MIP found nothing usable, in
    which case ``guess`` is the input guess unchanged and ``info`` is None.
    A failed polish is a lost improvement, never a lost run.
    """

    guess: CommitmentGuess
    info: SolveInfo | None
    succeeded: bool
    seconds: float
    notes: dict = field(default_factory=dict)


def screen_mask(
    guess: CommitmentGuess, screen: str = "unit", neighbourhood: int = 0
) -> pd.DataFrame:
    """Which entries the sub-MIP pins; everything else is left integral.

    ``entry`` pins exactly what the guess vouches for. ``unit`` pins only
    units the guess vouches for in *every* hour, freeing a contested unit's
    whole column — which on these instances is most of the model, since a
    single fractional hour condemns all 168.

    ``neighbourhood`` is the middle ground, and the one the structure of
    the error argues for: a commitment mistake is rarely an isolated hour,
    it is a run that starts too early or ends too late, so freeing this
    many hours either side of every contested entry lets the sub-MIP move a
    boundary or delete a short run — which pinning right up against the
    fractional hour forbids. It dilates cyclically, matching the model's
    own wrap.

    Entries the guess marked :attr:`~gridlock.heuristics.CommitmentGuess.soft`
    are never pinned. Those are the residue whose mistakes are known to
    concentrate, which makes them precisely what the sub-MIP should be
    deciding.
    """
    if screen not in SCREENS:
        raise ValueError(f"unknown polish screen '{screen}' (available: {SCREENS})")
    if neighbourhood < 0:
        raise ValueError("neighbourhood must be non-negative")
    certain = guess.certain
    if screen == "unit":
        certain = pd.DataFrame(
            np.tile(certain.all(axis=0).to_numpy(), (len(certain.index), 1)),
            index=certain.index,
            columns=certain.columns,
        )
    if guess.soft is not None:
        certain = certain & ~guess.soft
    if neighbourhood:
        free = ~certain.to_numpy()
        spread = free.copy()
        for shift in range(1, neighbourhood + 1):
            spread |= np.roll(free, shift, axis=0) | np.roll(free, -shift, axis=0)
        certain = pd.DataFrame(~spread, index=certain.index, columns=certain.columns)
    return certain


def polish_guess(
    model: pyo.ConcreteModel,
    guess: CommitmentGuess,
    config: RunConfig,
    session: HighsSession | None = None,
    screen: str = "unit",
    neighbourhood: int = 0,
    seconds: float = _DEFAULT_SECONDS,
    gap: float = _DEFAULT_GAP,
    highs_options: dict | None = None,
    warmstart: bool = True,
    baseline_objective: float | None = None,
) -> PolishResult:
    """Solve the restricted MIP, release every fixing, return the schedule.

    ``model`` must already hold a complete solution at the guess's own
    commitment (i.e. :func:`~gridlock.heuristics.complete_solution` has just
    run on it), because that solution is what seeds the sub-MIP. Pass the
    ``session`` that completed it and will run the real solve: the model is
    then translated once and the pinning arrives as an incremental bound
    update rather than a re-translation.

    ``warmstart=False`` for a completion that had to relax the
    minimum-output rows — the values on the model then violate the model the
    sub-MIP is solving, and offering them would only make HiGHS reject a
    start it had to read first.

    Returns the polished guess, or the original one if the restricted MIP
    could not be solved or came back dearer. In every case the call leaves
    no commitment variable fixed and leaves the model holding the solution
    that goes with the guess it returned — the caller hands that on as the
    warm start.
    """
    start = time.perf_counter()
    mask = screen_mask(guess, screen, neighbourhood)
    settings = SolverSettings(
        mip_gap=gap,
        time_limit=seconds,
        # The polish is a sub-solve of the run, not a separate experiment:
        # it has to respect the same thread budget.
        threads=config.solver.threads,
        # The sub-MIP is asked for an incumbent, not for a proof, so the
        # HiGHS knobs worth setting here are the ones that trade proving
        # time for improving time (``mip_heuristic_effort``). Nothing is set
        # by default: the run's own ``highs_options`` are *not* inherited,
        # because a main-solve setting chosen to close a gap is rarely the
        # setting that finds a better solution fastest.
        highs_options=dict(highs_options or {}),
    )

    # Variables the *model* fixed (a carried min up/down obligation does
    # that) are none of our business: they are recorded so they can be told
    # apart from ours, and only ours are released.
    ours = []
    for (g, t), var in model.u.items():
        if var.fixed or not bool(mask.at[t, g]):
            continue
        var.fix(float(guess.commitment.at[t, g]))
        ours.append((g, t))
    free = sum(1 for var in model.u.values() if not var.fixed)

    session = session or HighsSession(model)
    info = None
    failure = None
    try:
        info, _ = session.solve(settings, warmstart=warmstart)
    except RuntimeError as error:
        # A restricted problem can be infeasible where the true one is not
        # (the screen may pin an entry no feasible schedule agrees with), and
        # a time limit can expire before any incumbent is stored. Both cost
        # the improvement, neither costs the run.
        failure = f"{type(error).__name__}: {error}"
    finally:
        for key in ours:
            model.u[key].unfix()

    elapsed = time.perf_counter() - start
    notes = {
        "polish_screen": screen,
        "polish_neighbourhood": neighbourhood,
        "polish_fixed_vars": len(ours),
        "polish_free_vars": free,
        "polish_seconds": elapsed,
        "polish_gap": gap,
        "polish_time_limit": seconds,
    }
    if info is None:
        notes["polish_failed"] = failure
        # The guess itself is returned untouched; only its notes gain the
        # record of the attempt, which is what a sweep has to see.
        return PolishResult(
            replace(guess, notes={**guess.notes, **notes}), None, False, elapsed, notes
        )

    if warmstart and baseline_objective and info.objective > baseline_objective:
        # Only reachable if HiGHS declined the start it was handed (a start
        # it accepts is an incumbent it can only improve on). Keeping the
        # dearer schedule would make the polish a way to *lose* margin, so
        # the completion is restored — the model must end this call holding
        # the solution that goes with the guess being returned, since that
        # is what the caller hands on as the warm start.
        notes["polish_rejected"] = info.objective
        complete_solution(model, guess, session=session)
        return PolishResult(
            replace(guess, notes={**guess.notes, **notes}), None, False, elapsed, notes
        )

    # A restricted problem's dual bound is not a bound on the true one, and
    # the difference is invisible downstream: both are floats on a SolveInfo.
    # Dropping it here means no caller can route it into a bound or a
    # termination decision by accident. The gap it was measured against
    # survives in the notes, where nothing consumes it as arithmetic.
    info.bound = None

    polished = guess.commitment.copy()
    for (g, t), var in model.u.items():
        polished.at[t, g] = float(round(var.value))
    changed = polished != guess.commitment

    notes.update(
        {
            "polish_termination": info.termination,
            "polish_solve_seconds": info.solve_seconds,
            "polish_nodes": info.metrics.mip_nodes,
            "polish_final_gap": info.metrics.final_mip_gap,
            "polish_objective": info.objective,
            "polish_changed_entries": int(changed.to_numpy().sum()),
        }
    )
    if baseline_objective:
        notes["polish_objective_before"] = baseline_objective
        notes["polish_improvement"] = (
            baseline_objective - info.objective
        ) / baseline_objective

    result = CommitmentGuess(
        name=f"{guess.name}+polish",
        commitment=polished,
        # An entry the sub-MIP overruled is no longer the guess's verdict,
        # the same rule lp_relaxation_guess applies to whatever repair and
        # adequacy moved. Without it a downstream ``screen`` fixing would
        # pin a value the guess never vouched for.
        certain=guess.certain & ~changed,
        build_seconds=guess.build_seconds + elapsed,
        notes={**guess.notes, **notes},
        relaxation=guess.relaxation,
        soft=None if guess.soft is None else guess.soft & ~changed,
    )
    return PolishResult(result, info, True, elapsed, notes)
