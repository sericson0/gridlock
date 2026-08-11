"""Sub-MIP polish: it improves the start, and it gives every fixing back."""

import pytest

from gridlock import RunConfig, run
from gridlock.config import SolverSettings
from gridlock.heuristics import build_guess, complete_solution
from gridlock.model import build_model
from gridlock.polish import polish_guess, screen_mask
from gridlock.scoring import score_guess
from gridlock.solver import HighsSession, SolveInfo

from test_heuristics import merit_system


def uc_config(**kwargs):
    return RunConfig(
        unit_commitment=True, solver=SolverSettings(mip_gap=0.005), **kwargs
    )


def completed(system, hours, config=None, guess=None):
    """A model holding the completed guess — the polish's precondition."""
    config = config or uc_config(heuristic="lp")
    guess = guess or build_guess(system, config, hours)
    model = build_model(system, config, hours)
    session = HighsSession(model)
    info, _ = complete_solution(model, guess, session=session)
    return model, session, guess, info


def over_committed_guess(system, hours, config):
    """The LP guess with one unit forced on all week and left contested.

    The polish's whole reason for existing is that nothing in the guess
    pipeline ever *removes* a commitment, so a guess that commits a unit it
    does not need is the case to test against.
    """
    guess = build_guess(system, config, hours)
    guess.commitment["peak"] = 1.0
    guess.certain["peak"] = False
    return guess


# --------------------------------------------------------------- exactness


def test_every_fixing_is_released():
    """The one way this feature can do real damage: a leaked fixing.

    A commitment left pinned after the polish silently restricts the solve
    that follows, which turns an exact run into an inexact one with no
    other symptom.
    """
    system = merit_system(hours=48)
    hours = list(range(48))
    model, session, guess, _ = completed(system, hours)

    result = polish_guess(model, guess, uc_config(heuristic="lp"), session=session)
    assert result.succeeded
    assert result.notes["polish_fixed_vars"] > 0  # it really did pin things
    assert not any(var.fixed for var in model.u.values())


def test_fixings_are_released_even_when_the_sub_mip_fails():
    system = merit_system(hours=48)
    hours = list(range(48))
    model, session, guess, _ = completed(system, hours)

    class Failing:
        def solve(self, *args, **kwargs):
            raise RuntimeError("solver found no feasible solution")

    result = polish_guess(model, guess, uc_config(heuristic="lp"), session=Failing())
    assert not result.succeeded
    assert not any(var.fixed for var in model.u.values())
    assert result.guess.commitment.equals(guess.commitment)
    assert "polish_failed" in result.guess.notes


def test_a_dearer_answer_is_refused():
    """The polish may only ever lower the start's cost, never raise it."""
    system = merit_system(hours=48)
    hours = list(range(48))
    model, session, guess, completion = completed(system, hours)

    class Dearer:
        """HiGHS declining the start it was handed, in effect."""

        def solve(self, *args, **kwargs):
            return SolveInfo("optimal", completion.objective * 1.1, None, 0.0), None

    result = polish_guess(
        model,
        guess,
        uc_config(heuristic="lp"),
        session=Dearer(),
        baseline_objective=completion.objective,
    )
    assert not result.succeeded
    assert result.guess.commitment.equals(guess.commitment)
    assert result.guess.notes["polish_rejected"] > completion.objective
    assert not any(var.fixed for var in model.u.values())


def test_a_fixing_the_model_made_is_not_released():
    """Only the polish's own pins come off; a carried obligation stays."""
    system = merit_system(hours=48)
    hours = list(range(48))
    model, session, guess, _ = completed(system, hours)
    model.u["peak", 0].fix(1.0)

    polish_guess(model, guess, uc_config(heuristic="lp"), session=session)
    assert model.u["peak", 0].fixed
    assert model.u["peak", 0].value == pytest.approx(1.0)
    assert sum(var.fixed for var in model.u.values()) == 1


def test_a_polished_run_reaches_the_same_optimum():
    """End to end: the polish moves the start, never the answer."""
    system = merit_system(hours=48)
    plain = run(system, uc_config(heuristic="lp"))
    polished = run(system, uc_config(heuristic="lp", polish_options={}))
    assert polished.window_stats.iloc[0]["termination"] == "optimal"
    assert polished.objective_value == pytest.approx(plain.objective_value, rel=1e-9)


def test_the_solve_after_a_polish_can_overrule_it():
    """The restriction manufactures an incumbent; it must not survive it.

    The screen here vouches for a schedule that is *wrong* — the dear
    peaker pinned on all week — so the restricted optimum is strictly worse
    than the true one. If any fixing leaked, the solve that follows could
    not recover the optimum, and would do so silently.
    """
    system = merit_system(hours=48)
    hours = list(range(48))
    config = uc_config(heuristic="lp")
    guess = build_guess(system, config, hours)
    guess.commitment["peak"] = 1.0
    guess.certain.loc[:, :] = True  # vouches for every entry, wrongly

    model, session, guess, _ = completed(system, hours, config, guess)
    result = polish_guess(model, guess, config, session=session)
    assert result.succeeded
    assert (result.guess.commitment["peak"] == 1.0).all()  # the restriction held

    info, _ = session.solve(config.solver, warmstart=True)
    true_optimum = run(system, uc_config()).objective_value
    assert info.objective == pytest.approx(true_optimum, rel=1e-6)
    assert info.objective < result.info.objective


def test_the_restricted_bound_is_not_reported():
    """A restricted optimum bounds nothing; it must not look like it does."""
    system = merit_system(hours=48)
    hours = list(range(48))
    model, session, guess, _ = completed(system, hours)
    result = polish_guess(model, guess, uc_config(heuristic="lp"), session=session)
    assert result.info.objective is not None
    assert result.info.bound is None
    assert result.info.gap is None


def test_polish_is_off_by_default():
    assert RunConfig().polish_options is None
    with pytest.raises(ValueError):
        RunConfig(polish_options={}).validate()  # nothing to polish
    uc_config(heuristic="lp", polish_options={}).validate()


# ------------------------------------------------------------------ quality


def test_polish_decommits_what_the_guess_pipeline_cannot():
    system = merit_system(hours=48)
    hours = list(range(48))
    config = uc_config(heuristic="lp")
    guess = over_committed_guess(system, hours, config)
    model, session, guess, completion = completed(system, hours, config, guess)

    result = polish_guess(
        model, guess, config, session=session, baseline_objective=completion.objective
    )
    assert result.succeeded
    # The dear peaker was needed for none of it, and only the sub-MIP can
    # say so: rounding, repair and adequacy all commit in one direction.
    assert result.guess.commitment["peak"].sum() < 48
    assert result.info.objective < completion.objective
    assert result.notes["polish_improvement"] > 0


def test_polish_never_worsens_the_incumbent():
    """It starts from the completion, so it can only improve on it."""
    system = merit_system(hours=48)
    hours = list(range(48))
    config = uc_config(heuristic="lp")
    guess = over_committed_guess(system, hours, config)
    model, session, guess, completion = completed(system, hours, config, guess)

    result = polish_guess(model, guess, config, session=session)
    assert result.info.objective <= completion.objective + 1e-6


def test_screened_entries_keep_the_guess_value():
    system = merit_system(hours=48)
    hours = list(range(48))
    config = uc_config(heuristic="lp")
    guess = over_committed_guess(system, hours, config)
    model, session, guess, _ = completed(system, hours, config, guess)

    result = polish_guess(model, guess, config, session=session, screen="entry")
    mask = screen_mask(guess, "entry").to_numpy()
    before = guess.commitment.to_numpy()
    after = result.guess.commitment.to_numpy()
    assert (after[mask] == before[mask]).all()


def test_certainty_is_withdrawn_where_the_sub_mip_disagreed():
    """A polished entry is the MIP's verdict, not the guess's."""
    system = merit_system(hours=48)
    hours = list(range(48))
    config = uc_config(heuristic="lp")
    guess = over_committed_guess(system, hours, config)
    model, session, guess, _ = completed(system, hours, config, guess)

    result = polish_guess(model, guess, config, session=session)
    changed = result.guess.commitment != guess.commitment
    assert changed.to_numpy().any()
    assert not (result.guess.certain & changed).to_numpy().any()


# ------------------------------------------------------------------- screens


def test_unit_screen_pins_no_more_than_the_entry_screen():
    system = merit_system(hours=48)
    hours = list(range(48))
    guess = build_guess(system, uc_config(heuristic="lp"), hours)
    entry = screen_mask(guess, "entry")
    unit = screen_mask(guess, "unit")
    assert not (unit & ~entry).to_numpy().any()
    # A unit the LP left fractional anywhere is freed for the whole horizon.
    for g in guess.certain.columns:
        assert unit[g].all() == guess.certain[g].all()


def test_soft_entries_are_never_pinned():
    system = merit_system(hours=48)
    hours = list(range(48))
    guess = build_guess(system, uc_config(heuristic="ensemble"), hours)
    guess.soft = guess.certain.copy()
    assert not screen_mask(guess, "entry").to_numpy().any()


def test_a_neighbourhood_frees_the_hours_around_a_contested_entry():
    system = merit_system(hours=48)
    hours = list(range(48))
    guess = build_guess(system, uc_config(heuristic="lp"), hours)
    guess.certain.loc[:, :] = True
    guess.certain.at[10, "peak"] = False

    mask = screen_mask(guess, "entry", neighbourhood=2)
    free = ~mask["peak"]
    assert free[8:13].all()
    assert not free[13:].any() and not free[:8].any()
    # The dilation wraps, because so does the model it restricts.
    guess.certain.loc[:, :] = True
    guess.certain.at[0, "peak"] = False
    wrapped = ~screen_mask(guess, "entry", neighbourhood=1)["peak"]
    assert wrapped[47] and wrapped[0] and wrapped[1]


def test_unknown_screen_is_refused():
    system = merit_system(hours=24)
    guess = build_guess(system, uc_config(heuristic="lp"), list(range(24)))
    with pytest.raises(ValueError):
        screen_mask(guess, "column")


# ------------------------------------------------------------------- scoring


def test_scoring_reports_the_polish_and_its_cost():
    system = merit_system(hours=48)
    hours = list(range(48))
    plain = score_guess(system, uc_config(heuristic="lp"), hours)
    polished = score_guess(
        system, uc_config(heuristic="lp", polish_options={}), hours
    )
    # Same bound, so the margins are comparable and the polish cannot lose.
    assert polished.lp_bound == pytest.approx(plain.lp_bound)
    assert polished.threshold_margin <= plain.threshold_margin + 1e-9
    assert polished.notes["polish_seconds"] > 0
    assert polished.notes["polish_free_vars"] > 0
