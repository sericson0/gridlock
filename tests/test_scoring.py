"""Scoring a guess without solving the MIP."""

import pytest

from gridlock import RunConfig
from gridlock.config import SolverSettings
from gridlock.heuristics import build_guess, gap_threshold_objective
from gridlock.model import build_model
from gridlock.scoring import commitment_census, score_guess, shedding_hours

from conftest import gen, make_system

from test_heuristics import merit_system


def uc_config(**kwargs):
    return RunConfig(
        unit_commitment=True, solver=SolverSettings(mip_gap=0.005), **kwargs
    )


def test_score_reports_a_margin_against_the_lp_bound():
    system = merit_system(hours=48)
    config = uc_config(heuristic="lp")
    score = score_guess(system, config, list(range(48)))
    assert score.lp_bound > 0
    # A pinned feasible schedule cannot beat the relaxation it came from.
    assert score.completion_objective >= score.lp_bound
    assert score.threshold_objective == pytest.approx(score.lp_bound / 0.995)
    assert score.threshold_margin == pytest.approx(
        score.completion_objective / score.threshold_objective - 1.0
    )
    assert score.clears == (score.threshold_margin <= 0.0)


def test_score_matches_what_a_real_run_reports():
    """The cheap path and the full run must agree, or it measures nothing."""
    from gridlock import run

    system = merit_system(hours=48)
    config = uc_config(heuristic="lp")
    score = score_guess(system, config, list(range(48)))
    stats = run(system, uc_config(heuristic="lp")).window_stats.iloc[0]
    assert score.completion_objective == pytest.approx(
        stats["heuristic_completion_objective"], rel=1e-9
    )
    assert score.threshold_margin == pytest.approx(
        stats["heuristic_threshold_margin"], rel=1e-9
    )


def test_supplied_bound_lets_a_structural_guess_be_scored():
    """priority carries no bound of its own; one can be lent to it."""
    system = merit_system(hours=48)
    config = uc_config(heuristic="priority")
    unscored = score_guess(system, config, list(range(48)))
    assert unscored.threshold_margin is None

    lp = score_guess(system, uc_config(heuristic="lp"), list(range(48)))
    scored = score_guess(system, config, list(range(48)), lp_bound=lp.lp_bound)
    assert scored.threshold_margin == pytest.approx(
        scored.completion_objective / gap_threshold_objective(lp.lp_bound, 0.005) - 1.0
    )


def test_score_accepts_a_caller_built_guess():
    """New heuristics get measured before they are wired into build_guess."""
    system = merit_system(hours=48)
    config = uc_config(heuristic="lp")
    guess = build_guess(system, config, list(range(48)))
    guess.commitment.loc[:, :] = 1.0  # a deliberately worse schedule
    worse = score_guess(system, config, list(range(48)), guess=guess, name="all_on")
    best = score_guess(system, config, list(range(48)))
    assert worse.name == "all_on"
    assert worse.completion_objective >= best.completion_objective


def test_shed_is_reported_and_split_out_of_the_objective():
    """A guess whose completion cannot serve load must not read as merely dear."""
    system = make_system(
        [gen("a", "A", 10, 100, min_mw=0, startup_cost=100)],
        {"A": [500.0] * 6},  # demand far beyond the fleet: shedding is forced
    )
    config = uc_config(heuristic="priority", voll=10_000.0)
    score = score_guess(system, config, list(range(6)))
    assert score.shed_mwh > 0
    assert score.shed_cost == pytest.approx(score.shed_mwh * 10_000.0)
    assert score.objective_net_of_shed == pytest.approx(
        score.completion_objective - score.shed_cost
    )
    assert score.objective_net_of_shed < score.completion_objective


def test_shedding_hours_locates_the_failure():
    system = make_system(
        [gen("a", "A", 10, 100, min_mw=0, startup_cost=100)],
        {"A": [500.0] * 6},
    )
    config = uc_config(heuristic="priority")
    guess = build_guess(system, config, list(range(6)))
    model = build_model(system, config, list(range(6)))
    from gridlock.heuristics import complete_solution

    complete_solution(model, guess)
    rows = shedding_hours(model)
    assert rows and all(mw > 0 for _, _, mw in rows)
    # Worst first, so a repair pass can work down the list.
    assert rows == sorted(rows, key=lambda r: -r[2])


def test_census_counts_hours_and_startups():
    system = merit_system(hours=48)
    guess = build_guess(system, uc_config(heuristic="lp"), list(range(48)))
    committed, startups = commitment_census(guess)
    assert committed == pytest.approx(guess.commitment.to_numpy().round().sum())
    assert startups >= 0
