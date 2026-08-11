"""Repairing a guess against its completion: shed first, then over-commitment."""

import pandas as pd
import pytest

from gridlock import RunConfig, run
from gridlock.config import SolverSettings
from gridlock.heuristics import (
    CommitmentGuess,
    build_guess,
    complete_solution,
    enforce_adequacy,
    lp_relaxation_guess,
)
from gridlock.model import build_model
from gridlock.repair import repair_guess
from gridlock.scoring import score_guess

from conftest import gen, line, make_system


def uc_config(**kwargs):
    return RunConfig(
        unit_commitment=True, solver=SolverSettings(mip_gap=0.005), **kwargs
    )


def guess_from(system, hours, schedule):
    """A hand-built guess: a full commitment frame, everything vouched for."""
    frame = pd.DataFrame(schedule, index=hours, dtype=float)
    return CommitmentGuess(
        name="hand",
        commitment=frame,
        certain=pd.DataFrame(True, index=hours, columns=frame.columns),
    )


def shed_of(system, config, hours, guess):
    model = build_model(system, config, hours)
    complete_solution(model, guess)
    return float(sum(var.value or 0.0 for var in model.shed.values()))


# ------------------------------------------------------------- shed repair


def ramp_trap_system():
    """Enough capacity every hour, and still unable to serve the ramp.

    A static per-hour adequacy test passes this system trivially: the slow
    unit alone covers the peak. Only the dispatch knows it cannot get there
    from 50 MW at 100 MW/h.
    """
    return make_system(
        [
            gen("slow", "A", 10, 400, min_mw=0, no_load_cost=100, ramp=100),
            gen("fast", "A", 50, 300, min_mw=0, no_load_cost=10),
        ],
        {"A": [50.0, 50.0, 400.0, 400.0, 50.0, 50.0]},
    )


def test_static_adequacy_passes_the_schedule_the_dispatch_cannot_serve():
    """The premise of the whole pass: enforce_adequacy is not sufficient."""
    system = ramp_trap_system()
    hours = list(range(6))
    guess = guess_from(system, hours, {"slow": [1] * 6, "fast": [0] * 6})

    _, added = enforce_adequacy(guess.commitment, system, hours)
    assert added == 0  # the static test sees nothing wrong
    assert shed_of(system, uc_config(), hours, guess) > 0


def test_shed_repair_commits_where_the_completion_actually_failed():
    system = ramp_trap_system()
    hours = list(range(6))
    guess = guess_from(system, hours, {"slow": [1] * 6, "fast": [0] * 6})

    repaired = repair_guess(system, uc_config(), hours, guess, decommit=False)
    assert repaired.notes["repair_shed_before"] > 0
    assert repaired.notes["repair_shed_after"] == pytest.approx(0.0, abs=1e-6)
    # The fast unit is what the ramp-limited hours needed.
    assert repaired.commitment.at[2, "fast"] == 1.0
    assert shed_of(system, uc_config(), hours, repaired) == pytest.approx(0.0, abs=1e-6)


def test_shed_repair_stops_when_nothing_can_be_committed():
    """Load beyond the whole fleet is not a commitment error; it must not loop."""
    system = make_system(
        [gen("a", "A", 10, 100, min_mw=0, startup_cost=100)],
        {"A": [500.0] * 6},
    )
    hours = list(range(6))
    guess = guess_from(system, hours, {"a": [1] * 6})
    repaired = repair_guess(system, uc_config(), hours, guess, decommit=False)
    assert repaired.notes["repair_shed_after"] > 0
    assert repaired.notes["repair_added_unit_hours"] == 0
    pd.testing.assert_frame_equal(repaired.commitment, guess.commitment)


# ------------------------------------------------------------- decommitment


def surplus_system(hours=12):
    """A dear unit committed for no reason, held at its minimum by min_output."""
    return make_system(
        [
            gen("cheap", "A", 10, 300, min_mw=0, no_load_cost=100),
            gen("dear", "A", 40, 100, min_mw=50, no_load_cost=500, startup_cost=200),
        ],
        {"A": [100.0] * hours},
    )


def test_decommit_drops_a_run_that_only_costs_money():
    system = surplus_system()
    hours = list(range(12))
    guess = guess_from(system, hours, {"cheap": [1] * 12, "dear": [1] * 12})

    repaired = repair_guess(system, uc_config(), hours, guess)
    assert repaired.commitment["dear"].sum() == 0.0
    assert repaired.commitment["cheap"].sum() == 12.0  # the useful one stays
    assert repaired.notes["repair_objective_after"] < repaired.notes[
        "repair_objective_before"
    ]
    assert repaired.notes["repair_decommitted_unit_hours"] == 12


def test_a_cut_trims_the_shoulders_and_leaves_the_peak_alone():
    """The measured defect is extended runs, so a cut has to stop in time.

    Ranking a run by its average value cannot do this: the peaker below
    loses money in eight of its twelve hours and is indispensable in the
    other four.
    """
    system = make_system(
        [
            gen("base", "A", 10, 100, min_mw=0, no_load_cost=50),
            gen("peaker", "A", 30, 200, min_mw=20, no_load_cost=400,
                startup_cost=100),
        ],
        {"A": [40.0] * 4 + [260.0] * 4 + [40.0] * 4},
    )
    hours = list(range(12))
    guess = guess_from(system, hours, {"base": [1] * 12, "peaker": [1] * 12})

    repaired = repair_guess(system, uc_config(), hours, guess)
    peaker = repaired.commitment["peaker"]
    assert (peaker.loc[4:7] == 1.0).all()  # 260 MW needs it
    assert peaker.sum() < 12.0  # and the flat hours do not
    assert repaired.notes["repair_shed_after"] == pytest.approx(0.0, abs=1e-6)


def test_decommit_will_not_buy_its_saving_with_unserved_energy():
    """Capacity that is load-carrying stays, however dear it looks."""
    system = make_system(
        [
            gen("cheap_a", "A", 8, 600, min_mw=200, no_load_cost=300),
            gen("dear_b", "B", 60, 400, min_mw=100, no_load_cost=900,
                startup_cost=800),
        ],
        {"A": [150.0] * 12, "B": [320.0] * 12},
        lines=[line("A", "B", 100.0, 0.02, name="AB")],
    )
    hours = list(range(12))
    guess = guess_from(system, hours, {"cheap_a": [1] * 12, "dear_b": [1] * 12})

    repaired = repair_guess(system, uc_config(), hours, guess)
    # Node B can import at most 98 MW against 320 MW of load; dropping its
    # only unit would shed the rest at VOLL.
    assert (repaired.commitment["dear_b"] == 1.0).all()
    assert repaired.notes["repair_shed_after"] <= repaired.notes["repair_shed_before"]


def test_decommit_leaves_min_up_and_down_times_intact():
    """Whole runs only: removing one merges its neighbouring gaps, never splits."""
    system = make_system(
        [
            gen("cheap", "A", 10, 300, min_mw=0, no_load_cost=100),
            gen("dear", "A", 40, 100, min_mw=50, no_load_cost=500,
                startup_cost=200, min_up=3, min_down=3),
        ],
        {"A": [100.0] * 12},
    )
    hours = list(range(12))
    guess = guess_from(
        system, hours, {"cheap": [1] * 12, "dear": [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]}
    )
    repaired = repair_guess(system, uc_config(), hours, guess)
    series = repaired.commitment["dear"].tolist()
    # Either run survives whole or it is gone; no 1- or 2-hour stub is left.
    assert series in ([0.0] * 12, [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])


def test_repair_never_leaves_the_schedule_worse_than_it_found_it():
    """Every change is accepted on the completion's evidence, so this holds."""
    system = surplus_system(hours=24)
    hours = list(range(24))
    config = uc_config()
    guess = lp_relaxation_guess(system, config, hours)
    repaired = repair_guess(system, config, hours, guess)
    assert repaired.notes["repair_objective_after"] <= repaired.notes[
        "repair_objective_before"
    ] + 1e-6
    assert repaired.notes["repair_shed_after"] <= repaired.notes[
        "repair_shed_before"
    ] + 1e-6


# ----------------------------------------------------- delivery and defaults


def test_repair_is_off_unless_asked_for():
    system = surplus_system(hours=24)
    hours = list(range(24))
    config = uc_config()
    plain = lp_relaxation_guess(system, config, hours)
    assert not any(key.startswith("repair_") for key in plain.notes)
    assert RunConfig().heuristic_repair is False


def test_the_option_reaches_the_guess_both_ways():
    system = surplus_system(hours=24)
    hours = list(range(24))
    by_option = build_guess(
        system, uc_config(heuristic="lp", heuristic_options={"repair": True}), hours
    )
    by_config = build_guess(system, uc_config(heuristic="lp", heuristic_repair=True), hours)
    assert "repair_solves" in by_option.notes
    pd.testing.assert_frame_equal(by_option.commitment, by_config.commitment)


def test_repaired_entries_are_no_longer_vouched_for():
    """A screen must not pin a value the completion has since overruled."""
    system = ramp_trap_system()
    hours = list(range(6))
    guess = guess_from(system, hours, {"slow": [1] * 6, "fast": [0] * 6})
    repaired = repair_guess(system, uc_config(), hours, guess, decommit=False)

    moved = repaired.commitment != guess.commitment
    assert moved.to_numpy().any()
    assert not (repaired.certain & moved).to_numpy().any()


def test_a_repaired_start_still_reaches_the_same_optimum():
    system = surplus_system(hours=24)
    plain = run(system, uc_config())
    repaired = run(system, uc_config(heuristic="lp", heuristic_repair=True))
    assert repaired.total_cost == pytest.approx(plain.total_cost, rel=1e-4)


def test_scoring_sees_the_repaired_schedule():
    """The margin is the deliverable, so it has to move with the repair."""
    system = surplus_system(hours=24)
    hours = list(range(24))
    reference = lp_relaxation_guess(system, uc_config(), hours)
    bound = reference.notes["lp_objective"]
    base = score_guess(system, uc_config(heuristic="lp"), hours, guess=reference,
                       lp_bound=bound)
    repaired = score_guess(
        system,
        uc_config(heuristic="lp", heuristic_options={"repair": True}),
        hours,
        lp_bound=bound,
    )
    assert repaired.completion_objective <= base.completion_objective + 1e-6
    assert repaired.threshold_margin <= base.threshold_margin + 1e-9
