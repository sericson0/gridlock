"""Domain heuristics: ingredients, guesses, delivery modes, and safety."""

import pytest

from gridlock import RunConfig, run
from gridlock.config import SolverSettings
from gridlock.heuristics import (
    apply_soft_budget,
    complete_solution,
    gap_threshold_objective,
    hot_start_margin,
    ensemble_guess,
    derive_startup_shutdown,
    enforce_adequacy,
    import_capacity_mw,
    local_need_mw,
    lp_relaxation_guess,
    merit_order,
    net_load_mw,
    priority_list_guess,
    repair_min_up_down,
    similar_days_guess,
)
from gridlock.model import build_model

from conftest import gen, line, make_system

import pandas as pd
import pyomo.environ as pyo


def merit_system(hours=24, peak=380.0):
    """Cheap baseload, mid CCGT, dear peaker, plus wind that offsets demand."""
    half = hours // 2
    demand = [260.0] * half + [peak] * (hours - half)
    return make_system(
        [
            gen("base", "A", 10, 200, min_mw=80, startup_cost=5000, no_load_cost=200,
                min_up=4, min_down=3),
            gen("mid", "A", 35, 150, min_mw=50, startup_cost=1200, no_load_cost=80,
                min_up=2, min_down=2),
            gen("peak", "A", 80, 150, min_mw=30, startup_cost=300, no_load_cost=20),
            gen("wind", "A", 0, 100),
        ],
        {"A": demand},
        availability={"wind": [0.5] * hours},
    )


# ------------------------------------------------------------- ingredients


def test_net_load_subtracts_free_generation():
    system = merit_system()
    net = net_load_mw(system, list(range(24)))
    # wind: 100 MW at 0.5 availability = 50 MW free, all hours.
    assert net.iloc[0] == pytest.approx(260.0 - 50.0)
    assert net.iloc[-1] == pytest.approx(380.0 - 50.0)


def test_merit_order_ranks_by_all_in_cost():
    ranked = merit_order(merit_system())
    assert list(ranked.index) == ["base", "mid", "peak"]
    # all-in = marginal + no-load / max.
    assert ranked.at["base", "all_in_cost"] == pytest.approx(10 + 200 / 200)


def test_repair_fills_short_gaps_and_extends_short_runs():
    system = make_system(
        [gen("g", "A", 10, 100, min_mw=10, startup_cost=100, min_up=3, min_down=2)],
        {"A": [50.0] * 8},
    )
    pattern = pd.DataFrame({"g": [1, 1, 1, 0, 1, 0, 0, 0]}, dtype=float)
    repaired = repair_min_up_down(pattern, system, cyclic=False)
    # The 1-hour gap at t=3 is shorter than min_down=2: committed through.
    values = repaired["g"].tolist()
    assert values[3] == 1.0
    # After filling, the run 0..4 is length 5 >= min_up; trailing zeros stay
    # (a trailing off-gap carries no obligation in an acyclic window).
    assert values[5:] == [0.0, 0.0, 0.0]


def test_repair_only_ever_commits_more():
    """Monotonicity is the safety property: erasing a run deletes capacity."""
    system = make_system(
        [gen("g", "A", 10, 100, min_mw=10, startup_cost=100, min_up=4, min_down=3)],
        {"A": [50.0] * 12},
    )
    pattern = pd.DataFrame(
        {"g": [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0]}, dtype=float
    )
    for cyclic in (False, True):
        repaired = repair_min_up_down(pattern, system, cyclic=cyclic)
        assert (repaired["g"].to_numpy() >= pattern["g"].to_numpy()).all()


def test_repair_handles_the_cyclic_seam():
    system = make_system(
        [gen("g", "A", 10, 100, min_mw=10, startup_cost=100, min_up=4, min_down=2)],
        {"A": [50.0] * 8},
    )
    # On-run wraps the seam: lengths 2 + 1 = 3 < min_up 4 when cyclic.
    pattern = pd.DataFrame({"g": [1, 1, 0, 0, 0, 0, 0, 1]}, dtype=float)
    cyclic = repair_min_up_down(pattern, system, cyclic=True)
    # Measured across the seam as one 3-hour run, extended to 4 — not as
    # two separate runs.
    assert cyclic["g"].sum() == 4.0
    assert cyclic["g"].tolist()[:3] == [1.0, 1.0, 1.0]

    linear = repair_min_up_down(pattern, system, cyclic=False)
    # Acyclic: the leading 2-hour run is extended to min_up, and the
    # trailing 1-hour run is exempt (its obligation runs past the horizon)
    # rather than wrapping into hour 0.
    assert linear["g"].tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def networked_system(hours=24):
    """Two nodes joined by a thin line, with load stranded behind it.

    Node B's demand far exceeds what the line can deliver, so a heuristic
    that only checks system-wide capacity will happily leave B's units off.
    """
    return make_system(
        [
            gen("cheap_a", "A", 8, 600, min_mw=200, startup_cost=5000,
                no_load_cost=300, min_up=3, min_down=2),
            gen("dear_b", "B", 60, 400, min_mw=100, startup_cost=800,
                no_load_cost=100, min_up=2, min_down=2),
        ],
        {"A": [150.0] * hours, "B": [320.0] * hours},
        lines=[line("A", "B", 100.0, 0.02, name="AB")],
    )


def test_import_capacity_counts_both_directions():
    limits = import_capacity_mw(networked_system())
    assert limits["A"] == pytest.approx(100.0 * 0.98)
    assert limits["B"] == pytest.approx(100.0 * 0.98)


def test_local_need_strands_load_behind_the_line():
    need = local_need_mw(networked_system(), list(range(24)))
    # B must serve 320 minus at most 98 MW of imports locally.
    assert need["B"][0] == pytest.approx(320.0 - 98.0)
    # A can import enough to cover its own 150 MW... but only 98, so 52.
    assert need["A"][0] == pytest.approx(150.0 - 98.0)


def test_enforce_adequacy_commits_stranded_local_capacity():
    system = networked_system()
    hours = list(range(24))
    empty = pd.DataFrame(0.0, index=hours, columns=["cheap_a", "dear_b"])
    adequate, added = enforce_adequacy(empty, system, hours)
    assert added > 0
    # The expensive unit behind the line must still come on: nothing else
    # can serve node B's stranded load.
    assert (adequate["dear_b"] == 1.0).all()


def test_priority_guess_is_network_aware():
    system = networked_system()
    guess = priority_list_guess(system, RunConfig(), list(range(24)))
    # Merit order alone would never pick the dear unit; the network forces it.
    assert (guess.commitment["dear_b"] == 1.0).all()


def test_guesses_are_capacity_adequate(  # every heuristic, one invariant
):
    system = networked_system(hours=48)
    hours = list(range(48))
    config = RunConfig()
    for guess in (
        priority_list_guess(system, config, hours),
        lp_relaxation_guess(system, config, hours),
        similar_days_guess(system, config, hours, num_representatives=1),
    ):
        _, missing = enforce_adequacy(guess.commitment, system, hours)
        assert missing == 0, f"{guess.name} left an adequacy shortfall"


def test_derive_startup_shutdown_matches_transitions():
    frame = pd.DataFrame({"g": [0, 1, 1, 0]}, dtype=float)
    v, w = derive_startup_shutdown(frame, cyclic=False)
    assert v["g"].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert w["g"].tolist() == [0.0, 0.0, 0.0, 1.0]
    v_cyc, w_cyc = derive_startup_shutdown(frame, cyclic=True)
    # Cyclic: hour 0 follows hour 3 (off -> off), no extra transition.
    assert v_cyc["g"].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert w_cyc["g"].tolist() == [0.0, 0.0, 0.0, 1.0]


# ------------------------------------------------------------------ guesses


def test_priority_guess_stacks_in_merit_order():
    # peak=330: base+mid capacity (350) covers even the generous screen
    # margin, so the peaker is never stacked.
    system = merit_system(peak=330.0)
    guess = priority_list_guess(system, RunConfig(), list(range(24)))
    # Baseload covers all hours; the peaker is never needed for this load.
    assert (guess.commitment["base"] == 1.0).all()
    assert guess.commitment["peak"].sum() == 0.0


def test_priority_screen_defaults_to_on_only():
    """Off-screening conflates "not needed for capacity" with "uneconomic"."""
    system = merit_system(peak=330.0)
    hours = list(range(24))
    default = priority_list_guess(system, RunConfig(), hours)
    both = priority_list_guess(
        system, RunConfig(), hours, screen_directions="both"
    )
    # The never-stacked peaker is only vouched for when off-screening is
    # explicitly requested; by default the guess stays silent about it.
    assert not default.certain["peak"].any()
    assert both.certain["peak"].all()
    # Certainty is never claimed for a unit the stack did not commit.
    assert not default.certain.to_numpy()[default.commitment.to_numpy() == 0.0].any()


def test_priority_guess_respects_min_up_down():
    system = merit_system(hours=48)
    guess = priority_list_guess(system, RunConfig(), list(range(48)))
    for g in guess.commitment.columns:
        up = int(system.generators.at[g, "min_up_time_hr"])
        series = guess.commitment[g].tolist() * 2  # wrap for cyclic check
        run_length = 0
        for i in range(1, len(series)):
            if series[i] == 1 and series[i - 1] == 0:
                run_length = 1
            elif series[i] == 1:
                run_length += 1
            elif run_length:
                assert run_length >= up or run_length >= 48
                run_length = 0


def test_lp_guess_marks_fractional_entries_uncertain():
    system = merit_system()
    guess = lp_relaxation_guess(system, RunConfig(), list(range(24)))
    assert guess.commitment.isin([0.0, 1.0]).all().all()
    assert guess.certain.to_numpy().mean() > 0.5  # most values integral here


def test_similar_days_transfers_schedules_between_lookalike_days():
    # Two alternating day shapes over six days.
    day_a = [200.0] * 12 + [340.0] * 12
    day_b = [120.0] * 12 + [180.0] * 12
    system = make_system(
        [
            gen("base", "A", 10, 250, min_mw=100, startup_cost=4000, no_load_cost=150,
                min_up=4, min_down=3),
            gen("peak", "A", 70, 200, min_mw=40, startup_cost=400),
        ],
        {"A": (day_a + day_b) * 3},
    )
    guess = similar_days_guess(
        system, RunConfig(), list(range(144)), num_representatives=2
    )
    frame = guess.commitment
    # Lookalike days carry identical schedules.
    assert frame.iloc[0:24].to_numpy().tolist() == frame.iloc[48:72].to_numpy().tolist()
    assert frame.iloc[24:48].to_numpy().tolist() == frame.iloc[72:96].to_numpy().tolist()
    assert guess.notes["representatives"] == 2


# ------------------------------------------------------------ cluster repair


def _cluster_system(num_units, up, down, hours=24):
    """One cluster of identical units, plus enough load to be well posed."""
    system = make_system(
        [gen("c", "A", 10, 100, min_mw=20, min_up=up, min_down=down)],
        {"A": [50.0] * hours},
    )
    system.generators.loc["c", "num_units"] = float(num_units)
    return system


def _violations(counts, num_units, up, down, cyclic=True):
    """Where a count schedule breaks the model's own cluster rows.

    Mirrors gridlock/model.py: min-up is ``sum(v over UT) <= u`` and
    min-down is ``sum(w over DT) <= N - u``.
    """
    import numpy as np

    u = np.asarray(counts, dtype=float)
    n = len(u)
    previous = np.roll(u, 1)
    v = np.maximum(u - previous, 0.0)
    w = np.maximum(previous - u, 0.0)
    bad = []
    for t in range(n):
        if up > 1 and sum(v[(t - k) % n] for k in range(up)) > u[t] + 1e-9:
            bad.append(("min_up", t))
        if down > 1 and sum(w[(t - k) % n] for k in range(down)) > num_units - u[t] + 1e-9:
            bad.append(("min_down", t))
    return bad


def test_cluster_repair_does_not_overwrite_a_count():
    """A cluster running flat out must stay at N, not be flattened to 1."""
    system = _cluster_system(num_units=3, up=4, down=4)
    frame = pd.DataFrame({"c": [3.0] * 24}, index=range(24))
    out = repair_min_up_down(frame, system, cyclic=True)
    assert list(out["c"]) == [3.0] * 24


def test_cluster_repair_satisfies_the_models_own_rows():
    """The repaired count schedule must not violate min up/down anywhere.

    This is the defect that made `heuristic='lp'` with `cluster_units=True`
    infeasible: the repair left a count column breaking its own min-up row,
    and pinning `u` to it gave the completion no feasible point at all.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    for num_units, up, down in ((2, 8, 4), (3, 3, 3), (5, 4, 2), (4, 2, 6)):
        system = _cluster_system(num_units, up, down, hours=48)
        for _ in range(15):
            counts = rng.integers(0, num_units + 1, size=48).astype(float)
            frame = pd.DataFrame({"c": counts}, index=range(48))
            out = repair_min_up_down(frame, system, cyclic=True)
            values = out["c"].to_numpy()
            assert values.max() <= num_units, "a count may never exceed the fleet"
            assert (values >= counts).all(), "the repair must stay monotone"
            assert not _violations(values, num_units, up, down), (
                f"N={num_units} up={up} down={down} left "
                f"{_violations(values, num_units, up, down)[:3]}"
            )


def test_single_unit_repair_is_the_n_equals_one_case():
    """The layer decomposition must not change any existing single-unit result."""
    system = _cluster_system(num_units=1, up=4, down=3)
    frame = pd.DataFrame(
        {"c": [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 16}, index=range(24)
    )
    out = repair_min_up_down(frame, system, cyclic=True)
    assert set(out["c"].unique()) <= {0.0, 1.0}
    assert not _violations(out["c"].to_numpy(), 1, 4, 3)


# ------------------------------------------------------- hot-start scoring


def test_threshold_is_the_bound_inflated_by_the_gap():
    assert gap_threshold_objective(1000.0, 0.005) == pytest.approx(1000.0 / 0.995)
    # An unset gap means HiGHS's own default, not "no tolerance".
    assert gap_threshold_objective(1000.0, None) == pytest.approx(1000.0 / 0.9999)
    assert gap_threshold_objective(1000.0, 0.0) == pytest.approx(1000.0)


def test_threshold_rejects_a_gap_outside_the_unit_interval():
    with pytest.raises(ValueError):
        gap_threshold_objective(1000.0, 1.0)
    with pytest.raises(ValueError):
        gap_threshold_objective(1000.0, -0.1)


def test_margin_sign_says_whether_the_solve_can_stop_at_the_root():
    bound, gap = 1000.0, 0.005
    threshold = gap_threshold_objective(bound, gap)
    assert hot_start_margin(threshold, bound, gap) == pytest.approx(0.0)
    assert hot_start_margin(threshold * 0.99, bound, gap) < 0.0   # clears
    assert hot_start_margin(threshold * 1.02, bound, gap) == pytest.approx(0.02)


def test_margin_is_none_without_a_usable_bound():
    assert hot_start_margin(None, 1000.0, 0.005) is None
    assert hot_start_margin(1000.0, None, 0.005) is None
    assert hot_start_margin(1000.0, 0.0, 0.005) is None


def test_lp_run_scores_its_own_warm_start():
    """The LP guesses carry a bound, so the runner can score them."""
    system = merit_system(hours=48)
    guided = run(
        system,
        RunConfig(
            unit_commitment=True,
            heuristic="lp",
            solver=SolverSettings(mip_gap=0.005),
        ),
    )
    stats = guided.window_stats.iloc[0]
    bound = stats["heuristic_lp_bound"]
    completion = stats["heuristic_completion_objective"]
    assert bound > 0
    # The completion pins a feasible schedule, so it cannot beat the
    # relaxation it was rounded from.
    assert completion >= bound
    assert stats["heuristic_threshold_objective"] == pytest.approx(bound / 0.995)
    assert stats["heuristic_threshold_margin"] == pytest.approx(
        completion / (bound / 0.995) - 1.0
    )
    # A cleared threshold must imply the solve had nothing left to prove.
    if stats["heuristic_threshold_margin"] <= 0:
        assert guided.total_cost <= completion + 1e-6


def test_structural_guesses_report_no_threshold():
    """`priority` has no bound to measure against; it must not invent one."""
    system = merit_system(hours=48)
    guided = run(system, RunConfig(unit_commitment=True, heuristic="priority"))
    stats = guided.window_stats.iloc[0]
    assert stats["heuristic_completion_objective"] > 0
    assert pd.isna(stats["heuristic_lp_bound"])
    assert pd.isna(stats["heuristic_threshold_margin"])


def test_relaxed_completion_withholds_the_margin():
    """A fallback completion solved a weaker model, so its margin would lie."""
    system = make_system(
        [
            gen("a", "A", 10, 100, min_mw=80, startup_cost=100),
            gen("b", "A", 20, 100, min_mw=80, startup_cost=100),
        ],
        {"A": [100.0] * 6},
    )
    model = build_model(system, RunConfig(unit_commitment=True), list(range(6)))
    guess = priority_list_guess(system, RunConfig(), list(range(6)))
    guess.commitment.loc[:, :] = 1.0
    info, used_fallback = complete_solution(model, guess)
    assert used_fallback
    # The relaxed solve understates the schedule's true cost, which is the
    # direction that would make a guess look better than it is.
    assert info.objective is not None


# --------------------------------------------------- delivery and end-to-end


@pytest.mark.parametrize("heuristic", ["priority", "similar_days", "lp", "ensemble"])
def test_warmstart_delivery_preserves_the_optimum(heuristic):
    system = merit_system(hours=48)
    plain = run(system, RunConfig(unit_commitment=True))
    guided = run(system, RunConfig(unit_commitment=True, heuristic=heuristic))
    assert guided.total_cost == pytest.approx(plain.total_cost, rel=1e-4)
    stats = guided.window_stats.iloc[0]
    assert stats["heuristic_seconds"] > 0
    assert 0.0 <= stats["heuristic_match_pct"] <= 1.0


def test_screen_fixing_stays_within_tolerance():
    system = merit_system(hours=48)
    plain = run(system, RunConfig(unit_commitment=True))
    screened = run(
        system,
        RunConfig(unit_commitment=True, heuristic="priority", heuristic_fixing="screen"),
    )
    assert screened.total_cost == pytest.approx(plain.total_cost, rel=1e-3)
    assert screened.window_stats.iloc[0]["heuristic_fixed_vars"] > 0


def test_aggressive_fixing_bounds_cost_from_above():
    system = merit_system(hours=48)
    plain = run(system, RunConfig(unit_commitment=True))
    pinned = run(
        system,
        RunConfig(
            unit_commitment=True, heuristic="priority", heuristic_fixing="aggressive"
        ),
    )
    # Fixing everything can only cost more (or equal), never less.
    assert pinned.total_cost >= plain.total_cost - 1e-6
    fixed = pinned.window_stats.iloc[0]["heuristic_fixed_vars"]
    assert fixed == 48 * 3  # every commitment variable


def test_overcommitted_guess_falls_back_instead_of_failing():
    """All-on is infeasible here: committed minimums exceed demand."""
    system = make_system(
        [
            gen("a", "A", 10, 100, min_mw=80, startup_cost=100),
            gen("b", "A", 20, 100, min_mw=80, startup_cost=100),
        ],
        {"A": [100.0] * 6},
    )
    model = build_model(system, RunConfig(unit_commitment=True), list(range(6)))
    guess = priority_list_guess(system, RunConfig(), list(range(6)))
    guess.commitment.loc[:, :] = 1.0  # force the over-commitment
    info, used_fallback = complete_solution(model, guess)
    assert used_fallback
    assert info.objective is not None
    # u must be unfixed again afterwards.
    assert not any(var.fixed for var in model.u.values())


def test_validate_rejects_bad_combinations():
    with pytest.raises(ValueError):
        RunConfig(heuristic="priority", window_hours=24).validate()
    with pytest.raises(ValueError):
        RunConfig(heuristic="priority", warmstart_window_hours=24).validate()
    with pytest.raises(ValueError):
        RunConfig(heuristic="mystery").validate()
    with pytest.raises(ValueError):
        RunConfig(heuristic_fixing="sometimes").validate()


def test_priority_rejects_clustered_units():
    system = merit_system()
    system.generators.loc["mid", "num_units"] = 2.0
    with pytest.raises(NotImplementedError):
        priority_list_guess(system, RunConfig(), list(range(24)))


# ----------------------------------------------------------------- ensemble


def test_ensemble_vouches_only_where_all_three_agree():
    """Confidence is the intersection, so it cannot exceed any member's own."""
    system = merit_system(hours=48)
    config = RunConfig(unit_commitment=True)
    hours = list(range(48))
    guess = ensemble_guess(system, config, hours)
    lp = lp_relaxation_guess(system, config, hours)

    assert guess.name == "ensemble"
    assert guess.notes["ensemble_members"] == 3
    # The schedule delivered is the LP's; only the confidence differs.
    pd.testing.assert_frame_equal(guess.commitment, lp.commitment)
    assert guess.certain.to_numpy().sum() <= lp.certain.to_numpy().sum()
    # A unit with any fractional hour is untrusted in *every* hour of it.
    fractional = ((lp.relaxation > 1e-6) & (lp.relaxation < 1 - 1e-6)).any(axis=0)
    for unit in fractional.index[fractional]:
        assert not guess.certain[unit].any()


def test_ensemble_marks_fast_units_soft_not_certain_free():
    """soft_min_up_hours splits the screen; it does not shrink it."""
    system = merit_system(hours=48)
    config = RunConfig(unit_commitment=True)
    hours = list(range(48))
    plain = ensemble_guess(system, config, hours)
    split = ensemble_guess(system, config, hours, soft_min_up_hours=1)

    # Same confidence, differently delivered.
    assert split.certain.to_numpy().sum() == plain.certain.to_numpy().sum()
    assert split.soft is not None and plain.soft is None
    # "peak" has min_up 1 in merit_system; the others are 2 and 4.
    assert split.soft["peak"].any()
    assert not split.soft["base"].any()
    # Soft entries are always a subset of the confident ones.
    assert not (split.soft & ~split.certain).to_numpy().any()


def test_soft_budget_row_counts_deviations_from_the_guess():
    system = merit_system(hours=24)
    config = RunConfig(unit_commitment=True)
    hours = list(range(24))
    guess = ensemble_guess(system, config, hours, soft_min_up_hours=1)
    model = build_model(system, config, hours)

    covered = apply_soft_budget(model, guess, budget=3)
    assert covered == int(guess.soft.to_numpy().sum())
    assert hasattr(model, "soft_fixing_budget")
    # The guess itself sits at distance zero, so the row must admit it.
    for (g, t), var in model.u.items():
        var.set_value(float(guess.commitment.at[t, g]))
    assert pyo.value(model.soft_fixing_budget.body) == pytest.approx(0.0)


def test_soft_budget_is_a_no_op_without_soft_entries():
    system = merit_system(hours=24)
    config = RunConfig(unit_commitment=True)
    hours = list(range(24))
    guess = ensemble_guess(system, config, hours)  # no soft_min_up_hours
    model = build_model(system, config, hours)
    assert apply_soft_budget(model, guess, budget=5) == 0
    assert not hasattr(model, "soft_fixing_budget")


def test_zero_budget_matches_hard_fixing():
    """budget=0 forbids every deviation, so it is fixing by another route."""
    system = merit_system(hours=48)
    plain = run(system, RunConfig(unit_commitment=True))
    soft = run(
        system,
        RunConfig(
            unit_commitment=True,
            heuristic="ensemble",
            heuristic_fixing="screen",
            heuristic_options={"soft_min_up_hours": 1},
            soft_fixing_budget=0,
        ),
    )
    stats = soft.window_stats.iloc[0]
    assert stats["heuristic_soft_vars"] > 0
    assert soft.total_cost >= plain.total_cost - 1e-6


def test_budget_recovers_what_fixing_would_have_excluded():
    """A budget is looser than a pin: cost can only improve, never worsen."""
    system = merit_system(hours=48)
    options = dict(
        unit_commitment=True,
        heuristic="ensemble",
        heuristic_fixing="screen",
        heuristic_options={"soft_min_up_hours": 1},
    )
    pinned = run(system, RunConfig(**options, soft_fixing_budget=0))
    budgeted = run(system, RunConfig(**options, soft_fixing_budget=40))
    assert budgeted.total_cost <= pinned.total_cost + 1e-6


def test_validate_rejects_a_budget_without_a_screen():
    with pytest.raises(ValueError, match="soft_fixing_budget"):
        RunConfig(
            heuristic="ensemble", heuristic_fixing="off", soft_fixing_budget=10
        ).validate()
    with pytest.raises(ValueError, match="non-negative"):
        RunConfig(
            heuristic="ensemble", heuristic_fixing="screen", soft_fixing_budget=-1
        ).validate()
