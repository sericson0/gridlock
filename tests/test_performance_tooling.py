"""Performance-research tooling: log parsing, sessions, warm starts, acyclic mode."""

import json

import pytest

from gridlock import RunConfig, run
from gridlock.model import InitialState, build_model
from gridlock.profiling import parse_mip_progress
from gridlock.solver import HighsSession

from conftest import gen, make_system

# Excerpt of a real HiGHS 1.15 MIP log (uc week on the example system).
MIP_LOG = """\
Src  Proc. InQueue |  Leaves   Expl. | BestBound       BestSol              Gap |   Cuts   InLp Confl. | LpIters     Time

         0       0         0   0.00%   -3620064.7339   inf                  inf        0      0      0         0     0.5s
 R       0       0         0   0.00%   5730276.738555  35744963.2858     83.97%        0      0      0      3502     1.0s
 L       0       0         0   0.00%   5758161.689821  5760305.185049     0.04%     6201    686      0      5479    14.6s
         1       0         1 100.00%   5758161.689821  5760305.185049     0.04%     6201    686      0     20301    14.7s

Solving report
  Status            Optimal
  Timing            14.65
                    0.43 (Presolve)
                        MIP    time [calls] = 0.12 [1]
                        subMIP time [calls] = 0.31 [3]
                    14.22 (Solve)
                        MIP    time [calls] = 3.76 [1]
                        subMIP time [calls] = 10.47 [3]
                    0.00 (Postsolve)
  Nodes             1
  LP iterations     20301
                    0 (strong br.)
                    1977 (separation)
                    14822 (heuristics)
"""


def test_parse_mip_progress_extracts_root_story():
    out = parse_mip_progress(MIP_LOG)

    assert out["first_feasible_seconds"] == 1.0
    assert out["first_feasible_objective"] == pytest.approx(35744963.2858)
    assert out["final_cuts_in_lp"] == 686
    assert out["mip_restarts"] == 0
    assert out["solve_main_mip_seconds"] == pytest.approx(3.76)
    assert out["solve_submip_seconds"] == pytest.approx(10.47)
    assert out["submip_calls"] == 3
    assert out["lp_iters_separation"] == 1977
    assert out["lp_iters_heuristics"] == 14822
    assert out["lp_iters_strong_branching"] == 0

    timeline = json.loads(out["mip_timeline_json"])
    assert [row["src"] for row in timeline] == [None, "R", "L", None]
    assert timeline[0]["sol"] is None  # inf -> no incumbent yet
    assert timeline[-1]["lp_iters"] == 20301
    assert timeline[-1]["gap_pct"] == pytest.approx(0.04)


def test_parse_mip_progress_tolerates_lp_logs():
    out = parse_mip_progress("Presolving model\nSolving the presolved LP\n")
    assert out["mip_timeline_json"] is None
    assert out["first_feasible_seconds"] is None
    assert out["solve_submip_seconds"] is None


def uc_system(hours=24):
    """A committed unit plus peaker, demand shaped to force a nightly shutdown."""
    return make_system(
        [
            gen("big", "A", 10, 100, min_mw=40, startup_cost=1000, min_up=3, min_down=2),
            gen("peaker", "A", 100, 100),
        ],
        {"A": ([30.0] * (hours // 2) + [90.0] * (hours // 2))},
    )


def test_acyclic_monolithic_drops_only_wrap_rows():
    system = uc_system()
    hours = list(range(24))
    cyclic = build_model(system, RunConfig(), hours)
    acyclic = build_model(system, RunConfig(), hours, initial=InitialState())

    # The wrap removal frees exactly the first-hour commitment-logic and
    # ramp rows; everything else (including cyclic storage SOC) is intact.
    assert len(acyclic.commitment_logic) == len(cyclic.commitment_logic) - 1
    assert len(acyclic.soc_balance) == len(cyclic.soc_balance)


def test_acyclic_cost_lower_bounds_cyclic():
    system = uc_system()
    cyclic = run(system, RunConfig(unit_commitment=True, cyclic=True))
    acyclic = run(system, RunConfig(unit_commitment=True, cyclic=False))

    # Acyclic relaxes constraints, so it can only be cheaper.
    assert acyclic.total_cost <= cyclic.total_cost + 1e-6
    assert (acyclic.window_stats["termination"] == "optimal").all()


def test_warmstart_from_rolling_matches_plain_solve():
    system = uc_system()
    plain = run(system, RunConfig(unit_commitment=True))
    warm = run(
        system, RunConfig(unit_commitment=True, warmstart_window_hours=12)
    )

    assert warm.total_cost == pytest.approx(plain.total_cost, rel=1e-3)
    stats = warm.window_stats
    assert stats["warmstart_seconds"].iloc[0] > 0
    # The plain run never pays a warm-start pre-pass.
    assert plain.window_stats["warmstart_seconds"].isna().all()


def test_session_resolves_and_isolates_options():
    system = uc_system()
    model = build_model(system, RunConfig(unit_commitment=True), list(range(24)))
    session = HighsSession(model)

    first, _ = session.solve()
    off, _ = session.solve(settings=None)  # default settings again
    assert first.objective == pytest.approx(off.objective, rel=1e-6)

    # An option set on one solve must not leak into the next.
    from gridlock.config import SolverSettings

    with_off, _ = session.solve(SolverSettings(highs_options={"presolve": "off"}))
    after, _ = session.solve()
    assert with_off.objective == pytest.approx(first.objective, rel=1e-6)
    assert after.objective == pytest.approx(first.objective, rel=1e-6)


def test_session_reports_per_solve_run_time_not_a_cumulative_clock():
    """HiGHS's getRunTime() accumulates over an instance's life.

    Left raw it makes every re-solve look slower than the last, which
    silently inverts option sweeps. The session must report each solve's
    own time.
    """
    system = uc_system()
    model = build_model(system, RunConfig(unit_commitment=True), list(range(24)))
    session = HighsSession(model)

    first, _ = session.solve()
    second, _ = session.solve()

    # A warm re-solve is cheaper, and each reading must stay within its
    # own wall-clock measurement rather than growing without bound.
    assert second.metrics.highs_run_seconds <= first.metrics.highs_run_seconds
    for info in (first, second):
        assert info.metrics.highs_run_seconds <= info.solve_seconds + 1e-6


def test_cold_solve_discards_retained_solver_state():
    system = uc_system()
    model = build_model(system, RunConfig(unit_commitment=True), list(range(24)))
    session = HighsSession(model)

    first, _ = session.solve()
    warm, _ = session.solve()
    cold, _ = session.solve(cold=True)

    # Warm re-solves inherit the basis and incumbent; a cold one repeats
    # the original work, so its iteration count returns to the first solve's.
    assert warm.metrics.simplex_iterations < first.metrics.simplex_iterations
    assert cold.metrics.simplex_iterations == first.metrics.simplex_iterations
    assert cold.objective == pytest.approx(first.objective, rel=1e-6)


def test_session_writes_model_file(tmp_path):
    system = uc_system()
    model = build_model(system, RunConfig(unit_commitment=True), list(range(24)))
    path = HighsSession(model).write_model(tmp_path / "case.mps")
    assert path.is_file()
    assert path.stat().st_size > 0
