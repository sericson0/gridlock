"""Profiling and benchmark-harness tests.

Log-parser tests use verbatim samples captured from HiGHS 1.15 so parser
regressions are caught without solving anything; the solve tests run tiny
systems end-to-end and check the metrics land in window_stats.
"""

import pandas as pd
import pytest

from gridlock import RunConfig, run
from gridlock.bench import (
    SUITES,
    BenchCase,
    compare_records,
    get_suite,
    run_case,
    summarize_records,
)
from gridlock.profiling import (
    MemorySampler,
    append_jsonl,
    capture_environment,
    parse_highs_log,
    read_jsonl,
    total_memory_mb,
)

from conftest import gen, make_system

MIP_LOG = """\
MIP has 22 rows; 42 cols; 63 nonzeros; 21 integer variables (21 binary)
Coefficient ranges:
  Matrix  [1e+00, 3e+00]
  Cost    [1e+00, 1e+00]
  Bound   [1e+00, 1e+01]
  RHS     [4e+00, 1e+02]
Presolving model
22 rows, 21 cols, 42 nonzeros 0s
20 rows, 18 cols, 40 nonzeros 0s
Presolve reductions: rows 20(-2); columns 18(-24); nonzeros 40(-23)

Solving MIP model with:
   20 rows
   18 cols (5 binary, 0 integer, 0 implied int., 13 continuous)
   40 nonzeros

Solving report
  Status            Optimal
  Primal bound      -121
  Dual bound        -121
  Gap               0% (tolerance: 0.01%)
  Timing            0.03
                    0.01 (Presolve)
                    0.02 (Solve)
                    0.00 (Postsolve)
  Nodes             7
  LP iterations     55
"""

LP_LOG = """\
LP has 22 rows; 42 cols; 63 nonzeros
Coefficient ranges:
  Matrix  [1e+00, 3e+00]
  Cost    [1e+00, 1e+00]
  Bound   [1e+00, 1e+01]
  RHS     [4e+00, 1e+02]
Presolving model
22 rows, 21 cols, 42 nonzeros 0s
0 rows, 0 cols, 0 nonzeros 0s
Presolve reductions: rows 0(-22); columns 0(-42); nonzeros 0(-63) - Reduced to empty
Performed postsolve
Solving the original LP from the solution after postsolve

Model status        : Optimal
Objective value     : -1.2100000000e+02
HiGHS run time      :          0.00
"""

# Older HiGHS versions word the reductions line differently.
OLD_STYLE_LOG = "Presolve : Reductions: rows 297(-31); columns 229(-42); elements 785(-93)\n"


def two_gen_uc_system(hours=6):
    """Cheap unit vs expensive peaker; demand forces commitment churn."""
    demand = [50, 120, 50, 120, 50, 120][:hours]
    return make_system(
        generators=[
            gen("base", "a", marginal_cost=10, max_mw=100, min_mw=30,
                startup_cost=500, no_load_cost=50, min_up=2, min_down=2),
            gen("peak", "a", marginal_cost=80, max_mw=100),
        ],
        demand={"a": demand},
    )


# --------------------------------------------------------------------------
# Log parsing
# --------------------------------------------------------------------------


def test_parse_mip_log():
    parsed = parse_highs_log(MIP_LOG)
    assert parsed["presolved_rows"] == 20
    assert parsed["presolved_cols"] == 18
    assert parsed["presolved_nonzeros"] == 40
    assert parsed["presolved_binaries"] == 5
    assert parsed["presolve_seconds"] == pytest.approx(0.01)
    assert parsed["solve_phase_seconds"] == pytest.approx(0.02)
    assert parsed["postsolve_seconds"] == pytest.approx(0.00)
    assert parsed["matrix_coef_min"] == pytest.approx(1.0)
    assert parsed["matrix_coef_max"] == pytest.approx(3.0)
    assert parsed["rhs_coef_max"] == pytest.approx(100.0)


def test_parse_lp_log():
    parsed = parse_highs_log(LP_LOG)
    assert parsed["presolved_rows"] == 0
    assert parsed["presolved_cols"] == 0
    assert parsed["presolved_binaries"] is None
    assert parsed["presolve_seconds"] is None  # LP log has no phase timing block


def test_parse_old_style_reductions():
    parsed = parse_highs_log(OLD_STYLE_LOG)
    assert parsed["presolved_rows"] == 297
    assert parsed["presolved_cols"] == 229
    assert parsed["presolved_nonzeros"] == 785


def test_parse_garbage_log_returns_nones():
    parsed = parse_highs_log("nothing recognizable here")
    assert all(value is None for value in parsed.values())


# --------------------------------------------------------------------------
# Solve metrics end to end
# --------------------------------------------------------------------------


def test_lp_run_populates_metrics():
    system = two_gen_uc_system()
    results = run(system, RunConfig(unit_commitment=False))
    stats = results.window_stats.iloc[0]

    # 6 hours x (p base, p peak, u/v/w base, shed) = 36 columns.
    assert stats["num_cols"] == 36
    assert stats["num_rows"] > 0
    assert stats["num_nonzeros"] > 0
    assert stats["num_integer_vars"] == 0  # relaxed
    assert stats["highs_run_seconds"] >= 0
    assert stats["translate_seconds"] >= 0
    assert stats["extract_seconds"] >= 0
    # No log captured without profile mode.
    assert pd.isna(stats["presolved_rows"])


def test_uc_profile_run_captures_presolve_detail():
    system = two_gen_uc_system()
    results = run(system, RunConfig(unit_commitment=True, profile=True))
    stats = results.window_stats.iloc[0]

    assert stats["num_integer_vars"] == 6  # u for 6 hours; v/w stay continuous
    assert stats["presolved_rows"] is not None and not pd.isna(stats["presolved_rows"])
    assert stats["matrix_coef_max"] >= stats["matrix_coef_min"] > 0
    assert stats["mip_nodes"] >= 0
    assert results.total_highs_seconds >= 0
    assert results.total_solve_seconds >= results.total_highs_seconds


def test_component_stats_census():
    system = two_gen_uc_system()
    results = run(system, RunConfig(unit_commitment=True))
    census = results.component_stats

    constraints = census[census["kind"] == "constraint"]
    variables = census[census["kind"] == "variable"]
    assert "load_balance" in constraints.index
    assert constraints.loc["load_balance", "count"] == 6  # 1 node x 6 hours
    assert constraints.loc["commitment_logic", "count"] == 6
    assert variables.loc["u", "binary"] == 6
    assert variables.loc["p", "count"] == 12  # 2 gens x 6 hours

    # Census totals should reconcile with what HiGHS received.
    assert variables["count"].sum() == results.window_stats.iloc[0]["num_cols"]
    assert constraints["count"].sum() == results.window_stats.iloc[0]["num_rows"]


def test_rolling_run_has_per_window_metrics():
    system = two_gen_uc_system()
    config = RunConfig(unit_commitment=True, window_hours=3, lookahead_hours=1)
    results = run(system, config)
    assert len(results.window_stats) == 2
    assert results.window_stats["num_rows"].notna().all()
    assert results.window_stats["extract_seconds"].notna().all()


# --------------------------------------------------------------------------
# Records, environment, JSONL round trip
# --------------------------------------------------------------------------


def test_capture_environment_keys():
    env = capture_environment()
    for key in ("timestamp_utc", "python", "pyomo", "highspy", "cpu_count", "platform"):
        assert env[key] is not None


def test_memory_sampler_tracks_growth():
    with MemorySampler(interval=0.01) as memory:
        ballast = bytearray(180 * 1_048_576)
        assert len(ballast) > 0
        del ballast

    if memory.peak_mb is None:
        pytest.skip("RSS is not readable on this platform")
    assert memory.start_mb > 0
    assert memory.peak_mb >= memory.start_mb
    # The 180 MB allocation must show up in the peak.
    assert memory.growth_mb > 100


def test_memory_sampler_survives_unreadable_rss(monkeypatch):
    monkeypatch.setattr("gridlock.profiling._make_rss_reader", lambda: None)
    with MemorySampler() as memory:
        pass
    assert memory.peak_mb is None and memory.growth_mb is None


def test_total_memory_mb():
    total = total_memory_mb()
    assert total is None or total > 256


def test_record_roundtrip(tmp_path):
    system = two_gen_uc_system()
    case = BenchCase("tiny_uc", "test case", RunConfig(unit_commitment=True))
    record = run_case(system, case, trial=0, seed=None, tag="t", environment={"host": "x"})

    path = tmp_path / "records.jsonl"
    append_jsonl(path, record)
    append_jsonl(path, record)
    loaded = read_jsonl(path)

    assert len(loaded) == 2
    assert loaded[0]["case"] == "tiny_uc"
    assert loaded[0]["totals"]["build_seconds"] > 0
    assert loaded[0]["totals"]["termination"] == "optimal"
    assert loaded[0]["config"]["unit_commitment"] is True
    assert any(row["component"] == "load_balance" for row in loaded[0]["components"])
    # NaNs must have become JSON null, not the string "nan".
    for window in loaded[0]["windows"]:
        for value in window.values():
            assert value != "nan"


def _fake_record(case, cost, highs_seconds, mip_gap=0.0, trial=0, nodes=10):
    return {
        "case": case,
        "tag": None,
        "trial": trial,
        "seed": trial,
        "config": {"solver": {"mip_gap": mip_gap}},
        "totals": {
            "build_seconds": 1.0,
            "translate_seconds": 0.5,
            "solve_seconds": highs_seconds + 0.5,
            "highs_seconds": highs_seconds,
            "extract_seconds": 0.1,
            "wall_seconds": highs_seconds + 2.0,
            "simplex_iterations": 100,
            "mip_nodes": nodes,
            "total_cost": cost,
            "worst_final_gap": mip_gap,
            "termination": "optimal",
        },
    }


def test_compare_flags_speedup_and_cost_mismatch():
    baseline = [
        _fake_record("fast_case", cost=1000.0, highs_seconds=10.0, trial=0),
        _fake_record("fast_case", cost=1000.0, highs_seconds=10.5, trial=1),
        _fake_record("bad_case", cost=1000.0, highs_seconds=10.0, mip_gap=0.001),
    ]
    candidate = [
        _fake_record("fast_case", cost=1000.0, highs_seconds=5.0),
        _fake_record("bad_case", cost=1100.0, highs_seconds=9.0, mip_gap=0.001),
    ]
    comparison = compare_records(baseline, candidate)

    fast = comparison.loc["fast_case"]
    assert fast["delta_pct"] == pytest.approx(-50.0)
    assert bool(fast["significant"])
    assert fast["cost_check"] == "ok"

    bad = comparison.loc["bad_case"]
    assert bad["cost_check"] == "MISMATCH"  # 10% off with 0.1% gaps


def test_compare_within_noise_not_significant():
    baseline = [
        _fake_record("noisy", cost=500.0, highs_seconds=10.0, trial=0),
        _fake_record("noisy", cost=500.0, highs_seconds=13.0, trial=1),
    ]
    candidate = [_fake_record("noisy", cost=500.0, highs_seconds=11.5)]
    comparison = compare_records(baseline, candidate)
    row = comparison.loc["noisy"]
    assert not bool(row["significant"])  # 15% delta < 30% baseline spread


def test_suites_are_well_formed():
    """Case names must be unique across suites: they are the join key for compare."""
    seen = {}
    for suite_name in SUITES:
        cases = get_suite(suite_name)
        assert cases, f"suite {suite_name} is empty"
        for case in cases:
            case.config.validate()
            assert case.description
            if case.name in seen:
                assert seen[case.name] == case.config, (
                    f"case name '{case.name}' is reused with a different config; "
                    "records keyed on it would not be comparable"
                )
            seen[case.name] = case.config


def test_get_suite_rejects_unknown():
    with pytest.raises(ValueError, match="unknown suite"):
        get_suite("nope")


def test_summarize_records():
    records = [
        _fake_record("a", cost=100.0, highs_seconds=2.0, trial=0),
        _fake_record("a", cost=100.0, highs_seconds=3.0, trial=1),
    ]
    summary = summarize_records(records)
    assert summary.loc["a", "trials"] == 2
    assert summary.loc["a", "highs_s_min"] == pytest.approx(2.0)
    assert summary.loc["a", "highs_s_med"] == pytest.approx(2.5)
