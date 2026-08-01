"""Benchmark harness: a fixed case suite, run records, and comparison.

The workflow this supports:

1. ``gridlock profile --tag baseline`` — solve the standard suite, write
   one JSONL record per (case, trial) to ``benchmarks/``.
2. Change the formulation or solver settings.
3. ``gridlock profile --tag my-change`` and
   ``gridlock compare benchmarks/<baseline>.jsonl benchmarks/<my-change>.jsonl``.

Design choices that keep comparisons honest:

- The suite is *fixed*: same cases, same gaps, same time limits, so files
  from different days remain comparable. Experiments with solver options
  belong in ``gridlock benchmark`` (presets) or ``--highs-option``; the
  profile suite is for measuring *code* changes.
- MIP timings are seed-noisy, so trials beyond the first re-solve with a
  different HiGHS ``random_seed`` (disable with ``vary_seed=False``). The
  comparison uses the minimum across trials — the least machine-noise
  estimate — and reports the spread so a delta can be judged against it.
- Every record embeds the environment (git commit, package versions,
  hardware) and the full per-window metric table.
- A comparison first checks the *answers* match (costs within the summed
  MIP gaps): a speedup that changes the objective is a bug, not a win.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import RunConfig, SolverSettings
from .data import SystemData, load_system
from .profiling import (
    MemorySampler,
    append_jsonl,
    capture_environment,
    frame_to_records,
    read_jsonl,
)
from .runner import RunResults, run

DEFAULT_RESULTS_DIR = Path("benchmarks")

# Per-solve cap so a pathological change can't hang the harness; a case
# that hits it shows up as termination != optimal in the comparison.
_MIP_TIME_LIMIT = 300.0

# Scale cases deliberately run long, so they get a much larger cap. A case
# that hits it is still a useful data point: it marks where the horizon
# stops being tractable in that mode.
_SCALE_TIME_LIMIT = 1800.0

HOURS_PER_MONTH = 720


@dataclass
class BenchCase:
    """One named benchmark: a RunConfig template applied to the shared dataset."""

    name: str
    description: str
    config: RunConfig


def default_suite() -> list[BenchCase]:
    """The standard case matrix: LP vs MIP, monolithic vs rolling.

    Sized so one pass over the suite takes on the order of a minute on
    laptop hardware with the example dataset.
    """
    return [
        BenchCase(
            "lp_day_mono",
            "24 h monolithic LP — smoke test; overhead dominates, solve is trivial",
            RunConfig(unit_commitment=False, num_hours=24),
        ),
        BenchCase(
            "lp_month_mono",
            "720 h monolithic LP — build/translate scaling and pure LP solve time",
            RunConfig(unit_commitment=False, num_hours=720),
        ),
        BenchCase(
            "uc_week_mono",
            "168 h monolithic UC MIP at 0.1% gap — the core unit-commitment benchmark",
            RunConfig(
                unit_commitment=True,
                num_hours=168,
                solver=SolverSettings(mip_gap=0.001, time_limit=_MIP_TIME_LIMIT),
            ),
        ),
        BenchCase(
            "uc_month_rolling",
            "720 h UC in 168 h windows + 24 h lookahead at 0.5% gap — rolling machinery",
            RunConfig(
                unit_commitment=True,
                num_hours=720,
                window_hours=168,
                lookahead_hours=24,
                solver=SolverSettings(mip_gap=0.005, time_limit=_MIP_TIME_LIMIT),
            ),
        ),
    ]


def _uc_mono_case(name: str, hours: int, label: str, gap: float = 0.005) -> BenchCase:
    return BenchCase(
        name,
        f"{hours} h monolithic UC MIP at {gap:.1%} gap — {label}",
        RunConfig(
            unit_commitment=True,
            num_hours=hours,
            solver=SolverSettings(mip_gap=gap, time_limit=_SCALE_TIME_LIMIT),
        ),
    )


def _uc_rolling_case(
    name: str, hours: int, label: str, window: int = 168, lookahead: int = 24
) -> BenchCase:
    return BenchCase(
        name,
        f"{hours} h UC in {window} h windows (+{lookahead} h lookahead) — {label}",
        RunConfig(
            unit_commitment=True,
            num_hours=hours,
            window_hours=window,
            lookahead_hours=lookahead,
            solver=SolverSettings(mip_gap=0.005, time_limit=_MIP_TIME_LIMIT),
        ),
    )


# Horizons in the scaling sweep: (case stem, hours, prose label). The
# annual entry uses the full 8760 h year rather than 12 * 720 so it lines
# up with the dataset horizon and the README's reference timings.
SCALE_HORIZONS = [
    ("month", HOURS_PER_MONTH, "one month"),
    ("2month", 2 * HOURS_PER_MONTH, "two months"),
    ("quarter", 3 * HOURS_PER_MONTH, "one quarter"),
    ("6month", 6 * HOURS_PER_MONTH, "six months"),
    ("year", 8760, "a full year"),
]

# Case names already recorded under an older naming scheme; keeping the
# mapping means existing benchmark files stay comparable.
_SCALE_NAME_OVERRIDES = {"uc_month_rolling": "uc_month_rolling_scale"}


def scale_suite() -> list[BenchCase]:
    """Horizon-scaling cases: how UC cost grows with the modeled horizon.

    Pairs each horizon in monolithic and rolling mode, which is the
    decomposition question this model exists to study: monolithic sees the
    whole horizon (better solutions, superlinear cost), rolling trades
    foresight for a solve time that grows linearly in windows. Runs long
    by design — the full sweep is over an hour, and the annual monolithic
    case is expected to exhaust its time limit — so it is a deliberate
    experiment, not a per-change check. Use :func:`default_suite` for that.
    """
    cases = []
    for stem, hours, label in SCALE_HORIZONS:
        mono = f"uc_{stem}_mono"
        rolling = _SCALE_NAME_OVERRIDES.get(
            f"uc_{stem}_rolling", f"uc_{stem}_rolling"
        )
        cases.append(_uc_mono_case(mono, hours, f"{label} in a single model"))
        cases.append(_uc_rolling_case(rolling, hours, f"{label}, weekly windows"))
    return cases


SUITES = {"default": default_suite, "scale": scale_suite}


def get_suite(name: str) -> list[BenchCase]:
    if name not in SUITES:
        raise ValueError(f"unknown suite '{name}' (available: {list(SUITES)})")
    return SUITES[name]()


def default_output_path(tag: str | None, results_dir: str | Path = DEFAULT_RESULTS_DIR) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{tag}.jsonl" if tag else f"{stamp}.jsonl"
    return Path(results_dir) / name


# --------------------------------------------------------------------------
# Running the suite
# --------------------------------------------------------------------------


def run_suite(
    data_dir: str | Path,
    cases: list[BenchCase] | None = None,
    repeats: int = 1,
    vary_seed: bool = True,
    tag: str | None = None,
    output_path: str | Path | None = None,
    verbose: bool = True,
) -> tuple[list[dict], Path]:
    """Run every case ``repeats`` times, appending JSONL records to ``output_path``."""
    cases = cases if cases is not None else default_suite()
    output_path = Path(output_path) if output_path else default_output_path(tag)
    system = load_system(data_dir)
    environment = capture_environment()

    records = []
    for case in cases:
        for trial in range(repeats):
            seed = trial if (vary_seed and trial > 0) else None
            if verbose:
                seed_note = f", seed {seed}" if seed is not None else ""
                print(f"profile: {case.name} trial {trial + 1}/{repeats}{seed_note} ...")
            record = run_case(system, case, trial, seed, tag, environment)
            append_jsonl(output_path, record)
            records.append(record)
            if verbose:
                totals = record["totals"]
                memory_note = (
                    f"  peak {totals['peak_memory_mb']:,.0f}MB"
                    if totals.get("peak_memory_mb")
                    else ""
                )
                gap_note = (
                    f" gap {100 * totals['worst_final_gap']:.3f}%"
                    if totals.get("worst_final_gap")
                    else ""
                )
                print(
                    f"  build {totals['build_seconds']:.2f}s"
                    f"  translate {_fmt(totals['translate_seconds'], '.2f')}s"
                    f"  highs {_fmt(totals['highs_seconds'], '.2f')}s"
                    f"  extract {totals['extract_seconds']:.2f}s"
                    f"{memory_note}"
                    f"  | cost {totals['total_cost']:,.0f}"
                    f"  ({totals['termination']}{gap_note})"
                )
    return records, output_path


def run_case(
    system: SystemData,
    case: BenchCase,
    trial: int,
    seed: int | None,
    tag: str | None,
    environment: dict,
) -> dict:
    """Solve one (case, trial) and package everything measured into a record."""
    config = copy.deepcopy(case.config)
    config.profile = True
    if seed is not None:
        config.solver.highs_options.setdefault("random_seed", seed)

    wall_start = time.perf_counter()
    with MemorySampler() as memory:
        results = run(system, config)
    wall_seconds = time.perf_counter() - wall_start
    return make_record(
        case, config, results, wall_seconds, trial, seed, tag, environment, memory
    )


def make_record(
    case: BenchCase,
    config: RunConfig,
    results: RunResults,
    wall_seconds: float,
    trial: int,
    seed: int | None,
    tag: str | None,
    environment: dict,
    memory: MemorySampler | None = None,
) -> dict:
    stats = results.window_stats
    worst_gap = None
    if "final_mip_gap" in stats and stats["final_mip_gap"].notna().any():
        worst_gap = float(stats["final_mip_gap"].max())

    totals = {
        "wall_seconds": wall_seconds,
        "build_seconds": results.total_build_seconds,
        "solve_seconds": results.total_solve_seconds,
        "highs_seconds": _nan_none(stats["highs_run_seconds"].sum(min_count=1)),
        "translate_seconds": _nan_none(stats["translate_seconds"].sum(min_count=1)),
        "extract_seconds": results.total_extract_seconds,
        "total_cost": results.total_cost,
        "objective": results.objective_value,
        "simplex_iterations": _nan_none(stats["simplex_iterations"].sum(min_count=1)),
        "mip_nodes": _nan_none(stats["mip_nodes"].sum(min_count=1)),
        "worst_final_gap": worst_gap,
        "num_windows": len(stats),
        "termination": ";".join(stats["termination"].astype(str).unique()),
        "num_rows": _nan_none(stats["num_rows"].max()),
        "num_cols": _nan_none(stats["num_cols"].max()),
        "num_nonzeros": _nan_none(stats["num_nonzeros"].max()),
        "num_integer_vars": _nan_none(stats["num_integer_vars"].max()),
        "presolved_rows": _nan_none(stats["presolved_rows"].max()),
        "presolved_cols": _nan_none(stats["presolved_cols"].max()),
        "peak_memory_mb": memory.peak_mb if memory else None,
        "memory_growth_mb": memory.growth_mb if memory else None,
    }
    return {
        "schema_version": 1,
        "case": case.name,
        "description": case.description,
        "tag": tag,
        "trial": trial,
        "seed": seed,
        "config": asdict(config),
        "env": environment,
        "totals": totals,
        "windows": frame_to_records(stats),
        "components": frame_to_records(results.component_stats.reset_index()),
    }


def _nan_none(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


# --------------------------------------------------------------------------
# Summaries and comparison
# --------------------------------------------------------------------------

_SUMMARY_COLUMNS = [
    "build_seconds",
    "translate_seconds",
    "solve_seconds",
    "highs_seconds",
    "extract_seconds",
    "wall_seconds",
    "simplex_iterations",
    "mip_nodes",
    "total_cost",
    "worst_final_gap",
    "termination",
    "num_rows",
    "num_cols",
    "num_integer_vars",
    "presolved_rows",
    "num_windows",
    "peak_memory_mb",
]


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    """One row per (case, trial) with the headline totals."""
    rows = []
    for record in records:
        row = {
            "case": record["case"],
            "tag": record.get("tag"),
            "trial": record.get("trial"),
            "seed": record.get("seed"),
        }
        row.update({k: record["totals"].get(k) for k in _SUMMARY_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_records(records: list[dict]) -> pd.DataFrame:
    """Per-case summary: min (least-noise) and median timings across trials."""
    frame = records_to_frame(records)
    grouped = frame.groupby("case", sort=False)
    summary = grouped.agg(
        trials=("trial", "count"),
        highs_s_min=("highs_seconds", "min"),
        highs_s_med=("highs_seconds", "median"),
        build_s=("build_seconds", "min"),
        translate_s=("translate_seconds", "min"),
        extract_s=("extract_seconds", "min"),
        nodes=("mip_nodes", "min"),
        simplex_iters=("simplex_iterations", "min"),
        rows=("num_rows", "max"),
        int_vars=("num_integer_vars", "max"),
        windows=("num_windows", "max"),
        peak_mb=("peak_memory_mb", "max"),
        gap=("worst_final_gap", "max"),
        cost=("total_cost", "mean"),
        termination=("termination", lambda s: ";".join(s.astype(str).unique())),
    )
    return summary.round(
        {
            "highs_s_min": 2,
            "highs_s_med": 2,
            "build_s": 2,
            "translate_s": 2,
            "extract_s": 2,
            "gap": 5,
            "cost": 0,
        }
    )


def compare_records(
    baseline: list[dict], candidate: list[dict], time_metric: str = "highs_seconds"
) -> pd.DataFrame:
    """Case-by-case comparison of two record sets.

    ``time_metric`` picks the headline timing column (``highs_seconds`` by
    default: pure solver time, immune to build-side noise). Deltas are on
    the minimum across trials; ``significant`` marks deltas that exceed
    both 5% and the baseline's own cross-trial spread. ``cost_check`` is
    'MISMATCH' when total costs differ by more than the two runs' summed
    MIP gaps — i.e. the change altered the answer, not just the speed.
    """
    base_frame = records_to_frame(baseline)
    cand_frame = records_to_frame(candidate)
    base_gaps = _configured_gaps(baseline)
    cand_gaps = _configured_gaps(candidate)

    rows = []
    cases = list(dict.fromkeys(base_frame["case"]))
    for case in cases:
        base = base_frame[base_frame["case"] == case]
        cand = cand_frame[cand_frame["case"] == case]
        if cand.empty:
            continue

        base_time = base[time_metric].min()
        cand_time = cand[time_metric].min()
        delta = None
        if pd.notna(base_time) and pd.notna(cand_time) and base_time > 0:
            delta = (cand_time - base_time) / base_time

        spread = 0.0
        base_times = base[time_metric].dropna()
        if len(base_times) > 1 and base_times.min() > 0:
            spread = float((base_times.max() - base_times.min()) / base_times.min())

        base_cost = base["total_cost"].mean()
        cand_cost = cand["total_cost"].mean()
        tolerance = base_gaps.get(case, 0.0) + cand_gaps.get(case, 0.0) + 1e-5
        cost_ok = True
        if pd.notna(base_cost) and pd.notna(cand_cost) and base_cost != 0:
            cost_ok = abs(cand_cost - base_cost) / abs(base_cost) <= tolerance

        terminations = set(base["termination"]) | set(cand["termination"])
        clean_exit = terminations == {"optimal"}

        rows.append(
            {
                "case": case,
                f"base_{time_metric}": round(float(base_time), 3) if pd.notna(base_time) else None,
                f"cand_{time_metric}": round(float(cand_time), 3) if pd.notna(cand_time) else None,
                "delta_pct": round(100 * delta, 1) if delta is not None else None,
                "base_spread_pct": round(100 * spread, 1),
                "significant": (
                    delta is not None and abs(delta) > max(0.05, spread)
                ),
                "base_nodes": _nan_none(base["mip_nodes"].min()),
                "cand_nodes": _nan_none(cand["mip_nodes"].min()),
                "base_build_s": round(float(base["build_seconds"].min()), 2),
                "cand_build_s": round(float(cand["build_seconds"].min()), 2),
                "cost_check": "ok" if cost_ok else "MISMATCH",
                "termination": "optimal" if clean_exit else ";".join(sorted(terminations)),
            }
        )
    return pd.DataFrame(rows).set_index("case")


def _configured_gaps(records: list[dict]) -> dict[str, float]:
    gaps: dict[str, float] = {}
    for record in records:
        gap = (record.get("config", {}).get("solver", {}) or {}).get("mip_gap")
        gaps[record["case"]] = float(gap) if gap else 0.0
    return gaps


def compare_files(
    baseline_path: str | Path,
    candidate_path: str | Path,
    time_metric: str = "highs_seconds",
) -> pd.DataFrame:
    return compare_records(
        read_jsonl(baseline_path), read_jsonl(candidate_path), time_metric
    )


def format_comparison(comparison: pd.DataFrame) -> str:
    """Human-readable verdict lines under the comparison table."""
    lines = [comparison.to_string()]
    lines.append("")
    for case, row in comparison.iterrows():
        if row["cost_check"] != "ok":
            lines.append(
                f"  !! {case}: COST MISMATCH — the change altered the answer, "
                "not just the speed"
            )
        elif row["delta_pct"] is None:
            lines.append(f"  ?  {case}: no comparable timing")
        elif row["significant"]:
            direction = "faster" if row["delta_pct"] < 0 else "SLOWER"
            lines.append(f"  -> {case}: {abs(row['delta_pct']):.1f}% {direction}")
        else:
            lines.append(
                f"  ~  {case}: within noise "
                f"({row['delta_pct']:+.1f}% vs spread {row['base_spread_pct']:.1f}%)"
            )
    return "\n".join(lines)


def _fmt(value, spec: str) -> str:
    return format(value, spec) if value is not None else "?"
