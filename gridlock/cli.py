"""Command-line interface.

``gridlock run``        solve a case and write result CSVs
``gridlock benchmark``  solve the same case under several HiGHS option
                        presets and compare runtimes
``gridlock profile``    run the fixed benchmark suite and record profiling
                        metrics to a JSONL file (see docs/profiling.md)
``gridlock compare``    diff two profile record files case by case
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd

from .bench import (
    SUITES,
    compare_files,
    format_comparison,
    get_suite,
    run_suite,
    summarize_records,
)
from .config import RunConfig, SolverSettings
from .data import load_system
from .results import generation_by_technology, write_results
from .runner import run

# HiGHS option presets for quick solver experiments. Anything HiGHS accepts
# can also be passed ad hoc with --highs-option KEY=VALUE.
BENCHMARK_PRESETS: dict[str, dict] = {
    "default": {},
    "presolve_off": {"presolve": "off"},
    "simplex": {"solver": "simplex"},
    "ipm": {"solver": "ipm"},
    "no_symmetry": {"mip_detect_symmetry": False},
    "heuristics_high": {"mip_heuristic_effort": 0.5},
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _command_run(args)
    if args.command == "benchmark":
        return _command_benchmark(args)
    if args.command == "profile":
        return _command_profile(args)
    if args.command == "compare":
        return _command_compare(args)
    parser.print_help()
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gridlock", description="A small Pyomo + HiGHS production cost model."
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="solve a case and write results")
    _add_common_arguments(run_parser)
    run_parser.add_argument(
        "--output-dir", default="results", help="directory for result CSVs (default: results)"
    )

    bench_parser = subparsers.add_parser(
        "benchmark", help="compare HiGHS option presets on the same case"
    )
    _add_common_arguments(bench_parser)
    bench_parser.add_argument(
        "--presets",
        default="default,presolve_off,simplex,ipm",
        help="comma-separated preset names (available: %s)" % ", ".join(BENCHMARK_PRESETS),
    )
    bench_parser.add_argument(
        "--repeat", type=int, default=1, help="solves per preset (default: 1)"
    )
    bench_parser.add_argument(
        "--output-csv", default=None, help="optional path to save the benchmark table"
    )

    profile_parser = subparsers.add_parser(
        "profile",
        help="run the fixed benchmark suite and record profiling metrics",
    )
    profile_parser.add_argument(
        "--data-dir", default="data/example", help="input dataset (default: data/example)"
    )
    profile_parser.add_argument(
        "--suite",
        default="default",
        choices=sorted(SUITES),
        help="'default' = fast per-change suite; 'scale' = long horizon-scaling study",
    )
    profile_parser.add_argument(
        "--cases",
        default=None,
        help="comma-separated case names (default: all in the suite; --list to see them)",
    )
    profile_parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="trials per case (default: 1; use 3+ when comparing changes)",
    )
    profile_parser.add_argument(
        "--tag", default=None, help="label baked into the output filename and records"
    )
    profile_parser.add_argument(
        "--output", default=None, help="output JSONL path (default: benchmarks/<stamp>_<tag>.jsonl)"
    )
    profile_parser.add_argument(
        "--no-vary-seed",
        action="store_true",
        help="repeat trials with the same HiGHS random seed instead of varying it",
    )
    profile_parser.add_argument(
        "--list", action="store_true", help="list suite cases and exit"
    )

    compare_parser = subparsers.add_parser(
        "compare", help="compare two profile record files case by case"
    )
    compare_parser.add_argument("baseline", help="baseline JSONL file")
    compare_parser.add_argument("candidate", help="candidate JSONL file")
    compare_parser.add_argument(
        "--time-metric",
        default="highs_seconds",
        choices=["highs_seconds", "solve_seconds", "wall_seconds", "build_seconds"],
        help="headline timing to diff (default: highs_seconds — pure solver time)",
    )
    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", required=True, help="directory with the input CSVs")
    parser.add_argument(
        "--uc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="binary unit commitment (--no-uc relaxes commitment to an LP)",
    )
    parser.add_argument(
        "--hours", type=int, default=None, help="only model the first N hours"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="rolling-horizon window length in hours (omit for one monolithic solve)",
    )
    parser.add_argument(
        "--lookahead", type=int, default=24, help="rolling-horizon lookahead hours (default: 24)"
    )
    parser.add_argument(
        "--initial-soc-fraction",
        type=float,
        default=0.5,
        help="rolling mode: starting/terminal storage SOC fraction (default: 0.5)",
    )
    parser.add_argument(
        "--voll", type=float, default=10_000.0, help="value of lost load in $/MWh"
    )
    parser.add_argument("--mip-gap", type=float, default=None, help="relative MIP gap")
    parser.add_argument(
        "--time-limit", type=float, default=None, help="solver time limit in seconds"
    )
    parser.add_argument("--threads", type=int, default=None, help="solver threads")
    parser.add_argument(
        "--highs-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="raw HiGHS option (repeatable), e.g. --highs-option presolve=off",
    )
    parser.add_argument(
        "--stream", action="store_true", help="stream the HiGHS log to the console"
    )


def _config_from_args(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        unit_commitment=args.uc,
        num_hours=args.hours,
        window_hours=args.window,
        lookahead_hours=args.lookahead,
        initial_soc_fraction=args.initial_soc_fraction,
        voll=args.voll,
        solver=SolverSettings(
            mip_gap=args.mip_gap,
            time_limit=args.time_limit,
            threads=args.threads,
            stream_solver=args.stream,
            highs_options=_parse_highs_options(args.highs_option),
        ),
    )


def _parse_highs_options(pairs: list[str]) -> dict:
    options = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--highs-option expects KEY=VALUE, got: {pair}")
        key, raw = pair.split("=", 1)
        options[key.strip()] = _coerce(raw.strip())
    return options


def _coerce(raw: str):
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    return raw


def _command_run(args: argparse.Namespace) -> int:
    system = load_system(args.data_dir)
    config = _config_from_args(args)

    mode = "unit commitment (MIP)" if config.unit_commitment else "LP relaxation"
    horizon = (
        "monolithic"
        if config.window_hours is None
        else f"rolling {config.window_hours}h windows (+{config.lookahead_hours}h lookahead)"
    )
    print(f"gridlock: {mode}, {horizon}")

    results = run(system, config)
    output_dir = write_results(results, args.output_dir, system)

    print(f"\nresults written to {output_dir}/")
    print(f"\ncost summary ($):\n{results.cost_summary.map('{:,.0f}'.format).to_string()}")
    print(
        f"\ngeneration by technology (MWh):\n"
        f"{generation_by_technology(system, results).map('{:,.0f}'.format).to_string()}"
    )
    shed_mwh = results.shed.to_numpy().sum()
    if shed_mwh > 1e-3:
        print(f"\nwarning: {shed_mwh:,.1f} MWh of unserved energy")
    print(
        f"\nbuild {results.total_build_seconds:.1f}s, "
        f"solve {results.total_solve_seconds:.1f}s over "
        f"{len(results.window_stats)} window(s)"
    )
    return 0


def _command_benchmark(args: argparse.Namespace) -> int:
    system = load_system(args.data_dir)
    base_config = _config_from_args(args)

    preset_names = [name.strip() for name in args.presets.split(",") if name.strip()]
    unknown = [name for name in preset_names if name not in BENCHMARK_PRESETS]
    if unknown:
        raise SystemExit(
            f"unknown presets: {unknown} (available: {list(BENCHMARK_PRESETS)})"
        )

    rows = []
    for name in preset_names:
        for trial in range(args.repeat):
            config = copy.deepcopy(base_config)
            # Preset options are the experiment; user options are the baseline.
            merged = dict(config.solver.highs_options)
            merged.update(BENCHMARK_PRESETS[name])
            config.solver.highs_options = merged

            print(f"benchmark: preset '{name}' trial {trial + 1}/{args.repeat} ...")
            results = run(system, config)
            rows.append(
                {
                    "preset": name,
                    "trial": trial,
                    "build_seconds": results.total_build_seconds,
                    "solve_seconds": results.total_solve_seconds,
                    "total_cost": results.total_cost,
                    "windows": len(results.window_stats),
                    "termination": ";".join(
                        results.window_stats["termination"].unique()
                    ),
                }
            )

    table = pd.DataFrame(rows)
    summary = (
        table.groupby("preset", sort=False)
        .agg(
            solve_seconds_mean=("solve_seconds", "mean"),
            solve_seconds_min=("solve_seconds", "min"),
            total_cost=("total_cost", "mean"),
            termination=("termination", "first"),
        )
        .round({"solve_seconds_mean": 2, "solve_seconds_min": 2, "total_cost": 0})
    )
    print("\n" + summary.to_string())

    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output_csv, index=False)
        print(f"\nper-trial table written to {args.output_csv}")
    return 0


def _command_profile(args: argparse.Namespace) -> int:
    suite = get_suite(args.suite)
    if args.list:
        for case in suite:
            print(f"{case.name:20s} {case.description}")
        return 0

    if args.cases:
        wanted = [name.strip() for name in args.cases.split(",") if name.strip()]
        by_name = {case.name: case for case in suite}
        unknown = [name for name in wanted if name not in by_name]
        if unknown:
            raise SystemExit(f"unknown cases: {unknown} (available: {list(by_name)})")
        suite = [by_name[name] for name in wanted]

    if args.repeat < 2:
        print(
            "profile: single trial per case — timing noise is unquantified; "
            "prefer --repeat 3 when comparing changes\n"
        )

    records, output_path = run_suite(
        data_dir=args.data_dir,
        cases=suite,
        repeats=args.repeat,
        vary_seed=not args.no_vary_seed,
        tag=args.tag,
        output_path=args.output,
    )
    print(f"\n{summarize_records(records).to_string()}")
    print(f"\nrecords written to {output_path}")
    print(f"compare with: gridlock compare <baseline>.jsonl {output_path}")
    return 0


def _command_compare(args: argparse.Namespace) -> int:
    comparison = compare_files(args.baseline, args.candidate, args.time_metric)
    if comparison.empty:
        print("no overlapping cases between the two files")
        return 1
    print(format_comparison(comparison))
    return 0


if __name__ == "__main__":
    sys.exit(main())
