"""Solve a system one week at a time and save everything a commitment
estimator would want to learn from.

Each week is an independent monolithic cyclic UC over ``--hours`` hours,
warm started from the LP relaxation (``heuristic='lp'``,
``heuristic_fixing='off'``) so the answer is exact -- the solver may
overrule every guessed value -- while still skipping the sub-MIP
heuristic time that dominates the root.

Segments are either consecutive ``--hours`` blocks (``--weeks``) or the
first ``--hours`` of each calendar month (``--monthly``). Month starts do
not land on week boundaries -- February begins at hour 744, which is week
4.43 -- so a monthly sweep cannot be expressed as week indices and gets
its own segment list.

Per week we keep three aligned frames: what the LP *predicted* (raw
fractional ``u``), what the heuristic *proposed* after rounding, repair
and the adequacy pass, and what the MIP *decided*. Studying how to
estimate commitments means studying the disagreements between those, so
throwing the first two away and re-deriving them later would mean
re-solving every LP.

Runs are resumable: a week with a ``done.json`` is skipped, so the sweep
can be interrupted and relaunched.

    python scripts/run_weekly.py --data-dir data/rts_gmlc --label rts_gmlc
"""

from __future__ import annotations

import argparse
import calendar
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gridlock.config import RunConfig, SolverSettings
from gridlock.data import SystemData, build_system, load_system
from gridlock.heuristics import net_load_mw
from gridlock.model import InitialState
from gridlock.runner import _extract_state, run


@dataclass(frozen=True)
class Segment:
    """One solved slice of the horizon.

    ``label`` names its output directory, ``ordinal`` is the number the
    summary reports (week index, or month 1-12).
    """

    label: str
    start_hour: int
    ordinal: int


def week_segments(spec: str | None, total_weeks: int, hours: int) -> list[Segment]:
    """Consecutive ``hours``-long blocks, numbered from zero."""
    return [
        Segment(f"week{w:02d}", w * hours, w) for w in _parse_weeks(spec, total_weeks)
    ]


def monthly_segments(num_hours: int, hours: int) -> list[Segment]:
    """The first ``hours`` hours of each calendar month.

    Assumes the horizon starts at midnight on 1 January, which is how both
    shipped year-long datasets are indexed. Only month *lengths* matter, so
    the year is picked purely to match the leap/common day count. A month
    whose segment would run past the end of the data is dropped rather than
    truncated, so every segment covers the same number of hours.
    """
    days, remainder = divmod(num_hours, 24)
    if remainder or days not in (365, 366):
        raise SystemExit(
            f"--monthly needs a whole-year horizon starting 1 January; "
            f"got {num_hours} h ({num_hours / 24:.2f} days)"
        )
    year = 2020 if days == 366 else 2021
    segments = []
    start = 0
    for month in range(1, 13):
        if start + hours <= num_hours:
            segments.append(Segment(f"month{month:02d}", start, month))
        start += calendar.monthrange(year, month)[1] * 24
    return segments


def slice_hours(system: SystemData, start: int, stop: int) -> SystemData:
    """A SystemData covering only hours [start, stop), re-indexed from 0.

    Rebuilt through ``build_system`` rather than mutated so the slice gets
    the same validation (and derived columns) as any other input.
    """
    availability = system.availability
    if len(availability.columns):
        availability = availability.iloc[start:stop].reset_index(drop=True)
    else:
        availability = pd.DataFrame()
    return build_system(
        generators=system.generators.reset_index(),
        storage=system.storage.reset_index(),
        nodes=system.nodes.reset_index(),
        network=system.network.reset_index(),
        demand=system.demand.iloc[start:stop].reset_index(drop=True),
        availability=availability,
    )


def state_from_saved_week(
    system: SystemData, out_dir: Path, earlier: list[Segment]
) -> InitialState | None:
    """Rebuild a chain's carried state from the latest solved segment on disk.

    Searches backwards through ``earlier`` rather than requiring the
    calendar predecessor: a sampled sweep (every 4th week) still chains,
    just from the previous *solved* segment. That start is a plausible
    feasible condition rather than the physically correct one -- good
    enough to seed the solve, and flagged in the summary via
    ``had_initial_state``.

    Without this a chained sweep is not resumable: a segment skipped
    because it already has a ``done.json`` would leave the next one with no
    predecessor, silently downgrading it to a free start.
    """
    week_dir = next(
        (
            out_dir / segment.label
            for segment in reversed(earlier)
            if (out_dir / segment.label / "done.json").is_file()
        ),
        None,
    )
    if week_dir is None:
        return None
    try:
        frames = {
            name: pd.read_csv(week_dir / f"{name}.csv", index_col="hour")
            for name in ("commitment", "dispatch", "storage_soc")
        }
    except (OSError, ValueError):
        return None
    return _extract_state(system, frames, None)


def commitment_census(lp_values: pd.DataFrame, mip: pd.DataFrame) -> dict:
    """How the LP relaxation's verdicts compare with the MIP's answer.

    The headline numbers for an estimation study: how much of the schedule
    the LP already resolved, and how often it was right where it was sure.
    """
    lp = lp_values.to_numpy(dtype=float)
    optimum = mip.to_numpy(dtype=float).round()
    integral = np.abs(lp - lp.round()) < 1e-6
    total = lp.size
    if total == 0:
        return {}
    agree_integral = (lp.round() == optimum) & integral
    return {
        "binaries": int(total),
        "lp_at_zero_share": float((integral & (lp.round() == 0)).mean()),
        "lp_at_one_share": float((integral & (lp.round() == 1)).mean()),
        "lp_fractional_share": float((~integral).mean()),
        # Of the values the LP resolved, how many did the MIP confirm.
        "lp_integral_accuracy": (
            float(agree_integral.sum() / integral.sum()) if integral.sum() else None
        ),
        "mip_on_share": float(optimum.mean()),
        "mip_startups": int(
            np.maximum(0.0, optimum - np.roll(optimum, 1, axis=0)).sum()
        ),
        "units_never_switching": int(
            sum(1 for j in range(optimum.shape[1]) if len(np.unique(optimum[:, j])) == 1)
        ),
    }


def solve_week(
    system: SystemData,
    segment: Segment,
    args: argparse.Namespace,
    out_dir: Path,
    carried: InitialState | None = None,
) -> tuple[dict, InitialState | None]:
    """Solve one segment and write its frames; returns (summary row, end state).

    With ``--chain`` the segment starts from ``carried`` -- the previous
    segment's final hour -- instead of wrapping cyclically onto its own
    last hour. The two are alternative definitions of the hour before hour
    0, so chaining implies ``cyclic=False``.
    """
    start_hour = segment.start_hour
    sliced = slice_hours(system, start_hour, start_hour + args.hours)

    config = RunConfig(
        unit_commitment=True,
        cyclic=not args.chain,
        heuristic="lp",
        heuristic_fixing="off",
        tight_generation_limits=args.tight,
        tight_ramp_limits=args.tight,
        voll=args.voll,
        # Without this the HiGHS log is never parsed, so the
        # highs_run_seconds / translate_seconds split written below comes
        # out empty -- which is the only way to tell solver time from
        # Pyomo/appsi time after the fact.
        profile=args.profile,
        solver=SolverSettings(
            mip_gap=args.mip_gap, time_limit=args.time_limit, threads=args.threads
        ),
    )

    wall_start = time.perf_counter()
    results = run(sliced, config, initial_state=carried if args.chain else None)
    wall_seconds = time.perf_counter() - wall_start

    week_dir = out_dir / segment.label
    week_dir.mkdir(parents=True, exist_ok=True)

    guess = results.guess
    # Raw fractional u is the LP's actual prediction; the guess frame is
    # that rounded and then pushed around by repair and adequacy. Both are
    # kept because the difference is exactly what a screen gets wrong.
    lp_values = guess.relaxation

    frames = {
        "commitment": results.commitment,
        "lp_relaxation": lp_values,
        "guess_commitment": guess.commitment,
        "guess_certain": guess.certain.astype(int),
        "dispatch": results.dispatch,
        "shed": results.shed,
        "storage_soc": results.storage_soc,
    }
    for name, frame in frames.items():
        # Absolute hour, so weeks concatenate into a year without ambiguity.
        stamped = frame.copy()
        stamped.index = [start_hour + i for i in range(len(stamped))]
        stamped.index.name = "hour"
        stamped.to_csv(week_dir / f"{name}.csv")

    # Full solver telemetry: highs_run_seconds vs translate_seconds is the
    # only way to tell solver time from Pyomo/appsi time after the fact.
    results.window_stats.to_csv(week_dir / "window_stats.csv", index=False)

    hours = list(range(args.hours))
    net = net_load_mw(sliced, hours)
    pd.DataFrame(
        {
            "hour": [start_hour + h for h in hours],
            "demand_mw": sliced.demand.sum(axis=1).to_numpy(),
            "net_load_mw": net.to_numpy(),
        }
    ).set_index("hour").to_csv(week_dir / "load.csv")

    stats = results.window_stats.iloc[0]
    row = {
        "system": args.label,
        "segment": segment.label,
        "week": segment.ordinal,
        "first_hour": start_hour,
        "hours": args.hours,
        "termination": stats["termination"],
        "objective": _maybe_float(stats["objective"]),
        "bound": _maybe_float(stats["bound"]),
        "total_cost": float(results.total_cost),
        "shed_mwh": float(results.shed.to_numpy().sum()),
        "wall_seconds": wall_seconds,
        "build_seconds": float(stats["build_seconds"]),
        "solve_seconds": float(stats["solve_seconds"]),
        "extract_seconds": float(stats["extract_seconds"]),
        "heuristic_seconds": _maybe_float(stats["heuristic_seconds"]),
        "heuristic_match_pct": _maybe_float(stats["heuristic_match_pct"]),
        "heuristic_fallback": bool(stats["heuristic_fallback"]),
        "simplex_iterations": _maybe_float(stats.get("simplex_iterations")),
        "mip_nodes": _maybe_float(stats.get("mip_nodes")),
        "highs_run_seconds": _maybe_float(stats.get("highs_run_seconds")),
        "translate_seconds": _maybe_float(stats.get("translate_seconds")),
    }
    gap_objective, gap_bound = row["objective"], row["bound"]
    row["mip_gap"] = (
        abs(gap_objective - gap_bound) / abs(gap_objective)
        if gap_objective not in (None, 0.0) and gap_bound is not None
        else None
    )
    lp_objective = guess.notes.get("lp_objective")
    row["lp_objective"] = _maybe_float(lp_objective)
    row["integrality_gap"] = (
        (gap_objective - lp_objective) / abs(gap_objective)
        if lp_objective is not None and gap_objective not in (None, 0.0)
        else None
    )
    row.update({f"guess_{k}": v for k, v in guess.notes.items()})
    row.update(commitment_census(lp_values, results.commitment))
    row.update(
        {f"cost_{k}": float(v) for k, v in results.cost_summary.items()}
    )

    row["chained"] = bool(args.chain)
    row["had_initial_state"] = carried is not None

    end_state = None
    if args.chain:
        end_state = _extract_state(
            sliced,
            {
                "commitment": results.commitment,
                "dispatch": results.dispatch,
                "storage_soc": results.storage_soc,
            },
            carried,
        )

    (week_dir / "done.json").write_text(json.dumps(row, indent=2, default=str))
    return row, end_state


def _maybe_float(value) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--label", required=True, help="name for the output folder")
    parser.add_argument("--out-dir", default="results/weekly")
    parser.add_argument("--hours", type=int, default=168)
    parser.add_argument("--weeks", default=None, help="e.g. '0-51' or '0,5,9' (default: all)")
    parser.add_argument(
        "--monthly",
        action="store_true",
        help="solve the first --hours of each calendar month instead of "
        "consecutive week blocks (needs a whole-year horizon starting 1 January)",
    )
    parser.add_argument(
        "--no-profile",
        dest="profile",
        action="store_false",
        help="skip HiGHS log parsing (leaves highs_run_seconds empty)",
    )
    parser.add_argument("--mip-gap", type=float, default=1e-4)
    parser.add_argument("--time-limit", type=float, default=1800.0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--voll", type=float, default=10_000.0)
    parser.add_argument("--no-tight", dest="tight", action="store_false")
    parser.add_argument(
        "--chain",
        action="store_true",
        help="start each week from the previous week's final hour (implies "
        "cyclic=False) instead of wrapping each week onto itself; weeks are "
        "then solved in ascending order and must be contiguous to be "
        "physically meaningful",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="suffix for this process's summary file; weeks are independent, "
        "so several tagged shards can solve disjoint week ranges concurrently",
    )
    parser.set_defaults(tight=True, profile=True)
    args = parser.parse_args()

    system = load_system(args.data_dir)
    total_weeks = system.num_hours // args.hours
    if args.monthly:
        if args.weeks:
            raise SystemExit("--monthly and --weeks are alternative segment lists")
        segments = monthly_segments(system.num_hours, args.hours)
    else:
        segments = week_segments(args.weeks, total_weeks, args.hours)
    if args.chain:
        segments = sorted(segments, key=lambda s: s.start_hour)
        gaps = [
            b.label
            for a, b in zip(segments, segments[1:])
            if b.start_hour != a.start_hour + args.hours
        ]
        if gaps:
            print(
                f"[{args.label}] note: --chain with non-contiguous segments; "
                f"{gaps} start from the preceding *solved* segment, not the "
                "preceding calendar hours",
                flush=True,
            )

    out_dir = Path(args.out_dir) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"summary{args.tag}.jsonl"

    print(
        f"[{args.label}] {system.num_hours} h; solving {len(segments)} x "
        f"{args.hours} h segment(s): "
        f"{', '.join(f'{s.label}@{s.start_hour}' for s in segments)}",
        flush=True,
    )
    print(
        f"[{args.label}] tight={args.tight} mip_gap={args.mip_gap} "
        f"time_limit={args.time_limit}s cyclic={not args.chain} "
        f"warm start=LP relaxation (no fixing)",
        flush=True,
    )
    # A run is only comparable to another with the same knobs, so record them.
    (out_dir / f"run_config{args.tag}.json").write_text(json.dumps(vars(args), indent=2))

    failures = 0
    carried: InitialState | None = None
    for index, segment in enumerate(segments):
        week_dir = out_dir / segment.label
        if (week_dir / "done.json").is_file():
            print(f"[{args.label}] {segment.label} already done, skipping", flush=True)
            carried = None
            continue
        if args.chain and carried is None and index > 0:
            # Resuming mid-chain: recover the latest solved segment's end
            # state from disk rather than silently starting this one free.
            carried = state_from_saved_week(system, out_dir, segments[:index])
            if carried is not None:
                print(
                    f"[{args.label}] {segment.label} resuming chain from an "
                    "earlier solved segment",
                    flush=True,
                )
        try:
            row, carried = solve_week(system, segment, args, out_dir, carried)
        except Exception:
            carried = None
            failures += 1
            print(
                f"[{args.label}] {segment.label} FAILED:\n{traceback.format_exc()}",
                flush=True,
            )
            continue
        with summary_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
        print(
            f"[{args.label}] {segment.label}  {row['wall_seconds']:7.1f}s  "
            f"{row['termination']:>12s}  cost {row['total_cost']:,.0f}  "
            f"gap {_pct(row['mip_gap'])}  lp-frac {_pct(row.get('lp_fractional_share'))}  "
            f"match {_pct(row['heuristic_match_pct'])}",
            flush=True,
        )

    print(f"[{args.label}] finished; {failures} failure(s)", flush=True)
    return 1 if failures else 0


def _pct(value) -> str:
    return "  n/a" if value is None else f"{100 * value:5.2f}%"


def _parse_weeks(spec: str | None, total: int) -> list[int]:
    if not spec:
        return list(range(total))
    weeks: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            weeks.extend(range(int(lo), int(hi) + 1))
        else:
            weeks.append(int(part))
    invalid = [w for w in weeks if not 0 <= w < total]
    if invalid:
        raise SystemExit(f"weeks out of range 0..{total - 1}: {invalid}")
    return weeks


if __name__ == "__main__":
    raise SystemExit(main())
