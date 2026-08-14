"""Measure how much an external Lagrangian bound lifts over the LP bound.

The decisive cheap experiment for the dual side. On warm-started weeks the
residual gap is the bound (branch-and-bound improved repaired+polished
incumbents by exactly $0 across thousands of nodes), and HiGHS's own cut
loop lifts the LP bound by only 0.012-0.056% on RTS-GMLC. This probe
answers whether a *model-side* bound has more headroom: evaluate the
Lagrangian dual at the LP relaxation's own prices — one LP plus
milliseconds of per-unit DP, no iteration — and report the lift.

    python scripts/lagrangian_probe.py --data-dir data/rts_gmlc \\
        --weeks 0,9,44 --cluster

How to read it: the threshold a warm start must clear is
``bound / (1 - mip_gap)``. The current best starts sit +0.05% to +0.29%
above the LP threshold, so a lift in that range certifies them outright —
zero further MIP time — and any positive lift shrinks what the polish has
to buy. A lift of ~0 means the integrality gap is not in the per-unit
polytopes (commitment, min up/down, startup cost) and the bound direction
needs cuts that couple units instead.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from gridlock.config import RunConfig
from gridlock.data import cluster_identical_units, load_system
from gridlock.lagrangian import probe_at_lp_prices

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_weekly import slice_hours  # noqa: E402  (sibling script, same directory)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--weeks", default="0", help="comma list of week indices")
    parser.add_argument("--hours", type=int, default=168)
    parser.add_argument("--mip-gap", type=float, default=0.005)
    parser.add_argument("--voll", type=float, default=10_000.0)
    parser.add_argument("--no-tight", dest="tight", action="store_false")
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="pool identical generators first (match the clustered experiments)",
    )
    parser.add_argument("--out", default=None, help="write rows to this CSV")
    parser.set_defaults(tight=True)
    args = parser.parse_args()

    system = load_system(args.data_dir)
    config = RunConfig(
        tight_generation_limits=args.tight,
        tight_ramp_limits=args.tight,
        voll=args.voll,
    )
    weeks = [int(w) for w in args.weeks.split(",") if w.strip()]

    rows = []
    started = time.perf_counter()
    for week in weeks:
        start_hour = week * args.hours
        if start_hour + args.hours > system.num_hours:
            print(f"week{week:02d}: past the end of the data, skipped", flush=True)
            continue
        sliced = slice_hours(system, start_hour, start_hour + args.hours)
        if args.cluster:
            sliced = cluster_identical_units(sliced)

        result = probe_at_lp_prices(sliced, config, list(range(args.hours)))
        lp_bound = result["lp_bound"]
        bound = result["lagrangian"]
        lift = bound.lift_over(lp_bound)
        usable = result["usable_bound"]
        threshold_lp = lp_bound / (1.0 - args.mip_gap)
        threshold_new = usable / (1.0 - args.mip_gap)

        row = {
            "week": week,
            "first_hour": start_hour,
            "lp_bound": lp_bound,
            "lagrangian_bound": bound.total,
            "lift_pct": 100.0 * lift,
            "lift_dollars": bound.total - lp_bound,
            "usable_bound": usable,
            "threshold_lp": threshold_lp,
            "threshold_usable": threshold_new,
            "lp_seconds": result["lp_seconds"],
            "evaluate_seconds": bound.seconds,
        }
        row.update({f"part_{k}": v for k, v in bound.parts.items()})
        rows.append(row)

        print(
            f"week{week:02d}  lp {lp_bound:,.0f}  lagrangian {bound.total:,.0f}  "
            f"lift {100.0 * lift:+.4f}%  (${bound.total - lp_bound:+,.0f})  "
            f"[lp {result['lp_seconds']:.1f}s + eval {bound.seconds:.1f}s]",
            flush=True,
        )
        print(
            f"        threshold {threshold_lp:,.0f} -> {threshold_new:,.0f}  "
            f"(a start below the new number certifies with no MIP at all)",
            flush=True,
        )

    if args.out and rows:
        frame = pd.DataFrame(rows)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.out, index=False)
        print(f"\nwrote {len(frame)} rows to {args.out}")
    print(f"total {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
