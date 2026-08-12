"""A/B commitment guesses across several weeks without solving any MIP.

This is the iteration loop for hot-start work. Each (week, variant) pair
costs one guess plus one completion, so a five-variant comparison over six
weeks finishes in minutes rather than the ~12 hours the same comparison
would take through ``run_weekly.py``.

    python scripts/score_guesses.py --data-dir data/rts_gmlc \\
        --weeks 0,9,18,26,35,44 --variant lp --variant ens=ensemble

The headline column is ``margin`` — how far the guess sits from the
incumbent that would end the solve at the root node. Lower is better and
**at or below zero is a one-node solve**. Compare variants *within* a week;
the margin does not rank weeks against each other (a week can solve at one
node from a 5% start if HiGHS's root loop closes the gap unaided).

Two supporting columns exist because the baseline found defects the margin
alone hides: ``shed`` catches a guess whose completion cannot serve load
(week44 read as +56% for want of 259 MWh in six hours), and ``on-hrs``
tracks the pipeline's monotone over-commitment bias (+2.7% against the
optimum across the 12-month study).

One LP per week is shared by every variant: the relaxation's objective is
the bound all margins are measured against and does not depend on which
guess is being scored, so scoring N variants costs one LP, not N.

Variants are given on the command line, so adding one never means editing
this file:

    --variant lp                          # heuristic 'lp', default options
    --variant ens=ensemble                # labelled, no options
    --variant e3=ensemble:{"soft_min_up_hours":3}
    --variant pol=lp:{"polish":true}      # sub-MIP polish on top of the guess
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from gridlock.config import RunConfig, SolverSettings
from gridlock.data import (
    SystemData,
    build_system,
    cluster_identical_units,
    load_system,
)
from gridlock.heuristics import lp_relaxation_guess
from gridlock.scoring import score_guess

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_weekly import slice_hours  # noqa: E402  (sibling script, same directory)


def parse_variant(spec: str) -> tuple[str, str, dict, dict | None]:
    """``[name=]heuristic[:json]`` -> (name, heuristic, options, polish).

    ``polish`` is the one option that configures the *run* rather than the
    guess builder, so it is lifted out of the builder's kwargs here: ``true``
    enables the sub-MIP polish with its defaults, an object overrides them
    (``{"polish":{"screen":"entry","seconds":60}}``).
    """
    name, _, rest = spec.partition("=")
    if not rest:
        name, rest = spec, spec
    heuristic, _, raw = rest.partition(":")
    try:
        options = json.loads(raw) if raw else {}
    except json.JSONDecodeError as error:
        raise SystemExit(f"--variant {spec!r}: options are not valid JSON ({error})")
    if not isinstance(options, dict):
        raise SystemExit(f"--variant {spec!r}: options must be a JSON object")
    polish = options.pop("polish", None)
    if polish is True:
        polish = {}
    if polish is not None and not isinstance(polish, dict):
        raise SystemExit(f"--variant {spec!r}: 'polish' must be true or an object")
    return name, heuristic.strip(), options, polish


def make_config(
    args: argparse.Namespace, heuristic: str, options: dict, polish: dict | None = None
) -> RunConfig:
    return RunConfig(
        unit_commitment=True,
        cyclic=True,
        heuristic=heuristic,
        heuristic_options=options,
        polish_options=polish,
        tight_generation_limits=args.tight,
        tight_ramp_limits=args.tight,
        voll=args.voll,
        solver=SolverSettings(mip_gap=args.mip_gap, threads=args.threads),
    )


def score_week(
    system: SystemData, start_hour: int, args: argparse.Namespace, variants: list
) -> list[dict]:
    """Score every variant against one week, sharing a single LP bound."""
    sliced = slice_hours(system, start_hour, start_hour + args.hours)
    if args.cluster:
        # Pool interchangeable units into integer-count clusters. This is the
        # model the guess is built against *and* scored on, so the margin
        # stays internally consistent -- but a clustered bound is not the
        # unclustered one (pooling is a slight relaxation), so margins do not
        # compare across the flag.
        sliced = cluster_identical_units(sliced)
    hours = list(range(args.hours))

    # The bound every margin is measured against. Built once here; the 'lp'
    # variant reuses this very guess rather than solving the relaxation a
    # second time.
    reference_config = make_config(args, "lp", {})
    reference = lp_relaxation_guess(sliced, reference_config, hours)
    bound = reference.notes["lp_objective"]

    rows = []
    for name, heuristic, options, polish in variants:
        config = make_config(args, heuristic, options, polish)
        # The polish improves a guess rather than building one, so a polished
        # 'lp' variant reuses the shared relaxation too — which is what makes
        # the run-time difference between the two read as the polish's own.
        reuse = reference if (heuristic == "lp" and not options) else None
        try:
            score = score_guess(
                sliced,
                config,
                hours,
                guess=reuse,
                lp_bound=bound,
                name=name,
            )
        except Exception as error:  # one bad variant must not lose the sweep
            print(f"    {name:<16s} FAILED {type(error).__name__}: {error}", flush=True)
            rows.append({"name": name, "first_hour": start_hour, "error": repr(error)})
            continue

        row = score.as_row()
        row["first_hour"] = start_hour
        # The 'lp' variant reuses the reference guess, so its guess_seconds
        # would read as ~0 and flatter it against variants that built their
        # own. Charge it what the shared LP actually cost.
        if reuse is not None:
            row["guess_seconds"] = reference.build_seconds
        rows.append(row)
        margin = score.threshold_margin
        print(
            f"    {name:<16s} margin {'n/a' if margin is None else f'{100 * margin:+7.3f}%'}"
            f"  shed {score.shed_mwh:8.2f} MWh"
            f"  on-hrs {score.committed_unit_hours:7.0f}"
            f"  starts {score.startups:4.0f}"
            f"  {score.guess_seconds + score.completion_seconds:6.1f}s"
            # The polish's own cost, separately: a variant that clears the
            # threshold but costs more than the solve it saves is a negative
            # result, and the total above cannot show that.
            + (
                f" (polish {score.notes['polish_seconds']:.1f}s)"
                if "polish_seconds" in score.notes
                else ""
            )
            + ("  [FALLBACK]" if score.used_fallback else ""),
            flush=True,
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--weeks", default="0", help="comma list of week indices")
    parser.add_argument("--hours", type=int, default=168)
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        help="[name=]heuristic[:json options]; repeatable (default: lp)",
    )
    parser.add_argument("--mip-gap", type=float, default=0.005)
    parser.add_argument("--voll", type=float, default=10_000.0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--no-tight", dest="tight", action="store_false")
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="pool identical generators into integer-count clusters first",
    )
    parser.add_argument("--out", default=None, help="write rows to this CSV")
    parser.set_defaults(tight=True)
    args = parser.parse_args()

    variants = [parse_variant(spec) for spec in (args.variant or ["lp"])]
    weeks = [int(w) for w in args.weeks.split(",") if w.strip()]
    system = load_system(args.data_dir)

    print(
        f"scoring {len(variants)} variant(s) over {len(weeks)} week(s) of "
        f"{args.hours} h at mip_gap={args.mip_gap}",
        flush=True,
    )
    rows = []
    started = time.perf_counter()
    for week in weeks:
        start_hour = week * args.hours
        if start_hour + args.hours > system.num_hours:
            print(f"  week{week:02d}: past the end of the data, skipped", flush=True)
            continue
        print(f"  week{week:02d} @ hour {start_hour}", flush=True)
        rows.extend(score_week(system, start_hour, args, variants))

    frame = pd.DataFrame(rows)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.out, index=False)
        print(f"\nwrote {len(frame)} rows to {args.out}")

    scored = frame[frame.get("threshold_margin").notna()] if len(frame) else frame
    if len(scored):
        print("\nmean margin by variant (lower is better):")
        summary = (
            scored.groupby("name")["threshold_margin"].agg(["mean", "min", "max"]) * 100
        )
        summary["weeks"] = scored.groupby("name")["threshold_margin"].size()
        summary["shed_MWh"] = scored.groupby("name")["shed_mwh"].sum()
        print(summary.round(3).to_string())
    print(f"\ntotal {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
