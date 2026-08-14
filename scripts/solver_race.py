"""Race another solver on an exported gridlock MIP: is the slow bound ours or HiGHS's?

On warm-started clustered weeks the binding constraint is the dual bound
(HiGHS lifts it ~0.07-0.10%/hour and every knob tried moves it the wrong
way). This script measures whether that is a property of the *instance* or
of the *solver*: export the exact model HiGHS solves (via the appsi
translation, so it is byte-for-byte the same MIP), then run SCIP and a cold
HiGHS on the same file with the same gap and time limit, and compare where
the dual bound gets to.

    python scripts/solver_race.py --data-dir data/rts_gmlc --week 9 \\
        --cluster --time-limit 1800 --out results/race_wk09.csv

Reading it, for week09 clustered (numbers from results/tune_wk09_1512.csv):
HiGHS's root bound is ~6,497,839 and after 1,800 s of branching its bound
sits ~6,501,695. The best known incumbent (6,527,681) certifies at 0.5%
once the bound reaches ~6,495,043 — i.e. the *root* already certifies it —
while the polished incumbent (6,545,844) needs ~6,513,115, which HiGHS
never approaches. A racer whose root bound or 30-minute bound lands
materially above HiGHS's says the gap is closable with machinery HiGHS
lacks (cut families, stronger root processing); a racer that stalls at the
same numbers says the bound is structural and the remedy is model-side
(tightening, external bounds), not solver-side.

No MIP start is passed: bound progress is what is being raced, and the
measured HiGHS runs show its bound trajectory is essentially independent
of the incumbent it holds.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from gridlock.config import RunConfig, SolverSettings
from gridlock.data import cluster_identical_units, load_system
from gridlock.model import build_model
from gridlock.solver import HighsSession

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_weekly import slice_hours  # noqa: E402  (sibling script, same directory)


def export_week(args) -> Path:
    """Build the week's model exactly as the experiments do and write MPS."""
    system = load_system(args.data_dir)
    start = args.week * args.hours
    sliced = slice_hours(system, start, start + args.hours)
    if args.cluster:
        sliced = cluster_identical_units(sliced)
    config = RunConfig(
        unit_commitment=True,
        tight_generation_limits=args.tight,
        tight_ramp_limits=args.tight,
        voll=args.voll,
    )
    model = build_model(sliced, config, list(range(args.hours)))
    label = f"wk{args.week:02d}_{'cluster' if args.cluster else 'unit'}"
    label += "_tight" if args.tight else "_loose"
    path = Path(args.mps_dir) / f"{label}.mps"
    path.parent.mkdir(parents=True, exist_ok=True)
    HighsSession(model).write_model(path)
    print(f"wrote {path}", flush=True)
    return path


def race_scip(mps: Path, gap: float, seconds: float) -> dict:
    from pyscipopt import Model

    model = Model()
    model.hideOutput()
    model.readProblem(str(mps))
    model.setParam("limits/gap", gap)
    model.setParam("limits/time", seconds)
    started = time.perf_counter()
    model.optimize()
    wall = time.perf_counter() - started
    return {
        "solver": "scip",
        "status": model.getStatus(),
        "primal": model.getPrimalbound(),
        "dual_bound": model.getDualbound(),
        "root_dual_bound": model.getDualboundRoot(),
        "gap": model.getGap(),
        "nodes": model.getNNodes(),
        "wall_seconds": wall,
    }


def race_highs(mps: Path, gap: float, seconds: float) -> dict:
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.readModel(str(mps))
    h.setOptionValue("mip_rel_gap", gap)
    h.setOptionValue("time_limit", seconds)
    started = time.perf_counter()
    h.run()
    wall = time.perf_counter() - started
    info = h.getInfo()
    return {
        "solver": "highs",
        "status": h.modelStatusToString(h.getModelStatus()),
        "primal": info.objective_function_value,
        "dual_bound": info.mip_dual_bound,
        "root_dual_bound": None,  # in the HiGHS log only; tune CSVs carry it
        "gap": info.mip_gap,
        "nodes": info.mip_node_count,
        "wall_seconds": wall,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir")
    parser.add_argument("--week", type=int, default=9)
    parser.add_argument("--hours", type=int, default=168)
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--no-tight", dest="tight", action="store_false")
    parser.add_argument("--voll", type=float, default=10_000.0)
    parser.add_argument("--mps", help="race an existing MPS instead of exporting")
    parser.add_argument("--mps-dir", default="results/mps")
    parser.add_argument("--solvers", default="scip,highs")
    parser.add_argument("--gap", type=float, default=0.005)
    parser.add_argument("--time-limit", type=float, default=1800.0)
    parser.add_argument("--out", default=None)
    parser.set_defaults(tight=True)
    args = parser.parse_args()

    if args.mps:
        mps = Path(args.mps)
    else:
        if not args.data_dir:
            raise SystemExit("either --mps or --data-dir is required")
        mps = export_week(args)

    racers = {"scip": race_scip, "highs": race_highs}
    rows = []
    for name in [s.strip() for s in args.solvers.split(",") if s.strip()]:
        if name not in racers:
            raise SystemExit(f"unknown solver '{name}' (available: {list(racers)})")
        print(f"racing {name} on {mps.name} ({args.time_limit:.0f}s cap)...", flush=True)
        row = {"mps": mps.name, "gap_target": args.gap, **racers[name](mps, args.gap, args.time_limit)}
        rows.append(row)
        print(
            f"  {name}: {row['status']}  primal {row['primal']:,.0f}  "
            f"bound {row['dual_bound']:,.0f}"
            + (
                f"  root {row['root_dual_bound']:,.0f}"
                if row["root_dual_bound"] is not None
                else ""
            )
            + f"  nodes {row['nodes']}  {row['wall_seconds']:.0f}s",
            flush=True,
        )

    if args.out and rows:
        frame = pd.DataFrame(rows)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.out, index=False)
        print(f"wrote {len(frame)} rows to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
