"""A/B HiGHS option profiles against one prepared warm-started instance.

The recorded verdict that "no HiGHS knob moves the needle" was measured on
the example system, cold, in a regime with no warm start and no tree. Both
premises are now false, and the six real RTS-GMLC solves say where the time
goes: 13-45% of HiGHS's clock is *sub-MIP primal heuristics*, and on four of
those six runs the final objective equalled the warm start to the dollar --
so every one of those seconds provably bought nothing. Strong branching
costs another 57k-715k LP iterations. Both are tunable; neither was tuned.

This script isolates that question. The expensive part of a run -- the LP
guess, the repair and the sub-MIP polish, ~240 s on a clustered RTS-GMLC
week -- is paid *once*, and every profile then solves the identical model
from the identical warm start through one appsi translation:

    python scripts/tune_highs.py --data-dir data/rts_gmlc --week 44 \\
        --cluster --repair --polish --time-limit 1200

Fairness is the whole point of the design, and three things are needed for
it. Each solve passes ``cold=True`` so HiGHS drops the basis and incumbent
it kept from the previous profile (without this, profile N+1 inherits
profile N's answer and reads as miraculously fast). Each solve restores the
snapshotted variable values first, because appsi sends a *full* solution
vector and a variable left holding the previous solve's value would make
"the same warm start" a lie. And ``mip_start_status`` is recorded per
profile: a rejected start leaves no trace in the objective or the node
count, so without it a cold solve can masquerade as a tuned one.

Read the work counts, not just the clock. MIP timings swing 2x on seed
alone (see docs/profiling.md), so a profile is only interesting when nodes,
LP iterations or sub-MIP seconds move with the time. ``--seeds`` re-runs
every profile at several HiGHS random seeds for exactly that reason.

Profiles are named on the command line so adding one never means editing
this file:

    --profile default                     # a built-in (see PROFILES)
    --profile no_heur                     # another built-in
    --profile mine:{"mip_pscost_minreliable":2,"mip_detect_symmetry":false}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo

from gridlock.config import RunConfig, SolverSettings
from gridlock.data import SystemData, cluster_identical_units, load_system
from gridlock.heuristics import build_guess, complete_solution
from gridlock.model import build_model
from gridlock.polish import polish_guess
from gridlock.solver import HighsSession

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_weekly import slice_hours  # noqa: E402  (sibling script, same directory)


# Each profile is a HiGHS option dict. The groupings follow the three costs
# the run records actually show, so a result attributes to a mechanism
# rather than to "some setting".
PROFILES: dict[str, dict] = {
    # The control. Everything is measured against this, same process.
    "default": {},
    # --- the primal side: work on an incumbent we already believe --------
    # `mip_heuristic_effort` is kept only as a control. Measured on a 48 h
    # clustered RTS-GMLC slice, 0.0 and the 0.05 default produce *identical*
    # work -- 28,762 heuristic LP iterations and 10.4 s of sub-MIP either
    # way -- so it does not gate the root sub-MIP heuristics at all. That is
    # why the earlier option sweep, which tested effort 0.01/0.5/0.9 and
    # nothing else, concluded no knob mattered: it was varying a knob that
    # does not do what its name suggests.
    "effort0": {"mip_heuristic_effort": 0.0},
    # These are the switches that actually bite: each names one sub-MIP
    # heuristic and turns it off outright.
    "no_submip": {
        "mip_heuristic_run_rins": False,
        "mip_heuristic_run_rens": False,
        "mip_heuristic_run_root_reduced_cost": False,
    },
    # Feasibility jump exists to *find* a first feasible solution. We hand
    # one in, so it is answering a question that is already answered -- and
    # unlike the sub-MIP switches it costs no solution quality, because it
    # never improves an incumbent, it only produces a first one.
    "no_fjump": {"mip_heuristic_run_feasibility_jump": False},
    # no_submip and no_fjump together: everything that can be switched off
    # on the primal side.
    "no_primal": {
        "mip_heuristic_run_rins": False,
        "mip_heuristic_run_rens": False,
        "mip_heuristic_run_root_reduced_cost": False,
        "mip_heuristic_run_feasibility_jump": False,
    },
    # --- branching -------------------------------------------------------
    # Default 8 means eight strong-branching evaluations before a pseudocost
    # is trusted. Strong branching is 57k-715k LP iterations on these runs.
    "cheap_branch": {"mip_pscost_minreliable": 1},
    # --- structure HiGHS may be re-deriving ------------------------------
    # Clustering already removed the permutation symmetry, so detecting it
    # again is work with nothing to find. Expect this to matter *only* with
    # --cluster; unclustered, symmetry detection is earning its keep.
    "no_symmetry": {"mip_detect_symmetry": False},
    # A restart re-runs presolve on the cut-strengthened model. Worth having
    # when it tightens the formulation, pure cost when the answer is in hand.
    "no_restart": {"mip_allow_restart": False},
    # --- the dual side, which is what actually binds ---------------------
    # The bound crawls at 0.07-0.11%/hour and that is what censors weeks 00
    # and 09. Keeping cuts alive longer is the only family of knobs that
    # addresses it: age limits govern when a cut is dropped from the LP and
    # from the pool.
    "keep_cuts": {
        "mip_lp_age_limit": 30,
        "mip_pool_age_limit": 60,
        "mip_pool_soft_limit": 20000,
    },
    # --- combinations ----------------------------------------------------
    # Every primal saving redirected at the bound.
    "bound_focus": {
        "mip_heuristic_effort": 0.0,
        "mip_lp_age_limit": 30,
        "mip_pool_age_limit": 60,
        "mip_pool_soft_limit": 20000,
    },
    "combo": {
        "mip_heuristic_run_rins": False,
        "mip_heuristic_run_rens": False,
        "mip_heuristic_run_root_reduced_cost": False,
        "mip_heuristic_run_feasibility_jump": False,
        "mip_detect_symmetry": False,
        "mip_lp_age_limit": 30,
        "mip_pool_age_limit": 60,
    },
}

# Turning the sub-MIP heuristics off is not free, and the screen that says
# it is will mislead you. On a 48 h clustered slice warm started from a
# *repaired but unpolished* guess, `no_submip` cut HiGHS 18.9 s -> 14.9 s
# and left the objective at the warm start's 2,139,349, while the default
# spent that time improving it to 2,132,246 -- 0.33% cheaper. Both report
# "optimal" because both sit inside the 0.5% tolerance. So the switch trades
# money for time whenever the start still has room in it, and only pays
# where the start is already the answer (weeks 00/09/44 at 168 h with the
# polish on, where the final objective equalled the start to the dollar).
# Judge every profile on objective *and* clock, never the clock alone.
#
# Note also that single-seed differences on a 48 h slice are not readable:
# `no_primal` came in 4.4 s *slower* than `no_submip` despite doing strictly
# less work. Trust the counts (nodes, LP iterations, sub-MIP calls); use
# --seeds before trusting a time.


def parse_profile(spec: str) -> tuple[str, dict]:
    """``name`` (a built-in) or ``name:{json}`` (an ad-hoc profile)."""
    name, _, raw = spec.partition(":")
    name = name.strip()
    if not raw:
        if name not in PROFILES:
            raise SystemExit(
                f"unknown profile {name!r} (built-ins: {', '.join(PROFILES)})"
            )
        return name, dict(PROFILES[name])
    try:
        options = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SystemExit(f"--profile {spec!r}: options are not valid JSON ({error})")
    if not isinstance(options, dict):
        raise SystemExit(f"--profile {spec!r}: options must be a JSON object")
    return name, options


def prepare(system: SystemData, args: argparse.Namespace):
    """Build the instance and its warm start; return (session, model, notes).

    Everything expensive and profile-independent happens here exactly once.
    The model is left holding the completed (and optionally polished)
    solution, which is the warm start every profile will be handed.
    """
    sliced = slice_hours(system, args.first_hour, args.first_hour + args.hours)
    if args.cluster:
        sliced = cluster_identical_units(sliced)
    hours = list(range(args.hours))

    config = RunConfig(
        unit_commitment=True,
        cyclic=True,
        heuristic="lp",
        heuristic_repair=args.repair,
        polish_options={} if args.polish else None,
        tight_generation_limits=args.tight,
        tight_ramp_limits=args.tight,
        voll=args.voll,
        profile=True,
        solver=SolverSettings(mip_gap=args.mip_gap, threads=args.threads),
    )

    start = time.perf_counter()
    model = build_model(sliced, config, hours, None)
    session = HighsSession(model)
    guess = build_guess(sliced, config, hours, None)
    completion, fallback = complete_solution(model, guess, session=session)
    objective = completion.objective
    if args.polish:
        polished = polish_guess(
            model,
            guess,
            config,
            session=session,
            warmstart=not fallback,
            baseline_objective=objective,
        )
        if polished.succeeded:
            guess, objective, fallback = polished.guess, polished.info.objective, False
    notes = {
        "prepare_seconds": time.perf_counter() - start,
        "start_objective": objective,
        "used_fallback": bool(fallback),
        "commitment_rows": len(sliced.generators.index[sliced.generators.needs_commitment]),
    }
    return session, model, notes


def snapshot(model: pyo.ConcreteModel) -> dict:
    """Every variable's value, so each profile starts from the same vector.

    appsi sends a full solution vector as the MIP start and reads a ``None``
    value as 0.0, so a partial snapshot would silently hand later profiles a
    different -- and much worse -- start than the first one got.
    """
    return {id(var): var.value for var in model.component_data_objects(pyo.Var)}


def restore(model: pyo.ConcreteModel, values: dict) -> None:
    for var in model.component_data_objects(pyo.Var):
        var.set_value(values[id(var)], skip_validation=True)


def run_profile(
    session: HighsSession,
    model: pyo.ConcreteModel,
    values: dict,
    options: dict,
    args: argparse.Namespace,
    seed: int,
) -> dict:
    """One profile, one seed, from the same model state as every other."""
    restore(model, values)
    highs_options = dict(options)
    highs_options.setdefault("random_seed", seed)
    settings = SolverSettings(
        mip_gap=args.mip_gap,
        time_limit=args.time_limit,
        threads=args.threads,
        highs_options=highs_options,
    )
    info, _ = session.solve(settings, profile=True, warmstart=True, cold=True)
    m = info.metrics
    return {
        "termination": info.termination,
        "objective": info.objective,
        "bound": info.bound,
        "gap": info.gap,
        "highs_seconds": m.highs_run_seconds,
        "nodes": m.mip_nodes,
        "submip_seconds": m.solve_submip_seconds,
        "submip_calls": m.submip_calls,
        "main_mip_seconds": m.solve_main_mip_seconds,
        "it_separation": m.lp_iters_separation,
        "it_heuristics": m.lp_iters_heuristics,
        "it_strong_branching": m.lp_iters_strong_branching,
        "simplex_iterations": m.simplex_iterations,
        "cuts_in_lp": m.final_cuts_in_lp,
        "restarts": m.mip_restarts,
        "root_bound": m.root_bound,
        "presolve_seconds": m.presolve_seconds,
        # The silent-failure guard: a rejected start makes this a cold solve
        # wearing a tuned solve's name, and nothing else in the row shows it.
        "mip_start_status": m.mip_start_status,
        "mip_start_objective": m.mip_start_objective,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--week", type=int, default=None, help="week index (168 h blocks)")
    parser.add_argument("--first-hour", type=int, default=None)
    parser.add_argument("--hours", type=int, default=168)
    parser.add_argument(
        "--profile",
        action="append",
        default=None,
        dest="profiles",
        help="repeatable: a built-in name, or name:{json} (default: all built-ins)",
    )
    parser.add_argument(
        "--seeds",
        default="0",
        help="comma-separated HiGHS random seeds; every profile runs at each "
        "(default: 0). Use 3 seeds before trusting a timing difference.",
    )
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--polish", action="store_true")
    parser.add_argument("--no-tight", dest="tight", action="store_false")
    parser.add_argument("--mip-gap", type=float, default=0.005)
    parser.add_argument("--time-limit", type=float, default=1200.0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--voll", type=float, default=10_000.0)
    parser.add_argument("--out", default=None, help="CSV path (default: results/tune_<tag>.csv)")
    parser.add_argument("--tag", default="tune")
    parser.set_defaults(tight=True)
    args = parser.parse_args()

    if args.first_hour is None:
        if args.week is None:
            raise SystemExit("give --week or --first-hour")
        args.first_hour = args.week * args.hours

    specs = [parse_profile(s) for s in (args.profiles or list(PROFILES))]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    system = load_system(args.data_dir)
    print(
        f"preparing hour {args.first_hour}..{args.first_hour + args.hours} "
        f"(cluster={args.cluster} repair={args.repair} polish={args.polish})",
        flush=True,
    )
    session, model, notes = prepare(system, args)
    print(
        f"  warm start {notes['start_objective']:,.0f} in "
        f"{notes['prepare_seconds']:.0f}s over {notes['commitment_rows']} "
        f"commitment rows"
        + ("  [fallback: start may be rejected]" if notes["used_fallback"] else ""),
        flush=True,
    )
    values = snapshot(model)

    rows = []
    for name, options in specs:
        for seed in seeds:
            row = {"profile": name, "seed": seed, "options": json.dumps(options)}
            try:
                row.update(run_profile(session, model, values, options, args, seed))
            except Exception as error:  # a bad option must not lose the sweep
                row["termination"] = f"FAILED: {error}"
            rows.append(row)
            print(_format(row, notes), flush=True)

    frame = pd.DataFrame(rows)
    out = Path(args.out or f"results/{args.tag}_{args.first_hour}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"\nwrote {len(frame)} rows to {out}")

    ok = frame[frame.termination.isin(["optimal", "maxTimeLimit"])]
    if len(ok):
        best = ok.sort_values("highs_seconds")
        print("\nby HiGHS seconds (work counts corroborate, or it is noise):")
        cols = [
            "profile", "seed", "termination", "highs_seconds", "nodes",
            "submip_seconds", "it_strong_branching", "gap", "objective",
        ]
        print(best[cols].to_string(index=False))
    rejected = frame[frame.mip_start_status.notna() & ~frame.mip_start_status.astype(str).str.contains("feasible", case=False, na=False)]
    if len(rejected):
        print(f"\nWARNING: {len(rejected)} solve(s) did not report a feasible MIP "
              "start -- those rows measure a cold solve, not a tuned one.")
    return 0


def _format(row: dict, notes: dict) -> str:
    if str(row.get("termination", "")).startswith("FAILED"):
        return f"  {row['profile']:<14s} seed {row['seed']}  {row['termination']}"
    seconds = row.get("highs_seconds")
    return (
        f"  {row['profile']:<14s} seed {row['seed']}  "
        f"{str(row['termination']):>12s}  {seconds:7.1f}s  "
        f"nodes {row['nodes'] or 0:6.0f}  submip {row.get('submip_seconds') or 0:6.1f}s  "
        f"gap {100 * (row.get('gap') or 0):5.3f}%  obj {row['objective']:,.0f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
