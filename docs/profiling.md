# Profiling and benchmarking

gridlock exists to study how formulation and solver settings affect
runtime, so it ships with instrumentation to answer the only question that
matters when you change something: **did the solver's job get easier, or
did the clock just wobble?**

## The workflow

```bash
# 1. Record a baseline on the current code (3 trials per case)
gridlock profile --repeat 3 --tag baseline

# 2. Change the formulation / solver code

# 3. Record the candidate and diff it against the baseline
gridlock profile --repeat 3 --tag my-change
gridlock compare benchmarks/<stamp>_baseline.jsonl benchmarks/<stamp>_my-change.jsonl
```

`gridlock profile` solves a **fixed suite** of cases on the example dataset
(`--data-dir` points it elsewhere; `--cases` filters; `--list` shows the
matrix):

| case | what it isolates |
|---|---|
| `lp_day_mono` | 24 h LP — fixed overhead; the solve itself is trivial |
| `lp_month_mono` | 720 h LP — build/translate scaling and pure LP time |
| `uc_week_mono` | 168 h UC MIP, 0.1% gap — **the core UC benchmark** |
| `uc_month_rolling` | 720 h UC, weekly windows + lookahead, 0.5% gap — rolling machinery and state carry |

Keep the suite fixed across experiments: comparability is the whole point.
Ad-hoc solver-option exploration belongs in `gridlock benchmark`
(HiGHS presets) or `gridlock run --highs-option ...`; once an idea looks
good, measure it properly with `profile`/`compare`.

### The scale suite

`--suite scale` runs a separate, deliberately long horizon-scaling study:
one month / two months / one quarter / six months / a full year, each
solved both monolithically and in weekly rolling windows, at a 0.5% gap.

```bash
gridlock profile --suite scale --tag baseline-scale
gridlock profile --suite scale --cases uc_month_mono --repeat 3   # one case, 3 trials
```

This answers the decomposition question the model exists to study: how
fast does monolithic UC degrade as the horizon grows, and where does
rolling overtake it? Monolithic cases carry a 30-minute per-solve cap;
hitting it is itself a result (it marks the horizon where that mode stops
being tractable) and shows up as a non-optimal termination with the
achieved gap recorded. Run the whole sweep only when you mean it — it is
well over an hour — and prefer `--cases` to walk up the horizons one at a
time.

Each (case, trial) appends one JSON line to `benchmarks/<stamp>_<tag>.jsonl`
containing the headline totals, the full per-window metric table, a
per-component model census, the exact `RunConfig`, and the environment
(git commit + dirty flag, package versions, hardware). A record file is
self-describing: you can still interpret it months later.

## What is measured, and what each metric is for

**Time is split into four stages** so a regression can be located, not just
observed:

| metric | meaning |
|---|---|
| `build_seconds` | Pyomo model construction (`build_model`) |
| `translate_seconds` | appsi's Pyomo→HiGHS translation + interface overhead |
| `highs_run_seconds` | HiGHS's own clock: the actual solve |
| `extract_seconds` | pulling variable values into result frames |

On the example system the translation is several times the build — worth
knowing before attributing an LP "solve time" problem to the solver.

**Work counts are the low-noise signal.** `simplex_iterations`,
`ipm_iterations`, `mip_nodes`: if iterations/nodes drop, the algorithm
genuinely did less work; if seconds drop but work counts don't, you
probably measured machine noise. A `mip_nodes` of 0–1 with long
`highs_run_seconds` means the root node (cuts + heuristics + root LP)
dominates — improving the formulation's tightness or the root LP speed
will matter more than anything that helps tree search.

**Problem size, before and after presolve.** `num_rows/cols/nonzeros/
integer_vars` describe the model handed to HiGHS;
`presolved_rows/cols/nonzeros/binaries` (parsed from the HiGHS log,
`profile=True` runs only) describe what survives presolve. The presolved
size is the honest measure of a formulation: rows that presolve strips
were never hurting you, and a "tightening" that grows the presolved model
may not be a tightening at all. `RunResults.component_stats` attributes
the pre-presolve size to individual constraint blocks (`ramp_up`,
`min_up_time`, `load_balance`, ...).

**MIP quality:** `final_mip_gap`, `bound`, `primal_dual_integral`, and the
presolve/solve/postsolve phase-time split.

**Root-loop attribution** (`profile=True` MIP solves, HiGHS ≥ 1.15 log
format). The UC MIP solves at the root node, so knowing *which part* of
root processing is slow decides what to fix:

| metric | meaning |
|---|---|
| `solve_main_mip_seconds` / `solve_submip_seconds` | solve-phase time in the main model vs the sub-MIP heuristics (RINS/RENS-style); `submip_calls` counts them |
| `lp_iters_separation` / `lp_iters_heuristics` / `lp_iters_strong_branching` | LP iterations by purpose — "the cut loop is slow" and "the heuristics are slow" have entirely different fixes |
| `first_feasible_seconds` / `first_feasible_objective` | when the first incumbent appeared and how good it was — the primal-side story |
| `mip_restarts` | root restarts (each re-runs presolve on the cut-strengthened model) |
| `final_cuts_in_lp` | cut rows still in the LP at the end |
| `mip_timeline_json` | the progress table as JSON: bound/incumbent/gap/cuts/LP-iteration trajectory over time |

On the example system the sub-MIP heuristics — not the cut loop — dominate
the root: a warm start (below) removes exactly that cost.

**Numerics:** matrix/cost/bound/RHS coefficient ranges from the log. If
the cost range spans many orders of magnitude (VOLL vs small VOM values),
slow or unstable LPs may be a scaling problem, not a formulation problem.

**Peak memory** (`peak_memory_mb`, `memory_growth_mb` in benchmark
records) is sampled on a background thread during each case, with the
machine's total RAM recorded in `env`. On long monolithic horizons memory
becomes a binding constraint before time does, and "it thrashed" and "it
was slow" need different fixes. Sampling — rather than reading the OS
high-water counter — is what makes the figure per-case: that counter is a
process-lifetime peak and would otherwise report the biggest case for
every case after it.

All of this also lands in `RunResults.window_stats` (one row per window)
on any API run — set `RunConfig(profile=True)` to include the log-derived
fields:

```python
from gridlock import RunConfig, load_system, run

results = run(load_system("data/example"), RunConfig(profile=True, num_hours=168))
print(results.window_stats.T)       # every metric, one column per window
print(results.component_stats)      # rows/vars per constraint block
```

## Repeated solves of one model

Experiments that solve the *same* model many times (option sweeps, warm
starts) should not re-pay the Pyomo build and appsi translation each time —
at annual scale the translation alone rivals the LP solve. `HighsSession`
binds a model to a persistent solver interface; the first solve translates,
later solves reuse:

```python
from gridlock.config import RunConfig, SolverSettings
from gridlock.model import build_model
from gridlock.solver import HighsSession

model = build_model(system, RunConfig(), list(range(8760)))
session = HighsSession(model)
base, _ = session.solve(SolverSettings())                                # translates once
ipm, _ = session.solve(SolverSettings(highs_options={"solver": "hipo"})) # pure solver time
session.write_model("case.mps")   # export for Pyomo-free highspy experiments
```

Options are reset to HiGHS defaults between solves so settings can't leak
from one experiment into the next. After a solve the model holds its
solution, so `session.solve(..., warmstart=True)` re-starts from it.

## Warm starts and the cyclic wrap

Two `RunConfig` switches exist for root-node research on monolithic runs:

- `warmstart_window_hours=N` (CLI `--warmstart-window N`): solve the
  horizon in N-hour rolling windows first, then hand that solution to
  HiGHS as a MIP start for the monolithic solve. This attacks the primal
  side: at long horizons HiGHS's own heuristics fail to find a good
  incumbent (the annual case times out at a ~53% gap with the *bound*
  nearly converged). The pre-pass cost lands in `warmstart_seconds`.
- `cyclic=False` (CLI `--no-cyclic`): drop the first-hour wrap rows from
  commitment logic, min up/down and ramps, leaving the first hours free
  exactly like a rolling run's first window (storage SOC stays cyclic).
  Relaxing the wrap can only lower cost; measure what it buys the solver.

## How to read a comparison

```
  -> uc_week_mono: 18.2% faster
  ~  lp_month_mono: within noise (+1.1% vs spread 3.4%)
  !! uc_month_rolling: COST MISMATCH — the change altered the answer, not just the speed
```

- The headline diff is on `highs_seconds` (pure solver time) and uses the
  **minimum across trials** — the least-noise estimator for timings.
- `significant` requires the delta to beat both 5% and the baseline's own
  cross-trial spread. One trial per case gives no spread estimate, which
  is why `--repeat 3` is the recommended minimum for decisions.
- **`cost_check` is a correctness gate.** Costs must agree within the two
  runs' summed MIP gaps; a MISMATCH means the change altered the model's
  answer and the timing column is irrelevant until that's explained.
- Look at `base_nodes`/`cand_nodes` (and iterations in the record file)
  before believing a timing delta: work counts should move with time.

## MIP timing variance

MIP solve times are notoriously sensitive to random seeds — 2× swings
from performance variability alone are normal. The harness confronts this
instead of hiding it: trials after the first re-solve with a different
HiGHS `random_seed` (`--no-vary-seed` disables), so the cross-trial spread
*includes* seed sensitivity. A change is only real when it clears that
spread, or when the work counts corroborate it. Machine background load
still matters: close heavy applications, don't benchmark on battery
power, and prefer comparing files recorded on the same machine (the
records carry hostname and hardware so mixed comparisons are at least
visible).

## Extending the suite

Cases are plain objects; add system-specific ones in a script:

```python
from gridlock import BenchCase, RunConfig, SolverSettings
from gridlock.bench import default_suite, run_suite, compare_files

suite = default_suite() + [
    BenchCase(
        "uc_week_tight",
        "168 h UC at 0.01% gap — stresses tree search instead of the root",
        RunConfig(unit_commitment=True, num_hours=168,
                  solver=SolverSettings(mip_gap=0.0001, time_limit=600)),
    ),
]
records, path = run_suite("data/example", suite, repeats=3, tag="tight")
```

If you change the standard suite's cases, old record files stop being
comparable — bump the case name (e.g. `uc_week_mono_v2`) instead of
silently changing its definition.
