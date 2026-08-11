# Hot-start plan

How to give the UC MIP a starting schedule good enough that it never
branches. This is a research plan, not a spec: it records what is measured,
what follows from it, and the order the work should be attempted in.

## The target is a threshold, not accuracy

Branch-and-bound stops when `(incumbent - bound) / incumbent <= mip_gap`.
An incumbent at or below

```
threshold = root_bound / (1 - mip_gap)
```

proves the tolerance against the *root* bound, so the solve ends at one
node. That number — not model size, not variable count — governs solve
time. RTS-GMLC month01, identical unrestricted model, only the seed
schedule differs:

| seed | vs threshold | nodes | simplex | HiGHS |
|---|---|---|---|---|
| 5,814,078 (lp guess) | +2.38% | 1,560 | 1.41M | 1,732 s |
| 5,685,140 | +0.113% | 197 | 639k | 873 s |
| 5,676,660 | −0.036% | **1** | 57k | **28 s** |

There is no gradient. A start 0.113% from optimal buys 2x; clearing the
threshold buys 20x. Everything below is aimed at that one number.

**The corollary that matters most:** share-of-binaries-predicted-correctly
is the wrong scoreboard. The month01 guess matched 96.9% of commitments and
still landed 2.38% above the threshold. Optimise the margin, report the
match percentage as colour.

## What the guess gets wrong

Comparing the delivered guess against the solved schedule across all 12
RTS-GMLC months:

| | guess | MIP | delta |
|---|---|---|---|
| committed unit-hours | 34,738 | 33,840 | **+2.7%** |
| startups | 476 | 499 | −4.6% |
| entries guess=1, MIP=0 | 1,735 | | |
| entries guess=0, MIP=1 | 837 | | |
| raw LP on-hours (fractional sum) | 33,738 | 33,840 | **−0.3%** |

Two things follow.

**The LP relaxation is well calibrated; the pipeline downstream of it is
not.** The raw fractional `u` sums to within 0.3% of the optimum's
commitment. The delivered guess is 2.7% over. All of that bias is
introduced by `round()`, `repair_min_up_down` and `enforce_adequacy` — and
it is introduced *by design*, because each of those only ever commits more
(the monotone direction is what makes fixing safe: an under-committed guess
gets repaired by the solver as load shedding at VOLL). Nothing anywhere in
the pipeline ever removes a commitment.

**The excess is extended runs, not extra starts.** Startups actually come
in 4.6% *low*. So the wasted money is no-load plus min-output fuel on units
that stay committed too long, which is exactly the cost a decommitment pass
recovers.

## Step 0 — instrumentation (done)

Guess quality was previously unmeasurable without solving the MIP. It is
now a ~20 second LP. `RunResults.window_stats` carries, per window:

| column | meaning |
|---|---|
| `heuristic_completion_objective` | what the guessed schedule actually costs, dispatch optimised against it |
| `heuristic_lp_bound` | the LP relaxation's objective (`lp`/`ensemble` only) |
| `heuristic_threshold_objective` | `lp_bound / (1 - mip_gap)` |
| `heuristic_threshold_margin` | `completion / threshold - 1` — **≤ 0 is a one-node solve** |
| `root_bound` | LP bound *plus* the root cut loop, from the log (profile mode) |
| `mip_start_status` | HiGHS's own verdict on the start it was handed |
| `mip_start_objective` | the accepted start's objective, as HiGHS read it |

`scripts/run_weekly.py` writes all of them into `summary.jsonl`.

Three notes on reading them:

- **The reported margin is conservative.** It uses the LP relaxation, which
  is knowable *before* the solve — that is the point, since it lets a guess
  be scored without solving anything. HiGHS's real root bound is the LP
  plus whatever its cut loop adds, and that lift is not small: 0.48% on the
  example system, which turned a true margin of 2.19% into a reported
  2.68%. Grade retrospectively against `root_bound`; steer prospectively
  against `heuristic_threshold_margin`.
- **`mip_start_status` exists to catch a silent failure.** A completion
  that had to relax the minimum-output rows produces a start HiGHS rejects,
  and a rejected start leaves no trace in the objective or the node count —
  the run measures a cold solve wearing a warm start's name. It needs
  `config.profile=True` because it is read from the log. Verified against a
  real run: HiGHS logs `MIP start solution is feasible, objective value is
  ...` and that objective matches `heuristic_completion_objective` to 12
  digits.
- **The margin is withheld when `heuristic_fallback` is true.** A relaxed
  completion solved a weaker model, so its objective understates what the
  schedule really costs — optimistic in precisely the direction that
  misleads. The raw objective is still reported.

Also fixed in passing: `lp_relaxation_guess` and `similar_days_guess` built
their sub-configs without `config.voll`, so a non-default VOLL left the LP
pricing unserved energy differently from the MIP it is supposed to bound.
That would have made the threshold arithmetic invalid.

## Track 1 — structure

**1a. A decommitment pass.** The +2.7% bias has no counter-pass. Rank each
committed run by `(no-load + startup) - LP dispatch surplus`, tentatively
drop the worst, re-solve the fixed-`u` LP off the retained basis, keep if
feasible and cheaper. Greedy and monotone. Cheap now that the appsi
fixed-vars tax is gone (a re-solve is ~0.1–1 s, not ~1,000 s). This is the
smallest change with the largest expected effect, because it attacks the
one defect that has actually been measured.

**1b. Per-unit DP instead of round-and-repair.** The principled version of
1a. Take nodal prices from the LP duals (`want_duals` is already plumbed
through `HighsSession.solve`, just unused on the LP path) and solve each
unit's single-unit UC *exactly* by dynamic programming over `(hour, on/off,
hours-in-state)`, with startup cost and min up/down as transitions and
`(λ_{n,t} - mc)·p - no_load` as the per-hour value. Roughly 168 × 24 states
× 73 units — milliseconds. This is the thing merit-order stacking
structurally cannot do: a stack knows what capacity is *needed*, a DP knows
whether a three-hour start *pays for itself*, and per the ensemble study the
surviving errors are precisely whole missed starts on fast peakers in hours
where merit order says nothing needs to run.

**1c. Price iteration.** Three to five subgradient or bundle updates on λ,
one fixed-`u` LP per round. Textbook Lagrangian UC closes to ~0.5–1%
duality gap, which is the right order of magnitude for the threshold.

**1d. Counts over identities.** Established but unimplemented: constrain
`sum_t u[class,t] == round(LP class total)` for the near-symmetric classes
(CC-355, Steam-350) where that total is integral, leave identity free, and
add symmetry-ordering rows `u[i,t] >= u[i+1,t]` within the 32 chains of
units identical on node and every parameter. Unlike a better guess, this
also shrinks the search itself.

**1e. `deep_money >= 0.95` prescreen.** Units whose net load exceeds all
cheaper capacity plus their own output for ≥95% of hours commit all week:
zero errors over 12 weeks, ~630 binaries/week. Free.

## Track 2 — machine learning

The study already establishes two constraints that rule out the obvious
framings. Accuracy is not the bottleneck (the LP alone is 98.3%);
*calibration* is. And per-unit identity labels are permutation noise —
`315_STEAM_1` and `315_STEAM_4` are identical in every parameter at the same
bus, and one commits while the other never does.

**2a. Label canonicalization — the prerequisite.** Within each group
identical on node and all parameters, sort the solved commitment columns
lexicographically before using them as labels. This converts "which
arbitrary tie did the solver break" into a well-posed target. Without it
the standing advice — never train a per-unit estimator on these labels —
holds. With it, per-unit learning becomes legitimate. Do this first; it is
cheap and every downstream learner benefits.

**2b. A learned confidence model.** Replace the hand-tuned AND-of-screens in
`ensemble_guess` with a gradient-boosted classifier predicting
`P(LP-integral entry disagrees with the MIP)`. Features: column
fractionality statistics, hours to the nearest transition, merit rank,
`deep_money`, per-heuristic agreement flags, hourly capacity slack, week
load percentile. Yields a continuous coverage/error frontier instead of the
four discrete operating points measured so far, and trains on the 12 weeks
already on disk. Highest-confidence ML item.

**2c. kNN warm start over net-load profiles.** Retrieve the *k* most
load-similar solved instances and transfer their schedules — Xavier, Qiu &
Ahmed (INFORMS J. Computing, 2021), the approach behind UnitCommitment.jl.
The screen study already concluded that reference weeks should be selected
by load similarity rather than calendar adjacency, and that persistence is
92.7% accurate standalone; that *is* kNN, unimplemented. Composes with the
portfolio in 3d.

**2d. Class-count sequence model.** A temporal CNN or GRU over 168 h of net
load and availability predicting per-class hourly committed counts. Targets
are well-posed without 2a, and it pairs directly with 1d's count
constraints.

**2e. GNN / neural diving.** Gasse et al. (2019) for the bipartite
constraint-variable encoding; Nair et al. (2020) for learned partial
assignments with a coverage-controlled confidence head, which is 3a
automated. The research endpoint, and premature with 12 labelled weeks.
Revisit after 2f.

**2f. Data generation is the binding constraint** for anything past 2b.
Target ~10³ instances: perturb load scale and shape, renewable profiles,
forced outages, fuel prices. Two traps — label at a tighter gap than 0.005
or the model learns which incumbent the solver happened to accept inside the
tolerance band, and canonicalize (2a) before storing.

## Track 3 — other approaches

**3a. Sub-MIP polish.** Fix the screen, solve the restricted MIP to
optimality, **release every fixing**, then warm start the true model. The
fractional core is 3–6% of binaries (~400–800 free), so the restricted
problem is small. Releasing before the real solve is what keeps the run
exact while still using the restriction to manufacture an incumbent.

**3b. Fix-and-optimize over time blocks.** Complement to 3a: 3a decomposes
by variable, this by time. Free one overlapping 24–48 h block, fix the
rest, solve, slide; three to five sweeps, each step monotone improving and
each sub-MIP tiny.

**3c. Lift the heuristic ⊥ rolling-pre-pass restriction.** `RunConfig.validate`
forbids combining `heuristic` with `warmstart_window_hours`, but the rolling
pre-pass is the only *measured* near-optimal start producer in the codebase
(~0.03% above the monolithic optimum at annual scale, where it turned a
1,800 s timeout at a 52.9% gap into an optimal solve in 56.6 s). At week
scale, run 24–48 h windows with the LP guess fixed inside each.

**3d. Guess portfolio, best-of-N.** A completion now costs ~1–3 s. Build
5–20 diverse guesses (varying reserve margin, price vector, DP tie-breaks,
kNN neighbours), complete each, keep the minimum. min-of-N beats one draw
substantially, and 20 completions cost ~1% of a hard month's solve.

**3e. The other side of the ratio.** Clearing the threshold can come from
lifting the bound as well as lowering the incumbent. Fixing entries the LP
already resolved provably *cannot* move the bound (verified: root LP
identical to 7 decimals with 0 / 2,498 / 8,848 entries fixed, because the LP
optimum is already feasible for the fixed problem). Only formulation
tightening and cuts move it — and the root cut loop is already worth 0.48%
on the example system, so it is doing real work.

## Sequencing

| step | work | why here |
|---|---|---|
| 0 | threshold + margin + start-acceptance instrumentation | **done** — makes everything below measurable in seconds |
| 1 | decommit pass (1a) → DP rounding (1b) → price iteration (1c) | attacks the one confirmed defect; no ML, no new data |
| 2 | sub-MIP polish (3a) + portfolio (3d) + lift the 3c restriction | the named ceiling candidate, built from existing parts |
| 3 | canonicalization (2a) → confidence model (2b) → kNN (2c) | trains on the 12 weeks already on disk |
| 4 | data generation (2f) → sequence model (2d) → GNN (2e) | only if 1–3 leave the threshold uncleared |

Steps 1 and 2 are independent and can proceed in parallel.

## Caveats

- The +2.38% month01 figure predates the appsi fixed-vars fix. The
  *relative* seed comparison it supports is unaffected (same model, same
  session, only the seed differs), but do not compare its wall-clock
  against current runs.
- Per-month spread is wide: month08's guess is exactly right (over = under =
  0) and months 08/09 already finish at one node. Pick the evaluation set
  from the hard months (01, 02, 03, 10) deliberately rather than averaging
  over all twelve.
- Error rates quoted from the 12-week study are upper bounds: the labels are
  0.5%-gap incumbents, not proven optima.
- The 61.7x month01 result used an oracle start lifted from an earlier
  solve. It is the ceiling this plan aims at, not a result already achieved.
