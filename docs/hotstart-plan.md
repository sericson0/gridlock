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

**But the margin is a sufficient condition, not a difficulty predictor.**
The seed experiment above holds one model fixed and varies only the start,
and within that comparison it is decisive. *Across* instances it ranks
nothing — see the baseline below, where the two weeks that solved at one
node had *worse* margins (3.4%, 5.0%) than two that timed out (1.5%,
2.4%). What differs is whether HiGHS's root loop can close the gap on its
own from whatever start it is given. Our start matters only on the weeks
where it cannot — which are exactly the slow weeks, so the target is still
right, but a low margin on an easy week buys nothing and cross-week margin
comparisons mean nothing.

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

Guess quality was previously unmeasurable without solving the MIP. It now
costs one LP plus one completion — 45–90 s on RTS-GMLC at 168 h, against
solves of 270–1,200 s+. `RunResults.window_stats` carries, per window:

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

- **The reported margin is conservative, but barely so on RTS-GMLC.** It
  uses the LP relaxation, which is knowable *before* the solve — that is
  the point, since it lets a guess be scored without solving anything.
  HiGHS's real root bound is the LP plus whatever its cut loop adds. On the
  example system that lift is 0.48%, enough to turn a true margin of 2.19%
  into a reported 2.68%; on RTS-GMLC it is 0.012–0.056% across six weeks,
  so the two margins agree to within 0.06 points. Grade retrospectively
  against `root_bound` anyway — it is free — but the conservatism is not a
  practical obstacle on the real system.
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

## Baseline

Six RTS-GMLC weeks spread across the year, `heuristic=lp`, `fixing=off`,
`mip_gap=0.005`, 1,200 s limit, sequential on an otherwise idle machine
(`results/weekly/rts_gmlc_margin/`):

| week | margin | margin vs `root_bound` | cut lift | nodes | wall | end gap | termination |
|---|---|---|---|---|---|---|---|
| 00 | +2.384% | +2.341% | 0.042% | 247 | 1,293 s | 1.32% | time limit |
| 09 | +1.533% | +1.477% | 0.056% | 420 | 1,275 s | 0.96% | time limit |
| 18 | +5.421% | +5.389% | 0.031% | 754 | 1,269 s | 2.72% | time limit |
| 26 | +3.440% | +3.425% | 0.014% | **1** | 388 s | 0.28% | optimal |
| 35 | +5.034% | +5.021% | 0.012% | **1** | 271 s | 0.28% | optimal |
| 44 | +56.04% | +55.98% | 0.033% | 911 | 1,249 s | 1.15% | time limit |

Four observations, in descending order of consequence.

**The metric reproduces the hand-run experiment.** Week00 covers hours
0–167 — month01's segment — and scores +2.384% against the +2.38% that was
previously established by manually reseeding the model three times. The
automated number measures what the expensive experiment measured.

**Margin does not rank weeks by difficulty** (see the caveat above). Weeks
26 and 35 solved at one node from starts 3.4% and 5.0% out; weeks 00 and 09
timed out from 2.4% and 1.5%. HiGHS's first feasible incumbent is our start
in all six cases, so the warm start is always taken; what varies is whether
the root loop can improve on it unaided. On weeks 26/35 it closed to 0.28%
without branching. On the other four it could not.

**Week44's +56% is one bug, and step 0 found it.** Its completion sheds
259 MWh — $2.59 M at VOLL, which is 55 of the 56 points. Net of shed it is
+0.85%, ordinary. The schedule itself is fine (98.3% match, 1,414 committed
unit-hours against the optimum's 1,421), and no unit at any shedding node
was left off in an hour the optimum had it on, so this is not a missing
commitment. `enforce_adequacy` passed it: that test is static, per hour, and
bounds imports by *total incident line capacity*, which assumes the rest of
the system can spare it. Feasibility of the fixed-commitment dispatch also
needs ramps, simultaneous network flow and storage to work out, and on six
hours of week44 they do not. See 1f.

**Four of six weeks never reached the 0.5% tolerance in 1,200 s**, so the
node and wall columns are censored and are a weak baseline for measuring
track 1 against. Margins are unaffected — they are computed before the
solve. Re-run the timing baseline at a longer limit when a candidate is
ready to be measured.

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

**1f. Shed-driven adequacy repair.** `enforce_adequacy` estimates whether a
schedule *can* serve load; the completion LP *knows*. Read the shed off the
completed solution, commit more units at the shedding node-hours (cheapest
available locally first), re-complete, repeat two or three times. That
replaces a static optimistic bound with the dispatch problem's own verdict,
and it costs one extra LP per iteration. On week44 it is worth 55 points of
margin on its own — the largest single defect the baseline exposed, and the
only one that is a correctness *risk* rather than a quality one: with
`fixing=off` a shedding completion merely wastes the warm start, but a
screen that pinned such entries would hand the MIP a schedule that must
shed. The 12-week study's certain-mask happened not to cover the offending
hours here, which is luck rather than a guarantee.

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

**3a. Sub-MIP polish — built.** `gridlock/polish.py`, switched on with
`RunConfig.polish_options` (None, the default, is the old behaviour
exactly). Pin the screen, solve the restricted MIP, **release every
fixing**, warm start the true model with what comes back. Releasing before
the real solve is what keeps the run exact; a restricted optimum is not a
bound on the true problem, so the sub-MIP's dual bound is discarded rather
than returned.

Weeks 00/09/26 at 168 h, entry screen (pin what the guess vouches for,
entry by entry), sub-MIP gap 5e-4, on a contended machine:

| week | free binaries | baseline | 120 s budget | longer budget |
|---|---|---|---|---|
| 00 | 601 (4.9%) | +2.384% | +2.384% | **+0.618%** — 600 s, optimal in 550 s, 150 nodes |
| 09 | 702 (5.7%) | +1.533% | +1.533% | **+0.695%** — 900 s, optimal in 637 s, 430 nodes |
| 26 | 310 (2.5%) | +3.440% | **−0.211%** | not run: it already clears |

Four findings, none of which was visible before it was built.

**The budget binds, not the idea.** At 168 h the sub-MIP reaches 0–2 nodes
in 120 s — presolve, the root LP and the root cut loop eat the whole
budget — so two minutes bought nothing on the two weeks whose margin is
real commitment cost. Given ten, both solve their restricted problem to
optimality and hand back 1.77 of week00's 2.38 points and 0.84 of week09's
1.53. Week09's polished schedule commits 2,309 unit-hours against the
guess's 2,458: the monotone +2.7% over-commitment, removed directly. On a
48 h slice, where the sub-MIP converges in 38 s, +14.13% becomes +0.34%.

**The screen that is right for a fixing is wrong for a polish.**
Unit-level integrality is 7x safer to pin, but nothing pinned here
survives the call, so the only question is how much sub-MIP fits in the
budget: the unit screen frees 3,192 of week00's binaries against the entry
screen's 601, and at 168 h it lost (−0.044% vs −0.211% on week26, nothing
on 00/09). It edges ahead only at 48 h, where both fit in the budget and
freedom is worth more than speed (+0.254% vs +0.337%, at 3x the time and
without converging). Dilating the free set by ±6 hours around each
contested entry (1,132 free) also lost — +0.667% against +0.618%, and it
was still 0.83% from its own optimum when the 600 s ran out.

**The restriction has a floor, and on both hard weeks the floor is above
the threshold.** Solved to optimality the entry screen still leaves
+0.618% and +0.695% — week00's restricted optimum is 5,713,816 against a
threshold of 5,678,714 and the known feasible 5,676,660, so its pinned
values exclude every schedule that would clear. More time cannot fix that;
only a screen that frees more can, and the two screens that free more are
already too slow to converge. The polish lowers the margin a long way and
then stops, at a floor that is suspiciously similar on both weeks and is
worth measuring on more of them.

**Most of week26's win was shed, not commitment.** Its guess sheds 54.1
MWh — the 1f defect — and the sub-MIP's dispatch prices that at VOLL while
its free commitments can respond, so 99.4% of the $544k recovered was
unserved energy. That also explains the split: week26 clears in two
minutes because its money is sitting in the root LP, while weeks 00 and 09
(3.5 and 0 MWh shed) have to find theirs by branching.

What is *not* measured is the only question that decides whether this is
worth running: what the polished start does to the solve after it. The
margin says week26's would end at one node; for weeks 00 and 09 it says
only that the start is much better, and 550–640 s of polish has to be
weighed against solves that were still 1.32% and 0.96% from their bound
when 1,200 s ran out. Both need a real run at a limit long enough not to
censor it.

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
| 1 | shed repair (1f) → decommit pass (1a) → DP rounding (1b) → price iteration (1c) | 1f first: it is the largest measured defect and the only safety one |
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
- The six-week baseline ran with a 1,200 s limit, which four weeks hit. Its
  node and wall figures are lower bounds on the work those weeks need.
