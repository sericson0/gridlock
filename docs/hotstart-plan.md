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

## Composition (measured)

The three tracks were built independently and then stacked. Weeks 0/9/44,
`heuristic=lp`, 0.5% gap (`results/composition.csv`):

| week | baseline | + repair (1f/1a) | **+ repair + polish (3a)** | + repair + DP (1b) |
|---|---|---|---|---|
| 00 | +2.384% | +0.891% | **+0.478%** | +1.051% |
| 09 | +1.533% | +0.981% | **+0.347%** | +1.437% |
| 44 | +56.036% | +1.542% | **+0.199%** | +4.267% |
| mean | 19.984% | 1.138% | **0.341%** | 2.251% |

**Repair and polish compose; they are not substitutes.** Polish alone
reached +0.618% / +0.695% on weeks 00/09 and repair alone +0.891% /
+0.981%, but stacked they reach +0.478% / +0.347% — better than either,
because they recover different money. The polish also gets *cheaper* on a
repaired guess (322–417 s against 550–640 s standalone), since it starts
from a feasible schedule instead of spending its budget rediscovering one.
Committed unit-hours land at 1,911 on week00 against the relaxation's own
fractional sum of 1,907.

**The DP does not compose, and shed was not what was holding it back.**
Its author inferred from net-of-shed figures that a working adequacy repair
might flip the result. Measured, it does not: with shed at zero for both,
the DP is worse than repair alone on all three weeks. On week44 it is
worse by 2.7 points *and* commits more (1,516 unit-hours against 1,414,
98 startups against 81) — the DP decommits, the schedule then cannot serve
load, and the repair puts back more than the DP removed. The repair pass
alone already beats the DP's net-of-shed score, so the DP's apparent edge
was an artifact of comparing against an unrepaired baseline.

**Nothing clears yet.** The best result is +0.199%. On RTS-GMLC the
LP-derived threshold is conservative by only 0.03–0.06%, so the true
margins are near these. Week00's `rep_pol` start costs ~5,705,900 against a
schedule known to cost 5,676,660 — consistent with the entry screen
excluding it (see 3a), which is the next thing to try.

**Still unmeasured: whether any of this pays.** These are margins. The
repair costs ~120–160 s and the polish a further ~320–420 s, against
baseline solves that were themselves censored at 1,200 s. No MIP has been
solved from a repaired or polished start, so there is no node count and no
wall-clock. That is the next measurement, and it is the one that decides
whether the preprocessing earns its keep.

**One flag:** `rep_pol` on week44 leaves 0.08 MWh of shed where `rep` left
zero. Negligible against the margin ($800), but it means the polish can
undo the repair's feasibility guarantee — the sub-MIP is free to decommit
and answers to its own dispatch, not to the repair's invariant.

## The confirming solve: these weeks are bound-limited

Weeks 0/9/44 solved for real from repaired-and-polished starts, 0.5% gap,
3,600 s limit (`results/weekly/rts_gmlc_confirm/`):

| week | start | final objective | match | nodes | bound: LP → final | end gap | baseline (1,200 s) |
|---|---|---|---|---|---|---|---|
| 00 | 5,705,833 | **5,705,833** | 100% | 3,555 | +0.074% | 0.90% | 5,728,835, 1.32% |
| 09 | 6,549,442 | **6,549,442** | 100% | 3,820 | +0.105% | 0.74% | 6,562,759, 0.96% |
| 44 | 4,703,883 | **4,703,883** | 100% | 1,905 | +0.102% | 0.60% | 4,728,008, 1.15% |

**Branch-and-bound improved the incumbent by nothing at all.** Not "a
little" — the final objective equals the start to the dollar on all three
weeks, across 1,905 to 3,820 nodes and an hour of HiGHS each. Every gain
came from ~660 s of preprocessing.

**The binding constraint is the bound, not the incumbent.** To terminate,
week00 needed its bound up 0.404% and got 0.074%; week09 needed 0.242% and
got 0.105%; week44 needed 0.097% and got 0.102% — it very nearly closed.
The bound climbs at roughly 0.07–0.10% per hour on these instances, and
nothing in this plan addresses that.

This refines rather than contradicts the recorded finding that the gap here
is primal-side. Both hold, at different points on the same curve: *without*
a warm start the problem is primal — HiGHS's own heuristics fail and the
incumbent is terrible — but once the start is good, the residual is dual,
and further incumbent work buys literally zero. Every track in this plan
except 3e is incumbent work.

**The threshold theory survives intact.** Against week00's real root bound
of 5,652,683 anything at or below 5,681,088 terminates at one node, and a
schedule costing 5,676,660 is known to exist. The target has not moved; the
start is 0.44% short of it, and the solver demonstrably cannot close that
remainder itself — neither by finding a better incumbent (3,555 nodes, no
improvement) nor by lifting the bound (0.074% in an hour).

Two consequences:

- **The ensemble-screen polish is now the whole game, not a refinement.**
  The 47 LP-integral entries across 8 units that the entry screen pins
  wrongly are exactly what stands between +0.478% and a one-node solve, and
  the historical ensemble run — which pins only what its screen vouches for
  and budgets the residue — is the run that actually found 5,676,660.
- **3e stops being a footnote.** On a warm-started week the bound is the
  whole gap. Formulation tightening and cuts are the only things that move
  it; no guess ever will.

Worth keeping even so: the polished runs return answers 0.20–0.51% cheaper
than the censored baselines. If the goal is a good schedule rather than a
proof of optimality, the preprocessing pays even when the solve times out.
And a strong incumbent makes nodes cheap — 14x the nodes in 3x the time on
week00 — it just does not make the bound move.

## Clustering restores branch-and-bound (measured)

Weeks 0/9/44, repaired + polished starts, 0.5% gap, 3,600 s limit, with and
without `cluster_units` (`results/weekly/rts_gmlc_cluster/`):

| week | model | final | nodes | end gap | wall | termination |
|---|---|---|---|---|---|---|
| 00 | unclustered | 5,705,833 | 3,555 | 0.90% | 4,210 s | time limit |
| 00 | clustered | 5,699,877 | 3,526 | 0.78% | 3,790 s | time limit |
| 09 | unclustered | 6,549,442 | 3,820 | 0.74% | 3,774 s | time limit |
| 09 | **clustered** | **6,527,681** | 2,836 | **0.40%** | **2,326 s** | **optimal** |
| 44 | unclustered | 4,703,883 | 1,905 | 0.60% | 3,928 s | time limit |
| 44 | **clustered** | **4,699,316** | **3** | **0.50%** | **645 s** | **optimal** |

**Two of three timeouts become proven-optimal solves**, week44 in 645 s and
three nodes against 3,928 s and 1,905.

**The mechanism is primal, not dual.** The bound lift barely moves
(0.074% → 0.091%, 0.105% → 0.114%, 0.102% → 0.102%). What changes is that
the search can improve the incumbent again:

| week | B&B gain, unclustered | B&B gain, clustered |
|---|---|---|
| 00 | 0.000% | 0.025% |
| 09 | 0.000% | **0.622%** |
| 44 | 0.000% | 0.000% (terminated at 3 nodes) |

Week09 is the proof: its clustered start is *worse* (+0.639% against
+0.347%) and it still finishes optimal, because branch-and-bound closes
0.622% that it could not touch at all on the unit-level model. With 24
clusters covering 56 of 73 units, every improving move on the unit-level
model has a factorial number of equivalent relabellings, so the search
spends itself re-deriving schedules it has already seen. Remove the
symmetry and the same effort finds genuinely new ones.

This qualifies the bound-limited finding above rather than replacing it:
the unit-level model is bound-limited *because* its incumbent search is
paralysed by symmetry, not because the incumbent was already as good as it
could get.

**The clustered answers are exact, verified three ways.** Clustering is
documented as a relaxation (a cluster can shift ramp capability between
members), and the clustered costs come in 0.10–0.33% *below* the
unclustered ones — precisely the signature a relaxation would leave. It is
not one here. Every clustered MIP answer replays on the unit-level model at
the same cost to the cent (6,527,680.85 / 4,699,316.49 / 5,699,876.75),
with zero shed and no relaxed-row fallback.

Getting that right needed the correct disaggregation, and the obvious one
is wrong. Splitting a count by *staircase* (member k on wherever the count
reaches k) leaves members violating their own min-up rows and reads as a
relaxation that isn't there. The aggregate min-up row states that the units
started in the last UT hours never outnumber those running now, so the
recent starters can always be placed among the running units — provided the
member retired is the longest-serving one. A FIFO assignment does that and
produces zero violations on every cluster of both weeks tested.

Exactness follows from the clustering *key*: `cluster_identical_units`
pools only units agreeing on every parameter, at the same node, with the
same availability profile. For exactly identical units the optimal dispatch
splits evenly and the aggregate ramp row is the sum of the per-unit rows.
Loosen that key to pool near-identical units and the relaxation becomes
real immediately.

**Open:** the decommit pass skips clusters, so with clustering on it sees
17 of 73 units, and `rep` alone is 0.55 points worse on weeks 00/09. The
polish substitutes for it (it is a MIP over integer counts and can lower
one), which is why `rep_pol` still wins on 00 and 44 — week09 is where the
substitution fails. A cluster-aware decommit drops the top layer over a
run's shoulder, priced at `no_load - (price - mc)*(p/u)` per member. Also
unbuilt: nothing converts a clustered result frame back to unit-level
schedules, which the FIFO routine would supply.

## The sequencing experiments (measured 2026-08-14)

Four parallel probes, one day. Two positives, two negatives, and the
negatives are the informative ones.

**The margin gate is built and makes the right call on all three weeks.**
`polish_options={"gate": true}` (`gridlock/polish.py`) delivers the polish
only when its margin clears the LP threshold within a slack covering the
measured cut lift (default 0.05%). Weeks 00/09 (+0.227%/+0.292%): polish
discarded, repair start delivered — the delivery the week09 solves proved
faster *and* better. Week44 (+0.047%): kept, preserving the one-node
solve. The gate pays the polish cost either way; what it saves is the
delivery mistake. `results/sequencing_clustered.csv`.

**The ensemble-screen polish loses, clustered and unclustered.** Pinning
only what the ensemble vouches for and budgeting the soft residue inside
the sub-MIP (`polish_options={"soft_budget": N}`) frees too much for the
600 s budget: every run hit the limit and landed *above* the entry screen
— +0.845/+0.737/+0.379 against +0.478/+0.347/+0.199 unclustered, worse
still clustered, where the structural members refuse clusters and the
soft set collapses to single units. The hypothesis above ("the
ensemble-screen polish is now the whole game") fails as a polish at this
budget; whatever found 5,676,660 historically, this restriction is not
its cheap replica. `results/sequencing_unclustered.csv`.

**The one-shot Lagrangian bound is below the LP bound.** Exact per-unit
DPs at the LP's own duals (`gridlock/lagrangian.py`) land 0.10-0.54%
*below* the tight LP bound: the dropped ramp/startup-capability rows carry
more dual value than per-unit integrality adds, and the tight-vs-loose
comparison shows the tight rows already extract the per-unit hull money.
The residual dual gap is cross-unit; no per-unit decomposition reaches it.
`results/lagrangian_probe_*.csv`.

**The bound is structural — and week09's wall was never the bound.** On
the byte-identical exported MPS, cold SCIP's root matches HiGHS's on all
three weeks (±0.01%), killing the "better cut families" hope. But SCIP
*cracked week09 primal-side*: 6,515,801 at 0.24% gap in 1,381 s, clearing
the root threshold by 0.225% — a schedule neither this pipeline (floor
6,545,844) nor HiGHS's own heuristics (no incumbent in 7,000 s cold) ever
found. It does not generalize: SCIP loses to the pipeline on week00
(5,738,179 vs 5,699,877) and badly on week44. Per-week complementarity —
the portfolio argument of 3d, lifted to the solver level. Week00 remains
unbroken by everything. `results/race_wk*.csv`, `scripts/solver_race.py`.

## Track 1 — structure




**1a. A decommitment pass.** The +2.7% bias has no counter-pass. Rank each
committed run by `(no-load + startup) - LP dispatch surplus`, tentatively
drop the worst, re-solve the fixed-`u` LP off the retained basis, keep if
feasible and cheaper. Greedy and monotone. Cheap now that the appsi
fixed-vars tax is gone (a re-solve is ~0.1–1 s, not ~1,000 s). This is the
smallest change with the largest expected effect, because it attacks the
one defect that has actually been measured.

*Done* (`gridlock/repair.py`, `heuristic_repair`), but the ranking above
had to be rewritten twice before it bought anything. Whole-run removal was
refused on **every** candidate across weeks 00/09/44, shedding 100–5,000
MWh each time: a run that loses money averaged over 168 h is still
load-carrying in its peak hours. And priced on duals alone essentially the
whole committed fleet reads as waste, because LP energy prices never
recover no-load cost — the standard non-convexity — so the ranking
cheerfully proposed cutting 144 of a 168-hour run. What works is (i) cuts
that grow inward from one end of a run and stop at the first hour that pays
for itself, since "extended runs, not extra starts" means the money is in
the shoulders, and (ii) screening a candidate hour against the rest of the
fleet's ramp-limited headroom before proposing it. On top of 1f that is
worth 0.55–0.98 points of margin per week for ~24 LP re-solves — week00
+1.851% → +0.891%, week09 +1.533% → +0.981%, week44 +2.523% → +1.542%,
and it hands week44 back all 51 of the unit-hours 1f spent. Note it
also cuts startups hard (week09 45 → 32) — further in the direction the
census above says is *already* 4.6% low, so the objective improves while
that statistic moves away from the optimum's.

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

*Done* (`gridlock/repair.py`, `heuristic_repair`), and it is the largest
single win measured so far. Week44's 259 MWh goes to zero for 12 committed
unit-hours — 51 once the min up/down repair extends them — and one extra
LP: **+56.036% → +2.523%**. Week00's 3.50 MWh goes for one unit-hour,
+2.384% → +1.851%. Week09 sheds nothing and the pass correctly does nothing
to it, at a cost of one LP. Successive rounds widen the committed block
around the failing hour (`[t-k, t+k]`), which is what handles shed that is
a ramp shortfall rather than a capacity one; in practice one round at
`k = 0` sufficed on both weeks.

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
