# Mathematical formulation

gridlock solves a deterministic, hourly production cost problem. A model is
built for one *window* of contiguous hours; a monolithic run is a single
window over the whole horizon.

## Sets

| Symbol | Meaning |
|---|---|
| $t \in T$ | hours in the window (contiguous) |
| $g \in G$ | generators; $G^{UC} \subseteq G$ are units with commitment state |
| $s \in S$ | storage units |
| $n \in N$ | nodes |
| $\ell \in L$ | transmission lines (pipe-and-bubble) |

A generator belongs to $G^{UC}$ when on/off state matters: minimum stable
level, startup/shutdown cost, no-load cost, or min up/down time above one
hour. Other units (typically wind/solar) dispatch freely between zero and
their available capacity and get no commitment variables.

## Parameters

| Symbol | Source | Meaning |
|---|---|---|
| $D_{n,t}$ | demand.csv | demand (MW) |
| $A_{g,t}$ | availability_factors.csv | availability factor in $[0,1]$ |
| $\overline{P}_g, \underline{P}_g$ | generators.csv | max / min stable output (MW) |
| $c^{mc}_g$ | derived | marginal cost = fuel price × heat-rate slope + VOM ($/MWh) |
| $c^{nl}_g$ | derived | no-load cost = fuel price × heat-rate intercept ($/hr) |
| $c^{su}_g, c^{sd}_g$ | generators.csv | startup / shutdown cost ($) |
| $R_g$ | generators.csv | ramp rate (MW/hr) |
| $UT_g, DT_g$ | generators.csv | min up / down time (hr) |
| $\overline{F}_\ell, \lambda_\ell$ | network.csv | line rating (MW), loss factor |
| $\overline{C}_s, \overline{E}_s$ | storage.csv | power (MW) and energy (MWh) capacity |
| $\eta_s$ | derived | one-way efficiency $= \sqrt{\text{roundtrip}}$ |
| $\mathrm{VOLL}$ | config | value of lost load ($/MWh) |

## Variables

| Symbol | Domain | Meaning |
|---|---|---|
| $p_{g,t}$ | $[0,\ \overline{P}_g A_{g,t}]$ | generation (MW) |
| $u_{g,t}$ | $\{0,1\}$ or $[0,1]$ | committed (binary with unit commitment on; relaxed otherwise) |
| $v_{g,t}, w_{g,t}$ | $[0,1]$ | startup / shutdown indicators |
| $f^{+}_{\ell,t}, f^{-}_{\ell,t}$ | $[0,\ \overline{F}_\ell]$ | forward / reverse line flow (MW) |
| $c_{s,t}, d_{s,t}$ | $[0,\ \overline{C}_s]$ | grid-side charge / discharge (MW) |
| $e_{s,t}$ | $[0,\ \overline{E}_s]$ | state of charge (MWh) |
| $e^{0}_{s}$ | $[0,\ \overline{E}_s]$ | free starting SOC (cyclic windows only) |
| $\sigma_{n,t}$ | $[0,\ D_{n,t}]$ | unserved energy (MW) |

## Objective

$$
\min \sum_{t}\Bigg[
  \sum_{g} c^{mc}_g\, p_{g,t}
  + \sum_{g \in G^{UC}} \left( c^{nl}_g u_{g,t} + c^{su}_g v_{g,t} + c^{sd}_g w_{g,t} \right)
  + \mathrm{VOLL} \sum_{n} \sigma_{n,t}
\Bigg]
$$

## Constraints

### Load balance (every node, every hour)

$$
\sum_{g \in G_n} p_{g,t}
+ \sum_{s \in S_n} (d_{s,t} - c_{s,t})
+ \sum_{\ell \in L^{in}_n} (1-\lambda_\ell) f^{+}_{\ell,t}
+ \sum_{\ell \in L^{out}_n} (1-\lambda_\ell) f^{-}_{\ell,t}
- \sum_{\ell \in L^{out}_n} f^{+}_{\ell,t}
- \sum_{\ell \in L^{in}_n} f^{-}_{\ell,t}
+ \sigma_{n,t}
= D_{n,t}
$$

Losses are charged at the receiving end: a flow $f$ delivers
$(1-\lambda_\ell) f$. Its dual (available on LP solves) is the nodal price.

### Generator limits ($g \in G^{UC}$)

$$
\underline{P}_g A_{g,t}\, u_{g,t} \;\le\; p_{g,t} \;\le\; \overline{P}_g A_{g,t}\, u_{g,t}
$$

The availability factor derates both bounds, so a partially derated unit
may remain committed at reduced output, and a fully unavailable unit can
stay "on" through an outage (producing nothing) rather than being forced
through an artificial shutdown/startup cycle.

### Commitment logic ($g \in G^{UC}$)

$$
u_{g,t} - u_{g,t-1} = v_{g,t} - w_{g,t}
$$

At the first hour of a window this references the carried initial state
$u^0_g$; with no initial state (monolithic runs) the first hour is
unconstrained, so hour one incurs no startup cost.

### Minimum up/down times ($g \in G^{UC}$, when $UT_g$ or $DT_g > 1$)

$$
\sum_{\tau = t-UT_g+1}^{t} v_{g,\tau} \le u_{g,t}
\qquad\qquad
\sum_{\tau = t-DT_g+1}^{t} w_{g,\tau} \le 1 - u_{g,t}
$$

In rolling-horizon mode, a unit that enters a window with an unfinished
up/down obligation has its first hours fixed to the inherited state.

### Ramp limits (units with $R_g < \overline{P}_g$)

With startup/shutdown ramp allowance $SU_g = \max(\underline{P}_g, R_g)$:

$$
p_{g,t} - p_{g,t-1} \le R_g\, u_{g,t-1} + SU_g\, v_{g,t}
\qquad\qquad
p_{g,t-1} - p_{g,t} \le R_g\, u_{g,t} + SU_g\, w_{g,t}
$$

Units outside $G^{UC}$ use the plain form
$|p_{g,t} - p_{g,t-1}| \le R_g$. First-hour ramps reference the carried
previous output when a window inherits state.

### Storage (bathtub)

$$
e_{s,t} = e_{s,t-1} + \eta_s\, c_{s,t} - d_{s,t} / \eta_s
$$

The round-trip efficiency is split evenly: $\eta_s = \sqrt{\eta^{rt}_s}$,
applied once on the way in and once on the way out.

**Cyclic (monolithic runs):** the first hour references the free variable
$e^{0}_s$ and the last hour must return to it,
$e_{s,\,t_{last}} = e^{0}_s$. No starting SOC input is needed — the
optimizer chooses it, and cycling guarantees no free energy.

**Rolling windows:** the first hour references the carried SOC. The first
window starts from `initial_soc_fraction` × energy capacity, and the final
window must end at or above that level so the year cannot end drained.

### Network

Flow bounds are the only line constraints (pipe-and-bubble): each direction
independently limited by the rating. Simultaneous forward and reverse flow
on one line is never optimal when losses are positive, since it wastes
energy; the same argument keeps simultaneous charge/discharge of storage
out of optimal LP solutions except under local surplus, where it acts as
implicit curtailment.

## The unit commitment switch

`RunConfig.unit_commitment` toggles only the domain of $u$: binary (MIP)
versus $[0,1]$ (LP relaxation). Every constraint stays in the model, so the
relaxation is a true lower bound of the MIP on identical structure — useful
for isolating the integrality gap and for benchmarking solver settings on
comparable problems.
