# gridlock

A small, modular electricity **production cost model** built on
[Pyomo](https://www.pyomo.org/) and the **appsi HiGHS** solver interface.

gridlock dispatches generators, storage and transmission to minimize total
system cost over a year (or any horizon), with unit commitment that can be
switched between binary (MIP) and its LP relaxation. It is deliberately
small and readable: a testbed for studying how model formulation and solver
settings affect runtime, not a planning tool.

## Features

- **Least-cost dispatch** over an 8760-hour year (or any horizon length)
- **Unit commitment switch**: binary commitment (`--uc`, default) or the LP
  relaxation of the *same* model (`--no-uc`) — ideal for comparing a MIP
  against its relaxation on identical structure
- Commitment detail: minimum stable levels, startup/shutdown costs, no-load
  (heat-rate intercept) costs, ramp limits, minimum up/down times
- **Storage** as a bathtub state-of-charge model with round-trip efficiency;
  cyclic over the horizon (ending SOC = starting SOC, with the starting
  level a free decision variable)
- **Network** as a pipe-and-bubble transport model with per-line capacities
  and losses
- Hourly, unit-level **availability factors** (outages, derates, wind/solar)
- **Monolithic or rolling-horizon** solves: one 8760-hour model, or
  sequential windows with lookahead that carry commitment, ramp and SOC
  state across boundaries
- **Nodal prices** from load-balance duals on LP solves
- Unserved energy priced at a configurable VOLL so scarcity never breaks
  feasibility
- A **benchmark command** that solves the same case under different HiGHS
  option presets and compares runtimes

## Installation

```bash
git clone https://github.com/sericson0/gridlock.git
cd gridlock
pip install -e .          # add [dev] for pytest
```

Requires Python ≥ 3.10. HiGHS ships with the `highspy` wheel — no separate
solver install.

## Quickstart

A synthetic 3-node, 12-generator, 2-storage test system ships in
[data/example/](data/example/) (regenerate with
[scripts/make_example_data.py](scripts/make_example_data.py)).

```bash
# One week, LP relaxation, monolithic
gridlock run --data-dir data/example --no-uc --hours 168

# One week, binary unit commitment
gridlock run --data-dir data/example --uc --hours 168 --mip-gap 0.001

# Full-year unit commitment via weekly rolling horizon
gridlock run --data-dir data/example --uc --window 168 --lookahead 24 --mip-gap 0.005

# Compare HiGHS presets on the same case
gridlock benchmark --data-dir data/example --no-uc --hours 336 \
    --presets default,presolve_off,simplex,ipm

# Any raw HiGHS option can be passed through
gridlock run --data-dir data/example --no-uc --hours 168 \
    --highs-option solver=ipm --highs-option run_crossover=off
```

Results land in `results/` as CSVs (dispatch, commitment, storage, flows,
prices, shed, cost summary, per-window solve stats).

Reference timings (example system, laptop-class hardware): the monolithic
8760-hour LP builds in ~11 s and solves in ~100 s; full-year weekly rolling
MIP (0.5% gap) solves its 53 windows in ~320 s total.

### Python API

```python
from gridlock import RunConfig, SolverSettings, load_system, run

system = load_system("data/example")
config = RunConfig(
    unit_commitment=True,
    window_hours=168,
    lookahead_hours=24,
    solver=SolverSettings(mip_gap=0.005, highs_options={"presolve": "on"}),
)
results = run(system, config)

print(results.cost_summary)          # $ by component
print(results.dispatch)              # hourly MW per unit
print(results.window_stats)          # build/solve seconds per window
```

## Input data

Six CSVs in one directory. Profile files (`demand.csv`,
`availability_factors.csv`) may include a leading `hour` column; rows are
hours in order and both files must cover the same horizon.

| File | Contents |
|---|---|
| `generators.csv` | One row per unit: `name`, `node`, `technology`, `max_mw`, `min_mw`, `heat_rate_slope_mmbtu_per_mwh`, `fuel_cost_per_mmbtu`, `vom_cost_per_mwh`, `startup_cost`, `ramp_rate_mw_per_hr`; optional `heat_rate_intercept_mmbtu_per_hr`, `shutdown_cost`, `min_up_time_hr`, `min_down_time_hr` |
| `storage.csv` | One row per unit: `name`, `node`, `technology`, `power_mw`, `energy_mwh`, `roundtrip_efficiency` |
| `node_locations.csv` | `node`, `latitude`, `longitude` |
| `network.csv` | One row per line: `from_node`, `to_node`, `capacity_mw`, `loss_factor`; optional `line` name |
| `demand.csv` | Hourly MW, one column per node (missing nodes = zero load) |
| `availability_factors.csv` | Hourly fraction in [0, 1], one column per unit (missing units = fully available) |

Fuel burn when a unit is on is `intercept + slope × MW`, so fuel cost splits
into a no-load ($/hr while committed) and a marginal ($/MWh) component.
Availability factors derate both the maximum *and* the minimum stable
level, so a derated unit can stay committed at reduced output.

## Model

Full mathematical formulation in [docs/formulation.md](docs/formulation.md).
In brief, for every node and hour:

```
generation + storage discharge + imports·(1 − loss) + unserved energy
    = demand + storage charge + exports
```

subject to commitment logic (`u`, startup `v`, shutdown `w`), min/max
output, ramp limits with startup/shutdown allowances, min up/down times,
line ratings in each direction, and bathtub SOC accounting with the
round-trip efficiency split evenly between charge and discharge. The
objective minimizes fuel + VOM + no-load + startup/shutdown costs plus the
VOLL penalty on unserved energy.

**The unit-commitment switch** (`RunConfig.unit_commitment`) changes only
the domain of `u` — binary versus [0, 1] — leaving every constraint in
place. The LP relaxation is therefore a true lower bound on the MIP and a
clean baseline for solver experiments. Units that need no on/off state
(zero minimum, no startup/no-load costs — typically wind and solar) never
get commitment variables, keeping the MIP small.

**Horizon handling**: by default the whole horizon is one model with cyclic
storage and free initial commitment. With `window_hours` set, the horizon
splits into sequential windows solved with `lookahead_hours` of extra
foresight; commitment state, previous output (for ramps), unfinished min
up/down obligations and storage SOC carry across boundaries, and the final
window must end storage at or above the starting fraction.

## Repository layout

```
gridlock/
  config.py     RunConfig / SolverSettings (all run + solver options)
  data.py       CSV loading, validation, derived economics
  model.py      Pyomo model builder (one function per constraint block)
  solver.py     appsi HiGHS wrapper (options passthrough, duals, timing)
  runner.py     monolithic / rolling-horizon driver and state carrying
  results.py    frame extraction, cost accounting, CSV output
  cli.py        `gridlock run` and `gridlock benchmark`
data/example/   synthetic 3-node test system (seeded, reproducible)
scripts/        example-data generator
tests/          pytest suite on tiny analytic systems
docs/           mathematical formulation
```

## Tests

```bash
pip install -e .[dev]
pytest
```

The suite solves small systems with hand-checkable optima: merit order,
nodal prices under congestion, minimum-generation and min up/down behavior,
LP ≤ MIP bounds, storage bathtub/cyclic accounting, loss charging, rolling
state continuity.

## License

MIT
