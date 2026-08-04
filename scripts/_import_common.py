"""Shared helpers for converting third-party datasets into gridlock CSVs.

The two importers (RTS-GMLC, NREL-118) face the same three problems, so
the interesting decisions live here rather than being solved twice:

**Heat rates.** Both datasets describe fuel input as a piecewise-linear
curve — a base/average point plus incremental bands in BTU/kWh. gridlock
models it as ``intercept + slope * MW``, so the curve has to be collapsed.
:func:`fit_heat_rate` takes the secant between minimum and maximum stable
output, which reproduces total fuel burn exactly at both endpoints and
errs inside the range. That is deliberate: matching the endpoints keeps
full-load and min-load costs honest, which is what commitment decisions
turn on. Fitting a least-squares line instead would spread the error more
evenly but make both endpoints wrong.

**Availability.** Both give renewable output in absolute MW; gridlock
wants a 0-1 factor, so profiles are divided by unit capacity. Values are
clipped to [0, 1] because both datasets contain hours that exceed the
nameplate by rounding.

**Minimum up/down times.** RTS-GMLC has non-integer values (2.2, 4.5 h).
gridlock indexes these as whole hours, and its model builder truncates via
``int()`` — which would silently *relax* a 4.5 h obligation to 4. These
importers round **up** instead, so the converted system is never laxer
than the source.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

# gridlock's generators.csv column order (required, then optional).
GENERATOR_COLUMNS = [
    "name",
    "node",
    "technology",
    "max_mw",
    "min_mw",
    "heat_rate_slope_mmbtu_per_mwh",
    "heat_rate_intercept_mmbtu_per_hr",
    "fuel_cost_per_mmbtu",
    "vom_cost_per_mwh",
    "startup_cost",
    "shutdown_cost",
    "ramp_rate_mw_per_hr",
    "min_up_time_hr",
    "min_down_time_hr",
    "num_units",
]
STORAGE_COLUMNS = [
    "name",
    "node",
    "technology",
    "power_mw",
    "energy_mwh",
    "roundtrip_efficiency",
]
NETWORK_COLUMNS = ["line", "from_node", "to_node", "capacity_mw", "loss_factor"]
NODE_COLUMNS = ["node", "latitude", "longitude"]


def number(value, default: float = 0.0) -> float:
    """Coerce a CSV cell to a float, mapping blanks and NaN to ``default``.

    Do not reach for ``float(value or default)`` instead: **NaN is truthy
    in Python**, so that idiom passes NaN straight through. A single NaN
    reaching a cost column makes the whole objective NaN — the LP then
    "solves" with a NaN objective and the MIP reports infeasible, with
    nothing in the model pointing at the real cause.
    """
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(result) else result


def check_no_missing(frame: pd.DataFrame, label: str) -> None:
    """Fail loudly if a converted table still holds NaN.

    A guard rather than a formality: gridlock's loader tolerates missing
    optional columns, so a NaN can survive all the way into a solver
    coefficient before anything complains.
    """
    missing = frame.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        raise ValueError(
            f"{label} still contains missing values after conversion: "
            f"{missing.to_dict()}"
        )


def fit_heat_rate(
    breakpoints_mw: list[float], fuel_mmbtu_per_hr: list[float]
) -> tuple[float, float]:
    """Collapse a piecewise fuel-input curve to (slope, intercept).

    ``breakpoints_mw`` must be ascending with matching total fuel input in
    MMBtu/hr. Returns the secant through the first and last point, so both
    endpoints are reproduced exactly. A degenerate curve (single point, or
    zero span) falls back to the average heat rate with no no-load term.
    """
    if len(breakpoints_mw) < 2 or breakpoints_mw[-1] <= breakpoints_mw[0]:
        power = breakpoints_mw[-1] if breakpoints_mw else 0.0
        if power <= 0:
            return 0.0, 0.0
        return fuel_mmbtu_per_hr[-1] / power, 0.0

    low, high = breakpoints_mw[0], breakpoints_mw[-1]
    slope = (fuel_mmbtu_per_hr[-1] - fuel_mmbtu_per_hr[0]) / (high - low)
    intercept = fuel_mmbtu_per_hr[0] - slope * low
    # A sharply convex curve can put the secant's intercept below zero.
    # That is mathematically harmless inside the operating range but
    # poisonous in the objective: no-load cost becomes negative, so the
    # solver is *paid* to commit the unit. Clamp to zero and re-derive the
    # slope through the origin, accepting a worse fit for a sane model.
    if intercept < 0:
        return fuel_mmbtu_per_hr[-1] / high, 0.0
    return slope, intercept


def heat_rate_error(
    breakpoints_mw: list[float],
    fuel_mmbtu_per_hr: list[float],
    slope: float,
    intercept: float,
) -> float:
    """Worst relative fuel-input error of the fit across the breakpoints."""
    worst = 0.0
    for power, fuel in zip(breakpoints_mw, fuel_mmbtu_per_hr):
        if fuel <= 0:
            continue
        predicted = intercept + slope * power
        worst = max(worst, abs(predicted - fuel) / fuel)
    return worst


def ceil_hours(value: float) -> int:
    """Round a possibly fractional min up/down time up to whole hours.

    Rounding up keeps the converted system at least as constrained as the
    source; gridlock's ``int()`` cast would truncate and quietly relax it.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 1
    return max(1, int(math.ceil(float(value) - 1e-9)))


def availability_from_mw(
    profile_mw: pd.Series | np.ndarray, capacity_mw: float
) -> np.ndarray:
    """Absolute MW output -> availability factor in [0, 1]."""
    values = np.asarray(profile_mw, dtype=float)
    if capacity_mw <= 0:
        return np.zeros_like(values)
    return np.clip(values / capacity_mw, 0.0, 1.0)


def write_dataset(
    output_dir: str | Path,
    generators: pd.DataFrame,
    storage: pd.DataFrame,
    nodes: pd.DataFrame,
    network: pd.DataFrame,
    demand: pd.DataFrame,
    availability: pd.DataFrame,
    provenance: str,
) -> Path:
    """Write the six gridlock CSVs plus a PROVENANCE.md into ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generators = generators.reindex(columns=GENERATOR_COLUMNS)
    storage = storage.reindex(columns=STORAGE_COLUMNS)
    check_no_missing(generators, "generators")
    check_no_missing(storage, "storage")
    check_no_missing(demand, "demand")
    check_no_missing(availability, "availability_factors")

    generators.to_csv(output_dir / "generators.csv", index=False)
    storage.to_csv(output_dir / "storage.csv", index=False)
    nodes.reindex(columns=NODE_COLUMNS).to_csv(
        output_dir / "node_locations.csv", index=False
    )
    network.reindex(columns=NETWORK_COLUMNS).to_csv(
        output_dir / "network.csv", index=False
    )
    demand.round(4).rename_axis("hour").to_csv(output_dir / "demand.csv")
    availability.round(5).rename_axis("hour").to_csv(
        output_dir / "availability_factors.csv"
    )
    (output_dir / "PROVENANCE.md").write_text(provenance, encoding="utf-8")
    return output_dir


def summarize(
    label: str,
    generators: pd.DataFrame,
    storage: pd.DataFrame,
    nodes: pd.DataFrame,
    network: pd.DataFrame,
    demand: pd.DataFrame,
    availability: pd.DataFrame,
) -> None:
    """Print a short census so a conversion can be eyeballed for sanity."""
    committed = generators[
        (generators["min_mw"] > 0)
        | (generators["startup_cost"] > 0)
        | (generators["heat_rate_intercept_mmbtu_per_hr"] > 0)
        | (generators["min_up_time_hr"] > 1)
        | (generators["min_down_time_hr"] > 1)
    ]
    units = generators["num_units"].fillna(1)
    peak = demand.sum(axis=1).max()
    energy = demand.to_numpy().sum() / 1e6
    print(f"\n{label}")
    print(f"  nodes {len(nodes)} | lines {len(network)} | storage {len(storage)}")
    print(
        f"  generators {len(generators)} rows / {int(units.sum())} units"
        f" | {len(committed)} rows need commitment"
    )
    print(f"  capacity {(generators['max_mw'] * units).sum():,.0f} MW")
    print(f"  hours {len(demand)} | peak {peak:,.0f} MW | energy {energy:,.2f} TWh")
    print(f"  availability profiles: {len(availability.columns)} units")
