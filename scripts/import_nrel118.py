"""Convert the NREL-118 test system into a gridlock dataset.

    python scripts/fetch_external_data.py nrel-118
    python scripts/import_nrel118.py                    # 118-node system
    python scripts/import_nrel118.py --aggregate region # 3-node transport system

NREL-118 (Peña, Brancucci Martinez-Anido & Hodge, IEEE TPWRS 33(1), 2018)
is the IEEE 118-bus network re-generated from three WECC regions: 118
buses, 327 generators (192 thermal), 186 lines with explicit MW limits,
and 8784 hours of the leap year 2024.

Its attraction over RTS-GMLC is the heat rate. NREL-118 publishes
``Heat Rate Base (MMBTU/hr)`` — a genuine no-load fuel input — alongside
incremental bands, so gridlock's ``intercept + slope * MW`` form is read
off directly instead of being fitted to a curve.

**Licensing.** The mirror this reads from carries *no license file*, so
the data is all-rights-reserved by default. Convert and compute locally;
do not redistribute the result. (RTS-GMLC is the redistributable one.)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _import_common import (  # noqa: E402
    availability_from_mw,
    ceil_hours,
    fit_heat_rate,
    number,
    heat_rate_error,
    summarize,
    write_dataset,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "data" / "external" / "PowerSystemsTestData" / "118-Bus"

# Fuel prices ($/MMBtu) are not in any CSV — they are hard-coded in the
# repository's data_118bus.jl, along with the rule mapping each unit to a
# fuel. Both are reproduced here; see _fuel_price_for.
FUEL_PRICES = {
    "natural_gas": 5.4,
    "coal": 1.8,
    "oil": 21.0,
    "biomass": 2.4,
    "geothermal": 0.0,
}


def _fuel_price_for(prime_mover: str, name: str) -> tuple[str, float]:
    """Replicate data_118bus.jl's fuel assignment for a thermal unit."""
    if prime_mover == "OT":
        return "biomass", FUEL_PRICES["biomass"]
    if prime_mover == "CC" or name.startswith(("CT NG", "ICE NG", "ST NG")):
        return "natural_gas", FUEL_PRICES["natural_gas"]
    if name.startswith("CT Oil") or name.startswith("ST Other 01"):
        return "oil", FUEL_PRICES["oil"]
    if name.startswith("ST Coal"):
        return "coal", FUEL_PRICES["coal"]
    if name.startswith("Geo"):
        return "geothermal", FUEL_PRICES["geothermal"]
    if name.startswith("ST Other 02"):
        return "natural_gas", FUEL_PRICES["natural_gas"]
    # data_118bus.jl leaves anything else unmapped; natural gas is the
    # majority fuel and the least distorting default.
    return "natural_gas", FUEL_PRICES["natural_gas"]


def _strip_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace from column names and object values.

    Necessary, not cosmetic: this dataset ships ``'Thermal '`` with a
    trailing space as a type value and ``'Bus from '`` as a column name,
    so naive equality comparisons silently match nothing.
    """
    frame = frame.rename(columns=lambda c: str(c).strip())
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].str.strip()
    return frame


def _heat_rate(row: pd.Series) -> tuple[float, float, float]:
    """(slope, intercept, fit error) from base + incremental bands.

    ``Heat Rate Base`` is fuel input at zero output in MMBtu/h — exactly
    gridlock's intercept — and each band gives an incremental heat rate in
    BTU/kWh up to its load point. When only one band exists the mapping is
    exact and the fit error is zero; multi-band units are collapsed by the
    same secant rule the RTS-GMLC importer uses.
    """
    base = number(row.get("Heat Rate Base (MMBTU/hr)"))
    max_mw = number(row["Max Capacity (MW)"])
    min_mw = number(row.get("Min Stable Level (MW)"))

    powers, fuel = [0.0], [base]
    previous = 0.0
    for band in range(1, 6):
        increment = row.get(f"Heat Rate Inc Band {band} (BTU/kWh)")
        point = row.get(f"Load Point Band {band} (MW)")
        if pd.isna(increment) or pd.isna(point) or float(point) <= previous:
            continue
        point, increment = float(point), float(increment)
        fuel.append(fuel[-1] + (point - previous) * increment / 1000.0)
        powers.append(point)
        previous = point

    if len(powers) < 2:
        return 0.0, base, 0.0

    # Evaluate the curve over the operating range the model will use.
    def fuel_at(power: float) -> float:
        for index in range(1, len(powers)):
            if power <= powers[index] or index == len(powers) - 1:
                span = powers[index] - powers[index - 1]
                rate = (fuel[index] - fuel[index - 1]) / span if span else 0.0
                return fuel[index - 1] + (power - powers[index - 1]) * rate
        return fuel[-1]

    low = min_mw if min_mw > 0 else powers[0]
    high = max_mw if max_mw > low else powers[-1]
    slope, intercept = fit_heat_rate([low, high], [fuel_at(low), fuel_at(high)])
    inside = [p for p in powers if low <= p <= high] or [low, high]
    return slope, intercept, heat_rate_error(inside, [fuel_at(p) for p in inside], slope, intercept)


def _load_profile(path: Path) -> np.ndarray:
    return pd.read_csv(path)["value"].to_numpy(dtype=float)


def _align(values: np.ndarray, horizon: int, label: str) -> np.ndarray:
    """Force a profile onto the demand horizon.

    The hydro series carry 8808 rows against the load's 8784 — an extra
    day — so profiles are trimmed to the horizon. A profile that is too
    *short* would be padded, which is a real data problem rather than a
    formatting quirk, so it is announced rather than absorbed silently.
    """
    if len(values) == horizon:
        return values
    if len(values) > horizon:
        return values[:horizon]
    print(
        f"  warning: profile for {label} has {len(values)} hours < {horizon}; "
        "padding with its final value"
    )
    return np.concatenate([values, np.full(horizon - len(values), values[-1])])


def _profile_path(source: Path, kind: str, name: str) -> Path | None:
    """Map a generator name like 'Solar 07' to Solar/DA/Solar7DA.csv."""
    match = re.search(r"(\d+)", name)
    if not match:
        return None
    path = source / kind / "DA" / f"{kind}{int(match.group(1))}DA.csv"
    return path if path.is_file() else None


def build(source: Path, aggregate: str, hours: int | None) -> tuple[pd.DataFrame, ...]:
    gen = _strip_frame(pd.read_csv(source / "gen.csv"))
    buses = _strip_frame(pd.read_csv(source / "Buses.csv"))
    lines_raw = _strip_frame(pd.read_csv(source / "Lines.csv"))
    partfact = _strip_frame(pd.read_csv(source / "Load" / "partfact.csv"))

    buses["node"] = buses["Number"].apply(lambda n: f"bus{int(n):03d}")
    bus_region = buses.set_index("node")["Area"].to_dict()

    def node_of(bus_name: str) -> str:
        bus_name = str(bus_name).strip()
        return f"Region_{bus_region.get(bus_name, '?')}" if aggregate == "region" else bus_name

    # ---------------------------------------------------------------- nodes
    if aggregate == "region":
        nodes = pd.DataFrame(
            {
                "node": [f"Region_{r}" for r in sorted(buses["Area"].unique())],
                "latitude": 0.0,
                "longitude": 0.0,
            }
        )
    else:
        # NREL-118 ships no coordinates; the topology is what matters here.
        nodes = pd.DataFrame(
            {"node": buses["node"], "latitude": 0.0, "longitude": 0.0}
        )

    # -------------------------------------------------------------- network
    lines = pd.DataFrame(
        {
            "line": lines_raw["Line Name"],
            "from_node": lines_raw["Bus from"].map(node_of),
            "to_node": lines_raw["Bus to"].map(node_of),
            "capacity_mw": lines_raw["Max Flow (MW)"].astype(float),
        }
    )
    lines = lines[lines["from_node"] != lines["to_node"]]
    if aggregate == "region":
        lines = (
            lines.groupby(["from_node", "to_node"], as_index=False)["capacity_mw"]
            .sum()
            .assign(line=lambda d: d["from_node"] + "__" + d["to_node"])
        )
    lines["loss_factor"] = 0.0  # no losses published

    # The demand horizon is fixed first so every profile can be aligned to it.
    regional = {}
    for region in sorted(buses["Area"].unique()):
        path = source / "Load" / "DA" / f"Load{region}DA.csv"
        if path.is_file():
            regional[region] = _load_profile(path)
    if not regional:
        raise SystemExit(f"no regional load files under {source / 'Load' / 'DA'}")
    horizon = min(len(series) for series in regional.values())
    regional = {region: series[:horizon] for region, series in regional.items()}

    # ----------------------------------------------------------- generators
    records, availability_columns, fit_errors = [], {}, []
    for _, row in gen.iterrows():
        kind = str(row["type"])
        name = str(row["Generator Name"])
        max_mw = number(row["Max Capacity (MW)"])
        if max_mw <= 0:
            continue

        record = {
            "name": name,
            "node": node_of(row["bus of connection"]),
            "technology": kind.lower(),
            "max_mw": max_mw,
            "min_mw": 0.0,
            "heat_rate_slope_mmbtu_per_mwh": 0.0,
            "heat_rate_intercept_mmbtu_per_hr": 0.0,
            "fuel_cost_per_mmbtu": 0.0,
            "vom_cost_per_mwh": number(row.get("VO&M Charge (dollar/MWh)")),
            "startup_cost": 0.0,
            "shutdown_cost": 0.0,
            "ramp_rate_mw_per_hr": max_mw,
            "min_up_time_hr": 1,
            "min_down_time_hr": 1,
            "num_units": 1,
        }

        if kind == "Thermal":
            slope, intercept, error = _heat_rate(row)
            fit_errors.append((name, error))
            fuel, price = _fuel_price_for(str(row["PrimeMoveType"]), name)
            ramp = number(row.get("Max Ramp Up (MW/min)")) * 60.0
            record.update(
                technology=fuel,
                min_mw=number(row.get("Min Stable Level (MW)")),
                heat_rate_slope_mmbtu_per_mwh=slope,
                heat_rate_intercept_mmbtu_per_hr=intercept,
                fuel_cost_per_mmbtu=price,
                startup_cost=number(row.get("Start Cost (dollar)")),
                ramp_rate_mw_per_hr=ramp if ramp > 0 else max_mw,
                min_up_time_hr=ceil_hours(row.get("Min Up Time (h)")),
                min_down_time_hr=ceil_hours(row.get("Min Down Time (h)")),
            )
        else:
            kind_dir = {"Solar": "Solar", "Wind": "Wind"}.get(kind)
            if kind_dir:
                path = _profile_path(source, kind_dir, name)
                if path is not None:
                    availability_columns[name] = availability_from_mw(
                        _align(_load_profile(path), horizon, name), max_mw
                    )
            elif kind == "Hydro":
                path = source / "Hydro" / f"{name}.csv"
                if path.is_file():
                    frame = pd.read_csv(path)
                    column = "value" if "value" in frame.columns else frame.columns[-1]
                    availability_columns[name] = availability_from_mw(
                        _align(frame[column].to_numpy(dtype=float), horizon, name), max_mw
                    )

        records.append(record)

    generators = pd.DataFrame(records)

    # --------------------------------------------------------------- demand
    shares = partfact.set_index("Bus Name")["Load Participation Factor"].to_dict()
    demand = pd.DataFrame(index=pd.RangeIndex(horizon))
    for _, bus_row in buses.iterrows():
        region = bus_row["Area"]
        if region not in regional:
            continue
        share = float(shares.get(bus_row["node"], 0.0))
        if share <= 0:
            continue
        node = node_of(bus_row["node"])
        series = regional[region] * share
        demand[node] = demand[node].to_numpy() + series if node in demand else series

    availability = pd.DataFrame(availability_columns, index=pd.RangeIndex(horizon))

    if hours is not None:
        demand = demand.iloc[:hours].reset_index(drop=True)
        availability = availability.iloc[:hours].reset_index(drop=True)

    storage = pd.DataFrame(
        columns=["name", "node", "technology", "power_mw", "energy_mwh", "roundtrip_efficiency"]
    )

    if fit_errors:
        worst = max(fit_errors, key=lambda item: item[1])
        mean_error = float(np.mean([error for _, error in fit_errors]))
        exact = sum(1 for _, error in fit_errors if error < 1e-9)
        print(
            f"  heat-rate fit: worst {worst[1]:.2%} ({worst[0]}), mean {mean_error:.2%}; "
            f"{exact}/{len(fit_errors)} exact (single-band)"
        )

    return generators, storage, nodes, lines, demand, availability


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="118-Bus directory")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--aggregate",
        choices=["none", "region"],
        default="none",
        help="'none' keeps all 118 buses; 'region' collapses to the 3 WECC "
        "regions joined by summed inter-region capacity",
    )
    parser.add_argument("--hours", type=int, default=None, help="truncate to the first N hours")
    args = parser.parse_args(argv)

    source = Path(args.source)
    if not (source / "gen.csv").is_file():
        raise SystemExit(
            f"NREL-118 not found at {source}\n"
            "run: python scripts/fetch_external_data.py nrel-118"
        )

    label = "nrel118" if args.aggregate == "none" else "nrel118_region"
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "data" / label

    print(f"importing NREL-118 ({args.aggregate} aggregation)")
    frames = build(source, args.aggregate, args.hours)
    generators, storage, nodes, lines, demand, availability = frames

    provenance = f"""# Provenance

Converted from **NREL-118** by `scripts/import_nrel118.py`.

- Upstream mirror: https://github.com/Sienna-Platform/PowerSystemsTestData (`118-Bus/`)
- Original: Peña, Brancucci Martinez-Anido & Hodge, "An Extended IEEE
  118-Bus Test System With High Renewable Penetration", IEEE Transactions
  on Power Systems 33(1):281-289, 2018. DOI 10.1109/TPWRS.2017.2695963
- Aggregation: `{args.aggregate}` | Hours: {len(demand)}

## License — READ BEFORE REDISTRIBUTING

The upstream mirror carries **no license file**, so it is
all-rights-reserved by default. This converted dataset is for local
research only; do not redistribute it. Use RTS-GMLC (which carries an
explicit NREL grant) for anything published.

Note the original NREL download URLs are dead: NREL was renamed the
National Laboratory of the Rockies and `nrel.gov` no longer resolves.

## Conversion decisions

- Heat rates read directly: `Heat Rate Base (MMBTU/hr)` is a genuine
  no-load fuel input and maps onto gridlock's intercept with no fitting.
  Units with a single incremental band convert exactly; multi-band units
  are collapsed by a secant between minimum and maximum stable output.
- Fuel prices are **not** in any CSV. They are hard-coded in
  `data_118bus.jl` (natural gas 5.4, coal 1.8, oil 21, biomass 2.4,
  geothermal 0 $/MMBtu), and that file's unit-to-fuel rule (by prime
  mover and name prefix) is reproduced in the importer.
- Ramp rates converted from MW/min to MW/h; units with none recorded get
  a full-range hourly ramp.
- Minimum up/down times rounded up to whole hours.
- Hydro is modelled as profile-driven generation with no commitment
  state (gridlock has no reservoir model).
- Nodal demand = regional day-ahead load x the published per-bus
  participation factors (`Load/partfact.csv`).
- Line loss factors are zero (none published). No storage is included.
- Bus coordinates are zeroed: NREL-118 ships no geography, and gridlock
  only uses coordinates for reporting.

## Network caveat

Line ratings are thermal limits for a **DC power flow**. gridlock is a
transport model and ignores Kirchhoff's voltage law, so the nodal variant
lets flow route around congestion in ways the real system cannot.
`--aggregate region` collapses to the three WECC regions joined by summed
inter-region capacity, which is a defensible transport representation.
"""

    write_dataset(
        output_dir, generators, storage, nodes, lines, demand, availability, provenance
    )
    summarize(
        f"wrote {output_dir}", generators, storage, nodes, lines, demand, availability
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
