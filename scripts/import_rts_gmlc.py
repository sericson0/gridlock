"""Convert RTS-GMLC into a gridlock dataset.

    python scripts/fetch_external_data.py rts-gmlc
    python scripts/import_rts_gmlc.py                  # 73-node nodal system
    python scripts/import_rts_gmlc.py --aggregate area # 3-node transport system

RTS-GMLC is the only public system carrying real unit-commitment
parameters, a real network and a full year of hourly profiles: 73 buses,
158 generators (73 thermal), 120 AC branches plus one DC tie, 8784 hours
of 2020.

Everything is read from ``RTS_Data/SourceData`` rather than the bundled
openTEPES export. That export looks like a shortcut — its network, node
and demand tables already match gridlock's schema — but its generator
table zeroes the no-load term, VOM and shutdown cost for *every* unit, and
states startup cost in thousands of dollars. Using it would silently drop
the entire commitment cost structure this model exists to study.

**The network is a transport relaxation.** RTS-GMLC's line ratings are
thermal limits for a DC power flow with reactances. gridlock ignores
Kirchhoff's voltage law, so flows here route around congestion in ways the
real system cannot and dispatch cost comes out optimistically low. With
``--aggregate area`` the three areas are collapsed to single nodes joined
by their summed tie capacities, which *is* an honest transport model —
that mode is preferred for anything where congestion matters.
"""

from __future__ import annotations

import argparse
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
DEFAULT_SOURCE = REPO_ROOT / "data" / "external" / "RTS-GMLC" / "RTS_Data"

# Unit types that carry on/off state. Everything else is dispatched from a
# profile: hydro and CSP included, because gridlock has no reservoir or
# thermal-storage model and treating them as must-take shaped generation is
# the standard simplification (pglib-uc does the same).
THERMAL_TYPES = {"CC", "CT", "STEAM", "NUCLEAR"}
PROFILE_TYPES = {"WIND", "PV", "RTPV", "HYDRO", "ROR", "CSP"}

# (directory, filename) of the day-ahead profile per unit type, in MW.
PROFILE_FILES = {
    "WIND": ("WIND", "DAY_AHEAD_wind.csv"),
    "PV": ("PV", "DAY_AHEAD_pv.csv"),
    "RTPV": ("RTPV", "DAY_AHEAD_rtpv.csv"),
    "HYDRO": ("Hydro", "DAY_AHEAD_hydro.csv"),
    "ROR": ("Hydro", "DAY_AHEAD_hydro.csv"),
    "CSP": ("CSP", "DAY_AHEAD_Natural_Inflow.csv"),
}

STARTUP_TIERS = {
    "cold": "Start Heat Cold MBTU",
    "warm": "Start Heat Warm MBTU",
    "hot": "Start Heat Hot MBTU",
}

# The time-series files index hours by (Year, Month, Day, Period).
TIME_KEYS = ["Year", "Month", "Day", "Period"]


def _thermal_heat_rate(row: pd.Series) -> tuple[float, float, float]:
    """(slope, intercept, worst fit error) from the banded heat-rate curve.

    RTS-GMLC gives an *average* heat rate at the lowest load point
    (``HR_avg_0``, BTU/kWh) and *incremental* heat rates for each band
    above it. Integrating those gives total fuel input at each breakpoint,
    which the secant fit then collapses.
    """
    max_mw = number(row["PMax MW"])
    fractions, increments = [], []
    for index in range(5):
        fraction = row.get(f"Output_pct_{index}")
        if pd.isna(fraction):
            break
        fractions.append(float(fraction))
        if index > 0:
            increment = row.get(f"HR_incr_{index}")
            increments.append(0.0 if pd.isna(increment) else float(increment))

    if not fractions:
        return 0.0, 0.0, 0.0

    powers = [fraction * max_mw for fraction in fractions]
    # BTU/kWh -> MMBtu/MWh is a factor of 1000.
    fuel = [float(row["HR_avg_0"]) / 1000.0 * powers[0]]
    for index, increment in enumerate(increments, start=1):
        fuel.append(fuel[-1] + (powers[index] - powers[index - 1]) * increment / 1000.0)

    slope, intercept = fit_heat_rate(powers, fuel)
    return slope, intercept, heat_rate_error(powers, fuel, slope, intercept)


def _hours_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop the Year/Month/Day/Period columns, leaving one row per hour."""
    return frame.drop(columns=[c for c in TIME_KEYS if c in frame.columns]).reset_index(
        drop=True
    )


def build(
    source: Path, aggregate: str, startup_tier: str, hours: int | None
) -> tuple[pd.DataFrame, ...]:
    gen = pd.read_csv(source / "SourceData" / "gen.csv")
    bus = pd.read_csv(source / "SourceData" / "bus.csv")
    branch = pd.read_csv(source / "SourceData" / "branch.csv")
    dc_branch = pd.read_csv(source / "SourceData" / "dc_branch.csv")
    storage_source = pd.read_csv(source / "SourceData" / "storage.csv")

    bus["node"] = "Node_" + bus["Bus ID"].astype(str)
    bus_node = bus.set_index("Bus ID")["node"].to_dict()
    bus_area = bus.set_index("Bus ID")["Area"].to_dict()

    def node_of(bus_id) -> str:
        if aggregate == "area":
            return f"Area_{bus_area[bus_id]}"
        return bus_node[bus_id]

    # ---------------------------------------------------------------- nodes
    if aggregate == "area":
        grouped = bus.groupby("Area").agg(latitude=("lat", "mean"), longitude=("lng", "mean"))
        nodes = grouped.reset_index()
        nodes["node"] = "Area_" + nodes["Area"].astype(str)
    else:
        nodes = bus.rename(columns={"lat": "latitude", "lng": "longitude"})[
            ["node", "latitude", "longitude"]
        ]

    # -------------------------------------------------------------- network
    branch["from_node"] = branch["From Bus"].map(node_of)
    branch["to_node"] = branch["To Bus"].map(node_of)
    branch["capacity_mw"] = branch["Cont Rating"]
    branch["line"] = branch["UID"].astype(str)

    dc_branch["from_node"] = dc_branch["From Bus"].map(node_of)
    dc_branch["to_node"] = dc_branch["To Bus"].map(node_of)
    # The DC tie's rating column has varied across RTS-GMLC revisions.
    dc_rating = next(
        (c for c in ("MW Load", "Max Flow (MW)", "Cont Rating", "MW") if c in dc_branch.columns),
        None,
    )
    dc_branch["capacity_mw"] = (
        dc_branch[dc_rating] if dc_rating else 100.0
    )
    dc_branch["line"] = dc_branch["UID"].astype(str) if "UID" in dc_branch.columns else "DC_1"

    lines = pd.concat(
        [
            branch[["line", "from_node", "to_node", "capacity_mw"]],
            dc_branch[["line", "from_node", "to_node", "capacity_mw"]],
        ],
        ignore_index=True,
    )
    lines = lines[lines["from_node"] != lines["to_node"]]
    if aggregate == "area":
        # Parallel ties between the same pair collapse into one corridor.
        lines = (
            lines.groupby(["from_node", "to_node"], as_index=False)["capacity_mw"]
            .sum()
            .assign(line=lambda d: d["from_node"] + "__" + d["to_node"])
        )
    # RTS-GMLC publishes no per-line loss factor; the bundled openTEPES
    # export stamps a flat 1%, and we follow it rather than inventing a
    # value from line resistance at an unknown operating point.
    lines["loss_factor"] = 0.01

    # ----------------------------------------------------------- generators
    records, availability_columns, fit_errors = [], {}, []
    profiles: dict[str, pd.DataFrame] = {}
    for unit_type, (directory, filename) in PROFILE_FILES.items():
        path = source / "timeseries_data_files" / directory / filename
        if path.is_file():
            profiles[unit_type] = _hours_index(pd.read_csv(path))

    for _, row in gen.iterrows():
        unit_type = row["Unit Type"]
        if unit_type in ("SYNC_COND", "STORAGE"):
            continue  # zero-capacity condensers; storage handled separately
        name = str(row["GEN UID"])
        max_mw = number(row["PMax MW"])
        if max_mw <= 0:
            continue

        record = {
            "name": name,
            "node": node_of(row["Bus ID"]),
            "technology": str(row["Unit Type"]).lower(),
            "max_mw": max_mw,
            "min_mw": 0.0,
            "heat_rate_slope_mmbtu_per_mwh": 0.0,
            "heat_rate_intercept_mmbtu_per_hr": 0.0,
            "fuel_cost_per_mmbtu": number(row["Fuel Price $/MMBTU"]),
            "vom_cost_per_mwh": number(row["VOM"]),
            "startup_cost": 0.0,
            "shutdown_cost": 0.0,
            "ramp_rate_mw_per_hr": max_mw,
            "min_up_time_hr": 1,
            "min_down_time_hr": 1,
            "num_units": 1,
        }

        if unit_type in THERMAL_TYPES:
            slope, intercept, error = _thermal_heat_rate(row)
            fit_errors.append((name, error))
            start_heat = number(row[STARTUP_TIERS[startup_tier]])
            ramp = number(row["Ramp Rate MW/Min"]) * 60.0
            record.update(
                min_mw=number(row["PMin MW"]),
                heat_rate_slope_mmbtu_per_mwh=slope,
                heat_rate_intercept_mmbtu_per_hr=intercept,
                # Startup cost = fuel burned lighting off, plus the
                # non-fuel component RTS-GMLC lists separately.
                startup_cost=start_heat * number(row["Fuel Price $/MMBTU"])
                + number(row["Non Fuel Start Cost $"]),
                shutdown_cost=number(row.get("Non Fuel Shutdown Cost $")),
                # gridlock requires a positive ramp; a unit with none
                # recorded can traverse its full range in an hour.
                ramp_rate_mw_per_hr=ramp if ramp > 0 else max_mw,
                min_up_time_hr=ceil_hours(row["Min Up Time Hr"]),
                min_down_time_hr=ceil_hours(row["Min Down Time Hr"]),
            )
        elif unit_type in PROFILE_TYPES:
            frame = profiles.get(unit_type)
            if frame is not None and name in frame.columns:
                availability_columns[name] = availability_from_mw(frame[name], max_mw)

        records.append(record)

    generators = pd.DataFrame(records)

    # -------------------------------------------------------------- storage
    volumes = (
        storage_source[storage_source["position"] == "head"]
        .groupby("GEN UID")["Max Volume GWh"]
        .max()
    )
    storage_records = []
    for _, row in gen[gen["Unit Type"] == "STORAGE"].iterrows():
        name = str(row["GEN UID"])
        energy_gwh = float(volumes.get(name, 0.0))
        efficiency = number(row["Storage Roundtrip Efficiency"])
        storage_records.append(
            {
                "name": name,
                "node": node_of(row["Bus ID"]),
                "technology": "battery",
                "power_mw": float(row["PMax MW"]),
                "energy_mwh": energy_gwh * 1000.0,
                # Published as a percentage (85), not a fraction.
                "roundtrip_efficiency": efficiency / 100.0 if efficiency > 1 else efficiency,
            }
        )
    storage = pd.DataFrame(
        storage_records, columns=["name", "node", "technology", "power_mw", "energy_mwh", "roundtrip_efficiency"]
    )
    storage = storage[(storage["power_mw"] > 0) & (storage["energy_mwh"] > 0)]

    # --------------------------------------------------------------- demand
    regional = _hours_index(
        pd.read_csv(source / "timeseries_data_files" / "Load" / "DAY_AHEAD_regional_Load.csv")
    )
    # Regional totals are split across buses by their share of area load.
    bus["share"] = bus["MW Load"] / bus.groupby("Area")["MW Load"].transform("sum")
    demand = pd.DataFrame(index=regional.index)
    for _, bus_row in bus.iterrows():
        area = str(bus_row["Area"])
        if area not in regional.columns:
            continue
        node = node_of(bus_row["Bus ID"])
        series = regional[area].to_numpy(dtype=float) * float(bus_row["share"])
        demand[node] = demand[node].to_numpy() + series if node in demand else series

    availability = pd.DataFrame(availability_columns, index=regional.index)

    if hours is not None:
        demand = demand.iloc[:hours].reset_index(drop=True)
        availability = availability.iloc[:hours].reset_index(drop=True)

    if fit_errors:
        worst = max(fit_errors, key=lambda item: item[1])
        mean_error = float(np.mean([error for _, error in fit_errors]))
        print(
            f"  heat-rate fit: worst {worst[1]:.2%} ({worst[0]}), "
            f"mean {mean_error:.2%} across {len(fit_errors)} thermal units"
        )
        thermal = generators[generators["min_mw"] > 0]
        clamped = thermal[thermal["heat_rate_intercept_mmbtu_per_hr"] == 0]
        if len(clamped):
            print(
                f"  {len(clamped)} unit(s) had a convex curve whose secant implied a "
                "negative no-load term; clamped to zero (this is where the worst "
                f"fit error sits): {', '.join(clamped['name'])}"
            )
        flat = thermal[thermal["heat_rate_slope_mmbtu_per_mwh"] == 0]
        if len(flat):
            print(
                f"  {len(flat)} unit(s) have a flat heat rate in the source data "
                f"(zero marginal cost): {', '.join(flat['name'])}"
            )

    return generators, storage, nodes, lines, demand, availability


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="RTS_Data directory")
    parser.add_argument("--output-dir", default=None, help="where to write the gridlock CSVs")
    parser.add_argument(
        "--aggregate",
        choices=["none", "area"],
        default="none",
        help="'none' keeps all 73 buses; 'area' collapses to 3 nodes joined by "
        "summed tie capacities (an honest transport model)",
    )
    parser.add_argument(
        "--startup-tier",
        choices=["cold", "warm", "hot"],
        default="hot",
        help="which start-heat tier to use for the single startup_cost column "
        "(default: hot, what a cycling unit actually pays)",
    )
    parser.add_argument("--hours", type=int, default=None, help="truncate to the first N hours")
    args = parser.parse_args(argv)

    source = Path(args.source)
    if not (source / "SourceData" / "gen.csv").is_file():
        raise SystemExit(
            f"RTS-GMLC not found at {source}\n"
            "run: python scripts/fetch_external_data.py rts-gmlc"
        )

    label = "rts_gmlc" if args.aggregate == "none" else "rts_gmlc_area"
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "data" / label

    print(f"importing RTS-GMLC ({args.aggregate} aggregation, {args.startup_tier} starts)")
    frames = build(source, args.aggregate, args.startup_tier, args.hours)
    generators, storage, nodes, lines, demand, availability = frames

    provenance = f"""# Provenance

Converted from **RTS-GMLC** by `scripts/import_rts_gmlc.py`.

- Upstream: https://github.com/GridMod/RTS-GMLC
- Aggregation: `{args.aggregate}` | startup tier: `{args.startup_tier}`
- Hours: {len(demand)}

## License

RTS-GMLC carries no SPDX license file. Its README contains an NREL "Data
Use Disclaimer Agreement" granting "the right, without any fee or cost, to
use, copy, and distribute these Data for any purpose whatsoever, provided
that this entire notice appears in all copies", with attribution and an
indemnity clause. Reproduce that notice alongside any redistribution of
this converted data.

## Conversion decisions

- Heat rates come from `SourceData/gen.csv` (average heat rate at minimum
  load plus incremental bands), collapsed to slope + intercept by the
  secant between minimum and maximum stable output. The bundled openTEPES
  export was **not** used: it zeroes the no-load term, VOM and shutdown
  cost for every unit and states startup cost in thousands of dollars.
- Startup cost = start heat (`{args.startup_tier}` tier) x fuel price +
  `Non Fuel Start Cost $`.
- Ramp rates converted from MW/min to MW/h. Units with no rate recorded
  are given a full-range hourly ramp.
- Minimum up/down times rounded **up** to whole hours (the source has
  2.2 h and 4.5 h values; gridlock truncates, which would relax them).
- Hydro, run-of-river and CSP are modelled as profile-driven generators
  with no commitment state, since gridlock has no reservoir model.
  Synchronous condensers (0 MW) are dropped.
- Storage round-trip efficiency converted from percent; energy from
  `storage.csv` head volumes (GWh -> MWh).
- Line loss factors are a flat 1% (RTS-GMLC publishes none; matches the
  upstream openTEPES export).

## Network caveat

RTS-GMLC line ratings are thermal limits for a **DC power flow**. gridlock
is a transport model and ignores Kirchhoff's voltage law, so with
`--aggregate none` flows route around congestion in ways the real system
cannot and cost is optimistically low. `--aggregate area` collapses the
three areas to single nodes joined by summed tie capacities, which is a
defensible transport representation.
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
