"""Generate larger synthetic systems for size-scaling studies.

The committed example system has 12 generators across 3 nodes, which is
too small to show how UC cost grows with *system* size rather than
horizon length. This script scales that archetype up:

    python scripts/make_large_data.py --zones 10 --units-per-archetype 3
    python scripts/make_large_data.py --preset large

Each zone is a copy of the NORTH/SOUTH/EAST archetype mix with jittered
capacities and its own demand shape, wired into a ring plus a few chords.
``--units-per-archetype N`` duplicates every thermal archetype N times
within a zone with *identical* parameters, which is exactly the structure
integer clustering (``--cluster-units``) is meant to exploit: the copies
are interchangeable, so a per-unit model has N! equivalent schedules and
a clustered one has a single integer count.

Renewables are never duplicated (they carry unit-specific profiles and
need no commitment); their capacity is scaled instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HOURS = 8760
DEFAULT_SEED = 7

# (technology, max_mw, min_mw, hr_slope, hr_intercept, fuel, vom,
#  startup, shutdown, ramp, min_up, min_down)
THERMAL_ARCHETYPES = [
    ("coal", 650.0, 260.0, 9.90, 320.0, 2.15, 4.5, 45000.0, 5000.0, 260.0, 24, 12),
    ("gas_cc", 500.0, 160.0, 6.70, 95.0, 3.60, 3.4, 12000.0, 1500.0, 350.0, 6, 4),
    ("gas_cc", 420.0, 140.0, 7.00, 85.0, 3.60, 3.6, 10000.0, 1200.0, 300.0, 6, 4),
    ("gas_ct", 250.0, 70.0, 11.20, 28.0, 3.60, 5.5, 4500.0, 500.0, 250.0, 1, 1),
    ("gas_ct", 180.0, 50.0, 11.00, 20.0, 3.90, 5.4, 3500.0, 400.0, 180.0, 1, 1),
]
NUCLEAR = ("nuclear", 1000.0, 900.0, 10.45, 30.0, 0.72, 2.3, 120000.0, 10000.0, 500.0, 72, 48)

GENERATOR_COLUMNS = [
    "name", "node", "technology", "max_mw", "min_mw",
    "heat_rate_slope_mmbtu_per_mwh", "heat_rate_intercept_mmbtu_per_hr",
    "fuel_cost_per_mmbtu", "vom_cost_per_mwh", "startup_cost", "shutdown_cost",
    "ramp_rate_mw_per_hr", "min_up_time_hr", "min_down_time_hr", "num_units",
]

PRESETS = {
    # name: (zones, units per thermal archetype)
    "medium": (5, 2),
    "large": (10, 3),
    "huge": (20, 4),
}


def build(zones: int, units_per_archetype: int, seed: int, cluster_rows: bool):
    rng = np.random.default_rng(seed)
    hour = np.arange(HOURS)
    day = hour // 24
    hour_of_day = hour % 24
    is_weekend = (day % 7) >= 5

    diurnal = np.array(
        [0.82, 0.78, 0.76, 0.75, 0.76, 0.80, 0.88, 0.96, 1.00, 1.02, 1.03, 1.04,
         1.04, 1.03, 1.03, 1.04, 1.07, 1.12, 1.16, 1.15, 1.10, 1.03, 0.95, 0.87]
    )[hour_of_day]
    weekend = np.where(is_weekend, 0.93, 1.0)

    def demand_series(base_mw, amplitude, peak_day):
        seasonal = 1 + amplitude * np.cos(2 * np.pi * (day - peak_day) / 365)
        noise = np.zeros(HOURS)
        shocks = rng.normal(0, 0.012, HOURS)
        for t in range(1, HOURS):
            noise[t] = 0.95 * noise[t - 1] + shocks[t]
        return base_mw * seasonal * diurnal * weekend * (1 + noise)

    def wind_profile(mean_cf):
        z = np.zeros(HOURS)
        shocks = rng.normal(0, 0.35, HOURS)
        for t in range(1, HOURS):
            z[t] = 0.97 * z[t - 1] + shocks[t]
        tilt = 0.6 * np.cos(2 * np.pi * (day - 20) / 365)
        return np.clip(1 / (1 + np.exp(-(z + tilt + np.log(mean_cf / (1 - mean_cf))))), 0.02, 0.98)

    def solar_profile():
        daylight = 12 + 2.6 * np.cos(2 * np.pi * (day - 172) / 365)
        sunrise, sunset = 12 - daylight / 2, 12 + daylight / 2
        arc = np.sin(np.pi * (hour_of_day - sunrise) / np.maximum(sunset - sunrise, 1e-6))
        clear = np.where((hour_of_day > sunrise) & (hour_of_day < sunset), arc, 0.0)
        cloud = np.clip(0.85 + np.cumsum(rng.normal(0, 0.02, HOURS)) * 0.1, 0.15, 1.0)
        return np.clip(clear * (0.85 + 0.15 * np.cos(2 * np.pi * (day - 172) / 365)) * cloud, 0.0, 1.0)

    def outage_profile(maintenance_day, maintenance_days, events):
        af = np.ones(HOURS)
        start = maintenance_day * 24
        af[start : start + maintenance_days * 24] = 0.0
        for _ in range(events):
            begin = int(rng.integers(0, HOURS - 96))
            af[begin : begin + int(rng.integers(24, 96))] = float(rng.choice([0.0, 0.0, 0.6]))
        return af

    nodes, generators, storage, demand, availability = [], [], [], {}, {}

    for z in range(zones):
        node = f"Z{z:02d}"
        nodes.append((node, 30.0 + 1.5 * z, -120.0 + 2.0 * z))
        peak_day = 15 if z % 2 == 0 else 196
        demand[node] = demand_series(700.0 + 120.0 * (z % 4), 0.10 + 0.02 * (z % 4), peak_day)

        # One nuclear unit in every fourth zone; it is never duplicated.
        if z % 4 == 0:
            tech, mx, mn, slope, icpt, fuel, vom, su, sd, ramp, up, down = NUCLEAR
            name = f"{tech}_{node}"
            generators.append(
                (name, node, tech, mx, mn, slope, icpt, fuel, vom, su, sd, ramp, up, down, 1)
            )
            availability[name] = outage_profile(120, 21, 1)

        for a, arch in enumerate(THERMAL_ARCHETYPES):
            tech, mx, mn, slope, icpt, fuel, vom, su, sd, ramp, up, down = arch
            # Jitter across zones (so zones differ) but never within an
            # archetype's copies (so they stay perfectly interchangeable).
            scale = 1.0 + 0.1 * ((z * 7 + a * 3) % 5 - 2) / 5
            mx, mn, ramp = round(mx * scale, 1), round(mn * scale, 1), round(ramp * scale, 1)
            fuel = round(fuel * (1.0 + 0.05 * ((z + a) % 3 - 1)), 3)

            if cluster_rows:
                name = f"{tech}{a}_{node}"
                generators.append(
                    (name, node, tech, mx, mn, slope, icpt, fuel, vom, su, sd, ramp,
                     up, down, units_per_archetype)
                )
            else:
                for k in range(units_per_archetype):
                    name = f"{tech}{a}_{node}_{k}"
                    generators.append(
                        (name, node, tech, mx, mn, slope, icpt, fuel, vom, su, sd, ramp,
                         up, down, 1)
                    )

        wind_name = f"wind_{node}"
        generators.append(
            (wind_name, node, "wind", round(600.0 * (1 + 0.2 * (z % 3)), 1), 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 900.0, 1, 1, 1)
        )
        availability[wind_name] = wind_profile(0.34 + 0.02 * (z % 3))

        if z % 2 == 0:
            solar_name = f"solar_{node}"
            generators.append(
                (solar_name, node, "solar", round(500.0 * (1 + 0.2 * (z % 3)), 1), 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 700.0, 1, 1, 1)
            )
            availability[solar_name] = solar_profile()

        storage.append((f"battery_{node}", node, "battery", 200.0, 800.0, 0.88))

    # Ring plus chords: enough meshing that flows are not trivially determined.
    lines = []
    for z in range(zones):
        nxt = (z + 1) % zones
        if zones > 1 and (z != zones - 1 or zones > 2):
            lines.append((f"L_Z{z:02d}_Z{nxt:02d}", f"Z{z:02d}", f"Z{nxt:02d}", 700.0, 0.02))
    for z in range(0, zones - 2, 3):
        lines.append((f"C_Z{z:02d}_Z{z + 2:02d}", f"Z{z:02d}", f"Z{z + 2:02d}", 400.0, 0.03))

    return (
        pd.DataFrame(nodes, columns=["node", "latitude", "longitude"]),
        pd.DataFrame(generators, columns=GENERATOR_COLUMNS),
        pd.DataFrame(
            storage,
            columns=["name", "node", "technology", "power_mw", "energy_mwh",
                     "roundtrip_efficiency"],
        ),
        pd.DataFrame(
            lines, columns=["line", "from_node", "to_node", "capacity_mw", "loss_factor"]
        ),
        pd.DataFrame(demand).round(2),
        pd.DataFrame(availability).round(4),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None)
    parser.add_argument("--zones", type=int, default=10)
    parser.add_argument("--units-per-archetype", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--pre-clustered",
        action="store_true",
        help="emit one row per archetype with num_units set, instead of one "
        "row per physical unit (the same system, already pooled)",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    zones, units = args.zones, args.units_per_archetype
    if args.preset:
        zones, units = PRESETS[args.preset]

    nodes, generators, storage, network, demand, availability = build(
        zones, units, args.seed, args.pre_clustered
    )

    label = args.preset or f"z{zones}u{units}"
    out = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parents[1] / "data" / f"synthetic_{label}"
    )
    out.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(out / "node_locations.csv", index=False)
    network.to_csv(out / "network.csv", index=False)
    generators.to_csv(out / "generators.csv", index=False)
    storage.to_csv(out / "storage.csv", index=False)
    demand.rename_axis("hour").to_csv(out / "demand.csv")
    availability.rename_axis("hour").to_csv(out / "availability_factors.csv")

    committed = generators[
        (generators.min_mw > 0) | (generators.startup_cost > 0)
    ]
    print(f"wrote {out}")
    print(f"  {len(nodes)} nodes, {len(network)} lines, {len(storage)} storage")
    print(
        f"  {len(generators)} generator rows / "
        f"{int(generators.num_units.sum())} physical units; "
        f"{int(committed.num_units.sum())} need commitment"
    )
    print(f"  peak demand {demand.sum(axis=1).max():,.0f} MW")
    print(
        f"  capacity {(generators.max_mw * generators.num_units).sum():,.0f} MW"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
