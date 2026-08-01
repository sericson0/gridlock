"""Generate the synthetic 3-node example dataset in data/example/.

Deterministic (seeded) so the committed CSVs are reproducible:

    python scripts/make_example_data.py

The system: three nodes (NORTH winter-peaking, SOUTH summer-peaking with
solar, EAST in between), twelve generators across nuclear / coal / gas CC /
gas CT / wind / solar, two storage units, three transmission lines.
Availability factors carry planned maintenance, random forced outages and
synthetic wind/solar profiles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HOURS = 8760
SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "example"

rng = np.random.default_rng(SEED)

hour = np.arange(HOURS)
day = hour // 24
hour_of_day = hour % 24
day_of_week = day % 7
is_weekend = day_of_week >= 5


# ---------------------------------------------------------------------------
# Static tables
# ---------------------------------------------------------------------------

NODES = pd.DataFrame(
    [
        ("NORTH", 46.0, -94.2),
        ("SOUTH", 31.5, -97.2),
        ("EAST", 40.5, -77.5),
    ],
    columns=["node", "latitude", "longitude"],
)

NETWORK = pd.DataFrame(
    [
        ("NORTH_SOUTH", "NORTH", "SOUTH", 700.0, 0.020),
        ("SOUTH_EAST", "SOUTH", "EAST", 500.0, 0.025),
        ("NORTH_EAST", "NORTH", "EAST", 400.0, 0.015),
    ],
    columns=["line", "from_node", "to_node", "capacity_mw", "loss_factor"],
)

GENERATOR_COLUMNS = [
    "name", "node", "technology", "max_mw", "min_mw",
    "heat_rate_slope_mmbtu_per_mwh", "heat_rate_intercept_mmbtu_per_hr",
    "fuel_cost_per_mmbtu", "vom_cost_per_mwh", "startup_cost", "shutdown_cost",
    "ramp_rate_mw_per_hr", "min_up_time_hr", "min_down_time_hr",
]

GENERATORS = pd.DataFrame(
    [
        ("nuclear_north",  "NORTH", "nuclear", 1000.0, 900.0, 10.45,  30.0, 0.72, 2.3, 120000.0, 10000.0, 500.0, 72, 48),
        ("coal_north",     "NORTH", "coal",     650.0, 260.0,  9.90, 320.0, 2.15, 4.5,  45000.0,  5000.0, 260.0, 24, 12),
        ("ct_north",       "NORTH", "gas_ct",   220.0,  60.0, 10.80,  25.0, 3.60, 5.2,   4000.0,   500.0, 220.0,  1,  1),
        ("wind_north",     "NORTH", "wind",     900.0,   0.0,  0.00,   0.0, 0.00, 0.0,      0.0,     0.0, 900.0,  1,  1),
        ("ccgt_south_1",   "SOUTH", "gas_cc",   500.0, 160.0,  6.70,  95.0, 3.60, 3.4,  12000.0,  1500.0, 350.0,  6,  4),
        ("ccgt_south_2",   "SOUTH", "gas_cc",   420.0, 140.0,  7.00,  85.0, 3.60, 3.6,  10000.0,  1200.0, 300.0,  6,  4),
        ("ct_south",       "SOUTH", "gas_ct",   250.0,  70.0, 11.20,  28.0, 3.60, 5.5,   4500.0,   500.0, 250.0,  1,  1),
        ("solar_south",    "SOUTH", "solar",    700.0,   0.0,  0.00,   0.0, 0.00, 0.0,      0.0,     0.0, 700.0,  1,  1),
        ("coal_east",      "EAST",  "coal",     480.0, 200.0, 10.10, 260.0, 2.30, 4.8,  38000.0,  4000.0, 200.0, 24, 12),
        ("ccgt_east",      "EAST",  "gas_cc",   380.0, 130.0,  6.90,  80.0, 3.90, 3.5,  11000.0,  1300.0, 280.0,  6,  4),
        ("ct_east",        "EAST",  "gas_ct",   180.0,  50.0, 11.00,  20.0, 3.90, 5.4,   3500.0,   400.0, 180.0,  1,  1),
        ("wind_east",      "EAST",  "wind",     350.0,   0.0,  0.00,   0.0, 0.00, 0.0,      0.0,     0.0, 350.0,  1,  1),
    ],
    columns=GENERATOR_COLUMNS,
)

STORAGE = pd.DataFrame(
    [
        ("battery_south", "SOUTH", "battery", 250.0, 1000.0, 0.88),
        ("pumped_north",  "NORTH", "pumped_hydro", 300.0, 2400.0, 0.76),
    ],
    columns=["name", "node", "technology", "power_mw", "energy_mwh", "roundtrip_efficiency"],
)


# ---------------------------------------------------------------------------
# Demand profiles
# ---------------------------------------------------------------------------

DIURNAL_SHAPE = np.array(
    [0.82, 0.78, 0.76, 0.75, 0.76, 0.80, 0.88, 0.96, 1.00, 1.02, 1.03, 1.04,
     1.04, 1.03, 1.03, 1.04, 1.07, 1.12, 1.16, 1.15, 1.10, 1.03, 0.95, 0.87]
)


def make_demand(base_mw: float, seasonal_amplitude: float, peak_day: int) -> np.ndarray:
    """Multiplicative seasonal * diurnal * weekend shape with mild AR(1) noise."""
    seasonal = 1 + seasonal_amplitude * np.cos(2 * np.pi * (day - peak_day) / 365)
    diurnal = DIURNAL_SHAPE[hour_of_day]
    weekend = np.where(is_weekend, 0.93, 1.0)

    noise = np.empty(HOURS)
    noise[0] = 0.0
    shocks = rng.normal(0, 0.012, HOURS)
    for t in range(1, HOURS):
        noise[t] = 0.95 * noise[t - 1] + shocks[t]
    return base_mw * seasonal * diurnal * weekend * (1 + noise)


demand = pd.DataFrame(
    {
        "NORTH": make_demand(1000.0, 0.16, peak_day=15),    # winter peaking
        "SOUTH": make_demand(1300.0, 0.18, peak_day=196),   # summer peaking
        "EAST": make_demand(750.0, 0.10, peak_day=196),
    }
).round(2)


# ---------------------------------------------------------------------------
# Availability factors
# ---------------------------------------------------------------------------


def thermal_availability(maintenance_start_day: int, maintenance_days: int,
                         forced_outage_events: int) -> np.ndarray:
    """1.0 with a planned maintenance block and random forced outages."""
    af = np.ones(HOURS)
    start = maintenance_start_day * 24
    af[start : start + maintenance_days * 24] = 0.0
    for _ in range(forced_outage_events):
        outage_start = rng.integers(0, HOURS - 96)
        duration = int(rng.integers(24, 96))
        af[outage_start : outage_start + duration] = float(rng.choice([0.0, 0.0, 0.6]))
    return af


def wind_availability(mean_cf: float) -> np.ndarray:
    """AR(1) process mapped through a logistic to [0.02, 0.98], winter-tilted."""
    z = np.empty(HOURS)
    z[0] = 0.0
    shocks = rng.normal(0, 0.35, HOURS)
    for t in range(1, HOURS):
        z[t] = 0.97 * z[t - 1] + shocks[t]
    seasonal_tilt = 0.6 * np.cos(2 * np.pi * (day - 20) / 365)
    cf = 1 / (1 + np.exp(-(z + seasonal_tilt + np.log(mean_cf / (1 - mean_cf)))))
    return np.clip(cf, 0.02, 0.98)


def solar_availability() -> np.ndarray:
    """Clear-sky diurnal arc scaled by season, with autocorrelated cloud cover."""
    daylight_hours = 12 + 2.6 * np.cos(2 * np.pi * (day - 172) / 365)
    sunrise = 12 - daylight_hours / 2
    sunset = 12 + daylight_hours / 2
    arc = np.sin(np.pi * (hour_of_day - sunrise) / np.maximum(sunset - sunrise, 1e-6))
    clear_sky = np.where((hour_of_day > sunrise) & (hour_of_day < sunset), arc, 0.0)
    seasonal_strength = 0.85 + 0.15 * np.cos(2 * np.pi * (day - 172) / 365)

    cloud = np.empty(HOURS)
    cloud[0] = 0.85
    shocks = rng.normal(0, 0.08, HOURS)
    for t in range(1, HOURS):
        cloud[t] = 0.90 * cloud[t - 1] + 0.10 * 0.85 + shocks[t]
    cloud = np.clip(cloud, 0.15, 1.0)
    return np.clip(clear_sky * seasonal_strength * cloud, 0.0, 1.0)


availability = pd.DataFrame(
    {
        "nuclear_north": thermal_availability(120, 21, forced_outage_events=1),
        "coal_north": thermal_availability(90, 14, forced_outage_events=3),
        "ct_north": thermal_availability(250, 7, forced_outage_events=2),
        "wind_north": wind_availability(0.38),
        "ccgt_south_1": thermal_availability(75, 7, forced_outage_events=2),
        "ccgt_south_2": thermal_availability(260, 7, forced_outage_events=2),
        "ct_south": thermal_availability(285, 7, forced_outage_events=2),
        "solar_south": solar_availability(),
        "coal_east": thermal_availability(105, 14, forced_outage_events=3),
        "ccgt_east": thermal_availability(270, 7, forced_outage_events=2),
        "ct_east": thermal_availability(300, 7, forced_outage_events=2),
        "wind_east": wind_availability(0.34),
    }
).round(4)


# ---------------------------------------------------------------------------
# Write files
# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    NODES.to_csv(OUTPUT_DIR / "node_locations.csv", index=False)
    NETWORK.to_csv(OUTPUT_DIR / "network.csv", index=False)
    GENERATORS.to_csv(OUTPUT_DIR / "generators.csv", index=False)
    STORAGE.to_csv(OUTPUT_DIR / "storage.csv", index=False)
    demand.rename_axis("hour").to_csv(OUTPUT_DIR / "demand.csv")
    availability.rename_axis("hour").to_csv(OUTPUT_DIR / "availability_factors.csv")

    peak = demand.sum(axis=1).max()
    print(f"wrote {OUTPUT_DIR}")
    print(f"coincident peak demand: {peak:,.0f} MW")
    print(f"total demand: {demand.to_numpy().sum() / 1e6:.2f} TWh")
    print(f"thermal capacity: {GENERATORS.loc[GENERATORS.min_mw > 0, 'max_mw'].sum():,.0f} MW")
    print(f"renewable capacity: {GENERATORS.loc[GENERATORS.min_mw == 0, 'max_mw'].sum():,.0f} MW")
    print(f"storage power: {STORAGE.power_mw.sum():,.0f} MW")


if __name__ == "__main__":
    main()
