"""Shared helpers for building tiny in-memory test systems."""

from pathlib import Path

import pandas as pd
import pytest

from gridlock.data import build_system

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DATA_DIR = REPO_ROOT / "data" / "example"


def gen(
    name,
    node,
    marginal_cost,
    max_mw,
    min_mw=0.0,
    startup_cost=0.0,
    shutdown_cost=0.0,
    no_load_cost=0.0,
    ramp=None,
    min_up=1,
    min_down=1,
):
    """A generator row whose marginal cost is set directly via VOM."""
    return {
        "name": name,
        "node": node,
        "technology": "test",
        "max_mw": max_mw,
        "min_mw": min_mw,
        "heat_rate_slope_mmbtu_per_mwh": 0.0,
        "heat_rate_intercept_mmbtu_per_hr": no_load_cost,  # priced at $1/MMBtu
        "fuel_cost_per_mmbtu": 1.0,
        "vom_cost_per_mwh": marginal_cost,
        "startup_cost": startup_cost,
        "shutdown_cost": shutdown_cost,
        "ramp_rate_mw_per_hr": ramp if ramp is not None else max_mw,
        "min_up_time_hr": min_up,
        "min_down_time_hr": min_down,
    }


def store(name, node, power_mw, energy_mwh, roundtrip_efficiency):
    return {
        "name": name,
        "node": node,
        "technology": "battery",
        "power_mw": power_mw,
        "energy_mwh": energy_mwh,
        "roundtrip_efficiency": roundtrip_efficiency,
    }


def line(from_node, to_node, capacity_mw, loss_factor, name=None):
    row = {
        "from_node": from_node,
        "to_node": to_node,
        "capacity_mw": capacity_mw,
        "loss_factor": loss_factor,
    }
    if name is not None:
        row["line"] = name
    return row


def make_system(generators, demand, storage=None, lines=None, availability=None):
    """Assemble a SystemData from plain dict/list inputs.

    ``demand`` is a dict of node -> list of hourly MW; nodes are inferred
    from its keys. ``availability`` is an optional dict of unit -> list.
    """
    demand_df = pd.DataFrame(demand, dtype=float)
    nodes_df = pd.DataFrame(
        {"node": list(demand_df.columns), "latitude": 0.0, "longitude": 0.0}
    )
    storage_cols = ["name", "node", "technology", "power_mw", "energy_mwh", "roundtrip_efficiency"]
    line_cols = ["from_node", "to_node", "capacity_mw", "loss_factor"]
    availability_df = (
        pd.DataFrame(availability, dtype=float)
        if availability
        else pd.DataFrame(index=range(len(demand_df)))
    )
    return build_system(
        generators=pd.DataFrame(generators),
        storage=pd.DataFrame(storage) if storage else pd.DataFrame(columns=storage_cols),
        nodes=nodes_df,
        network=pd.DataFrame(lines) if lines else pd.DataFrame(columns=line_cols),
        demand=demand_df,
        availability=availability_df,
    )


@pytest.fixture(scope="session")
def example_data_dir():
    if not EXAMPLE_DATA_DIR.is_dir():
        pytest.skip("example dataset not generated (run scripts/make_example_data.py)")
    return EXAMPLE_DATA_DIR
