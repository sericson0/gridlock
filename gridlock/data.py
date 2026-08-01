"""Load and validate the input CSVs into a :class:`SystemData` object.

Input files (see README for full column specifications):

- ``generators.csv``            one row per thermal/renewable unit
- ``storage.csv``               one row per storage unit
- ``node_locations.csv``        one row per node (name, latitude, longitude)
- ``network.csv``               one row per line (pipe-and-bubble transport line)
- ``demand.csv``                hourly MW demand, one column per node
- ``availability_factors.csv``  hourly availability in [0, 1]; optional, and
                                only units that need derating need a column
                                (absent units are fully available)

All validation lives here so the model builder can assume clean data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

GENERATOR_REQUIRED_COLUMNS = [
    "name",
    "node",
    "technology",
    "max_mw",
    "min_mw",
    "heat_rate_slope_mmbtu_per_mwh",
    "fuel_cost_per_mmbtu",
    "vom_cost_per_mwh",
    "startup_cost",
    "ramp_rate_mw_per_hr",
]

# Optional generator columns and the value used when a column is absent.
GENERATOR_OPTIONAL_COLUMNS = {
    "heat_rate_intercept_mmbtu_per_hr": 0.0,
    "shutdown_cost": 0.0,
    "min_up_time_hr": 1,
    "min_down_time_hr": 1,
}

STORAGE_REQUIRED_COLUMNS = [
    "name",
    "node",
    "technology",
    "power_mw",
    "energy_mwh",
    "roundtrip_efficiency",
]

NETWORK_REQUIRED_COLUMNS = ["from_node", "to_node", "capacity_mw", "loss_factor"]

NODE_REQUIRED_COLUMNS = ["node", "latitude", "longitude"]


@dataclass
class SystemData:
    """A validated power system ready to be handed to the model builder.

    generators / storage / nodes / network are indexed by unit, unit, node
    and line name respectively. demand and availability are hourly frames
    (row = hour, 0-based); demand's columns cover every node, while
    availability holds only the units that have a profile — units without
    a column are fully available.
    """

    generators: pd.DataFrame
    storage: pd.DataFrame
    nodes: pd.DataFrame
    network: pd.DataFrame
    demand: pd.DataFrame
    availability: pd.DataFrame

    @property
    def num_hours(self) -> int:
        return len(self.demand)


def load_system(data_dir: str | Path) -> SystemData:
    """Read the six input CSVs from ``data_dir`` and validate them."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data directory not found: {data_dir}")

    def read(name: str) -> pd.DataFrame:
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"required input file not found: {path}")
        return pd.read_csv(path)

    def read_optional(name: str) -> pd.DataFrame:
        path = data_dir / name
        return pd.read_csv(path) if path.is_file() else pd.DataFrame()

    return build_system(
        generators=read("generators.csv"),
        storage=read("storage.csv"),
        nodes=read("node_locations.csv"),
        network=read("network.csv"),
        demand=_strip_hour_column(read("demand.csv")),
        availability=_strip_hour_column(read_optional("availability_factors.csv")),
    )


def build_system(
    generators: pd.DataFrame,
    storage: pd.DataFrame,
    nodes: pd.DataFrame,
    network: pd.DataFrame,
    demand: pd.DataFrame,
    availability: pd.DataFrame,
) -> SystemData:
    """Normalize, derive and validate raw input frames into a SystemData.

    This is the single entry point for both CSV loading and programmatic
    construction (used heavily by the test suite).
    """
    nodes = _prepare_nodes(nodes)
    generators = _prepare_generators(generators, nodes)
    storage = _prepare_storage(storage, nodes)
    network = _prepare_network(network, nodes)
    demand = _prepare_demand(demand, nodes)
    availability = _prepare_availability(availability, generators, len(demand))

    return SystemData(
        generators=generators,
        storage=storage,
        nodes=nodes,
        network=network,
        demand=demand,
        availability=availability,
    )


def _strip_hour_column(df: pd.DataFrame) -> pd.DataFrame:
    """Drop a leading hour/index column from a profile file, if present."""
    if len(df.columns) == 0:
        return df
    first = str(df.columns[0]).strip().lower()
    if first in ("hour", "hours", "unnamed: 0", "index", "t"):
        df = df.drop(columns=df.columns[0])
    return df.reset_index(drop=True)


def _require_columns(df: pd.DataFrame, required: list[str], file_label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{file_label} is missing required columns: {missing}")


def _prepare_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    _require_columns(nodes, NODE_REQUIRED_COLUMNS, "node_locations.csv")
    nodes = nodes.copy()
    nodes["node"] = nodes["node"].astype(str)
    if nodes["node"].duplicated().any():
        dupes = nodes.loc[nodes["node"].duplicated(), "node"].tolist()
        raise ValueError(f"duplicate node names in node_locations.csv: {dupes}")
    return nodes.set_index("node")


def _prepare_generators(generators: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    _require_columns(generators, GENERATOR_REQUIRED_COLUMNS, "generators.csv")
    gens = generators.copy()
    gens["name"] = gens["name"].astype(str)
    if gens["name"].duplicated().any():
        dupes = gens.loc[gens["name"].duplicated(), "name"].tolist()
        raise ValueError(f"duplicate generator names in generators.csv: {dupes}")

    for column, default in GENERATOR_OPTIONAL_COLUMNS.items():
        if column not in gens.columns:
            gens[column] = default
        gens[column] = gens[column].fillna(default)

    unknown_nodes = sorted(set(gens["node"].astype(str)) - set(nodes.index))
    if unknown_nodes:
        raise ValueError(f"generators.csv references unknown nodes: {unknown_nodes}")

    gens = gens.set_index("name")
    numeric = [c for c in gens.columns if c not in ("node", "technology")]
    gens[numeric] = gens[numeric].astype(float)

    if (gens["max_mw"] < 0).any() or (gens["min_mw"] < 0).any():
        raise ValueError("generator capacities must be non-negative")
    bad_min = gens.index[gens["min_mw"] > gens["max_mw"]].tolist()
    if bad_min:
        raise ValueError(f"generators with min_mw > max_mw: {bad_min}")
    bad_ramp = gens.index[gens["ramp_rate_mw_per_hr"] <= 0].tolist()
    if bad_ramp:
        raise ValueError(
            f"generators with non-positive ramp_rate_mw_per_hr: {bad_ramp} "
            "(set ramp_rate_mw_per_hr >= max_mw for an unconstrained unit)"
        )
    bad_time = gens.index[
        (gens["min_up_time_hr"] < 1) | (gens["min_down_time_hr"] < 1)
    ].tolist()
    if bad_time:
        raise ValueError(f"generators with min up/down time below 1 hour: {bad_time}")

    # Derived economics: $/MWh at the margin, and $/hr while committed.
    gens["marginal_cost"] = (
        gens["fuel_cost_per_mmbtu"] * gens["heat_rate_slope_mmbtu_per_mwh"]
        + gens["vom_cost_per_mwh"]
    )
    gens["no_load_cost"] = (
        gens["fuel_cost_per_mmbtu"] * gens["heat_rate_intercept_mmbtu_per_hr"]
    )

    # A unit only needs commitment variables when on/off state matters.
    gens["needs_commitment"] = (
        (gens["min_mw"] > 0)
        | (gens["startup_cost"] > 0)
        | (gens["shutdown_cost"] > 0)
        | (gens["no_load_cost"] > 0)
        | (gens["min_up_time_hr"] > 1)
        | (gens["min_down_time_hr"] > 1)
    )
    return gens


def _prepare_storage(storage: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    _require_columns(storage, STORAGE_REQUIRED_COLUMNS, "storage.csv")
    stor = storage.copy()
    stor["name"] = stor["name"].astype(str)
    if stor["name"].duplicated().any():
        dupes = stor.loc[stor["name"].duplicated(), "name"].tolist()
        raise ValueError(f"duplicate storage names in storage.csv: {dupes}")

    unknown_nodes = sorted(set(stor["node"].astype(str)) - set(nodes.index))
    if unknown_nodes:
        raise ValueError(f"storage.csv references unknown nodes: {unknown_nodes}")

    stor = stor.set_index("name")
    numeric = [c for c in stor.columns if c not in ("node", "technology")]
    stor[numeric] = stor[numeric].astype(float)

    if (stor["power_mw"] < 0).any() or (stor["energy_mwh"] < 0).any():
        raise ValueError("storage power_mw and energy_mwh must be non-negative")
    bad_rte = stor.index[
        (stor["roundtrip_efficiency"] <= 0) | (stor["roundtrip_efficiency"] > 1)
    ].tolist()
    if bad_rte:
        raise ValueError(f"storage roundtrip_efficiency must be in (0, 1]: {bad_rte}")

    # The round trip is split evenly between charge and discharge.
    stor["one_way_efficiency"] = stor["roundtrip_efficiency"].apply(math.sqrt)
    return stor


def _prepare_network(network: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    _require_columns(network, NETWORK_REQUIRED_COLUMNS, "network.csv")
    net = network.copy()
    net["from_node"] = net["from_node"].astype(str)
    net["to_node"] = net["to_node"].astype(str)

    if "line" not in net.columns:
        net["line"] = net["from_node"] + "__" + net["to_node"]
    net["line"] = net["line"].astype(str)
    if net["line"].duplicated().any():
        dupes = net.loc[net["line"].duplicated(), "line"].tolist()
        raise ValueError(f"duplicate line names in network.csv: {dupes}")

    unknown_nodes = sorted(
        (set(net["from_node"]) | set(net["to_node"])) - set(nodes.index)
    )
    if unknown_nodes:
        raise ValueError(f"network.csv references unknown nodes: {unknown_nodes}")
    self_loops = net.loc[net["from_node"] == net["to_node"], "line"].tolist()
    if self_loops:
        raise ValueError(f"network.csv lines with from_node == to_node: {self_loops}")

    net = net.set_index("line")
    net[["capacity_mw", "loss_factor"]] = net[["capacity_mw", "loss_factor"]].astype(float)
    if (net["capacity_mw"] < 0).any():
        raise ValueError("line capacity_mw must be non-negative")
    bad_loss = net.index[(net["loss_factor"] < 0) | (net["loss_factor"] >= 1)].tolist()
    if bad_loss:
        raise ValueError(f"line loss_factor must be in [0, 1): {bad_loss}")
    return net


def _prepare_demand(demand: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    demand = demand.copy().reset_index(drop=True)
    demand.columns = [str(c) for c in demand.columns]
    unknown = sorted(set(demand.columns) - set(nodes.index))
    if unknown:
        raise ValueError(f"demand.csv has columns that are not known nodes: {unknown}")
    if len(demand) == 0:
        raise ValueError("demand.csv contains no rows")

    # Nodes without a demand column carry zero load.
    demand = demand.reindex(columns=list(nodes.index), fill_value=0.0).astype(float)
    if demand.isna().any().any():
        raise ValueError("demand.csv contains missing values")
    if (demand < 0).any().any():
        raise ValueError("demand.csv contains negative values")
    return demand


def _prepare_availability(
    availability: pd.DataFrame, generators: pd.DataFrame, num_hours: int
) -> pd.DataFrame:
    """Validate whatever availability columns were provided.

    Only provided columns are kept (in generator order): units without a
    column are fully available and never materialized, so the frame — and
    the input file — stays small or can be omitted entirely.
    """
    availability = availability.copy().reset_index(drop=True)
    availability.columns = [str(c) for c in availability.columns]
    unknown = sorted(set(availability.columns) - set(generators.index))
    if unknown:
        raise ValueError(
            f"availability_factors.csv has columns that are not known generators: {unknown}"
        )
    if len(availability.columns) == 0:
        return pd.DataFrame(index=pd.RangeIndex(num_hours))
    if len(availability) != num_hours:
        raise ValueError(
            "availability_factors.csv and demand.csv must cover the same hours "
            f"(got {len(availability)} vs {num_hours} rows)"
        )

    ordered = [g for g in generators.index if g in availability.columns]
    availability = availability[ordered].astype(float)
    if availability.isna().any().any():
        raise ValueError("availability_factors.csv contains missing values")
    if ((availability < -1e-9) | (availability > 1 + 1e-9)).any().any():
        raise ValueError("availability factors must be within [0, 1]")
    return availability.clip(lower=0.0, upper=1.0)
