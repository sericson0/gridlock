"""Extract solved models into pandas frames, cost summaries and output CSVs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pyomo.environ as pyo

from .data import SystemData

if TYPE_CHECKING:
    from .runner import RunResults


def extract_window(
    model: pyo.ConcreteModel,
    system: SystemData,
    kept_hours: list[int],
    duals: dict | None = None,
) -> dict[str, pd.DataFrame]:
    """Pull variable values for the kept (non-lookahead) hours of a window."""

    def frame(var, units) -> pd.DataFrame:
        data = {name: [var[name, t].value for t in kept_hours] for name in units}
        return pd.DataFrame(data, index=kept_hours, dtype=float)

    uc_units = list(model.G_UC)
    frames = {
        "dispatch": frame(model.p, list(model.G)),
        "commitment": frame(model.u, uc_units),
        "startup": frame(model.v, uc_units),
        "shutdown": frame(model.w, uc_units),
        "storage_charge": frame(model.charge, list(model.S)),
        "storage_discharge": frame(model.discharge, list(model.S)),
        "storage_soc": frame(model.soc, list(model.S)),
        "shed": frame(model.shed, list(model.N)),
    }

    flow_data = {
        line: [
            model.flow_fwd[line, t].value - model.flow_rev[line, t].value
            for t in kept_hours
        ]
        for line in model.L
    }
    frames["flows"] = pd.DataFrame(flow_data, index=kept_hours, dtype=float)

    if duals is not None:
        price_data = {
            n: [duals.get(model.load_balance[n, t]) for t in kept_hours]
            for n in model.N
        }
        frames["prices"] = pd.DataFrame(price_data, index=kept_hours, dtype=float)
    return frames


def compute_cost_summary(system: SystemData, results: "RunResults") -> pd.Series:
    """Recompute cost components from the dispatch frames.

    This is the authoritative cost accounting: in rolling-horizon mode the
    per-window model objectives overlap (lookahead hours are re-solved), so
    summing them would double count. Summing over the realized dispatch
    counts every hour exactly once.
    """
    gens = system.generators
    fuel_rate = gens["fuel_cost_per_mmbtu"] * gens["heat_rate_slope_mmbtu_per_mwh"]

    dispatch = results.dispatch
    uc_columns = results.commitment.columns

    costs = pd.Series(dtype=float)
    costs["fuel"] = (
        (dispatch * fuel_rate).to_numpy().sum()
        + (results.commitment * gens.loc[uc_columns, "no_load_cost"]).to_numpy().sum()
    )
    costs["vom"] = (dispatch * gens["vom_cost_per_mwh"]).to_numpy().sum()
    costs["startup"] = (
        (results.startup * gens.loc[uc_columns, "startup_cost"]).to_numpy().sum()
    )
    costs["shutdown"] = (
        (results.shutdown * gens.loc[uc_columns, "shutdown_cost"]).to_numpy().sum()
    )
    costs["unserved_energy"] = results.shed.to_numpy().sum() * results.config.voll
    costs["total"] = costs.sum()
    return costs.rename("cost_usd")


def generation_by_technology(system: SystemData, results: "RunResults") -> pd.Series:
    """Total MWh generated per technology over the run."""
    tech = system.generators["technology"]
    return (
        results.dispatch.sum()
        .groupby(tech)
        .sum()
        .sort_values(ascending=False)
        .rename("generation_mwh")
    )


def write_results(results: "RunResults", output_dir: str | Path, system: SystemData) -> Path:
    """Write all result frames and summaries as CSVs under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = {
        "dispatch": results.dispatch,
        "commitment": results.commitment,
        "startup": results.startup,
        "shutdown": results.shutdown,
        "storage_charge": results.storage_charge,
        "storage_discharge": results.storage_discharge,
        "storage_soc": results.storage_soc,
        "flows": results.flows,
        "shed": results.shed,
    }
    if results.prices is not None:
        frames["prices"] = results.prices

    for name, frame in frames.items():
        frame.rename_axis("hour").to_csv(output_dir / f"{name}.csv", float_format="%.4f")

    results.cost_summary.rename_axis("component").to_csv(output_dir / "cost_summary.csv")
    generation_by_technology(system, results).rename_axis("technology").to_csv(
        output_dir / "generation_by_technology.csv"
    )
    results.window_stats.to_csv(output_dir / "window_stats.csv", index=False)
    if results.component_stats is not None:
        results.component_stats.to_csv(output_dir / "component_stats.csv")
    return output_dir
