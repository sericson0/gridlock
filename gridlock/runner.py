"""Run driver: monolithic or rolling-horizon solves over the input horizon.

Monolithic mode builds one model spanning every hour with cyclic storage
and a free initial commitment state. Rolling mode splits the horizon into
sequential windows (plus lookahead), carrying commitment state, output,
min up/down obligations and storage state of charge between windows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from .config import RunConfig
from .data import SystemData
from .model import InitialState, build_model
from .results import compute_cost_summary, extract_window
from .solver import SolveInfo, solve_model


@dataclass
class RunResults:
    """Hourly result frames (rows = hours) plus solve statistics."""

    config: RunConfig
    dispatch: pd.DataFrame
    commitment: pd.DataFrame
    startup: pd.DataFrame
    shutdown: pd.DataFrame
    storage_charge: pd.DataFrame
    storage_discharge: pd.DataFrame
    storage_soc: pd.DataFrame
    flows: pd.DataFrame
    shed: pd.DataFrame
    prices: pd.DataFrame | None
    window_stats: pd.DataFrame
    objective_value: float | None  # model objective (monolithic runs only)
    cost_summary: pd.Series = field(default=None)

    @property
    def total_cost(self) -> float:
        return float(self.cost_summary["total"])

    @property
    def total_build_seconds(self) -> float:
        return float(self.window_stats["build_seconds"].sum())

    @property
    def total_solve_seconds(self) -> float:
        return float(self.window_stats["solve_seconds"].sum())


def run(system: SystemData, config: RunConfig | None = None) -> RunResults:
    """Solve the production cost problem described by ``system`` and ``config``."""
    config = config or RunConfig()
    config.validate()

    total_hours = system.num_hours
    if config.num_hours is not None:
        total_hours = min(total_hours, config.num_hours)

    windows = _make_windows(total_hours, config)
    monolithic = config.window_hours is None
    # Duals (nodal prices) are only meaningful without binary variables.
    want_duals = not config.unit_commitment or not system.generators["needs_commitment"].any()

    collected: list[dict[str, pd.DataFrame]] = []
    stats_rows: list[dict] = []
    state: InitialState | None = None

    for index, (hours, kept_hours) in enumerate(windows):
        initial = _initial_state_for_window(
            system, config, index, len(windows), state, monolithic
        )

        build_start = time.perf_counter()
        model = build_model(system, config, hours, initial)
        build_seconds = time.perf_counter() - build_start

        info, duals = solve_model(model, config.solver, want_duals=want_duals)
        _warn_if_not_optimal(info, index)

        frames = extract_window(model, system, kept_hours, duals)
        collected.append(frames)
        stats_rows.append(
            {
                "window": index,
                "first_hour": kept_hours[0],
                "hours_kept": len(kept_hours),
                "hours_modeled": len(hours),
                "build_seconds": build_seconds,
                "solve_seconds": info.solve_seconds,
                "termination": info.termination,
                "objective": info.objective,
                "bound": info.bound,
            }
        )

        if index < len(windows) - 1:
            state = _extract_state(system, frames, state)

    def concat(name: str) -> pd.DataFrame:
        return pd.concat([frames[name] for frames in collected], axis=0)

    prices = concat("prices") if "prices" in collected[0] else None
    results = RunResults(
        config=config,
        dispatch=concat("dispatch"),
        commitment=concat("commitment"),
        startup=concat("startup"),
        shutdown=concat("shutdown"),
        storage_charge=concat("storage_charge"),
        storage_discharge=concat("storage_discharge"),
        storage_soc=concat("storage_soc"),
        flows=concat("flows"),
        shed=concat("shed"),
        prices=prices,
        window_stats=pd.DataFrame(stats_rows),
        objective_value=stats_rows[0]["objective"] if monolithic else None,
    )
    results.cost_summary = compute_cost_summary(system, results)
    return results


def _make_windows(total_hours: int, config: RunConfig) -> list[tuple[list[int], list[int]]]:
    """Return (modeled hours, kept hours) pairs covering [0, total_hours)."""
    if config.window_hours is None:
        hours = list(range(total_hours))
        return [(hours, hours)]

    windows = []
    start = 0
    while start < total_hours:
        keep_end = min(start + config.window_hours, total_hours)
        model_end = min(keep_end + config.lookahead_hours, total_hours)
        windows.append((list(range(start, model_end)), list(range(start, keep_end))))
        start = keep_end
    return windows


def _initial_state_for_window(
    system: SystemData,
    config: RunConfig,
    index: int,
    num_windows: int,
    carried: InitialState | None,
    monolithic: bool,
) -> InitialState | None:
    if monolithic:
        return None  # free initial commitment, cyclic storage

    energy = system.storage["energy_mwh"]
    if index == 0:
        state = InitialState(
            soc=(config.initial_soc_fraction * energy).to_dict()
        )
    else:
        state = carried

    if index == num_windows - 1:
        # Don't let the final window drain storage below the year's start.
        state.min_terminal_soc = (config.initial_soc_fraction * energy).to_dict()
    return state


def _extract_state(
    system: SystemData,
    frames: dict[str, pd.DataFrame],
    previous: InitialState | None,
) -> InitialState:
    """Build the InitialState for the next window from this window's kept hours."""
    commitment_frame = frames["commitment"]
    on_off = (commitment_frame >= 0.5).astype(int)  # robust to LP-relaxed values

    state_hours = {}
    for g in on_off.columns:
        series = on_off[g].to_list()
        run_length = _trailing_run_length(series)
        is_on = series[-1] == 1
        # If the whole window shares one state, extend the previous count.
        if run_length == len(series) and previous is not None and previous.state_hours:
            prior = previous.state_hours.get(g, 0)
            if (prior > 0) == is_on and prior != 0:
                run_length += abs(prior)
        state_hours[g] = run_length if is_on else -run_length

    return InitialState(
        commitment={g: float(on_off[g].iloc[-1]) for g in on_off.columns},
        output=frames["dispatch"].iloc[-1].to_dict(),
        state_hours=state_hours,
        soc=frames["storage_soc"].iloc[-1].to_dict(),
    )


def _trailing_run_length(series: list[int]) -> int:
    last = series[-1]
    run = 0
    for value in reversed(series):
        if value != last:
            break
        run += 1
    return run


def _warn_if_not_optimal(info: SolveInfo, window_index: int) -> None:
    if info.termination != "optimal":
        print(
            f"warning: window {window_index} finished with termination "
            f"'{info.termination}' (objective {info.objective}, bound {info.bound})"
        )
