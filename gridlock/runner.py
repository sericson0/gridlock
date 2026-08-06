"""Run driver: monolithic or rolling-horizon solves over the input horizon.

Monolithic mode builds one cyclic model spanning every hour: the hour
before hour 0 is the last hour for commitment, ramp and storage
constraints alike. Rolling mode splits the horizon into
sequential windows (plus lookahead), carrying commitment state, output,
min up/down obligations and storage state of charge between windows.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import pandas as pd
import pyomo.environ as pyo

from .config import RunConfig
from .data import SystemData, cluster_identical_units
from .heuristics import (
    CommitmentGuess,
    apply_fixing,
    build_guess,
    complete_solution,
    match_fraction,
)
from .model import InitialState, build_model
from .profiling import model_stats
from .results import compute_cost_summary, extract_window
from .solver import HighsSession, SolveInfo, solve_model


@dataclass
class RunResults:
    """Hourly result frames (rows = hours) plus solve statistics.

    ``window_stats`` has one row per solved window with build/solve/extract
    timings, termination, objective/bound and every
    :class:`~gridlock.profiling.SolveMetrics` field (problem size,
    iterations, nodes, presolve detail in profile mode).
    ``component_stats`` is a per-component size census of the first
    window's Pyomo model (see :func:`gridlock.profiling.model_stats`).
    ``system`` is the system as actually modeled — the same object that
    was passed in, unless ``config.cluster_units`` pooled its generators,
    in which case result columns carry the *cluster* names and this is the
    frame that describes them.
    ``guess`` is the :class:`~gridlock.heuristics.CommitmentGuess` that
    seeded the solve, kept because it is the expensive part of a heuristic
    run and the only record of what the heuristic *predicted* (as opposed
    to the ``heuristic_match_pct`` summary of how well it did). None when
    no heuristic ran.
    """

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
    component_stats: pd.DataFrame | None = None
    system: SystemData | None = None
    guess: CommitmentGuess | None = None

    @property
    def total_cost(self) -> float:
        return float(self.cost_summary["total"])

    @property
    def total_build_seconds(self) -> float:
        return float(self.window_stats["build_seconds"].sum())

    @property
    def total_solve_seconds(self) -> float:
        return float(self.window_stats["solve_seconds"].sum())

    @property
    def total_highs_seconds(self) -> float:
        """Pure HiGHS run time (excludes Pyomo -> HiGHS translation)."""
        return float(self.window_stats["highs_run_seconds"].fillna(0.0).sum())

    @property
    def total_extract_seconds(self) -> float:
        return float(self.window_stats["extract_seconds"].sum())


def run(
    system: SystemData,
    config: RunConfig | None = None,
    initial_state: InitialState | None = None,
) -> RunResults:
    """Solve the production cost problem described by ``system`` and ``config``.

    ``initial_state`` pins the hour before hour 0 for a *monolithic* run:
    commitment, output, min up/down obligations and storage SOC carried in
    from a previously solved horizon. It is the sequential alternative to
    the cyclic wrap, so it requires ``config.cyclic=False`` — the wrap
    defines that same hour as the horizon's last, and the two cannot both
    hold. Rolling runs derive their own per-window state and ignore this.
    """
    config = config or RunConfig()
    config.validate()
    if initial_state is not None:
        if config.window_hours is not None:
            raise ValueError(
                "initial_state applies to monolithic runs; a rolling run "
                "derives each window's state from the previous window"
            )
        if config.cyclic:
            raise ValueError(
                "initial_state requires cyclic=False: a carried-in initial "
                "hour and a wrap to the final hour are alternative "
                "definitions of the hour before hour 0"
            )

    if config.cluster_units:
        system = cluster_identical_units(system)

    total_hours = system.num_hours
    if config.num_hours is not None:
        total_hours = min(total_hours, config.num_hours)

    windows = _make_windows(total_hours, config)
    monolithic = config.window_hours is None
    # Duals (nodal prices) are only meaningful without binary variables.
    want_duals = not config.unit_commitment or not system.generators["needs_commitment"].any()

    # Rolling pre-pass whose solution seeds the monolithic solve as a MIP
    # start. Only meaningful when the main solve is a MIP.
    warmstart_results: RunResults | None = None
    warmstart_seconds: float | None = None
    if (
        monolithic
        and config.warmstart_window_hours is not None
        and config.unit_commitment
    ):
        pre_config = copy.deepcopy(config)
        pre_config.warmstart_window_hours = None
        pre_config.window_hours = config.warmstart_window_hours
        pre_config.cluster_units = False  # `system` is already clustered
        pre_start = time.perf_counter()
        warmstart_results = run(system, pre_config)
        warmstart_seconds = time.perf_counter() - pre_start

    # Domain-heuristic commitment guess (see gridlock/heuristics.py):
    # built once here, completed against the model inside the loop.
    guess = None
    if monolithic and config.heuristic is not None and config.unit_commitment:
        # The oracle has to start where the model starts. build_model reads
        # cyclic-ness from the initial state, so a guess built without it
        # solves a different problem and can contradict carried min up/down
        # obligations -- which surfaces as an infeasible completion, not as
        # a merely poor guess.
        guess_initial = initial_state or _initial_state_for_window(
            system, config, 0, len(windows), None, monolithic
        )
        guess_start = time.perf_counter()
        guess = build_guess(system, config, list(range(total_hours)), guess_initial)
        guess_seconds = time.perf_counter() - guess_start

    collected: list[dict[str, pd.DataFrame]] = []
    stats_rows: list[dict] = []
    state: InitialState | None = None
    component_stats: pd.DataFrame | None = None

    for index, (hours, kept_hours) in enumerate(windows):
        initial = _initial_state_for_window(
            system, config, index, len(windows), state, monolithic
        )
        if monolithic and initial_state is not None:
            initial = initial_state

        build_start = time.perf_counter()
        model = build_model(system, config, hours, initial)
        build_seconds = time.perf_counter() - build_start

        if index == 0:
            component_stats = model_stats(model)

        warm = warmstart_results is not None
        if warm:
            _seed_model_from_results(model, warmstart_results)

        heuristic_seconds = heuristic_fallback = heuristic_fixed = None
        completion_failed = False
        session = None
        if guess is not None:
            # One session serves completion and main solve, so the model is
            # translated once and the unfixing arrives as an incremental
            # update — at annual scale translation rivals the LP solve.
            session = HighsSession(model)
            completion_start = time.perf_counter()
            try:
                _, heuristic_fallback = complete_solution(model, guess, session=session)
                heuristic_fixed = apply_fixing(model, guess, config.heuristic_fixing)
            except RuntimeError:
                # An uncompletable guess is a lost warm start, not a lost
                # run: solve cold rather than failing the whole horizon.
                print(
                    f"warning: heuristic '{guess.name}' could not be completed "
                    "into a feasible solution; solving without a warm start"
                )
                for var in model.u.values():
                    var.unfix()
                # Keep the guess itself -- it is still the record of what the
                # heuristic predicted -- and drop only the warm start.
                completion_failed = True
                session = None
                heuristic_fixed = 0
            heuristic_seconds = (
                guess_seconds + time.perf_counter() - completion_start
            )
            if heuristic_fallback:
                print(
                    f"note: heuristic '{guess.name}' over-committed; completion "
                    "relaxed minimum-output rows (warm start may be rejected)"
                )

        if session is not None:
            try:
                info, duals = session.solve(
                    config.solver,
                    want_duals=want_duals,
                    profile=config.profile,
                    warmstart=True,
                )
            except RuntimeError:
                # Fixing can over-constrain the model into infeasibility.
                # Losing the run to a heuristic would be worse than losing
                # the speedup, so unfix and solve as a plain warm start.
                if not heuristic_fixed:
                    raise
                print(
                    f"warning: heuristic '{guess.name}' fixing left no feasible "
                    "solution; retrying as warm start only"
                )
                for var in model.u.values():
                    var.unfix()
                heuristic_fixed = 0
                info, duals = session.solve(
                    config.solver,
                    want_duals=want_duals,
                    profile=config.profile,
                    warmstart=True,
                )
        else:
            info, duals = solve_model(
                model,
                config.solver,
                want_duals=want_duals,
                profile=config.profile,
                warmstart=warm,
            )
        _warn_if_not_optimal(info, index)

        extract_start = time.perf_counter()
        frames = extract_window(model, system, kept_hours, duals)
        extract_seconds = time.perf_counter() - extract_start
        collected.append(frames)
        row = {
            "window": index,
            "first_hour": kept_hours[0],
            "hours_kept": len(kept_hours),
            "hours_modeled": len(hours),
            "build_seconds": build_seconds,
            "solve_seconds": info.solve_seconds,
            "translate_seconds": info.translate_seconds,
            "extract_seconds": extract_seconds,
            "warmstart_seconds": warmstart_seconds if warm else None,
            "heuristic_seconds": heuristic_seconds,
            "heuristic_fixed_vars": heuristic_fixed,
            "heuristic_fallback": heuristic_fallback,
            "heuristic_completion_failed": completion_failed,
            "heuristic_match_pct": (
                None if guess is None else match_fraction(model, guess)
            ),
            "termination": info.termination,
            "objective": info.objective,
            "bound": info.bound,
        }
        row.update(info.metrics.as_row())
        stats_rows.append(row)

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
        component_stats=component_stats,
        system=system,
        guess=guess,
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
        if config.cyclic:
            return None  # every time link wraps to the last hour
        # An empty InitialState relaxes the wrap: commitment logic, min
        # up/down and ramps leave the first hours free (exactly how a
        # rolling run treats its first window) while storage SOC stays
        # cyclic because no starting SOC is supplied.
        return InitialState()

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


def _seed_model_from_results(model: pyo.ConcreteModel, results: RunResults) -> None:
    """Give *every* variable a value from a prior run's hourly frames.

    appsi's warm start sends HiGHS a full solution vector in which any
    variable still at None becomes 0.0 — a silent, usually infeasible
    "start" — so completeness here is not optional. Commitment is rounded
    to a whole count; continuous values are clamped to their sign
    conventions so solver noise (-1e-9) can't leak in. Net line flows are
    split into the forward/reverse pair (with losses, an optimal solution
    never uses both directions at once, so the split is exact).
    """

    def seed(var, frame, transform):
        for name in frame.columns:
            values = frame[name]
            for t in values.index:
                var[name, t].set_value(transform(float(values.at[t])), skip_validation=True)

    def clamp(x):
        return max(0.0, x)

    def unit_interval(x):
        return min(1.0, max(0.0, x))

    def to_count(x):
        return float(round(x))

    seed(model.p, results.dispatch, clamp)
    seed(model.u, results.commitment, to_count)
    seed(model.v, results.startup, unit_interval)
    seed(model.w, results.shutdown, unit_interval)
    seed(model.charge, results.storage_charge, clamp)
    seed(model.discharge, results.storage_discharge, clamp)
    seed(model.soc, results.storage_soc, clamp)
    seed(model.shed, results.shed, clamp)

    for line in model.L:
        net = results.flows[line]
        for t in net.index:
            x = float(net.at[t])
            model.flow_fwd[line, t].set_value(max(0.0, x), skip_validation=True)
            model.flow_rev[line, t].set_value(max(0.0, -x), skip_validation=True)


def _extract_state(
    system: SystemData,
    frames: dict[str, pd.DataFrame],
    previous: InitialState | None,
) -> InitialState:
    """Build the InitialState for the next window from this window's kept hours."""
    # Round rather than threshold: a cluster carries a committed *count*,
    # and LP-relaxed runs carry fractional values that must land on one.
    on_off = frames["commitment"].round().astype(int)

    state_hours = {}
    for g in on_off.columns:
        series = on_off[g].to_list()
        run_length = _trailing_run_length(series)
        is_on = series[-1] > 0
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
