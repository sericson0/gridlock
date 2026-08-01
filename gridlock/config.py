"""Run configuration objects.

Everything that controls *how* a run is performed (as opposed to the input
system data) lives here, so a run is fully described by a
(SystemData, RunConfig) pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SolverSettings:
    """Settings passed to the appsi HiGHS interface.

    ``mip_gap``, ``time_limit`` and ``threads`` are conveniences for the
    most common HiGHS options; ``highs_options`` accepts any raw HiGHS
    option by name (e.g. ``{"presolve": "off", "solver": "ipm"}``) and
    takes precedence over the conveniences if the same option appears in
    both.
    """

    mip_gap: float | None = None
    time_limit: float | None = None
    threads: int | None = None
    stream_solver: bool = False
    highs_options: dict = field(default_factory=dict)

    def resolved_options(self) -> dict:
        """Merge convenience settings and raw options into one HiGHS option dict."""
        options: dict = {}
        if self.mip_gap is not None:
            options["mip_rel_gap"] = float(self.mip_gap)
        if self.time_limit is not None:
            options["time_limit"] = float(self.time_limit)
        if self.threads is not None:
            options["threads"] = int(self.threads)
            # HiGHS only uses multiple threads when parallel is on.
            options.setdefault("parallel", "on")
        options.update(self.highs_options)
        return options


@dataclass
class RunConfig:
    """Configuration for a production cost run.

    unit_commitment
        True: commitment variables are binary (MIP). False: the same model
        is built but commitment variables are relaxed to [0, 1] (LP).
    num_hours
        Truncate the input horizon to its first ``num_hours`` hours
        (None = use every hour in the input data). Handy for quick tests.
    cyclic
        Monolithic mode only. True (default): time-linked constraints wrap
        the first hour back to the last (the hour before hour 0 is the
        horizon's final hour). False: commitment logic, min up/down times
        and ramps leave the first hours unconstrained instead — the same
        free-initial-state treatment a rolling run gives its first window.
        Storage SOC stays cyclic either way, so the horizon cannot end
        with drained storage. Relaxing the wrap can only lower cost;
        the point is to measure what the wrap rows cost the solver.
    warmstart_window_hours
        Monolithic mode only. If set, first solve the horizon in rolling
        windows of this many hours (with ``lookahead_hours`` of lookahead),
        then hand that solution to HiGHS as a MIP start for the monolithic
        solve. Attacks the incumbent problem: at long horizons HiGHS's own
        heuristics struggle to find a good feasible solution.
    window_hours
        None solves the whole horizon as one model (monolithic). An integer
        switches to rolling-horizon mode: the horizon is split into
        sequential windows of this many hours, each solved with
        ``lookahead_hours`` of extra foresight, carrying commitment, output
        and storage state into the next window.
    lookahead_hours
        Extra hours appended to each rolling window to mitigate end-of-window
        effects. Lookahead results are discarded. Ignored in monolithic mode.
    initial_soc_fraction
        Rolling mode only: state of charge (as a fraction of energy
        capacity) at the start of the first window, and the minimum state of
        charge at the end of the last window. In monolithic mode storage is
        cyclic and the starting state of charge is a free decision variable.
    voll
        Value of lost load ($/MWh): penalty applied to unserved energy so the
        model stays feasible under scarcity.
    profile
        Capture and parse the HiGHS log of every solve, adding presolve
        reductions, solve-phase timings and coefficient ranges to
        ``RunResults.window_stats``. Basic metrics (problem size,
        iterations, nodes, HiGHS run time) are collected regardless.
    """

    unit_commitment: bool = True
    num_hours: int | None = None
    cyclic: bool = True
    warmstart_window_hours: int | None = None
    window_hours: int | None = None
    lookahead_hours: int = 24
    initial_soc_fraction: float = 0.5
    voll: float = 10_000.0
    profile: bool = False
    solver: SolverSettings = field(default_factory=SolverSettings)

    def validate(self) -> None:
        if self.num_hours is not None and self.num_hours < 1:
            raise ValueError("num_hours must be a positive integer")
        if self.window_hours is not None and self.window_hours < 1:
            raise ValueError("window_hours must be a positive integer")
        if self.warmstart_window_hours is not None:
            if self.warmstart_window_hours < 1:
                raise ValueError("warmstart_window_hours must be a positive integer")
            if self.window_hours is not None:
                raise ValueError(
                    "warmstart_window_hours only applies to monolithic runs "
                    "(window_hours must be None)"
                )
        if self.lookahead_hours < 0:
            raise ValueError("lookahead_hours must be non-negative")
        if not 0.0 <= self.initial_soc_fraction <= 1.0:
            raise ValueError("initial_soc_fraction must be in [0, 1]")
