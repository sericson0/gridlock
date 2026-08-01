"""Thin wrapper around the Pyomo appsi HiGHS interface.

All HiGHS options are passed through ``SolverSettings`` (see config.py), so
solver experiments only touch configuration, never model code.

Every solve also collects :class:`~gridlock.profiling.SolveMetrics` —
problem size, iteration/node counts and HiGHS's own run time — which is
nearly free. With ``profile=True`` the HiGHS log is additionally captured
to a temporary file and parsed for presolve reductions, phase timings and
coefficient ranges.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field

import pyomo.environ as pyo
from pyomo.contrib.appsi.solvers.highs import Highs

from .config import SolverSettings
from .profiling import SolveMetrics, collect_highs_metrics


@dataclass
class SolveInfo:
    """What happened during one solve."""

    termination: str
    objective: float | None
    bound: float | None
    solve_seconds: float
    metrics: SolveMetrics = field(default_factory=SolveMetrics)

    @property
    def gap(self) -> float | None:
        if self.objective is None or self.bound is None or self.objective == 0:
            return None
        return abs(self.objective - self.bound) / abs(self.objective)

    @property
    def translate_seconds(self) -> float | None:
        """Pyomo -> HiGHS translation and interface overhead inside the solve call."""
        if self.metrics.highs_run_seconds is None:
            return None
        return max(0.0, self.solve_seconds - self.metrics.highs_run_seconds)


def solve_model(
    model: pyo.ConcreteModel,
    settings: SolverSettings | None = None,
    want_duals: bool = False,
    profile: bool = False,
) -> tuple[SolveInfo, dict | None]:
    """Solve ``model`` with appsi HiGHS.

    Returns the solve info and, when ``want_duals`` is True and duals are
    available (LP only), a dict mapping constraint data objects to dual
    values. Raises RuntimeError if no feasible solution was found.

    ``profile=True`` captures and parses the HiGHS log (see module
    docstring); a user-supplied ``log_file`` HiGHS option is respected and
    the file is then left in place.
    """
    settings = settings or SolverSettings()
    options = settings.resolved_options()

    temp_log_path = None
    log_path = options.get("log_file")
    if profile and not log_path:
        fd, temp_log_path = tempfile.mkstemp(suffix=".highs.log", prefix="gridlock_")
        os.close(fd)
        options["log_file"] = log_path = temp_log_path

    opt = Highs()
    opt.config.stream_solver = settings.stream_solver
    # Load the solution manually so a time-limited-but-feasible solve
    # doesn't raise inside appsi.
    opt.config.load_solution = False
    opt.highs_options = options

    start = time.perf_counter()
    results = opt.solve(model)
    solve_seconds = time.perf_counter() - start

    termination = results.termination_condition.name
    objective = results.best_feasible_objective
    bound = results.best_objective_bound

    log_text = _read_log(opt, log_path, cleanup=temp_log_path is not None)
    metrics = collect_highs_metrics(opt, log_text)

    if objective is None:
        raise RuntimeError(
            f"solver found no feasible solution (termination: {termination})"
        )
    opt.load_vars()

    duals = None
    if want_duals:
        try:
            duals = opt.get_duals()
        except Exception:
            duals = None

    info = SolveInfo(
        termination=termination,
        objective=objective,
        bound=bound,
        solve_seconds=solve_seconds,
        metrics=metrics,
    )
    return info, duals


def _read_log(opt: Highs, log_path: str | None, cleanup: bool) -> str | None:
    if not log_path:
        return None
    try:
        with open(log_path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return None
    if cleanup:
        try:
            # HiGHS holds the log file open; pointing it elsewhere releases
            # the handle so the temp file can be removed on Windows.
            opt._solver_model.setOptionValue("log_file", "")
        except Exception:
            pass
        try:
            os.unlink(log_path)
        except OSError:
            pass
    return text
