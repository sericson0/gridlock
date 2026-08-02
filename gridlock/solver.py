"""Thin wrapper around the Pyomo appsi HiGHS interface.

All HiGHS options are passed through ``SolverSettings`` (see config.py), so
solver experiments only touch configuration, never model code.

Every solve also collects :class:`~gridlock.profiling.SolveMetrics` —
problem size, iteration/node counts and HiGHS's own run time — which is
nearly free. With ``profile=True`` the HiGHS log is additionally captured
to a temporary file and parsed for presolve reductions, phase timings and
coefficient ranges.

:class:`HighsSession` keeps one appsi interface bound to one model so the
Pyomo -> HiGHS translation (tens of seconds at annual scale) is paid once;
re-solving the same model with different options or a warm start is then
pure solver time. :func:`solve_model` remains the one-shot convenience.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

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


class HighsSession:
    """A model bound to a persistent appsi HiGHS interface.

    The first :meth:`solve` translates the Pyomo model into HiGHS; later
    solves reuse the translated model (appsi only pushes incremental
    updates), so repeated solves of one model — option sweeps, warm-start
    experiments — skip the translation cost entirely.

    HiGHS options are reset to defaults before each solve, so settings from
    one solve never leak into the next.

    Two things about a *reused* solver instance that silently corrupt
    experiments, both handled here:

    - HiGHS's ``getRunTime()`` is cumulative over the instance's life, not
      per solve. :attr:`SolveMetrics.highs_run_seconds` is reported as the
      difference from the previous solve so it always means "this solve".
    - HiGHS keeps the previous solve's basis and incumbent, so a second
      solve of the same model is dramatically faster than a cold one.
      That is the point for warm-start work and *fatal* for option
      sweeps, where every configuration must start from the same state.
      Pass ``cold=True`` to clear the retained solution first.
    """

    def __init__(self, model: pyo.ConcreteModel):
        self.model = model
        self._opt: Highs | None = None
        self._run_seconds_consumed = 0.0

    def solve(
        self,
        settings: SolverSettings | None = None,
        want_duals: bool = False,
        profile: bool = False,
        warmstart: bool = False,
        cold: bool = False,
    ) -> tuple[SolveInfo, dict | None]:
        """Solve the session's model with appsi HiGHS.

        Returns the solve info and, when ``want_duals`` is True and duals
        are available (LP only), a dict mapping constraint data objects to
        dual values. Raises RuntimeError if no feasible solution was found.

        ``profile=True`` captures and parses the HiGHS log (see module
        docstring); a user-supplied ``log_file`` HiGHS option is respected
        and the file is then left in place.

        ``warmstart=True`` passes the current variable values to HiGHS as a
        MIP start. appsi sends a *full* solution vector in which any
        variable whose ``.value`` is None becomes 0.0, so the caller must
        give every variable a value for the start to mean what it thinks
        it means. After a successful solve the model holds the solution,
        so a re-solve with ``warmstart=True`` starts from it automatically.

        ``cold=True`` discards the basis and incumbent HiGHS kept from the
        previous solve, so this solve starts from the same state a fresh
        process would. Required for honest option or seed sweeps.
        """
        settings = settings or SolverSettings()
        options = settings.resolved_options()

        temp_log_path = None
        log_path = options.get("log_file")
        if profile and not log_path:
            fd, temp_log_path = tempfile.mkstemp(suffix=".highs.log", prefix="gridlock_")
            os.close(fd)
            options["log_file"] = log_path = temp_log_path

        if self._opt is None:
            self._opt = Highs()
        opt = self._opt
        self._reset_options()
        if cold:
            self._clear_solver_state()
        opt.config.stream_solver = settings.stream_solver
        opt.config.warmstart = warmstart
        # Load the solution manually so a time-limited-but-feasible solve
        # doesn't raise inside appsi.
        opt.config.load_solution = False
        opt.highs_options = options

        start = time.perf_counter()
        results = opt.solve(self.model)
        solve_seconds = time.perf_counter() - start

        termination = results.termination_condition.name
        objective = results.best_feasible_objective
        bound = results.best_objective_bound

        log_text = _read_log(opt, log_path, cleanup=temp_log_path is not None)
        metrics = collect_highs_metrics(opt, log_text)
        self._charge_run_time(metrics)

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

    def write_model(self, path: str | Path) -> Path:
        """Write the translated HiGHS model to ``path`` (.mps or .lp).

        Translates first if no solve has happened yet. The written file
        lets pure solver experiments (option sweeps on a fixed model)
        bypass Pyomo entirely: load it straight into highspy.
        """
        path = Path(path)
        if self._opt is None:
            self._opt = Highs()
        if (
            getattr(self._opt, "_solver_model", None) is None
            or getattr(self._opt, "_model", None) is not self.model
        ):
            self._opt.set_instance(self.model)
        status = self._opt._solver_model.writeModel(str(path))
        # kWarning just means generated row/column names (appsi passes none).
        if int(status) < 0:
            raise RuntimeError(f"HiGHS could not write model to {path} (status {status})")
        return path

    def _reset_options(self) -> None:
        """Restore HiGHS option defaults so per-solve options don't leak."""
        highs = getattr(self._opt, "_solver_model", None)
        if highs is None:
            return
        try:
            highs.resetOptions()
        except Exception:
            pass

    def _clear_solver_state(self) -> None:
        """Drop the retained basis/incumbent, leaving the model in place."""
        highs = getattr(self._opt, "_solver_model", None)
        if highs is None:
            return
        try:
            highs.clearSolver()
        except Exception:
            pass

    def _charge_run_time(self, metrics: SolveMetrics) -> None:
        """Turn HiGHS's cumulative clock into this solve's own run time."""
        total = metrics.highs_run_seconds
        if total is None:
            return
        elapsed = total - self._run_seconds_consumed
        # A negative delta means HiGHS restarted its clock; trust the raw
        # reading in that case rather than reporting nonsense.
        metrics.highs_run_seconds = elapsed if elapsed >= 0 else total
        self._run_seconds_consumed = total


def solve_model(
    model: pyo.ConcreteModel,
    settings: SolverSettings | None = None,
    want_duals: bool = False,
    profile: bool = False,
    warmstart: bool = False,
) -> tuple[SolveInfo, dict | None]:
    """One-shot solve of ``model`` with a throwaway :class:`HighsSession`."""
    return HighsSession(model).solve(
        settings, want_duals=want_duals, profile=profile, warmstart=warmstart
    )


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
