"""Thin wrapper around the Pyomo appsi HiGHS interface.

All HiGHS options are passed through ``SolverSettings`` (see config.py), so
solver experiments only touch configuration, never model code.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pyomo.environ as pyo
from pyomo.contrib.appsi.solvers.highs import Highs

from .config import SolverSettings


@dataclass
class SolveInfo:
    """What happened during one solve."""

    termination: str
    objective: float | None
    bound: float | None
    solve_seconds: float

    @property
    def gap(self) -> float | None:
        if self.objective is None or self.bound is None or self.objective == 0:
            return None
        return abs(self.objective - self.bound) / abs(self.objective)


def solve_model(
    model: pyo.ConcreteModel,
    settings: SolverSettings | None = None,
    want_duals: bool = False,
) -> tuple[SolveInfo, dict | None]:
    """Solve ``model`` with appsi HiGHS.

    Returns the solve info and, when ``want_duals`` is True and duals are
    available (LP only), a dict mapping constraint data objects to dual
    values. Raises RuntimeError if no feasible solution was found.
    """
    settings = settings or SolverSettings()

    opt = Highs()
    opt.config.stream_solver = settings.stream_solver
    # Load the solution manually so a time-limited-but-feasible solve
    # doesn't raise inside appsi.
    opt.config.load_solution = False
    opt.highs_options = settings.resolved_options()

    start = time.perf_counter()
    results = opt.solve(model)
    solve_seconds = time.perf_counter() - start

    termination = results.termination_condition.name
    objective = results.best_feasible_objective
    bound = results.best_objective_bound

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
    )
    return info, duals
