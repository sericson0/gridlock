"""gridlock: a small, modular electricity production cost model.

Built on Pyomo with the appsi HiGHS solver interface. Designed as a
readable testbed for experimenting with model formulations and solver
settings.
"""

from .bench import (
    BenchCase,
    compare_files,
    compare_records,
    default_suite,
    formulation_suite,
    get_suite,
    run_suite,
    scale_suite,
    summarize_records,
)
from .config import RunConfig, SolverSettings
from .data import SystemData, build_system, cluster_identical_units, load_system
from .model import InitialState, build_model
from .profiling import (
    MemorySampler,
    SolveMetrics,
    capture_environment,
    model_stats,
    parse_highs_log,
    parse_mip_progress,
)
from .heuristics import (
    CommitmentGuess,
    build_guess,
    lp_relaxation_guess,
    merit_order,
    net_load_mw,
    priority_list_guess,
    similar_days_guess,
)
from .runner import RunResults, run
from .solver import HighsSession, SolveInfo, solve_model

__version__ = "0.1.0"

__all__ = [
    "RunConfig",
    "SolverSettings",
    "SystemData",
    "build_system",
    "load_system",
    "cluster_identical_units",
    "InitialState",
    "build_model",
    "CommitmentGuess",
    "build_guess",
    "net_load_mw",
    "merit_order",
    "priority_list_guess",
    "similar_days_guess",
    "lp_relaxation_guess",
    "RunResults",
    "run",
    "SolveInfo",
    "solve_model",
    "HighsSession",
    "SolveMetrics",
    "MemorySampler",
    "model_stats",
    "parse_highs_log",
    "parse_mip_progress",
    "capture_environment",
    "BenchCase",
    "default_suite",
    "scale_suite",
    "formulation_suite",
    "get_suite",
    "run_suite",
    "summarize_records",
    "compare_records",
    "compare_files",
    "__version__",
]
