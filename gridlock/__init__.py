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
    get_suite,
    run_suite,
    scale_suite,
    summarize_records,
)
from .config import RunConfig, SolverSettings
from .data import SystemData, build_system, load_system
from .model import InitialState, build_model
from .profiling import (
    MemorySampler,
    SolveMetrics,
    capture_environment,
    model_stats,
    parse_highs_log,
)
from .runner import RunResults, run
from .solver import SolveInfo, solve_model

__version__ = "0.1.0"

__all__ = [
    "RunConfig",
    "SolverSettings",
    "SystemData",
    "build_system",
    "load_system",
    "InitialState",
    "build_model",
    "RunResults",
    "run",
    "SolveInfo",
    "solve_model",
    "SolveMetrics",
    "MemorySampler",
    "model_stats",
    "parse_highs_log",
    "capture_environment",
    "BenchCase",
    "default_suite",
    "scale_suite",
    "get_suite",
    "run_suite",
    "summarize_records",
    "compare_records",
    "compare_files",
    "__version__",
]
