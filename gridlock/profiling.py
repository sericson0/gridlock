"""Profiling instrumentation: solver metrics, model census, run records.

Everything here answers one question: *did a change make the solver's job
easier?* Wall-clock time alone can't answer that — it moves with machine
load, and MIP times move with random seeds. So each solve is described by
a layered set of metrics:

- **Problem size** (rows / columns / nonzeros / integer columns) as HiGHS
  received it, straight from the highspy model.
- **Presolved size** parsed from the HiGHS log: how much of the model
  survives presolve. A formulation change that presolves away is free; one
  that shrinks the presolved model usually pays off.
- **Work counts** (simplex/IPM iterations, branch-and-bound nodes): these
  are far less noisy than seconds and separate "the algorithm did less
  work" from "the machine was faster today".
- **Time split**: pure HiGHS run time versus the Pyomo build and
  appsi translation overhead around it, so model-construction cost and
  solver cost are never conflated.
- **Numerics** (coefficient ranges from the log): wide ranges warn that
  scaling — not formulation — may be the bottleneck.

:func:`model_stats` adds a Pyomo-level census (constraint rows and
variables per component) so size changes can be attributed to a specific
constraint block.

Records of whole runs are stored as JSON lines via :func:`append_jsonl`,
one file per benchmark session, with :func:`capture_environment` metadata
(git commit, package versions, hardware) so numbers from different days
stay interpretable.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import threading
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo


# --------------------------------------------------------------------------
# Per-solve metrics from the HiGHS instance and log
# --------------------------------------------------------------------------


@dataclass
class SolveMetrics:
    """Everything measured about one HiGHS solve beyond termination/objective.

    Fields are None when the quantity does not apply (e.g. MIP node counts
    on an LP) or could not be read. ``presolved_*`` and later fields come
    from parsing the HiGHS log and are only populated when the solve
    captured one (``RunConfig.profile`` / ``solve_model(profile=True)``).
    """

    # Model as handed to HiGHS.
    num_rows: int | None = None
    num_cols: int | None = None
    num_nonzeros: int | None = None
    num_integer_vars: int | None = None

    # Pure solver wall time (HiGHS's own clock). The difference between
    # the solve wall time and this is Pyomo -> HiGHS translation overhead.
    highs_run_seconds: float | None = None

    # Work counts.
    simplex_iterations: int | None = None
    ipm_iterations: int | None = None
    crossover_iterations: int | None = None
    pdlp_iterations: int | None = None
    mip_nodes: int | None = None
    final_mip_gap: float | None = None
    primal_dual_integral: float | None = None

    # From the HiGHS log (profile mode only).
    presolved_rows: int | None = None
    presolved_cols: int | None = None
    presolved_nonzeros: int | None = None
    presolved_binaries: int | None = None
    presolve_seconds: float | None = None
    solve_phase_seconds: float | None = None
    postsolve_seconds: float | None = None

    # Root-loop attribution (MIP log only). The UC MIP solves at the root,
    # so these split root time between the main model and the sub-MIP
    # heuristics, and LP iterations between separation and heuristics —
    # the difference between "the cut loop is slow" and "the heuristics
    # are slow", which have entirely different fixes.
    solve_main_mip_seconds: float | None = None
    solve_submip_seconds: float | None = None
    submip_calls: int | None = None
    lp_iters_separation: int | None = None
    lp_iters_heuristics: int | None = None
    lp_iters_strong_branching: int | None = None
    mip_restarts: int | None = None
    first_feasible_seconds: float | None = None
    first_feasible_objective: float | None = None
    final_cuts_in_lp: int | None = None
    # Progress-table rows as a JSON list of dicts (time, bound, sol, gap,
    # cuts in LP, cumulative LP iterations, source tag) — the bound and
    # incumbent trajectory of the root loop, for research analysis.
    mip_timeline_json: str | None = None
    matrix_coef_min: float | None = None
    matrix_coef_max: float | None = None
    cost_coef_min: float | None = None
    cost_coef_max: float | None = None
    bound_coef_min: float | None = None
    bound_coef_max: float | None = None
    rhs_coef_min: float | None = None
    rhs_coef_max: float | None = None

    def as_row(self) -> dict:
        """Flat dict of every field, for use as DataFrame columns."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _clean_count(value) -> int | None:
    """HiGHS reports -1 for counters that don't apply."""
    if value is None:
        return None
    value = int(value)
    return value if value >= 0 else None


def _clean_float(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def collect_highs_metrics(opt, log_text: str | None = None) -> SolveMetrics:
    """Read metrics off a solved appsi ``Highs`` interface.

    ``opt._solver_model`` is the underlying ``highspy.Highs``; every read
    is guarded so a Pyomo/highspy version that lacks an attribute degrades
    to None rather than failing the solve.
    """
    metrics = SolveMetrics()
    highs = getattr(opt, "_solver_model", None)
    if highs is not None:
        try:
            metrics.num_rows = int(highs.getNumRow())
            metrics.num_cols = int(highs.getNumCol())
            metrics.num_nonzeros = int(highs.getNumNz())
        except Exception:
            pass
        try:
            integrality = highs.getLp().integrality_
            metrics.num_integer_vars = int(
                sum(1 for kind in integrality if int(kind) != 0)
            )
        except Exception:
            pass
        try:
            metrics.highs_run_seconds = float(highs.getRunTime())
        except Exception:
            pass
        try:
            info = highs.getInfo()
            metrics.simplex_iterations = _clean_count(
                getattr(info, "simplex_iteration_count", None)
            )
            metrics.ipm_iterations = _clean_count(
                getattr(info, "ipm_iteration_count", None)
            )
            metrics.crossover_iterations = _clean_count(
                getattr(info, "crossover_iteration_count", None)
            )
            metrics.pdlp_iterations = _clean_count(
                getattr(info, "pdlp_iteration_count", None)
            )
            metrics.mip_nodes = _clean_count(getattr(info, "mip_node_count", None))
            metrics.final_mip_gap = _clean_float(getattr(info, "mip_gap", None))
            metrics.primal_dual_integral = _clean_float(
                getattr(info, "primal_dual_integral", None)
            )
        except Exception:
            pass

    if log_text:
        parsed = parse_highs_log(log_text)
        for name, value in parsed.items():
            if value is not None:
                setattr(metrics, name, value)
    return metrics


# Tolerant patterns for the HiGHS log. Formats drift between HiGHS
# versions, so every parse is optional; unmatched fields stay None.
_RE_REDUCTIONS = re.compile(
    r"[Rr]eductions:\s*rows\s+(\d+)\(.*?\);\s*columns\s+(\d+)\(.*?\);\s*"
    r"(?:elements|nonzeros)\s+(\d+)"
)
_RE_MIP_WITH_BINARIES = re.compile(r"\((\d+)\s+binary")
_RE_PHASE_TIME = {
    "presolve_seconds": re.compile(r"([\d.]+)\s+\(Presolve\)"),
    "solve_phase_seconds": re.compile(r"([\d.]+)\s+\(Solve\)"),
    "postsolve_seconds": re.compile(r"([\d.]+)\s+\(Postsolve\)"),
}
_NUM = r"([\d.]+e[+-]\d+|[\d.]+)"

# One row of the MIP progress table, e.g.
#  " L       0       0         0   0.00%   5758161.689821  5760305.185049
#    0.04%     6201    686      0      5479    14.6s"
# Columns: Src, nodes processed, in queue, leaves, explored %, best bound,
# best solution, gap, cut-pool size, cuts in LP, conflicts, LP iterations
# (may carry a k/M suffix), time. Objective fields may be inf/-inf.
_MIP_OBJ = r"(-?inf|[-+]?[\d.]+(?:e[+-]?\d+)?)"
_RE_MIP_PROGRESS = re.compile(
    r"^\s*(?P<src>[A-Za-z]?)\s+(?P<proc>\d+)\s+(?P<inqueue>\d+)\s+(?P<leaves>\d+)\s+"
    r"(?P<expl>[\d.]+)%\s+" + _MIP_OBJ.replace("(", "(?P<bound>", 1) + r"\s+"
    + _MIP_OBJ.replace("(", "(?P<sol>", 1) + r"\s+(?P<gap>\S+)\s+"
    r"(?P<cuts>\d+)\s+(?P<inlp>\d+)\s+(?P<confl>\d+)\s+"
    r"(?P<lpiters>[\d.]+[kM]?)\s+(?P<time>[\d.]+)s\s*$",
    re.MULTILINE,
)
_RE_RESTART = re.compile(r"^\s*Restarting", re.MULTILINE | re.IGNORECASE)
# The end-of-solve report splits the Solve phase into main-MIP and sub-MIP
# time, and LP iterations by purpose.
_RE_SOLVE_SPLIT = re.compile(
    r"\(Solve\)\s*\n\s*MIP\s+time \[calls\] = ([\d.]+) \[(\d+)\]"
    r"\s*\n\s*subMIP time \[calls\] = ([\d.]+) \[(\d+)\]"
)
_RE_LP_ITERS_KIND = {
    "lp_iters_strong_branching": re.compile(r"(\d+)\s+\(strong br\.\)"),
    "lp_iters_separation": re.compile(r"(\d+)\s+\(separation\)"),
    "lp_iters_heuristics": re.compile(r"(\d+)\s+\(heuristics\)"),
}


def _parse_count_suffix(raw: str) -> int:
    """LP-iteration counts may be abbreviated ('129k'). Expand to an int."""
    scale = 1
    if raw.endswith("k"):
        scale, raw = 1_000, raw[:-1]
    elif raw.endswith("M"):
        scale, raw = 1_000_000, raw[:-1]
    return int(float(raw) * scale)


def _parse_objective(raw: str) -> float | None:
    if "inf" in raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_mip_progress(text: str) -> dict:
    """Extract the root-loop story from a HiGHS MIP log.

    Returns a dict of :class:`SolveMetrics` field names. Every parse is
    tolerant: a log without the expected blocks (LP solves, older HiGHS
    versions) yields Nones.
    """
    out: dict = {
        "solve_main_mip_seconds": None,
        "solve_submip_seconds": None,
        "submip_calls": None,
        "lp_iters_separation": None,
        "lp_iters_heuristics": None,
        "lp_iters_strong_branching": None,
        "mip_restarts": None,
        "first_feasible_seconds": None,
        "first_feasible_objective": None,
        "final_cuts_in_lp": None,
        "mip_timeline_json": None,
    }

    timeline = []
    for match in _RE_MIP_PROGRESS.finditer(text):
        row = {
            "src": match.group("src") or None,
            "time": float(match.group("time")),
            "bound": _parse_objective(match.group("bound")),
            "sol": _parse_objective(match.group("sol")),
            "gap_pct": (
                float(match.group("gap")[:-1])
                if match.group("gap").endswith("%")
                else None
            ),
            "cuts_in_lp": int(match.group("inlp")),
            "lp_iters": _parse_count_suffix(match.group("lpiters")),
            "nodes": int(match.group("proc")),
        }
        timeline.append(row)

    if timeline:
        out["mip_timeline_json"] = json.dumps(timeline)
        out["final_cuts_in_lp"] = timeline[-1]["cuts_in_lp"]
        for row in timeline:
            if row["sol"] is not None:
                out["first_feasible_seconds"] = row["time"]
                out["first_feasible_objective"] = row["sol"]
                break

    restarts = len(_RE_RESTART.findall(text))
    if timeline or restarts:
        out["mip_restarts"] = restarts

    split = _RE_SOLVE_SPLIT.search(text)
    if split:
        out["solve_main_mip_seconds"] = float(split.group(1))
        out["solve_submip_seconds"] = float(split.group(3))
        out["submip_calls"] = int(split.group(4))

    for name, pattern in _RE_LP_ITERS_KIND.items():
        match = pattern.search(text)
        if match:
            out[name] = int(match.group(1))
    return out


_RE_COEF_RANGE = {
    ("matrix_coef_min", "matrix_coef_max"): re.compile(
        r"Matrix\s+\[" + _NUM + r",\s*" + _NUM + r"\]"
    ),
    ("cost_coef_min", "cost_coef_max"): re.compile(
        r"Cost\s+\[" + _NUM + r",\s*" + _NUM + r"\]"
    ),
    ("bound_coef_min", "bound_coef_max"): re.compile(
        r"Bound\s+\[" + _NUM + r",\s*" + _NUM + r"\]"
    ),
    ("rhs_coef_min", "rhs_coef_max"): re.compile(
        r"RHS\s+\[" + _NUM + r",\s*" + _NUM + r"\]"
    ),
}


def parse_highs_log(text: str) -> dict:
    """Extract presolve, timing and numerics detail from a HiGHS log.

    Returns a dict whose keys match :class:`SolveMetrics` field names;
    anything the log doesn't contain maps to None. When the log holds
    several presolve passes (MIP restarts), the last one — the model the
    search actually ran on — wins.
    """
    out: dict = {
        "presolved_rows": None,
        "presolved_cols": None,
        "presolved_nonzeros": None,
        "presolved_binaries": None,
        "presolve_seconds": None,
        "solve_phase_seconds": None,
        "postsolve_seconds": None,
    }

    reductions = _RE_REDUCTIONS.findall(text)
    if reductions:
        rows, cols, nonzeros = reductions[-1]
        out["presolved_rows"] = int(rows)
        out["presolved_cols"] = int(cols)
        out["presolved_nonzeros"] = int(nonzeros)

    with_block = text.rfind("Solving MIP model with:")
    if with_block != -1:
        match = _RE_MIP_WITH_BINARIES.search(text, with_block)
        if match:
            out["presolved_binaries"] = int(match.group(1))

    for name, pattern in _RE_PHASE_TIME.items():
        matches = pattern.findall(text)
        if matches:
            out[name] = float(matches[-1])

    for (low_name, high_name), pattern in _RE_COEF_RANGE.items():
        match = pattern.search(text)
        out[low_name] = float(match.group(1)) if match else None
        out[high_name] = float(match.group(2)) if match else None

    out.update(parse_mip_progress(text))
    return out


# --------------------------------------------------------------------------
# Pyomo model census
# --------------------------------------------------------------------------


def model_stats(model: pyo.ConcreteModel) -> pd.DataFrame:
    """Count constraint rows and variables per named component.

    This attributes problem size to formulation blocks (ramp_up,
    min_up_time, load_balance, ...) so a size regression can be traced to
    the constraint that caused it. Columns: kind, count, binary, fixed
    (variable components only); indexed by component name.
    """
    rows = []
    for component in model.component_objects(pyo.Constraint, active=True):
        rows.append(
            {
                "component": component.name,
                "kind": "constraint",
                "count": len(component),
                "binary": None,
                "fixed": None,
            }
        )
    for component in model.component_objects(pyo.Var, active=True):
        variables = list(component.values())
        rows.append(
            {
                "component": component.name,
                "kind": "variable",
                "count": len(variables),
                "binary": sum(1 for v in variables if v.is_binary()),
                "fixed": sum(1 for v in variables if v.fixed),
            }
        )
    return pd.DataFrame(rows).set_index("component")


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------


def _make_rss_reader():
    """Return a callable giving current process RSS in bytes, or None.

    Deliberately dependency-free: psutil if it happens to be installed,
    otherwise the platform API. Returns None when nothing works, and every
    caller treats that as "memory not measured" rather than failing.
    """
    try:  # optional, not a declared dependency
        import psutil

        process = psutil.Process()
        return lambda: process.memory_info().rss
    except Exception:
        pass

    if os.name == "nt":
        try:
            import ctypes
            import ctypes.wintypes as wintypes

            class _MemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.windll.kernel32
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            query = kernel32.K32GetProcessMemoryInfo
            query.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(_MemoryCounters),
                wintypes.DWORD,
            ]
            query.restype = wintypes.BOOL

            def read_rss() -> int | None:
                counters = _MemoryCounters()
                counters.cb = ctypes.sizeof(_MemoryCounters)
                handle = kernel32.GetCurrentProcess()
                if not query(handle, ctypes.byref(counters), counters.cb):
                    return None
                return int(counters.WorkingSetSize)

            if read_rss() is not None:
                return read_rss
        except Exception:
            pass
        return None

    try:  # POSIX
        import resource

        # ru_maxrss is KB on Linux, bytes on macOS — and it is a high-water
        # mark, not current usage, so /proc is preferred where available.
        if Path("/proc/self/statm").is_file():
            page_size = os.sysconf("SC_PAGE_SIZE")

            def read_rss() -> int | None:
                with open("/proc/self/statm") as handle:
                    return int(handle.read().split()[1]) * page_size

            return read_rss

        scale = 1 if platform.system() == "Darwin" else 1024
        return lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    except Exception:
        return None


class MemorySampler:
    """Context manager sampling peak process RSS over a block of work.

    Polls on a daemon thread rather than reading the OS peak counter,
    because that counter is a process-lifetime high-water mark: with
    several benchmark cases in one process it would report the largest
    case for every case that followed it. Sampling gives a per-block
    figure. ``peak_mb``/``start_mb`` are None when RSS is unreadable.
    """

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self._read = _make_rss_reader()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.start_bytes: int | None = None
        self.peak_bytes: int | None = None

    def _poll(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                current = self._read()
            except Exception:
                return
            if current is not None and (
                self.peak_bytes is None or current > self.peak_bytes
            ):
                self.peak_bytes = current

    def __enter__(self) -> "MemorySampler":
        if self._read is None:
            return self
        try:
            self.start_bytes = self._read()
        except Exception:
            self._read = None
            return self
        self.peak_bytes = self.start_bytes
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            current = self._read()
        except Exception:
            current = None
        if current is not None and (
            self.peak_bytes is None or current > self.peak_bytes
        ):
            self.peak_bytes = current

    @staticmethod
    def _mb(value: int | None) -> float | None:
        return None if value is None else round(value / 1_048_576, 1)

    @property
    def peak_mb(self) -> float | None:
        return self._mb(self.peak_bytes)

    @property
    def start_mb(self) -> float | None:
        return self._mb(self.start_bytes)

    @property
    def growth_mb(self) -> float | None:
        """Peak minus the RSS on entry: memory attributable to this block."""
        if self.peak_bytes is None or self.start_bytes is None:
            return None
        return round((self.peak_bytes - self.start_bytes) / 1_048_576, 1)


# --------------------------------------------------------------------------
# Environment capture
# --------------------------------------------------------------------------


def _git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent.parent)] + args,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except Exception:
        return None


def total_memory_mb() -> float | None:
    """Physical RAM in MB, so a peak-memory figure has something to sit against."""
    try:
        if os.name == "nt":
            import ctypes
            import ctypes.wintypes as wintypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return round(status.ullTotalPhys / 1_048_576, 1)
        return round(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1_048_576, 1
        )
    except Exception:
        return None


def capture_environment() -> dict:
    """Snapshot everything that could explain a timing difference later."""
    status = _git(["status", "--porcelain"])
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "total_memory_mb": total_memory_mb(),
        "python": platform.python_version(),
        "pyomo": _package_version("pyomo"),
        "highspy": _package_version("highspy"),
        "pandas": _package_version("pandas"),
        "numpy": _package_version("numpy"),
        "git_sha": _git(["rev-parse", "--short", "HEAD"]),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": None if status is None else bool(status),
    }


# --------------------------------------------------------------------------
# JSONL record storage
# --------------------------------------------------------------------------


def frame_to_records(frame: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe list of row dicts (NaN becomes None)."""
    return json.loads(frame.to_json(orient="records"))


def append_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
