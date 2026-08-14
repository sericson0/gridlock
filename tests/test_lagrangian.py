"""The Lagrangian dual bound: subproblem exactness and the bound sandwich.

The bound's validity rests on the single-unit DP being *exact*: an
underestimate of a subproblem's maximum profit would overstate the bound,
which is the invalid direction. So the DP is pinned against brute-force
enumeration of every legal cyclic schedule, and the assembled bound is
checked to land between the LP relaxation and the MIP optimum on a system
that exercises every subproblem (clusters, storage, lines, shed pricing).
"""

import itertools

import numpy as np
import pandas as pd
import pytest

from gridlock import RunConfig, run
from gridlock.heuristics import _runs_of
from gridlock.lagrangian import lagrangian_bound, probe_at_lp_prices, schedule_value
from gridlock.rounding import single_unit_schedule

from conftest import gen, line, make_system, store


def legal_cyclic(schedule: np.ndarray, min_up: int, min_down: int) -> bool:
    on_runs = _runs_of(schedule, 1, cyclic=True)
    off_runs = _runs_of(schedule, 0, cyclic=True)
    if schedule.all() or not schedule.any():
        return True
    return all(length >= min_up for _, length in on_runs) and all(
        length >= min_down for _, length in off_runs
    )


def brute_force_best(value_on, min_up, min_down, startup, shutdown) -> float:
    horizon = len(value_on)
    best = -np.inf
    for bits in itertools.product((0, 1), repeat=horizon):
        schedule = np.array(bits)
        if not legal_cyclic(schedule, min_up, min_down):
            continue
        best = max(best, schedule_value(schedule, value_on, startup, shutdown))
    return best


def test_single_unit_dp_matches_brute_force():
    """The DP's schedule must earn exactly the enumerated maximum.

    Load-bearing for the bound (see module docstring): equality, not just
    feasibility, on every parameter combination tried.
    """
    rng = np.random.default_rng(7)
    horizon = 8
    for min_up, min_down in [(1, 1), (2, 3), (3, 2), (4, 4)]:
        for startup, shutdown in [(0.0, 0.0), (5.0, 0.0), (3.0, 2.0)]:
            for _ in range(6):
                value_on = rng.uniform(-10, 10, size=horizon)
                schedule = single_unit_schedule(
                    value_on,
                    min_up=min_up,
                    min_down=min_down,
                    startup_cost=startup,
                    shutdown_cost=shutdown,
                    cyclic=True,
                )
                assert legal_cyclic(schedule.round().astype(int), min_up, min_down)
                achieved = schedule_value(schedule, value_on, startup, shutdown)
                expected = brute_force_best(
                    value_on, min_up, min_down, startup, shutdown
                )
                assert achieved == pytest.approx(expected, abs=1e-9)


def bound_system():
    """Two nodes, a lossy line, storage, a cluster, and real UC economics.

    Ramps are left at their default (= max_mw), so no ramp rows exist and
    the "Lagrangian >= LP" guarantee holds exactly — which is what the
    sandwich test asserts.
    """
    base = gen(
        "coal",
        "A",
        10,
        60,
        min_mw=25,
        startup_cost=400,
        no_load_cost=120,
        min_up=3,
        min_down=3,
    )
    base["num_units"] = 2
    peaker = gen("gt", "B", 45, 80, min_mw=10, startup_cost=150, no_load_cost=40)
    demand = {
        "A": [40.0, 35.0, 30.0, 55.0, 90.0, 110.0, 95.0, 60.0] * 3,
        "B": [20.0, 15.0, 15.0, 30.0, 55.0, 70.0, 60.0, 35.0] * 3,
    }
    return make_system(
        [base, peaker],
        demand,
        storage=[store("batt", "B", 25, 80, 0.81)],
        lines=[line("A", "B", 45, 0.02)],
    )


def test_lagrangian_sits_between_lp_and_mip():
    system = bound_system()
    config = RunConfig(unit_commitment=True, voll=10_000.0)

    mip = run(system, config)
    result = probe_at_lp_prices(system, config)
    bound = result["lagrangian"]

    # Valid: never above the MIP optimum (allow the solver's own gap).
    assert bound.total <= mip.total_cost * (1 + 2e-4) + 1e-6
    # At the LP's own duals, with no dropped rows, it dominates the LP bound.
    assert bound.total >= result["lp_bound"] - 1e-6 * abs(result["lp_bound"])
    # The parts are an accounting, not an estimate.
    assert sum(bound.parts.values()) == pytest.approx(bound.total)
    assert bound.parts["thermal_uc"] <= 1e-9
    assert bound.parts["storage"] <= 1e-9
    assert bound.parts["lines"] <= 1e-9


def test_lagrangian_valid_at_arbitrary_prices():
    """L(lambda) must lower-bound the MIP for ANY prices, not just the LP's."""
    system = bound_system()
    config = RunConfig(unit_commitment=True, voll=10_000.0)
    mip = run(system, config)

    hours = list(range(system.num_hours))
    nodes = list(system.demand.columns)
    rng = np.random.default_rng(11)
    for _ in range(3):
        prices = pd.DataFrame(
            rng.uniform(0.0, 120.0, size=(len(hours), len(nodes))),
            index=hours,
            columns=nodes,
        )
        bound = lagrangian_bound(system, hours, prices, config.voll)
        assert bound.total <= mip.total_cost * (1 + 2e-4) + 1e-6
