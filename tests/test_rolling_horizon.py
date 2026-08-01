"""Rolling-horizon runs: window stitching and state carried across boundaries."""

import math

import pytest

from gridlock import RunConfig, run

from conftest import gen, make_system, store
from test_storage_network import arbitrage_system
from test_unit_commitment import assert_min_times_respected


def test_rolling_covers_horizon_and_soc_is_continuous():
    hours = 96
    config = RunConfig(window_hours=24, lookahead_hours=12)
    results = run(arbitrage_system(hours), config)
    eta = math.sqrt(0.81)

    assert len(results.dispatch) == hours
    assert list(results.dispatch.index) == list(range(hours))
    assert len(results.window_stats) == 4

    soc = results.storage_soc["battery"]
    charge = results.storage_charge["battery"]
    discharge = results.storage_discharge["battery"]

    # The bathtub holds across every hour, including window boundaries.
    for t in range(1, hours):
        expected = soc.iloc[t - 1] + eta * charge.iloc[t] - discharge.iloc[t] / eta
        assert soc.iloc[t] == pytest.approx(expected, abs=1e-6), f"hour {t}"

    # First window starts from the configured fraction ...
    initial_soc = 0.5 * 500
    expected_first = initial_soc + eta * charge.iloc[0] - discharge.iloc[0] / eta
    assert soc.iloc[0] == pytest.approx(expected_first, abs=1e-6)
    # ... and the final hour cannot end below it.
    assert soc.iloc[-1] >= initial_soc - 1e-6


def test_rolling_carries_min_up_down_obligations():
    demand = ([30.0] * 6 + [120.0] * 2 + [30.0] * 10 + [120.0] * 2 + [30.0] * 4) * 2
    system = make_system(
        [
            gen("mid", "A", 10, 100, min_mw=20, startup_cost=500, min_up=6, min_down=6),
            gen("peaker", "A", 50, 200),
        ],
        {"A": demand},
    )
    # 8-hour windows guarantee up/down runs span window boundaries.
    config = RunConfig(unit_commitment=True, window_hours=8, lookahead_hours=4)
    results = run(system, config)

    assert_min_times_respected(results, "mid", min_up=6, min_down=6)
    assert (results.commitment["mid"] >= 0.5).any()
    assert results.shed.to_numpy().sum() == pytest.approx(0.0)


def test_rolling_close_to_monolithic_for_lp():
    hours = 96
    monolithic = run(arbitrage_system(hours), RunConfig(unit_commitment=False))
    # Start (and allow ending) empty so the rolling run faces no terminal
    # refill obligation that the cyclic monolithic run doesn't have; with a
    # daily-periodic pattern and day-aligned windows the two should agree.
    rolling = run(
        arbitrage_system(hours),
        RunConfig(
            unit_commitment=False,
            window_hours=24,
            lookahead_hours=12,
            initial_soc_fraction=0.0,
        ),
    )

    # Rolling can't beat perfect foresight, but with lookahead it lands close.
    assert rolling.total_cost >= monolithic.total_cost - 1e-6
    assert rolling.total_cost <= monolithic.total_cost * 1.02
