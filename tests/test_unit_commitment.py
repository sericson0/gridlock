"""Unit commitment behavior: on/off logic, min generation, min up/down times."""

import pytest

from gridlock import RunConfig, run

from conftest import gen, make_system


def uc_system():
    """A cheap unit with a 40 MW minimum and an expensive free-ranging peaker."""
    return make_system(
        [
            gen("big", "A", 10, 100, min_mw=40, startup_cost=1000),
            gen("peaker", "A", 100, 100),
        ],
        {"A": [30.0] * 12 + [90.0] * 12},
    )


def test_commitment_respects_minimum_generation():
    results = run(uc_system(), RunConfig(unit_commitment=True))

    on = results.commitment["big"] >= 0.5
    # When demand (30) is below the minimum (40) the unit cannot run: with
    # supply == demand there is nowhere for surplus energy to go.
    assert not on.iloc[:12].any()
    assert on.iloc[12:].all()
    # Committed hours sit between min and max; off hours produce nothing.
    dispatched = results.dispatch["big"]
    assert (dispatched[on] >= 40 - 1e-6).all()
    assert dispatched[~on].to_numpy() == pytest.approx(0.0)
    assert dispatched.iloc[12:].to_numpy() == pytest.approx([90.0] * 12)


def test_lp_relaxation_lower_bounds_mip():
    mip = run(uc_system(), RunConfig(unit_commitment=True))
    lp = run(uc_system(), RunConfig(unit_commitment=False))

    assert lp.total_cost <= mip.total_cost + 1e-6
    # The relaxation exploits fractional commitment in the low-demand hours.
    assert lp.dispatch["big"].iloc[:12].to_numpy() == pytest.approx([30.0] * 12)
    u = lp.commitment["big"]
    assert ((u > 1e-6) & (u < 1 - 1e-6)).iloc[:12].all()


def assert_min_times_respected(results, unit, min_up, min_down):
    """Every startup keeps the unit on for min_up hours (and vice versa)."""
    on = (results.commitment[unit] >= 0.5).astype(int).to_list()
    started = (results.startup[unit] >= 0.5).to_list()
    stopped = (results.shutdown[unit] >= 0.5).to_list()
    horizon = len(on)
    for t in range(horizon):
        if started[t]:
            assert all(on[t : min(t + min_up, horizon)]), f"min up violated at {t}"
        if stopped[t]:
            assert not any(on[t : min(t + min_down, horizon)]), f"min down violated at {t}"


def test_min_up_and_down_times():
    demand = [30.0] * 6 + [120.0] * 2 + [30.0] * 10 + [120.0] * 2 + [30.0] * 4
    system = make_system(
        [
            gen("mid", "A", 10, 100, min_mw=20, startup_cost=500, min_up=4, min_down=4),
            gen("peaker", "A", 50, 200),
        ],
        {"A": demand},
    )
    results = run(system, RunConfig(unit_commitment=True))

    assert_min_times_respected(results, "mid", min_up=4, min_down=4)
    # The unit is actually exercised: it turns on for the demand spikes.
    assert (results.commitment["mid"] >= 0.5).any()
    assert results.shed.to_numpy().sum() == pytest.approx(0.0)


def test_no_load_cost_paid_only_when_committed():
    system = make_system(
        [
            gen("costly_idle", "A", 10, 100, min_mw=10, no_load_cost=200),
            gen("peaker", "A", 30, 100),
        ],
        {"A": [50.0] * 8},
    )
    results = run(system, RunConfig(unit_commitment=True))

    # Serving 50 MW from 'costly_idle' costs 10*50 + 200 = 700/h; the peaker
    # costs 30*50 = 1500/h, so the committed unit should win and pay no-load.
    assert (results.commitment["costly_idle"] >= 0.5).all()
    expected = 8 * (10 * 50 + 200)
    assert results.total_cost == pytest.approx(expected, rel=1e-6)
