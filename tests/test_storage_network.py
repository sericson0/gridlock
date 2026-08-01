"""Storage bathtub accounting and pipe-and-bubble network behavior."""

import math

import pytest

from gridlock import RunConfig, run

from conftest import gen, line, make_system, store


def arbitrage_system(hours=24):
    """Cheap capacity-limited energy plus a battery to cover evening peaks."""
    pattern = [50.0] * 12 + [130.0] * 12
    demand = (pattern * math.ceil(hours / 24))[:hours]
    return make_system(
        [
            gen("cheap", "A", 10, 100),
            gen("backstop", "A", 200, 100),
        ],
        {"A": demand},
        storage=[store("battery", "A", power_mw=50, energy_mwh=500, roundtrip_efficiency=0.81)],
    )


def test_storage_shifts_energy_and_avoids_expensive_unit():
    results = run(arbitrage_system(), RunConfig())

    # Peak deficit (130 - 100) is covered by discharge, not the backstop.
    assert results.storage_discharge["battery"].iloc[12:].to_numpy() == pytest.approx(
        [30.0] * 12
    )
    assert results.dispatch["backstop"].to_numpy().sum() == pytest.approx(0.0)
    # Charging happens in the cheap hours only.
    assert results.storage_charge["battery"].iloc[12:].to_numpy() == pytest.approx(0.0)


def test_storage_soc_bathtub_accounting():
    results = run(arbitrage_system(), RunConfig())
    eta = math.sqrt(0.81)

    soc = results.storage_soc["battery"]
    charge = results.storage_charge["battery"]
    discharge = results.storage_discharge["battery"]

    for t in range(1, 24):
        expected = soc.iloc[t - 1] + eta * charge.iloc[t] - discharge.iloc[t] / eta
        assert soc.iloc[t] == pytest.approx(expected, abs=1e-6)
    assert (soc <= 500 + 1e-6).all() and (soc >= -1e-6).all()


def test_cyclic_storage_conserves_energy():
    """Cyclic SOC means losses are the only gap between charge and discharge."""
    results = run(arbitrage_system(), RunConfig())
    eta = math.sqrt(0.81)

    charged = results.storage_charge["battery"].sum()
    discharged = results.storage_discharge["battery"].sum()
    assert charged * eta == pytest.approx(discharged / eta, rel=1e-6)


def two_node_system():
    return make_system(
        [
            gen("cheap_a", "A", 10, 200),
            gen("costly_b", "B", 100, 100),
        ],
        {"A": [0.0] * 24, "B": [80.0] * 24},
        lines=[line("A", "B", capacity_mw=50, loss_factor=0.1, name="AB")],
    )


def test_line_capacity_limits_flow_and_losses_are_charged():
    results = run(two_node_system(), RunConfig())

    # Importing costs 10 / 0.9 = 11.1 per delivered MWh, far below 100, so
    # the line runs at its 50 MW rating and delivers 45 MW.
    assert results.flows["AB"].to_numpy() == pytest.approx([50.0] * 24)
    assert results.dispatch["cheap_a"].to_numpy() == pytest.approx([50.0] * 24)
    assert results.dispatch["costly_b"].to_numpy() == pytest.approx([35.0] * 24)


def test_congestion_separates_nodal_prices():
    results = run(two_node_system(), RunConfig())

    assert results.prices is not None
    assert results.prices["A"].to_numpy() == pytest.approx([10.0] * 24)
    assert results.prices["B"].to_numpy() == pytest.approx([100.0] * 24)


def test_reverse_flow_uses_same_line():
    system = make_system(
        [
            gen("costly_a", "A", 100, 200),
            gen("cheap_b", "B", 10, 200),
        ],
        {"A": [80.0] * 12, "B": [0.0] * 12},
        lines=[line("A", "B", capacity_mw=50, loss_factor=0.1, name="AB")],
    )
    results = run(system, RunConfig())

    # Net flow is negative: B exports to A on the same line.
    assert results.flows["AB"].to_numpy() == pytest.approx([-50.0] * 12)
    assert results.dispatch["cheap_b"].to_numpy() == pytest.approx([50.0] * 12)
    assert results.dispatch["costly_a"].to_numpy() == pytest.approx([35.0] * 12)
