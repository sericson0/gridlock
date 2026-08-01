"""Economic dispatch fundamentals: merit order, prices, unserved energy."""

import pytest

from gridlock import RunConfig, run

from conftest import gen, make_system


def test_merit_order_dispatch():
    system = make_system(
        [
            gen("cheap", "A", 10, 50),
            gen("expensive", "A", 50, 100),
        ],
        {"A": [80.0] * 24},
    )
    results = run(system, RunConfig())

    assert results.dispatch["cheap"].to_numpy() == pytest.approx([50.0] * 24)
    assert results.dispatch["expensive"].to_numpy() == pytest.approx([30.0] * 24)
    assert results.shed.to_numpy().sum() == pytest.approx(0.0)


def test_prices_equal_marginal_unit_cost():
    system = make_system(
        [
            gen("cheap", "A", 10, 50),
            gen("expensive", "A", 50, 100),
        ],
        {"A": [80.0] * 24},
    )
    results = run(system, RunConfig())

    # No commitment state anywhere, so duals are available even with uc=True.
    assert results.prices is not None
    assert results.prices["A"].to_numpy() == pytest.approx([50.0] * 24)


def test_availability_derates_capacity():
    system = make_system(
        [
            gen("cheap", "A", 10, 100),
            gen("expensive", "A", 50, 100),
        ],
        {"A": [80.0] * 4},
        availability={"cheap": [1.0, 0.5, 0.0, 1.0]},
    )
    results = run(system, RunConfig())

    assert results.dispatch["cheap"].to_numpy() == pytest.approx([80.0, 50.0, 0.0, 80.0])
    assert results.dispatch["expensive"].to_numpy() == pytest.approx([0.0, 30.0, 80.0, 0.0])


def test_cyclic_ramp_links_first_and_last_hour():
    """Monolithic windows wrap, so the last-hour output constrains hour 0."""
    system = make_system(
        [
            gen("slow", "A", 10, 200, ramp=50),
            gen("fast", "A", 100, 200),
        ],
        {"A": [50.0, 200.0, 200.0, 200.0, 200.0, 200.0]},
    )
    results = run(system, RunConfig())

    # Hour 0 pins slow to 50 MW, and the wrap-around ramp caps the last
    # hour at 100 MW even though demand there is 200 MW.
    assert results.dispatch["slow"].to_numpy() == pytest.approx(
        [50.0, 100.0, 150.0, 200.0, 150.0, 100.0]
    )
    p = results.dispatch["slow"]
    assert abs(p.iloc[0] - p.iloc[-1]) <= 50 + 1e-6


def test_unserved_energy_when_capacity_short():
    system = make_system([gen("only", "A", 10, 60)], {"A": [100.0] * 6})
    results = run(system, RunConfig(voll=5000.0))

    assert results.shed["A"].to_numpy() == pytest.approx([40.0] * 6)
    assert results.cost_summary["unserved_energy"] == pytest.approx(40 * 6 * 5000.0)
    # Scarcity price equals VOLL.
    assert results.prices["A"].to_numpy() == pytest.approx([5000.0] * 6)


def test_objective_matches_cost_summary_monolithic():
    system = make_system(
        [
            gen("cheap", "A", 10, 50),
            gen("expensive", "A", 50, 100),
        ],
        {"A": [80.0] * 24},
    )
    results = run(system, RunConfig())
    assert results.objective_value == pytest.approx(results.total_cost, rel=1e-6)
