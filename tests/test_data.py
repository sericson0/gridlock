"""Input loading and validation."""

import pytest

from gridlock.data import load_system

from conftest import gen, make_system


def test_load_example_dataset(example_data_dir):
    system = load_system(example_data_dir)
    assert system.num_hours == 8760
    assert len(system.generators) == 12
    assert len(system.storage) == 2
    assert len(system.network) == 3
    # The example provides a profile for every unit; columns follow
    # generator order.
    assert list(system.availability.columns) == list(system.generators.index)
    # Derived economics exist and renewables need no commitment state.
    assert (system.generators["marginal_cost"] >= 0).all()
    assert not system.generators.loc["wind_north", "needs_commitment"]
    assert system.generators.loc["nuclear_north", "needs_commitment"]


def test_unknown_generator_node_rejected():
    with pytest.raises(ValueError, match="unknown nodes"):
        make_system([gen("g1", "NOWHERE", 10, 100)], {"A": [50] * 4})


def test_min_above_max_rejected():
    with pytest.raises(ValueError, match="min_mw > max_mw"):
        make_system([gen("g1", "A", 10, 100, min_mw=150)], {"A": [50] * 4})


def test_unknown_availability_column_rejected():
    with pytest.raises(ValueError, match="not known generators"):
        make_system(
            [gen("g1", "A", 10, 100)],
            {"A": [50] * 4},
            availability={"mystery_unit": [1.0] * 4},
        )


def test_availability_length_mismatch_rejected():
    with pytest.raises(ValueError, match="same hours"):
        make_system(
            [gen("g1", "A", 10, 100)],
            {"A": [50] * 4},
            availability={"g1": [1.0] * 3},
        )


def test_missing_availability_stays_unmaterialized():
    """Units without a profile column get no column at all (implicitly 1.0)."""
    system = make_system(
        [gen("g1", "A", 10, 100), gen("g2", "A", 20, 100)],
        {"A": [50] * 4},
        availability={"g2": [1.0, 0.5, 1.0, 1.0]},
    )
    assert list(system.availability.columns) == ["g2"]


def test_no_availability_input_at_all():
    system = make_system([gen("g1", "A", 10, 100)], {"A": [50] * 4})
    assert list(system.availability.columns) == []
    assert len(system.availability) == 4
