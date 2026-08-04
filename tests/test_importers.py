"""Dataset importer helpers, and end-to-end checks when the data is present.

The pure helpers are always tested. The conversions themselves need the
third-party clones under data/external/, so those tests skip when it is
absent (see scripts/fetch_external_data.py).
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _import_common import (  # noqa: E402
    availability_from_mw,
    ceil_hours,
    check_no_missing,
    fit_heat_rate,
    heat_rate_error,
    number,
)


# ------------------------------------------------------------------ number


@pytest.mark.parametrize(
    "value,expected",
    [
        (3.5, 3.5),
        ("2.5", 2.5),
        (None, 0.0),
        (float("nan"), 0.0),
        (np.nan, 0.0),
        ("", 0.0),
        ("not a number", 0.0),
    ],
)
def test_number_coerces_missing_to_default(value, expected):
    assert number(value) == expected


def test_number_exists_because_nan_is_truthy():
    """The idiom this replaces silently passes NaN through.

    A NaN reaching a cost column makes the entire objective NaN: the LP
    then reports a NaN objective and the MIP reports *infeasible*, with
    nothing pointing at the real cause. This is a regression guard for a
    bug that actually happened during the NREL-118 import.
    """
    assert math.isnan(float(float("nan") or 0.0))  # the trap
    assert number(float("nan")) == 0.0  # the fix


def test_number_honours_a_custom_default():
    assert number(None, default=1.0) == 1.0
    assert number(float("nan"), default=-1.0) == -1.0


# ---------------------------------------------------------- heat rate fit


def test_fit_heat_rate_reproduces_both_endpoints():
    # Fuel input 100 MMBtu/h at 20 MW rising to 400 at 100 MW.
    slope, intercept = fit_heat_rate([20.0, 60.0, 100.0], [100.0, 260.0, 400.0])
    assert intercept + slope * 20.0 == pytest.approx(100.0)
    assert intercept + slope * 100.0 == pytest.approx(400.0)
    assert slope == pytest.approx(3.75)
    assert intercept == pytest.approx(25.0)


def test_fit_heat_rate_is_exact_for_a_straight_curve():
    powers = [10.0, 50.0, 90.0]
    fuel = [55.0, 175.0, 295.0]  # intercept 25, slope 3
    slope, intercept = fit_heat_rate(powers, fuel)
    assert slope == pytest.approx(3.0)
    assert intercept == pytest.approx(25.0)
    assert heat_rate_error(powers, fuel, slope, intercept) == pytest.approx(0.0)


def test_fit_heat_rate_reports_error_on_a_convex_curve():
    powers = [10.0, 50.0, 90.0]
    fuel = [50.0, 150.0, 350.0]  # convex: the secant cuts under the middle
    slope, intercept = fit_heat_rate(powers, fuel)
    assert heat_rate_error(powers, fuel, slope, intercept) > 0.05


def test_fit_heat_rate_never_returns_a_negative_no_load():
    """A negative intercept would pay a unit to stay committed."""
    # Concave curve whose secant would extrapolate below zero.
    slope, intercept = fit_heat_rate([10.0, 100.0], [10.0, 500.0])
    assert intercept >= 0.0
    assert slope > 0.0


def test_fit_heat_rate_handles_a_degenerate_curve():
    assert fit_heat_rate([50.0], [200.0]) == (4.0, 0.0)
    assert fit_heat_rate([], []) == (0.0, 0.0)


# ------------------------------------------------------------ other helpers


@pytest.mark.parametrize(
    "value,expected", [(1.0, 1), (2.2, 3), (4.5, 5), (8.0, 8), (0.0, 1), (None, 1)]
)
def test_ceil_hours_rounds_up(value, expected):
    """Rounding up keeps the converted system no laxer than the source.

    gridlock's model builder truncates with int(), so a 4.5 h obligation
    would silently become 4 h if the importer did not round up first.
    """
    assert ceil_hours(value) == expected


def test_availability_clips_to_the_unit_interval():
    values = availability_from_mw([0.0, 50.0, 100.0, 101.0, -1.0], 100.0)
    assert list(values) == [0.0, 0.5, 1.0, 1.0, 0.0]


def test_availability_of_a_zero_capacity_unit_is_zero():
    assert list(availability_from_mw([5.0, 10.0], 0.0)) == [0.0, 0.0]


def test_check_no_missing_rejects_nan():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, float("nan")]})
    with pytest.raises(ValueError, match="missing values"):
        check_no_missing(frame, "generators")
    check_no_missing(frame.dropna(), "generators")  # no raise


# ------------------------------------------------- converted-dataset checks

CONVERTED = ["rts_gmlc", "rts_gmlc_area", "nrel118", "nrel118_region"]


def _dataset(name: str):
    directory = REPO_ROOT / "data" / name
    if not (directory / "generators.csv").is_file():
        pytest.skip(f"{name} not imported (see scripts/fetch_external_data.py)")
    return directory


@pytest.mark.parametrize("name", CONVERTED)
def test_converted_dataset_loads(name):
    from gridlock import load_system

    system = load_system(_dataset(name))
    assert len(system.generators) > 0
    assert system.num_hours >= 8760
    assert system.generators["needs_commitment"].any()


@pytest.mark.parametrize("name", CONVERTED)
def test_converted_dataset_has_no_missing_values(name):
    directory = _dataset(name)
    for filename in ("generators.csv", "storage.csv", "network.csv", "demand.csv"):
        frame = pd.read_csv(directory / filename)
        assert not frame.isna().any().any(), f"{name}/{filename} has NaN"


@pytest.mark.parametrize("name", CONVERTED)
def test_converted_dataset_min_up_down_are_whole_hours(name):
    generators = pd.read_csv(_dataset(name) / "generators.csv")
    for column in ("min_up_time_hr", "min_down_time_hr"):
        values = generators[column].dropna()
        assert (values == values.round()).all(), f"{name}: {column} not integral"
        assert (values >= 1).all()


@pytest.mark.parametrize("name", CONVERTED)
def test_converted_dataset_solves(name):
    """Short horizon at a tight gap: this checks convertibility, not speed.

    Six hours rather than a day because the nodal NREL-118 system is
    genuinely hard — a full day at a 1% gap exceeds ten minutes. The gap
    is tight *because* the horizon is short: at a loose tolerance the
    solver stops on a suboptimal incumbent that sheds a trace of load,
    which would make the no-shed assertion below flap.
    """
    from gridlock import RunConfig, SolverSettings, load_system, run

    system = load_system(_dataset(name))
    results = run(
        system,
        RunConfig(
            unit_commitment=True,
            num_hours=6,
            tight_generation_limits=True,
            tight_ramp_limits=True,
            solver=SolverSettings(mip_gap=0.001, time_limit=300),
        ),
    )
    assert results.total_cost > 0
    # A converted system that cannot serve its own load points at a
    # conversion error (lost capacity, mangled profiles) rather than at
    # a genuinely scarce system.
    assert results.shed.to_numpy().sum() == pytest.approx(0.0, abs=1e-3)


def test_rts_gmlc_thermal_units_carry_commitment_parameters():
    generators = pd.read_csv(_dataset("rts_gmlc_area") / "generators.csv")
    thermal = generators[generators["min_mw"] > 0]
    assert len(thermal) == 73  # matches the published RTS-GMLC thermal fleet
    assert (thermal["startup_cost"] > 0).all()
    assert (thermal["fuel_cost_per_mmbtu"] > 0).all()
    # Neither coefficient may be negative: a negative slope inverts merit
    # order, and a negative no-load term pays units to stay committed.
    assert (thermal["heat_rate_slope_mmbtu_per_mwh"] >= 0).all()
    assert (thermal["heat_rate_intercept_mmbtu_per_hr"] >= 0).all()
    # Every unit must have at least one of them, or it generates for free.
    assert (
        (thermal["heat_rate_slope_mmbtu_per_mwh"] > 0)
        | (thermal["heat_rate_intercept_mmbtu_per_hr"] > 0)
    ).all()


def test_rts_gmlc_zero_coefficients_are_explained_not_accidental():
    """The handful of zero heat-rate coefficients each have a known cause."""
    generators = pd.read_csv(_dataset("rts_gmlc_area") / "generators.csv")
    thermal = generators[generators["min_mw"] > 0]

    # RTS-GMLC gives the nuclear unit a flat average heat rate with a zero
    # incremental band: fuel burn does not vary with output. Carried
    # through faithfully that is zero marginal cost plus a large no-load
    # term, which is how nuclear should sit in merit order.
    flat = thermal[thermal["heat_rate_slope_mmbtu_per_mwh"] == 0]
    assert list(flat["technology"]) == ["nuclear"]
    assert (flat["heat_rate_intercept_mmbtu_per_hr"] > 0).all()

    # A few units have curves convex enough that the min-to-max secant
    # implies a negative no-load term; the fitter clamps those to zero.
    clamped = thermal[thermal["heat_rate_intercept_mmbtu_per_hr"] == 0]
    assert len(clamped) <= 5
    assert (clamped["heat_rate_slope_mmbtu_per_mwh"] > 0).all()


def test_nrel118_no_load_heat_rate_comes_through_unfitted():
    """NREL-118 publishes a real no-load term; it must survive conversion."""
    generators = pd.read_csv(_dataset("nrel118_region") / "generators.csv")
    thermal = generators[generators["min_mw"] > 0]
    assert len(thermal) > 100
    assert (thermal["heat_rate_intercept_mmbtu_per_hr"] > 0).any()
