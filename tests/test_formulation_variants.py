"""Tight formulations and unit clustering: validity first, tightness second.

A tightening is only interesting if it leaves the integer-feasible set
alone, so every variant is checked against the baseline MIP optimum before
its LP bound is compared.
"""

import itertools

import pytest

from gridlock import RunConfig, run
from gridlock.data import cluster_identical_units
from gridlock.model import build_model
from gridlock.solver import solve_model

from conftest import gen, make_system

VARIANTS = {
    "base": {},
    "tight_gen": {"tight_generation_limits": True},
    "tight_ramp": {"tight_ramp_limits": True},
    "both": {"tight_generation_limits": True, "tight_ramp_limits": True},
}


def ramped_system(hours=24):
    """Ramp- and commitment-constrained enough that the tightenings bite."""
    demand = [220.0, 180.0, 140.0, 120.0, 160.0, 260.0, 340.0, 400.0] * (hours // 8)
    return make_system(
        [
            gen("base", "A", 12, 300, min_mw=120, startup_cost=4000, shutdown_cost=800,
                no_load_cost=250, ramp=60, min_up=4, min_down=3),
            gen("mid", "A", 30, 200, min_mw=80, startup_cost=1500, shutdown_cost=300,
                no_load_cost=90, ramp=50, min_up=2, min_down=2),
            # min_up == 1: exercises the split startup/shutdown output rows.
            gen("fast", "A", 65, 150, min_mw=40, startup_cost=400, ramp=75),
            gen("peaker", "A", 200, 400),
        ],
        {"A": demand},
    )


def solve_variant(system, unit_commitment, **flags):
    config = RunConfig(unit_commitment=unit_commitment, **flags)
    return run(system, config)


@pytest.mark.parametrize("name", [n for n in VARIANTS if n != "base"])
def test_tightening_preserves_the_mip_optimum(name):
    system = ramped_system()
    base = solve_variant(system, True, **VARIANTS["base"])
    variant = solve_variant(system, True, **VARIANTS[name])

    assert variant.total_cost == pytest.approx(base.total_cost, rel=1e-6)


@pytest.mark.parametrize("name", [n for n in VARIANTS if n != "base"])
def test_tightening_never_weakens_the_lp_bound(name):
    system = ramped_system()
    base = solve_variant(system, False, **VARIANTS["base"])
    variant = solve_variant(system, False, **VARIANTS[name])

    # A valid tightening removes fractional points, so the minimization's
    # relaxation value can only rise (or stay put).
    assert variant.total_cost >= base.total_cost - 1e-6


def test_tight_generation_limits_raise_the_lp_bound_strictly():
    system = ramped_system()
    base = solve_variant(system, False)
    tight = solve_variant(system, False, tight_generation_limits=True)
    mip = solve_variant(system, True)

    assert tight.total_cost > base.total_cost + 1e-6
    assert tight.total_cost <= mip.total_cost + 1e-6


def test_tightening_adds_rows_only_for_short_min_up_units():
    system = ramped_system()
    hours = list(range(24))
    base = build_model(system, RunConfig(), hours)
    tight = build_model(system, RunConfig(tight_generation_limits=True), hours)

    # 'fast' has min_up == 1, so its output row splits in two; the other
    # committed units keep a single (tighter) row.
    assert len(tight.max_output) == len(base.max_output) - 24
    assert len(tight.max_output_startup) == 24
    assert len(tight.max_output_shutdown) == 24


def clustered_system(copies=3):
    """Identical peakers that a cluster can pool into one integer variable."""
    units = [
        gen("base", "A", 10, 400, min_mw=150, startup_cost=6000, no_load_cost=200,
            ramp=100, min_up=4, min_down=3)
    ]
    units += [
        gen(f"peaker_{i}", "A", 45, 120, min_mw=40, startup_cost=800, no_load_cost=50,
            ramp=60, min_up=2, min_down=2)
        for i in range(copies)
    ]
    units.append(gen("backstop", "A", 300, 500))
    demand = [200.0, 260.0, 380.0, 520.0, 610.0, 480.0, 330.0, 240.0] * 3
    return make_system(units, {"A": demand})


def test_clustering_pools_identical_units():
    system = clustered_system(copies=3)
    clustered = cluster_identical_units(system)

    assert len(clustered.generators) == 3  # base, peaker cluster, backstop
    counts = clustered.generators["num_units"].to_dict()
    assert sorted(counts.values()) == [1, 1, 3]
    # Per-unit parameters stay per unit.
    cluster_row = clustered.generators[clustered.generators["num_units"] == 3].iloc[0]
    assert cluster_row["max_mw"] == pytest.approx(120.0)


def test_clustering_is_idempotent():
    once = cluster_identical_units(clustered_system())
    twice = cluster_identical_units(once)

    assert list(twice.generators.index) == list(once.generators.index)
    assert twice.generators["num_units"].to_dict() == once.generators["num_units"].to_dict()


def test_clustered_run_matches_unit_level_cost():
    system = clustered_system(copies=3)
    per_unit = run(system, RunConfig(unit_commitment=True))
    pooled = run(system, RunConfig(unit_commitment=True, cluster_units=True))

    # Pooling relaxes ramp coupling between interchangeable members, so it
    # can only be cheaper — and on this system nothing exploits that.
    assert pooled.total_cost <= per_unit.total_cost + 1e-6
    assert pooled.total_cost == pytest.approx(per_unit.total_cost, rel=1e-3)


def test_clustered_commitment_is_an_integer_count():
    system = clustered_system(copies=3)
    results = run(system, RunConfig(unit_commitment=True, cluster_units=True))

    cluster = [c for c in results.commitment.columns if "cluster" in c]
    assert cluster, "expected a pooled column"
    values = results.commitment[cluster[0]]
    assert values.max() > 1.0  # genuinely counting, not a binary
    assert values.max() <= 3.0
    for value in values:
        assert value == pytest.approx(round(value), abs=1e-6)


def test_clustering_shrinks_the_integer_model():
    system = clustered_system(copies=3)
    hours = list(range(24))
    per_unit = build_model(system, RunConfig(unit_commitment=True), hours)
    pooled = build_model(
        cluster_identical_units(system), RunConfig(unit_commitment=True), hours
    )

    def binaries(model):
        return sum(v.is_binary() or v.is_integer() for v in model.u.values())

    assert binaries(pooled) < binaries(per_unit)


@pytest.mark.parametrize(
    "tight_gen,tight_ramp", list(itertools.product([False, True], repeat=2))
)
def test_tightenings_compose_with_clustering(tight_gen, tight_ramp):
    system = clustered_system(copies=3)
    baseline = run(system, RunConfig(unit_commitment=True, cluster_units=True))
    variant = run(
        system,
        RunConfig(
            unit_commitment=True,
            cluster_units=True,
            tight_generation_limits=tight_gen,
            tight_ramp_limits=tight_ramp,
        ),
    )
    assert variant.total_cost == pytest.approx(baseline.total_cost, rel=1e-6)


def test_tight_formulation_survives_rolling_windows():
    system = ramped_system(hours=48)
    plain = run(system, RunConfig(unit_commitment=True, window_hours=12, lookahead_hours=6))
    tight = run(
        system,
        RunConfig(
            unit_commitment=True,
            window_hours=12,
            lookahead_hours=6,
            tight_generation_limits=True,
            tight_ramp_limits=True,
        ),
    )
    assert tight.total_cost == pytest.approx(plain.total_cost, rel=1e-6)


def test_tight_formulation_survives_acyclic_windows():
    system = ramped_system()
    plain = run(system, RunConfig(unit_commitment=True, cyclic=False))
    tight = run(
        system,
        RunConfig(
            unit_commitment=True,
            cyclic=False,
            tight_generation_limits=True,
            tight_ramp_limits=True,
        ),
    )
    assert tight.total_cost == pytest.approx(plain.total_cost, rel=1e-6)


def test_availability_derating_keeps_tight_rows_valid():
    """A derated hour must not make the tight rows cut off the true optimum."""
    system = make_system(
        [
            gen("derated", "A", 15, 200, min_mw=80, startup_cost=1000, no_load_cost=100,
                ramp=50, min_up=3, min_down=2),
            gen("peaker", "A", 90, 300),
        ],
        {"A": [150.0, 170.0, 190.0, 120.0, 100.0, 160.0, 200.0, 210.0]},
        availability={"derated": [1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0]},
    )
    plain = run(system, RunConfig(unit_commitment=True))
    tight = run(
        system,
        RunConfig(
            unit_commitment=True,
            tight_generation_limits=True,
            tight_ramp_limits=True,
        ),
    )
    assert tight.total_cost == pytest.approx(plain.total_cost, rel=1e-6)
