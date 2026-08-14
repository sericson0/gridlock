"""The per-unit DP that discretises a fractional commitment."""

import itertools

import numpy as np
import pandas as pd
import pytest

from gridlock import RunConfig, run
from gridlock.heuristics import enforce_adequacy, lp_relaxation_guess
from gridlock.rounding import (
    commitment_value,
    dispatch_ceiling,
    dp_commitment,
    nodal_prices,
    restrict_to_decommitments,
    single_unit_schedule,
)

from conftest import gen, line, make_system


# ------------------------------------------------------ the single-unit DP


def schedule_value(u, value_on, startup_cost, shutdown_cost, cyclic, initial=None):
    """What a 0/1 schedule is worth, by the same accounting the DP uses."""
    u = np.asarray(u, dtype=float)
    total = float((u * np.asarray(value_on, dtype=float)).sum())
    for t in range(len(u)):
        if t == 0:
            if cyclic:
                previous = u[-1]
            elif initial is None:
                continue  # a free initial hour charges no transition
            else:
                previous = float(initial[0])
        else:
            previous = u[t - 1]
        total -= startup_cost if u[t] > previous else 0.0
        total -= shutdown_cost if u[t] < previous else 0.0
    return total


def obeys_min_times(u, min_up, min_down, cyclic, initial=None):
    """Whether every run the window *closes* meets its minimum time.

    A run left open at the horizon's end has no obligation inside the
    window, and neither has the opening run of a window whose initial state
    is free — which is exactly what the model's truncated lookback says.
    """
    u = [int(round(v)) for v in u]
    n = len(u)
    if cyclic:
        if len(set(u)) == 1:
            return True
        seam = next(i for i in range(n) if u[i] != u[i - 1])
        sequence = [u[(seam + k) % n] for k in range(n)]
        closes_last = True
    else:
        sequence = u
        closes_last = False

    index = 0
    while index < n:
        end = index
        while end < n and sequence[end] == sequence[index]:
            end += 1
        length = end - index
        opening = index == 0 and not cyclic
        if opening and initial is not None:
            if u[0] == initial[0]:
                length += initial[1]
            elif initial[1] < (min_up if initial[0] else min_down):
                return False  # transition at hour 0 before the state was served
        closed = end < n or closes_last
        if closed and not (opening and initial is None):
            if length < (min_up if sequence[index] else min_down):
                return False
        index = end
    return True


def brute_force(value_on, min_up, min_down, startup_cost, shutdown_cost, cyclic, initial=None):
    best = -np.inf
    for bits in itertools.product((0, 1), repeat=len(value_on)):
        if not obeys_min_times(bits, min_up, min_down, cyclic, initial):
            continue
        best = max(
            best,
            schedule_value(bits, value_on, startup_cost, shutdown_cost, cyclic, initial),
        )
    return best


@pytest.mark.parametrize("cyclic", [True, False])
def test_dp_matches_exhaustive_enumeration(cyclic):
    """The DP claims to be exact, so hold it to every schedule there is."""
    rng = np.random.default_rng(7)
    for _ in range(40):
        hours = int(rng.integers(4, 11))
        value_on = np.round(rng.normal(0.0, 40.0, size=hours), 2)
        min_up, min_down = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        startup, shutdown = float(rng.integers(0, 120)), float(rng.integers(0, 40))
        states = [None] if cyclic else [None, (1, 2), (0, 3)]
        for initial in states:
            u = single_unit_schedule(
                value_on, min_up, min_down, startup, shutdown, cyclic, initial
            )
            assert obeys_min_times(u, min_up, min_down, cyclic, initial)
            assert schedule_value(
                u, value_on, startup, shutdown, cyclic, initial
            ) == pytest.approx(
                brute_force(
                    value_on, min_up, min_down, startup, shutdown, cyclic, initial
                )
            )


def test_dp_respects_minimum_up_and_down_times():
    """A one-hour spike cannot buy a one-hour run out of a four-hour unit."""
    value_on = np.array([-10.0] * 6 + [900.0] + [-10.0] * 5)
    u = single_unit_schedule(
        value_on, min_up=4, min_down=3, startup_cost=0.0, cyclic=False
    )
    assert u.sum() == 4.0  # the spike is worth taking, but only in fours
    assert u[6] == 1.0
    assert obeys_min_times(u, 4, 3, cyclic=False)

    # Two spikes two hours apart: the gap is shorter than the minimum down
    # time, so the unit either runs through it or skips one of them.
    value_on = np.array([-10.0] * 4 + [900.0, -10.0, -10.0, 900.0] + [-10.0] * 4)
    u = single_unit_schedule(
        value_on, min_up=2, min_down=3, startup_cost=0.0, cyclic=False
    )
    assert obeys_min_times(u, 2, 3, cyclic=False)
    assert u[5] == u[6] == 1.0  # cheaper to idle through than to shut down


def test_dp_declines_a_start_it_cannot_pay_for():
    """The question a merit-order stack cannot ask, in one test.

    Two profitable hours are worth 300; the unit's minimum up time drags a
    third, loss-making hour along with them. Whether to start turns on the
    startup cost, not on whether the capacity is wanted.
    """
    value_on = np.array([-40.0, -60.0, 150.0, 150.0, -40.0, -40.0, -40.0])
    # Starting from a unit that is off and free to start: a window with no
    # carried state opens for free, which would give the start away.
    common = dict(min_up=3, min_down=2, cyclic=False, initial_state=(0, 9))

    profitable = single_unit_schedule(value_on, startup_cost=100.0, **common)
    assert profitable.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]

    # 300 - 40 of running profit against a 400 start: decline it entirely.
    declined = single_unit_schedule(value_on, startup_cost=400.0, **common)
    assert declined.sum() == 0.0


def test_dp_measures_a_run_across_the_cyclic_seam():
    """Hour 0's predecessor is the last hour, so a run may wrap — once."""
    value_on = np.array([80.0, 80.0] + [-100.0] * 8 + [80.0, 80.0])
    wrapped = single_unit_schedule(
        value_on, min_up=4, min_down=2, startup_cost=30.0, cyclic=True
    )
    # The four profitable hours form one legal run across the seam and are
    # charged one startup; as two separate two-hour runs they would be
    # illegal, and extending either to four hours would cost more than it
    # earns.
    assert wrapped.tolist() == [1.0, 1.0] + [0.0] * 8 + [1.0, 1.0]
    assert obeys_min_times(wrapped, 4, 2, cyclic=True)

    # Break the wrap and the same profile cannot assemble that run: the
    # leading pair would have to close inside the window, which costs two
    # loss-making hours it never recovers. Only the trailing pair — whose
    # obligation runs past the horizon — survives.
    linear = single_unit_schedule(
        value_on, min_up=4, min_down=2, startup_cost=30.0, cyclic=False,
        initial_state=(0, 9),
    )
    assert linear.tolist() == [0.0] * 10 + [1.0, 1.0]


def test_dp_honours_a_carried_obligation():
    """A window that starts mid-run may not shut down before it is served."""
    value_on = np.array([-100.0] * 8)
    held = single_unit_schedule(
        value_on, min_up=5, min_down=2, cyclic=False, initial_state=(1, 2)
    )
    # Two hours of the five-hour obligation are already served, so three
    # loss-making hours are compulsory and the fourth is not.
    assert held.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    value_on = np.array([100.0] * 8)
    blocked = single_unit_schedule(
        value_on, min_up=2, min_down=6, cyclic=False, initial_state=(0, 4)
    )
    assert blocked.tolist() == [0.0, 0.0] + [1.0] * 6


def test_dp_is_free_to_decommit():
    """The property the whole exercise exists for: it can also say *off*."""
    value_on = np.array([-5.0] * 12)
    assert single_unit_schedule(value_on, 1, 1, cyclic=True).sum() == 0.0
    assert single_unit_schedule(-value_on, 1, 1, cyclic=True).sum() == 12.0


# ------------------------------------------------------------- across a fleet


def priced_system(hours=24):
    """One node, one cheap unit and one dear one, flat load."""
    return make_system(
        [
            gen("cheap", "A", 10, 200, min_mw=50, startup_cost=1000,
                no_load_cost=200, min_up=3, min_down=2),
            gen("dear", "A", 90, 100, min_mw=20, startup_cost=200, no_load_cost=50),
        ],
        {"A": [120.0] * hours},
    )


def test_commitment_value_prices_output_at_the_better_end():
    system = priced_system()
    hours = list(range(4))
    prices = pd.DataFrame({"A": [200.0, 200.0, 0.0, 0.0]}, index=hours)
    value = commitment_value(system, "cheap", hours, prices)
    # Above marginal cost: full output. Below: minimum stable level, which
    # is a loss the unit takes only because it must be somewhere.
    assert value[0] == pytest.approx((200 - 10) * 200 - 200)
    assert value[2] == pytest.approx((0 - 10) * 50 - 200)


def test_a_ceiling_caps_the_upside_but_never_the_minimum():
    """The regression that made the DP hold runs open through troughs.

    Capping both ends prices a loss-making hour as if the unit could sit
    through it at zero output. It cannot: a committed unit carries its
    minimum stable level, and that loss is the reason to shut down.
    """
    system = priced_system()
    hours = list(range(2))
    prices = pd.DataFrame({"A": [200.0, 0.0]}, index=hours)
    capped = commitment_value(
        system, "cheap", hours, prices, ceiling=np.array([60.0, 60.0])
    )
    assert capped[0] == pytest.approx((200 - 10) * 60 - 200)  # upside capped
    assert capped[1] == pytest.approx((0 - 10) * 50 - 200)  # floor untouched


def test_dispatch_ceiling_unscales_a_fractional_commitment():
    relaxation = pd.DataFrame({"g": [0.5, 1.0, 0.0]})
    output = pd.DataFrame({"g": [60.0, 90.0, 0.0]})
    ceiling = dispatch_ceiling(relaxation, output)
    # Half committed and running 60 MW is one unit running 120.
    assert ceiling["g"].tolist() == [120.0, 90.0, 0.0]


def test_restricting_to_decommitments_never_adds_a_commitment():
    reference = pd.DataFrame({"g": [1.0, 1.0, 0.0, 0.0]})
    schedule = pd.DataFrame({"g": [0.0, 1.0, 1.0, 0.0]})
    kept = restrict_to_decommitments(schedule, reference)
    assert kept["g"].tolist() == [0.0, 1.0, 0.0, 0.0]

    # ...and with a contested mask, only where the relaxation was unsure.
    contested = pd.DataFrame({"g": [False, True, True, True]})
    guarded = restrict_to_decommitments(schedule, reference, contested)
    assert guarded["g"].tolist() == [1.0, 1.0, 0.0, 0.0]


def test_dp_commitment_leaves_clusters_to_the_fallback():
    """A count is not a binary, and one price cannot resolve one."""
    system = priced_system()
    system.generators.loc["dear", "num_units"] = 3.0
    hours = list(range(24))
    prices = pd.DataFrame({"A": [500.0] * 24}, index=hours)
    fallback = pd.DataFrame(
        {"cheap": [0.0] * 24, "dear": [1.6] * 24}, index=hours
    )
    schedule, notes = dp_commitment(
        system, ["cheap", "dear"], hours, prices, cyclic=True, fallback=fallback
    )
    assert notes == {"dp_units": 1, "dp_clustered_units": 1, **{"dp_seconds": notes["dp_seconds"]}}
    # The cluster keeps the rounded relaxation; the single unit is DP'd.
    assert schedule["dear"].tolist() == [2.0] * 24
    assert schedule["cheap"].tolist() == [1.0] * 24


def test_nodal_prices_reject_a_solve_that_carried_no_duals():
    system = priced_system()
    with pytest.raises(RuntimeError):
        nodal_prices(None, None, list(range(4)))


# ---------------------------------------------------- inside the lp heuristic


def stranded_system(hours=48):
    """Load behind a thin line, so capacity adequacy has real work to do."""
    return make_system(
        [
            gen("cheap_a", "A", 8, 600, min_mw=200, startup_cost=5000,
                no_load_cost=300, min_up=3, min_down=2),
            gen("dear_b", "B", 60, 400, min_mw=100, startup_cost=800,
                no_load_cost=100, min_up=2, min_down=2),
        ],
        {"A": [150.0] * hours, "B": [320.0] * hours},
        lines=[line("A", "B", 100.0, 0.02, name="AB")],
    )


def test_dp_rounding_is_off_by_default():
    """Today's behaviour has to survive the option being added."""
    system = stranded_system()
    hours = list(range(48))
    plain = lp_relaxation_guess(system, RunConfig(), hours)
    assert "dp_units" not in plain.notes


@pytest.mark.parametrize("bad", [("dp_output", "vibes"), ("dp_scope", "vibes")])
def test_an_unknown_setting_is_refused(bad):
    system = stranded_system(hours=24)
    with pytest.raises(ValueError, match=bad[0]):
        lp_relaxation_guess(
            system, RunConfig(), list(range(24)), dp_rounding=True, **{bad[0]: bad[1]}
        )


@pytest.mark.parametrize("scope", ["decommit", "contested"])
def test_restricted_scopes_deliver_an_adequate_schedule(scope):
    """The narrowed directions still go through the same safety net."""
    system = stranded_system()
    hours = list(range(48))
    limited = lp_relaxation_guess(
        system, RunConfig(), hours, dp_rounding=True, dp_scope=scope
    )
    assert limited.notes["dp_scope"] == scope
    assert limited.commitment.isin([0.0, 1.0]).all().all()
    _, missing = enforce_adequacy(limited.commitment, system, hours)
    assert missing == 0


def test_dp_guess_is_still_capacity_adequate():
    """The DP can decommit, so the safety net matters more here, not less.

    An under-committed guess is not merely inaccurate: the solver repairs it
    by shedding load at VOLL, which costs multiples of the over-commitment
    the DP exists to remove.
    """
    system = stranded_system()
    hours = list(range(48))
    guess = lp_relaxation_guess(system, RunConfig(), hours, dp_rounding=True)
    assert guess.notes["dp_units"] == 2
    assert guess.commitment.isin([0.0, 1.0]).all().all()
    _, missing = enforce_adequacy(guess.commitment, system, hours)
    assert missing == 0
    # Nothing can serve node B but the unit behind the line.
    assert (guess.commitment["dear_b"] == 1.0).all()


def test_dp_guess_respects_min_up_down_across_the_wrap():
    system = stranded_system()
    hours = list(range(48))
    guess = lp_relaxation_guess(system, RunConfig(), hours, dp_rounding=True)
    for unit in guess.commitment.columns:
        assert obeys_min_times(
            guess.commitment[unit].tolist(),
            int(system.generators.at[unit, "min_up_time_hr"]),
            int(system.generators.at[unit, "min_down_time_hr"]),
            cyclic=True,
        )


def test_the_option_travels_through_run_config_untouched():
    """``heuristic_options`` is the whole delivery path — no dispatcher change."""
    system = stranded_system(hours=24)
    plain = run(system, RunConfig(unit_commitment=True))
    guided = run(
        system,
        RunConfig(
            unit_commitment=True,
            heuristic="lp",
            heuristic_options={"dp_rounding": True},
        ),
    )
    assert guided.guess.notes["dp_units"] == 2
    # A warm start is exact whatever built it.
    assert guided.total_cost == pytest.approx(plain.total_cost, rel=1e-4)


def test_dp_guess_withdraws_confidence_where_it_overrules_the_relaxation():
    """A fixing may only ever be staked on a value the LP itself resolved."""
    system = stranded_system()
    hours = list(range(48))
    guess = lp_relaxation_guess(system, RunConfig(), hours, dp_rounding=True)
    disagrees = guess.commitment != guess.relaxation.round()
    assert not (guess.certain & disagrees).to_numpy().any()
