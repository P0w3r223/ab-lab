"""Simulation validation - the claim that this package computes what it says.

Reference implementations agree with the *formulas*. These tests check the
formulas against generated worlds where the true answer is known: no effect
means the rejection rate must equal alpha, a known effect means it must equal
the power that :mod:`ab_lab.power` promised.

Every assertion is expressed in multiples of the Monte Carlo error so that the
suite fails on a broken method, not on an unlucky seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from ab_lab.power import power_t, power_z, sample_size_for_proportion
from ab_lab.sequential import tau_from_mde
from ab_lab.simulate import (
    binary_draw,
    lognormal_draw,
    msprt_p_value,
    normal_draw,
    peeking_curve,
    proportion_p_value,
    run_experiments,
    run_with_peeking,
    welch_p_value,
)

ALPHA = 0.05


def _within_monte_carlo_error(summary, target: float, sigmas: float = 4.0) -> bool:
    """True when the empirical rate is indistinguishable from ``target``."""
    return abs(summary.rejection_rate - target) < sigmas * summary.monte_carlo_error


def test_welch_holds_its_nominal_type_i_error_on_aa_data():
    summary = run_experiments(
        draw=normal_draw(n_per_group=500, mean=10.0, std_dev=3.0),
        p_value_fn=welch_p_value,
        n_experiments=3_000,
        rng=np.random.default_rng(101),
        alpha=ALPHA,
        label="A/A Welch",
    )
    assert _within_monte_carlo_error(summary, ALPHA)
    assert summary.mean_estimate == pytest.approx(0.0, abs=0.05)


def test_proportion_test_holds_its_nominal_type_i_error_on_aa_data():
    summary = run_experiments(
        draw=binary_draw(n_per_group=3_000, baseline_rate=0.10),
        p_value_fn=proportion_p_value,
        n_experiments=3_000,
        rng=np.random.default_rng(102),
        alpha=ALPHA,
    )
    assert _within_monte_carlo_error(summary, ALPHA)


def test_empirical_power_matches_the_promised_power_for_means():
    n_per_group, effect_size, std_dev = 400, 0.2, 4.0
    summary = run_experiments(
        draw=normal_draw(n_per_group, mean=0.0, std_dev=std_dev,
                         absolute_lift=effect_size * std_dev),
        p_value_fn=welch_p_value,
        n_experiments=3_000,
        rng=np.random.default_rng(103),
        alpha=ALPHA,
    )
    assert _within_monte_carlo_error(summary, power_t(effect_size, n_per_group, alpha=ALPHA))


def test_a_sample_size_solved_for_eighty_percent_power_delivers_it():
    """The full loop: design an experiment, then run it 2 000 times."""
    baseline_rate, mde = 0.10, 0.01
    design = sample_size_for_proportion(baseline_rate, mde, alpha=ALPHA, power=0.8)
    summary = run_experiments(
        draw=binary_draw(design.per_group, baseline_rate, absolute_lift=mde),
        p_value_fn=proportion_p_value,
        n_experiments=2_000,
        rng=np.random.default_rng(104),
        alpha=ALPHA,
    )
    assert _within_monte_carlo_error(summary, 0.8)


def test_the_normal_approximation_is_the_reason_power_z_is_used_for_rates():
    """A cross-check that the two power functions agree where both apply."""
    assert power_z(0.2, 400, alpha=ALPHA) == pytest.approx(
        power_t(0.2, 400, alpha=ALPHA), abs=0.005
    )


def test_peeking_inflates_the_false_positive_rate():
    """The headline result: same data, same test, ten looks instead of one."""
    draw = normal_draw(n_per_group=2_000, mean=0.0, std_dev=1.0)
    rng = np.random.default_rng(105)

    single_look = run_with_peeking(
        draw, welch_p_value, [2_000], 1_500, rng, alpha=ALPHA, label="1 look"
    )
    ten_looks = run_with_peeking(
        draw, welch_p_value, list(range(200, 2_001, 200)), 1_500, rng, alpha=ALPHA
    )

    assert _within_monte_carlo_error(single_look, ALPHA)
    assert ten_looks.rejection_rate > 3 * ALPHA


def test_the_type_i_error_climbs_with_every_extra_look():
    summaries = peeking_curve(
        draw=normal_draw(n_per_group=1_000, mean=0.0, std_dev=1.0),
        p_value_fn=welch_p_value,
        max_n=1_000,
        look_counts=[1, 2, 5, 10],
        n_experiments=1_200,
        rng=np.random.default_rng(106),
        alpha=ALPHA,
    )
    rates = [summary.rejection_rate for summary in summaries]

    assert rates == sorted(rates)
    assert rates[0] < 2 * ALPHA < rates[-1]
    assert [summary.label for summary in summaries] == [
        "1 look(s)", "2 look(s)", "5 look(s)", "10 look(s)"
    ]


def test_peeking_biases_the_effect_it_stops_on():
    """Stopping early on a large difference does not just break the p-value -
    it inflates the effect size that gets reported to the business."""
    draw = normal_draw(n_per_group=1_000, mean=0.0, std_dev=1.0)
    rng = np.random.default_rng(107)

    honest = run_with_peeking(draw, welch_p_value, [1_000], 1_000, rng, alpha=ALPHA)
    peeked = run_with_peeking(
        draw, welch_p_value, list(range(100, 1_001, 100)), 1_000, rng, alpha=ALPHA
    )

    assert abs(peeked.mean_estimate) > abs(honest.mean_estimate)


def test_the_sequential_test_survives_the_same_peeking():
    """mSPRT under continuous monitoring stays at or under alpha - that is the
    whole point, and it is checked rather than asserted in prose."""
    summary = run_with_peeking(
        draw=normal_draw(n_per_group=2_000, mean=0.0, std_dev=1.0),
        p_value_fn=msprt_p_value(tau=tau_from_mde(0.1)),
        look_sizes=list(range(200, 2_001, 200)),
        n_experiments=1_500,
        rng=np.random.default_rng(108),
        alpha=ALPHA,
        label="A/A mSPRT, 10 looks",
    )
    assert summary.rejection_rate <= ALPHA + 3 * summary.monte_carlo_error


def test_the_sequential_test_still_finds_a_real_effect():
    """Validity is worthless without power: the same monitor on a true effect."""
    summary = run_with_peeking(
        draw=normal_draw(n_per_group=2_000, mean=0.0, std_dev=1.0, absolute_lift=0.15),
        p_value_fn=msprt_p_value(tau=tau_from_mde(0.15)),
        look_sizes=list(range(200, 2_001, 200)),
        n_experiments=600,
        rng=np.random.default_rng(109),
        alpha=ALPHA,
    )
    assert summary.rejection_rate > 0.5


def test_a_skewed_metric_strains_the_t_test_at_small_samples():
    """Why the bootstrap exists: with 40 units of lognormal revenue per arm the
    t-test's real false positive rate is not the one on the label."""
    summary = run_experiments(
        draw=lognormal_draw(n_per_group=40, mean_log=0.0, sigma_log=2.0),
        p_value_fn=welch_p_value,
        n_experiments=3_000,
        rng=np.random.default_rng(110),
        alpha=ALPHA,
    )
    assert summary.rejection_rate < ALPHA - 2 * summary.monte_carlo_error


def test_summary_reports_its_own_noise_floor():
    summary = run_experiments(
        draw=normal_draw(n_per_group=50),
        p_value_fn=welch_p_value,
        n_experiments=400,
        rng=np.random.default_rng(111),
    )
    assert summary.n_experiments == 400
    assert 0.0 <= summary.rejection_rate <= 1.0
    assert summary.monte_carlo_error == pytest.approx(
        (summary.rejection_rate * (1 - summary.rejection_rate) / 400) ** 0.5
    )


@pytest.mark.parametrize(
    ("look_sizes", "message"),
    [
        ([], "at least one look"),
        ([1, 100], "at least 2 observations"),
        ([400, 200], "must be increasing"),
        ([200, 5_000], "but the last look needs"),
    ],
)
def test_peeking_run_rejects_incoherent_look_schedules(look_sizes, message):
    with pytest.raises(ValueError, match=message):
        run_with_peeking(
            draw=normal_draw(n_per_group=500),
            p_value_fn=welch_p_value,
            look_sizes=look_sizes,
            n_experiments=5,
            rng=np.random.default_rng(0),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_per_group": 1}, "n_per_group must be at least 2"),
        ({"n_per_group": 10, "std_dev": 0.0}, "std_dev must be positive"),
    ],
)
def test_draws_reject_impossible_specifications(kwargs, message):
    with pytest.raises(ValueError, match=message):
        normal_draw(**kwargs)


def test_binary_draw_rejects_a_rate_outside_the_unit_interval():
    with pytest.raises(ValueError, match="treatment rate must be in"):
        binary_draw(n_per_group=100, baseline_rate=0.98, absolute_lift=0.05)


def test_run_rejects_a_meaningless_number_of_experiments():
    with pytest.raises(ValueError, match="n_experiments must be positive"):
        run_experiments(normal_draw(50), welch_p_value, 0, np.random.default_rng(0))
