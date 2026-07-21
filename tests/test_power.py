"""Design functions checked against statsmodels, then against their own inverse.

statsmodels is the oracle here, never the implementation: if these two agree to
four decimals on the power equation, the equation is right.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize

from ab_lab.power import (
    cohens_h,
    mde_for_mean,
    mde_for_proportion,
    power_t,
    power_z,
    sample_size_for_mean,
    sample_size_for_proportion,
)

ALTERNATIVE_PAIRS = [("two-sided", "two-sided"), ("one-sided", "larger")]


@pytest.mark.parametrize("effect_size", [0.05, 0.2, 0.5, 0.8])
@pytest.mark.parametrize("n_per_group", [30, 200, 5_000])
@pytest.mark.parametrize(("ours", "theirs"), ALTERNATIVE_PAIRS)
def test_power_z_matches_statsmodels(effect_size, n_per_group, ours, theirs):
    expected = NormalIndPower().power(
        effect_size=effect_size, nobs1=n_per_group, alpha=0.05, ratio=1.0, alternative=theirs
    )
    assert power_z(effect_size, n_per_group, alternative=ours) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("effect_size", [0.05, 0.2, 0.5, 0.8])
@pytest.mark.parametrize("n_per_group", [5, 30, 200])
@pytest.mark.parametrize(("ours", "theirs"), ALTERNATIVE_PAIRS)
def test_power_t_matches_statsmodels(effect_size, n_per_group, ours, theirs):
    expected = TTestIndPower().power(
        effect_size=effect_size, nobs1=n_per_group, alpha=0.05, ratio=1.0, alternative=theirs
    )
    assert power_t(effect_size, n_per_group, alternative=ours) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("ratio", [0.5, 1.0, 3.0])
def test_power_handles_unequal_group_sizes(ratio):
    expected = TTestIndPower().power(effect_size=0.3, nobs1=400, alpha=0.05, ratio=ratio)
    assert power_t(0.3, 400, ratio=ratio) == pytest.approx(expected, abs=1e-10)


def test_t_power_is_below_z_power_at_small_samples():
    """Estimating the variance costs power - that cost is what nct captures."""
    assert power_t(0.5, 20) < power_z(0.5, 20)
    # ...and it vanishes as the sample grows.
    assert power_t(0.5, 50_000) == pytest.approx(power_z(0.5, 50_000), abs=1e-6)


@pytest.mark.parametrize(("mde", "std_dev"), [(2.5, 40.0), (0.1, 1.0), (12.0, 100.0)])
def test_sample_size_for_mean_matches_statsmodels(mde, std_dev):
    expected = TTestIndPower().solve_power(
        effect_size=mde / std_dev, alpha=0.05, power=0.8, ratio=1.0, alternative="two-sided"
    )
    result = sample_size_for_mean(mde, std_dev)
    assert result.exact_per_group == pytest.approx(expected, abs=1e-4)
    assert result.per_group == math.ceil(expected)
    assert result.total == 2 * result.per_group


@pytest.mark.parametrize(
    ("baseline_rate", "mde"), [(0.05, 0.005), (0.5, 0.02), (0.012, 0.003), (0.30, -0.03)]
)
def test_sample_size_for_proportion_matches_statsmodels(baseline_rate, mde):
    effect_size = abs(proportion_effectsize(baseline_rate + mde, baseline_rate))
    expected = NormalIndPower().solve_power(
        effect_size=effect_size, alpha=0.05, power=0.8, ratio=1.0, alternative="two-sided"
    )
    result = sample_size_for_proportion(baseline_rate, mde)
    assert result.exact_per_group == pytest.approx(expected, abs=1e-4)


def test_cohens_h_matches_statsmodels_up_to_sign_convention():
    assert cohens_h(0.05, 0.055) == pytest.approx(-proportion_effectsize(0.05, 0.055), abs=1e-12)


def test_sample_size_realises_the_power_it_promises():
    """The round trip: feed the solved n back into the power function."""
    result = sample_size_for_mean(mde=2.5, std_dev=40.0, power=0.8)
    assert power_t(2.5 / 40.0, result.exact_per_group) == pytest.approx(0.8, abs=1e-6)
    # Rounding up can only help.
    assert power_t(2.5 / 40.0, result.per_group) >= 0.8


def test_detecting_a_smaller_effect_costs_more_sample():
    big = sample_size_for_proportion(0.05, 0.010).per_group
    small = sample_size_for_proportion(0.05, 0.005).per_group
    # Halving the detectable effect roughly quadruples the required sample.
    assert small / big == pytest.approx(4.0, rel=0.05)


def test_the_same_lift_is_dearer_near_fifty_percent():
    """A proportion's variance peaks at 0.5, so the same absolute lift needs a
    much larger sample there than on a low-converting baseline."""
    near_zero = sample_size_for_proportion(0.01, 0.005).per_group
    near_half = sample_size_for_proportion(0.50, 0.005).per_group
    assert near_half > 10 * near_zero


@pytest.mark.parametrize("n_per_group", [500, 5_000, 50_000])
def test_mde_for_mean_inverts_sample_size(n_per_group):
    mde = mde_for_mean(n_per_group, std_dev=40.0)
    assert sample_size_for_mean(mde, std_dev=40.0).exact_per_group == pytest.approx(
        n_per_group, rel=1e-6
    )


@pytest.mark.parametrize("baseline_rate", [0.02, 0.15, 0.60])
def test_mde_for_proportion_inverts_sample_size(baseline_rate):
    mde = mde_for_proportion(20_000, baseline_rate)
    assert sample_size_for_proportion(baseline_rate, mde).exact_per_group == pytest.approx(
        20_000, rel=1e-6
    )


@settings(max_examples=50, deadline=None)
@given(
    effect_size=st.floats(min_value=0.01, max_value=2.0),
    n_per_group=st.floats(min_value=5.0, max_value=10_000.0),
    growth=st.floats(min_value=1.01, max_value=10.0),
)
def test_power_is_monotone_in_sample_size(effect_size, n_per_group, growth):
    assert power_t(effect_size, n_per_group * growth) >= power_t(effect_size, n_per_group)
    assert power_z(effect_size, n_per_group * growth) >= power_z(effect_size, n_per_group)


@settings(max_examples=50, deadline=None)
@given(
    effect_size=st.floats(min_value=0.01, max_value=2.0),
    n_per_group=st.floats(min_value=5.0, max_value=10_000.0),
)
def test_power_is_bounded_and_beats_alpha(effect_size, n_per_group):
    power = power_t(effect_size, n_per_group, alpha=0.05)
    assert 0.05 <= power <= 1.0


@settings(max_examples=30, deadline=None)
@given(effect_size=st.floats(min_value=0.05, max_value=1.5))
def test_one_sided_tests_are_more_powerful_than_two_sided(effect_size):
    assert power_t(effect_size, 300, alternative="one-sided") >= power_t(
        effect_size, 300, alternative="two-sided"
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mde": 2.5, "std_dev": 0.0}, "std_dev must be positive"),
        ({"mde": 0.0, "std_dev": 1.0}, "effect_size must be non-zero"),
        ({"mde": 2.5, "std_dev": 1.0, "alpha": 0.0}, "alpha must be in"),
        ({"mde": 2.5, "std_dev": 1.0, "power": 1.0}, "power must be in"),
    ],
)
def test_sample_size_for_mean_rejects_impossible_designs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        sample_size_for_mean(**kwargs)


def test_proportion_design_rejects_rates_outside_the_unit_interval():
    with pytest.raises(ValueError, match=r"must stay in \[0, 1\]"):
        sample_size_for_proportion(0.98, 0.05)


def test_undetectable_effect_fails_loudly_instead_of_running_forever():
    with pytest.raises(ValueError, match="too small to measure"):
        sample_size_for_mean(mde=1e-7, std_dev=1.0)


def test_mde_reports_when_no_lift_is_detectable():
    with pytest.raises(ValueError, match="under-powered"):
        mde_for_proportion(n_per_group=2, baseline_rate=0.5)
