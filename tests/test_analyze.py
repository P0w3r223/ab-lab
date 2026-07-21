"""Analysis functions checked against statsmodels and against hand arithmetic."""

from __future__ import annotations

import numpy as np
import pytest
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

from ab_lab.analyze import bootstrap_diff, mann_whitney, proportion_test, welch_t_test


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260721)


@pytest.fixture
def two_normal_samples(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    control = rng.normal(100.0, 20.0, 400)
    treatment = rng.normal(103.0, 25.0, 380)
    return control, treatment


def test_welch_p_value_and_statistic_match_statsmodels(two_normal_samples):
    control, treatment = two_normal_samples
    comparison = CompareMeans(DescrStatsW(treatment), DescrStatsW(control))
    expected_statistic, expected_p, _ = comparison.ttest_ind(usevar="unequal")

    result = welch_t_test(control, treatment)

    assert result.statistic == pytest.approx(expected_statistic, rel=1e-12)
    assert result.p_value == pytest.approx(expected_p, rel=1e-12)


def test_welch_interval_matches_statsmodels(two_normal_samples):
    control, treatment = two_normal_samples
    comparison = CompareMeans(DescrStatsW(treatment), DescrStatsW(control))
    expected_low, expected_high = comparison.tconfint_diff(alpha=0.05, usevar="unequal")

    result = welch_t_test(control, treatment, alpha=0.05)

    assert result.ci is not None
    assert result.ci.low == pytest.approx(expected_low, rel=1e-12)
    assert result.ci.high == pytest.approx(expected_high, rel=1e-12)
    assert result.ci.level == 0.95


def test_welch_estimate_is_treatment_minus_control(two_normal_samples):
    control, treatment = two_normal_samples
    result = welch_t_test(control, treatment)
    assert result.estimate == pytest.approx(treatment.mean() - control.mean())
    assert result.estimate > 0


def test_one_sided_welch_halves_the_p_value_in_the_expected_direction(two_normal_samples):
    control, treatment = two_normal_samples
    two_sided = welch_t_test(control, treatment).p_value
    greater = welch_t_test(control, treatment, alternative="greater").p_value
    assert greater == pytest.approx(two_sided / 2.0, rel=1e-12)


def test_a_wider_interval_is_the_price_of_more_confidence(two_normal_samples):
    control, treatment = two_normal_samples
    narrow = welch_t_test(control, treatment, alpha=0.10).ci
    wide = welch_t_test(control, treatment, alpha=0.01).ci
    assert wide.low < narrow.low and wide.high > narrow.high


def test_proportion_p_value_matches_statsmodels():
    result = proportion_test(
        control_successes=480, control_total=10_000,
        treatment_successes=545, treatment_total=10_000,
    )
    expected_statistic, expected_p = proportions_ztest(
        count=[545, 480], nobs=[10_000, 10_000]
    )
    assert result.statistic == pytest.approx(expected_statistic, rel=1e-12)
    assert result.p_value == pytest.approx(expected_p, rel=1e-12)


def test_proportion_interval_is_the_unpooled_wald_interval():
    """Hand arithmetic, so the interval is pinned to a formula and not to a library."""
    result = proportion_test(480, 10_000, 545, 10_000, alpha=0.05)

    rate_control, rate_treatment = 0.0480, 0.0545
    standard_error = np.sqrt(
        rate_control * (1 - rate_control) / 10_000
        + rate_treatment * (1 - rate_treatment) / 10_000
    )
    margin = 1.959963984540054 * standard_error

    assert result.estimate == pytest.approx(0.0065, abs=1e-12)
    assert result.ci.low == pytest.approx(0.0065 - margin, rel=1e-12)
    assert result.ci.high == pytest.approx(0.0065 + margin, rel=1e-12)


def test_proportion_test_rejects_impossible_counts():
    with pytest.raises(ValueError, match="control_successes must be in"):
        proportion_test(101, 100, 50, 100)
    with pytest.raises(ValueError, match="treatment_total must be positive"):
        proportion_test(10, 100, 0, 0)


def test_proportion_test_reports_a_degenerate_experiment():
    with pytest.raises(ValueError, match="no variation in either group"):
        proportion_test(0, 500, 0, 500)


@pytest.mark.parametrize(
    ("control", "treatment"),
    [
        (np.zeros(50), np.zeros(50)),          # nothing happened in either arm
        (np.zeros(50), np.ones(50)),           # everyone converted in exactly one arm
    ],
)
def test_welch_refuses_two_constant_groups(control, treatment):
    """Regression: the Welch-Satterthwaite df is 0/0 here. Left alone the test
    returned a NaN p-value that reads as 'not significant', or a p-value of
    exactly 0 from two degenerate samples. A sparse binary metric reaches this."""
    with pytest.raises(ValueError, match="no variance to test against"):
        welch_t_test(control, treatment)


def test_mann_whitney_estimates_probability_of_superiority():
    control = np.array([1.0, 2.0, 3.0, 4.0])
    treatment = np.array([5.0, 6.0, 7.0, 8.0])
    result = mann_whitney(control, treatment)
    # Every treatment value beats every control value.
    assert result.estimate == pytest.approx(1.0)
    assert result.ci is None


def test_mann_whitney_sees_a_shift_the_t_test_misses(rng):
    """A heavy-tailed metric: the ranks move, the means drown in the tail."""
    control = rng.lognormal(0.0, 1.5, 300)
    treatment = rng.lognormal(0.45, 1.5, 300)

    assert mann_whitney(control, treatment).p_value < 0.01
    assert welch_t_test(control, treatment).p_value > mann_whitney(control, treatment).p_value


def test_bootstrap_mean_agrees_with_welch_on_well_behaved_data(two_normal_samples, rng):
    control, treatment = two_normal_samples
    welch = welch_t_test(control, treatment)
    boot = bootstrap_diff(control, treatment, n_resamples=4_000, rng=rng)

    # Endpoints are compared against the interval's own width: an endpoint that
    # sits near zero would otherwise fail a relative check on pure resampling
    # noise, which says nothing about whether the two methods agree.
    tolerance = 0.05 * (welch.ci.high - welch.ci.low)
    assert boot.estimate == pytest.approx(welch.estimate, rel=1e-12)
    assert boot.ci.low == pytest.approx(welch.ci.low, abs=tolerance)
    assert boot.ci.high == pytest.approx(welch.ci.high, abs=tolerance)


def test_bootstrap_is_reproducible_for_a_given_seed(two_normal_samples):
    control, treatment = two_normal_samples
    first = bootstrap_diff(control, treatment, n_resamples=1_000, rng=np.random.default_rng(7))
    second = bootstrap_diff(control, treatment, n_resamples=1_000, rng=np.random.default_rng(7))
    assert first == second


def test_bootstrap_handles_a_statistic_no_closed_form_test_covers(rng):
    """The median: the reason the bootstrap earns its resampling cost."""
    control = rng.lognormal(0.0, 1.0, 500)
    treatment = rng.lognormal(0.6, 1.0, 500)
    result = bootstrap_diff(control, treatment, statistic=np.median, n_resamples=2_000, rng=rng)

    assert "median" in result.test
    assert result.ci.low > 0.0
    assert result.p_value < 0.05


def test_bootstrap_p_value_cannot_beat_its_own_resolution(rng):
    """A guard against reading 'p < 0.001' off a 1000-resample bootstrap."""
    control = rng.normal(0.0, 1.0, 300)
    treatment = rng.normal(5.0, 1.0, 300)
    result = bootstrap_diff(control, treatment, n_resamples=1_000, rng=rng)

    assert result.p_value == pytest.approx(2.0 / 1_001.0)
    assert "resolution" in " ".join(result.assumptions)


def test_bootstrap_finds_no_effect_when_there_is_none(rng):
    control = rng.normal(0.0, 1.0, 400)
    treatment = rng.normal(0.0, 1.0, 400)
    result = bootstrap_diff(control, treatment, n_resamples=2_000, rng=rng)
    assert result.ci.low < 0.0 < result.ci.high
    assert not result.ci.excludes_zero


def test_bootstrap_rejects_a_resample_count_too_small_to_mean_anything(two_normal_samples):
    control, treatment = two_normal_samples
    with pytest.raises(ValueError, match="n_resamples must be at least 100"):
        bootstrap_diff(control, treatment, n_resamples=50)


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (np.array([1.0]), "at least 2 observations"),
        (np.array([[1.0, 2.0], [3.0, 4.0]]), "one-dimensional"),
        (np.array([1.0, np.nan, 3.0]), "NaN or infinite"),
    ],
)
def test_every_test_rejects_malformed_input(bad, message):
    good = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match=message):
        welch_t_test(bad, good)
    with pytest.raises(ValueError, match=message):
        mann_whitney(good, bad)


def test_assumptions_travel_with_the_result(two_normal_samples):
    """The estimate is not the whole result - the caveats ship with it."""
    control, treatment = two_normal_samples
    for result in (
        welch_t_test(control, treatment),
        proportion_test(480, 10_000, 545, 10_000),
        mann_whitney(control, treatment),
    ):
        assert result.assumptions
        assert all(isinstance(line, str) and line for line in result.assumptions)
