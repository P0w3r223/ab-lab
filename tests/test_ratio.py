"""Ratio metrics, checked against a component this package has already verified.

There is no `statsmodels` oracle for a delta-method ratio test, so the oracle
here is `cluster_robust_t_test` - which *is* checked against statsmodels, to
1e-12, in `test_cluster.py`. Setting every denominator to one turns a ratio of
totals into a mean, so the two must agree; a special case that fails to reduce
is the clearest sign a general case is wrong.

The simulations answer two separate questions, and the second one has an answer
that is easy to assume and wrong. Validity: does the A/A rate hold at alpha?
Yes. Does the choice of estimand matter? **Only when exposure predicts the
rate.** When a unit's rate is independent of how much it is exposed, the ratio
of totals and the mean of per-unit ratios agree to the fourth decimal and both
analyses have the same power - measured here rather than asserted, because the
first version of this file claimed a difference that was not there.
"""

from __future__ import annotations

import numpy as np
import pytest

from ab_lab.cluster import ClusteredSample, cluster_robust_t_test
from ab_lab.ratio import RatioSample, ratio_metric_test
from ab_lab.results import TestResult as _TestResult
from ab_lab.simulate import (
    clustered_ratio_draw,
    naive_ratio_p_value,
    poisson_cluster_size,
    ratio_p_value,
    run_ratio_experiments,
)

ALPHA = 0.05
RUNS = 1_500


def _exposed(lift: float = 0.0, scales: bool = False):
    """Unequal exposure per unit, which is what makes a ratio a ratio."""
    return clustered_ratio_draw(
        n_clusters_per_group=400,
        trials_per_cluster=poisson_cluster_size(mean_size=12.0),
        base_rate=0.20,
        rate_dispersion=0.08,
        absolute_lift=lift,
        lift_scales_with_exposure=scales,
    )


def test_a_denominator_of_one_is_the_cluster_robust_t_test():
    """The reduction that makes the general case believable.

    `y / 1` is `y`, and the linearised contribution `y - R*1` is the residual
    `y - ybar`, so every quantity must match. The test statistic is compared to
    a relative tolerance rather than bit-for-bit: the ratio path divides by a
    mean denominator of exactly 1.0, which is one extra floating-point operation
    and lands one unit in the last place away.
    """
    rng = np.random.default_rng(3)
    control_values, treatment_values = rng.normal(0.0, 1.0, 300), rng.normal(0.2, 1.0, 300)
    control_ids = np.repeat(np.arange(60), 5)
    treatment_ids = np.repeat(np.arange(60), 5) + 1_000

    as_ratio = ratio_metric_test(
        RatioSample.from_arrays(control_values, np.ones(300), control_ids),
        RatioSample.from_arrays(treatment_values, np.ones(300), treatment_ids),
    )
    as_cluster = cluster_robust_t_test(
        ClusteredSample.from_arrays(control_values, control_ids),
        ClusteredSample.from_arrays(treatment_values, treatment_ids),
    )

    assert as_ratio.estimate == as_cluster.estimate
    assert as_ratio.p_value == as_cluster.p_value
    assert as_ratio.df == as_cluster.df
    assert as_ratio.statistic == pytest.approx(as_cluster.statistic, rel=1e-12)
    assert isinstance(as_ratio, _TestResult)


def test_the_estimand_is_the_ratio_of_totals_not_the_mean_of_ratios():
    """Two users, wildly unequal exposure, chosen so the difference is obvious.

    One user clicks 1 of 1; the other clicks 10 of 100. The ratio of totals is
    11/101 = 10.9%, because it weights a user by their impressions. The mean of
    per-user rates is (100% + 10%) / 2 = 55%, because it does not. Both are
    defensible numbers; only one of them is the click-through rate.
    """
    sample = RatioSample.from_arrays(
        numerator=[1.0, 10.0], denominator=[1.0, 100.0], cluster_ids=[0, 1]
    )
    assert sample.ratio == pytest.approx(11.0 / 101.0)
    per_unit_mean = float(np.mean(sample.numerator / sample.denominator))
    assert per_unit_mean == pytest.approx(0.55)
    assert abs(sample.ratio - per_unit_mean) > 0.4


def test_the_result_reports_the_relative_effect_beside_the_absolute_one():
    # Alternating counts rather than identical ones: with every unit the same
    # there is no between-unit variance and the test correctly refuses to run.
    # The totals are chosen so the two ratios are exactly 0.20 and 0.24.
    control = RatioSample.from_arrays([18.0, 22.0] * 25, [100.0] * 50, list(range(50)))
    treatment = RatioSample.from_arrays(
        [22.0, 26.0] * 25, [100.0] * 50, list(range(100, 150))
    )
    result = ratio_metric_test(control, treatment)

    assert result.control_ratio == pytest.approx(0.20)
    assert result.treatment_ratio == pytest.approx(0.24)
    assert result.estimate == pytest.approx(0.04)
    assert result.relative_effect == pytest.approx(0.20)
    assert result.n_clusters == 100
    assert any("ratio of totals" in note for note in result.assumptions)


def test_the_contributions_sum_to_zero_by_construction():
    """R is defined as the value that makes them, so they need no centring."""
    sample = RatioSample.from_arrays([3.0, 7.0, 1.0], [10.0, 20.0, 4.0], [0, 1, 2])
    assert float(sample.contributions.sum()) == pytest.approx(0.0, abs=1e-12)


def test_the_delta_method_holds_its_nominal_error_rate():
    summary = run_ratio_experiments(
        _exposed(),
        ratio_p_value,
        RUNS,
        np.random.default_rng(701),
        alpha=ALPHA,
        label="A/A ratio metric, unequal exposure",
    )
    assert summary.agrees_with(ALPHA)


def test_when_exposure_does_not_predict_the_rate_the_two_analyses_agree():
    """A finding, and the reason the next test exists.

    It is tempting to assume the per-unit t-test must be worse. With a lift that
    lands equally on everybody it is not: both estimands move by the same amount
    and the two tests have indistinguishable power. Asserting a difference here
    would be asserting something false.
    """
    delta = run_ratio_experiments(
        _exposed(lift=0.02), ratio_p_value, RUNS, np.random.default_rng(702), alpha=ALPHA
    )
    per_unit = run_ratio_experiments(
        _exposed(lift=0.02), naive_ratio_p_value, RUNS, np.random.default_rng(702), alpha=ALPHA
    )
    combined = (delta.monte_carlo_error**2 + per_unit.monte_carlo_error**2) ** 0.5
    assert abs(delta.rejection_rate - per_unit.rejection_rate) < 3.0 * combined


def test_when_engaged_users_respond_more_the_two_analyses_answer_differently():
    """The case the delta method exists for.

    A treatment that helps engaged users more moves the business metric - the
    ratio of totals, which weights by exposure - further than it moves the
    average of per-user rates. The analysis targeting the business metric finds
    it correspondingly more often.
    """
    delta = run_ratio_experiments(
        _exposed(lift=0.02, scales=True),
        ratio_p_value,
        RUNS,
        np.random.default_rng(703),
        alpha=ALPHA,
    )
    per_unit = run_ratio_experiments(
        _exposed(lift=0.02, scales=True),
        naive_ratio_p_value,
        RUNS,
        np.random.default_rng(703),
        alpha=ALPHA,
    )
    combined = (delta.monte_carlo_error**2 + per_unit.monte_carlo_error**2) ** 0.5
    assert delta.rejection_rate > per_unit.rejection_rate + 4.0 * combined


@pytest.mark.parametrize(
    ("numerator", "denominator", "ids", "message"),
    [
        ([[1.0]], [[1.0]], [[0]], "one-dimensional"),
        ([1.0, 2.0], [1.0], [0, 1], "same shape"),
        ([1.0, np.inf], [1.0, 1.0], [0, 1], "NaN or infinite"),
        ([1.0, 2.0], [1.0, -1.0], [0, 1], "non-negative"),
        ([1.0, 2.0], [0.0, 0.0], [0, 1], "sums to zero"),
        ([1.0, 2.0], [1.0, 1.0], [0.5, 1.5], "integers"),
    ],
)
def test_a_ratio_sample_refuses_data_it_cannot_interpret(numerator, denominator, ids, message):
    with pytest.raises(ValueError, match=message):
        RatioSample.from_arrays(numerator, denominator, ids)


def test_a_unit_cannot_be_in_both_arms():
    control = RatioSample.from_arrays([1.0, 2.0], [4.0, 4.0], [1, 2])
    treatment = RatioSample.from_arrays([2.0, 3.0], [4.0, 4.0], [2, 3])
    with pytest.raises(ValueError, match="both arms"):
        ratio_metric_test(control, treatment)


def test_one_unit_per_arm_is_refused_rather_than_estimated():
    control = RatioSample.from_arrays([1.0, 2.0], [4.0, 4.0], [7, 7])
    treatment = RatioSample.from_arrays([2.0, 3.0], [4.0, 4.0], [8, 8])
    with pytest.raises(ValueError, match="at least 2 clusters"):
        ratio_metric_test(control, treatment)
