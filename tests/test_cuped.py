"""CUPED: does it buy what the algebra promises, and does the trap bite?

The oracle here is arithmetic rather than a library. Theta is a covariance over
a variance and can be checked against numpy directly; the adjusted difference is
``(Ybar_t - Ybar_c) - theta * (Xbar_t - Xbar_c)`` by construction and can be
checked the same way.

What needs simulating is the claim that matters: ``Var(Y_adjusted)`` is
``Var(Y) * (1 - rho**2)``. That is checked against the *realised* variance
reduction and against realised power, not against a restatement of the formula.

The last two tests are the ones worth having. A covariate the experiment touched
does not announce itself: the estimate shrinks toward zero and the experiment
reports a null, which is indistinguishable from an experiment that simply found
nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from ab_lab.cuped import CupedSample, cuped_t_test, cuped_theta
from ab_lab.results import TestResult as _TestResult
from ab_lab.simulate import (
    covariate_draw,
    cuped_test,
    run_cuped_experiments,
    unadjusted_test,
)

ALPHA = 0.05
RUNS = 2_000
LIFT = 0.10


def test_theta_is_the_covariance_over_the_variance():
    rng = np.random.default_rng(1)
    covariate = rng.normal(0.0, 1.0, 400)
    metric = 0.6 * covariate + rng.normal(0.0, 0.8, 400)
    control = CupedSample.from_arrays(metric[:200], covariate[:200])
    treatment = CupedSample.from_arrays(metric[200:], covariate[200:])

    expected = np.cov(metric, covariate, ddof=1)[0, 1] / covariate.var(ddof=1)
    assert cuped_theta(control, treatment) == pytest.approx(expected, rel=1e-12)


def test_the_adjusted_estimate_is_the_difference_minus_theta_times_the_imbalance():
    """By construction, and worth pinning: this is why the estimate is unbiased.

    Randomisation makes the covariate imbalance small, so the correction to the
    estimate is small - but it is not zero, and knowing which direction it goes
    is how the adjusted and unadjusted figures get reconciled.
    """
    rng = np.random.default_rng(2)
    control = CupedSample.from_arrays(rng.normal(0, 1, 300), rng.normal(0, 1, 300))
    treatment = CupedSample.from_arrays(rng.normal(0.2, 1, 300), rng.normal(0, 1, 300))

    theta = cuped_theta(control, treatment)
    by_hand = (treatment.metric.mean() - control.metric.mean()) - theta * (
        treatment.covariate.mean() - control.covariate.mean()
    )
    result = cuped_t_test(control, treatment)

    assert result.estimate == pytest.approx(by_hand, rel=1e-12)
    assert isinstance(result, _TestResult)


@pytest.mark.parametrize("correlation", [0.3, 0.5, 0.7, 0.9])
def test_the_variance_reduction_is_the_squared_correlation(correlation):
    """The promise, checked against realised variance on a large sample."""
    draw = covariate_draw(n_per_group=4_000, correlation=correlation)
    result = cuped_t_test(*draw(np.random.default_rng(11)))

    assert result.variance_reduction == pytest.approx(correlation**2, abs=0.02)
    assert result.correlation == pytest.approx(correlation, abs=0.02)
    assert result.effective_sample_multiplier == pytest.approx(
        1.0 / (1.0 - correlation**2), rel=0.1
    )


def test_a_useless_covariate_costs_nothing_and_buys_nothing():
    """Correlation zero: the adjustment must be a no-op, not a penalty."""
    draw = covariate_draw(n_per_group=800, correlation=0.0, absolute_lift=LIFT)
    adjusted = run_cuped_experiments(draw, cuped_test, RUNS, np.random.default_rng(21), ALPHA)
    plain = run_cuped_experiments(draw, unadjusted_test, RUNS, np.random.default_rng(21), ALPHA)

    assert adjusted.rejection_rate == pytest.approx(plain.rejection_rate, abs=0.02)


def test_a_correlated_covariate_buys_power_at_the_same_sample_size():
    draw = covariate_draw(n_per_group=800, correlation=0.8, absolute_lift=LIFT)
    adjusted = run_cuped_experiments(draw, cuped_test, RUNS, np.random.default_rng(22), ALPHA)
    plain = run_cuped_experiments(draw, unadjusted_test, RUNS, np.random.default_rng(22), ALPHA)

    combined = (adjusted.monte_carlo_error**2 + plain.monte_carlo_error**2) ** 0.5
    assert adjusted.rejection_rate > plain.rejection_rate + 10.0 * combined
    # And it bought that without moving the answer.
    assert adjusted.mean_estimate == pytest.approx(LIFT, abs=0.01)


def test_the_adjustment_holds_its_nominal_error_rate():
    draw = covariate_draw(n_per_group=800, correlation=0.8)
    summary = run_cuped_experiments(draw, cuped_test, RUNS, np.random.default_rng(23), ALPHA)
    assert summary.agrees_with(ALPHA)


@pytest.mark.parametrize(
    ("leak", "worst_acceptable_bias"),
    [(0.5, -0.02), (1.0, -0.05)],
)
def test_a_covariate_the_experiment_touched_shrinks_the_effect(leak, worst_acceptable_bias):
    """The trap, measured. Half the effect leaking costs about a third of it.

    The bias bounds here are deliberately loose: the point is the *direction* and
    the order of magnitude, and a tight bound would make this a regression test
    for a number nobody should rely on.
    """
    draw = covariate_draw(
        n_per_group=800,
        correlation=0.7,
        absolute_lift=LIFT,
        treatment_leaks_into_covariate=leak,
    )
    summary = run_cuped_experiments(draw, cuped_test, RUNS, np.random.default_rng(31), ALPHA)

    bias = summary.mean_estimate - LIFT
    assert bias < worst_acceptable_bias


def test_the_leak_reports_a_null_rather_than_a_false_positive():
    """Which is why it survives review: it looks like an experiment that failed.

    With the whole effect leaking, the rejection rate collapses. Nothing about
    the output says "your covariate was contaminated" - it says "no effect".
    """
    clean = run_cuped_experiments(
        covariate_draw(n_per_group=800, correlation=0.7, absolute_lift=LIFT),
        cuped_test,
        RUNS,
        np.random.default_rng(32),
        ALPHA,
    )
    leaking = run_cuped_experiments(
        covariate_draw(
            n_per_group=800,
            correlation=0.7,
            absolute_lift=LIFT,
            treatment_leaks_into_covariate=1.0,
        ),
        cuped_test,
        RUNS,
        np.random.default_rng(32),
        ALPHA,
    )
    assert clean.rejection_rate > 0.7
    assert leaking.rejection_rate < 0.3


def test_a_constant_covariate_is_refused_rather_than_divided_by():
    control = CupedSample.from_arrays([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])
    treatment = CupedSample.from_arrays([2.0, 3.0, 4.0], [5.0, 5.0, 5.0])
    with pytest.raises(ValueError, match="constant"):
        cuped_t_test(control, treatment)


def test_metric_and_covariate_must_describe_the_same_units():
    with pytest.raises(ValueError, match="same units"):
        CupedSample.from_arrays([1.0, 2.0, 3.0], [1.0, 2.0])
