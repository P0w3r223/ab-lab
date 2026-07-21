"""The mSPRT: its algebra checked by numerical integration, its promise by simulation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate, stats

from ab_lab.sequential import (
    SequentialMonitor,
    always_valid_p_value,
    log_mixture_likelihood_ratio,
    mixture_likelihood_ratio,
    msprt,
    tau_from_mde,
)


def _likelihood_ratio_by_integration(estimate: float, variance: float, tau: float) -> float:
    """The definition, computed the slow way: integrate the mixture directly.

    This is the non-circular check on the closed form - if the algebra were
    wrong, quadrature would disagree.
    """
    scale = np.sqrt(variance)

    def integrand(effect: float) -> float:
        return stats.norm.pdf(estimate, loc=effect, scale=scale) * stats.norm.pdf(
            effect, loc=0.0, scale=tau
        )

    # Finite limits with the two peaks declared: the integrand is a product of
    # two narrow bells, and adaptive quadrature on an infinite interval can
    # step straight over them.
    limit = 12.0 * tau + 8.0 * scale + abs(estimate)
    numerator, _ = integrate.quad(integrand, -limit, limit, points=[0.0, estimate], limit=200)
    denominator = stats.norm.pdf(estimate, loc=0.0, scale=scale)
    return numerator / denominator


@pytest.mark.parametrize("estimate", [0.0, 0.002, 0.01, 0.05])
@pytest.mark.parametrize(("variance", "tau"), [(1e-5, 0.01), (4e-4, 0.02), (1e-3, 0.005)])
def test_closed_form_matches_numerical_integration(estimate, variance, tau):
    expected = _likelihood_ratio_by_integration(estimate, variance, tau)
    assert mixture_likelihood_ratio(estimate, variance, tau) == pytest.approx(expected, rel=1e-8)


def test_no_observed_difference_is_evidence_for_the_null():
    """With zero difference the ratio drops below one: the mixture is penalised
    for spreading its bet over effects that did not show up."""
    ratio = mixture_likelihood_ratio(estimate=0.0, variance=1e-4, tau=0.01)
    assert ratio < 1.0
    assert always_valid_p_value(0.0, 1e-4, 0.01) == 1.0


def test_evidence_grows_with_the_observed_effect():
    ratios = [mixture_likelihood_ratio(effect, 1e-4, 0.01) for effect in (0.0, 0.01, 0.03, 0.06)]
    assert ratios == sorted(ratios)
    assert ratios[-1] > 1_000


def test_the_always_valid_p_value_is_more_conservative_than_a_fixed_horizon_one():
    """The price of being allowed to peek, made visible on one dataset."""
    rng = np.random.default_rng(3)
    control = rng.normal(0.0, 1.0, 2_000)
    treatment = rng.normal(0.1, 1.0, 2_000)

    sequential = msprt(control, treatment, tau=0.1).p_value
    fixed = stats.ttest_ind(treatment, control, equal_var=False).pvalue
    assert sequential > fixed


def test_a_large_true_effect_is_still_detected():
    rng = np.random.default_rng(4)
    control = rng.normal(0.0, 1.0, 4_000)
    treatment = rng.normal(0.2, 1.0, 4_000)

    result = msprt(control, treatment, tau=0.2)
    assert result.should_stop
    assert result.estimate == pytest.approx(treatment.mean() - control.mean())
    assert result.n_control == 4_000


def test_monitor_never_walks_back_a_decision():
    """Once the boundary is crossed the p-value stays crossed - otherwise the
    experiment owner sees a win on Tuesday and a non-win on Wednesday."""
    rng = np.random.default_rng(5)
    control = rng.normal(0.0, 1.0, 6_000)
    treatment = rng.normal(0.15, 1.0, 6_000)
    monitor = SequentialMonitor(tau=0.15, alpha=0.05)

    p_values = [
        monitor.look(control[:size], treatment[:size]).p_value
        for size in (500, 1_000, 2_000, 4_000, 6_000)
    ]

    assert p_values == sorted(p_values, reverse=True)
    assert monitor.stopped
    assert len(monitor.looks) == 5


def test_tau_trades_power_between_effect_sizes_not_validity():
    """A tau far from the truth costs detection speed, never correctness."""
    rng = np.random.default_rng(6)
    control = rng.normal(0.0, 1.0, 3_000)
    treatment = rng.normal(0.1, 1.0, 3_000)

    well_matched = msprt(control, treatment, tau=0.1).p_value
    badly_matched = msprt(control, treatment, tau=5.0).p_value
    assert well_matched < badly_matched


def test_tau_from_mde_is_the_mde():
    assert tau_from_mde(0.005) == 0.005
    with pytest.raises(ValueError, match="mde must be positive"):
        tau_from_mde(0.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"variance": 0.0, "tau": 0.1}, "variance must be positive"),
        ({"variance": 1e-4, "tau": 0.0}, "tau must be positive"),
    ],
)
def test_mixture_rejects_degenerate_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        mixture_likelihood_ratio(estimate=0.01, **kwargs)


def test_monitor_rejects_degenerate_configuration():
    with pytest.raises(ValueError, match="tau must be positive"):
        SequentialMonitor(tau=-1.0)
    with pytest.raises(ValueError, match="alpha must be in"):
        SequentialMonitor(tau=0.1, alpha=1.0)


def test_constant_data_fails_loudly():
    with pytest.raises(ValueError, match="no variance to test against"):
        msprt(np.zeros(50), np.zeros(50), tau=0.1)


@pytest.mark.parametrize("z_score", [40.0, 100.0, 1_000.0])
def test_overwhelming_evidence_underflows_rather_than_overflowing(z_score):
    """Regression: the ratio grows like exp(z^2/2) and leaves float range at
    |z| ~ 37.7. Computing it directly raised OverflowError - at the moment the
    test should have been stopping, on the winning arm."""
    variance = 1e-6
    estimate = z_score * np.sqrt(variance)

    assert always_valid_p_value(estimate, variance, tau=0.1) == 0.0
    assert mixture_likelihood_ratio(estimate, variance, tau=0.1) == np.inf
    assert np.isfinite(log_mixture_likelihood_ratio(estimate, variance, tau=0.1))


def test_the_monitor_survives_an_experiment_that_keeps_running_after_it_wins():
    """The same regression, reached the way a user would: a large experiment
    with a real effect, looked at three times."""
    rng = np.random.default_rng(1)
    control = rng.normal(0.0, 1.0, 50_000)
    treatment = rng.normal(0.24, 1.0, 50_000)
    monitor = SequentialMonitor(tau=0.1)

    p_values = [monitor.look(control[:n], treatment[:n]).p_value for n in (5_000, 20_000, 50_000)]

    assert p_values[-1] == 0.0
    assert p_values == sorted(p_values, reverse=True)
    assert monitor.looks[-1].likelihood_ratio == np.inf
    assert monitor.looks[-1].log_likelihood_ratio > 700.0


def test_the_finite_ratio_is_the_exponential_of_the_log_ratio():
    log_ratio = log_mixture_likelihood_ratio(0.02, 1e-4, 0.01)
    assert mixture_likelihood_ratio(0.02, 1e-4, 0.01) == pytest.approx(np.exp(log_ratio))
