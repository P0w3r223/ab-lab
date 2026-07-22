"""The paired bootstrap: what pairing buys, and what it does not.

Two checks in the house style. Agreement: on data with a known answer the interval has to
land where arithmetic says it should. Simulation: the false positive rate under a true null
has to sit at alpha, judged in multiples of the Monte Carlo error rather than against a bare
threshold.
"""

from __future__ import annotations

import numpy as np
import pytest

from ab_lab.analyze import bootstrap_diff, paired_bootstrap

SIMULATIONS = 2_000
RESAMPLES = 2_000
ALPHA = 0.05
# Standard error of a proportion estimated from SIMULATIONS draws.
MONTE_CARLO_SIGMA = (ALPHA * (1 - ALPHA) / SIMULATIONS) ** 0.5


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260722)


@pytest.fixture
def correlated_pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Same units measured twice: large spread between units, small effect within them."""
    unit = rng.normal(100.0, 30.0, 500)
    control = unit + rng.normal(0.0, 2.0, 500)
    treatment = unit + rng.normal(1.5, 2.0, 500)
    return control, treatment


def test_a_constant_shift_is_recovered_exactly(rng: np.random.Generator):
    """Treatment = control + 5 has no within-unit variance, so the interval collapses."""
    control = rng.normal(50.0, 10.0, 300)
    treatment = control + 5.0

    result = paired_bootstrap(control, treatment, n_resamples=RESAMPLES, rng=rng)

    assert result.estimate == pytest.approx(5.0)
    assert result.ci.low == pytest.approx(5.0, abs=1e-9)
    assert result.ci.high == pytest.approx(5.0, abs=1e-9)


def test_pairing_buys_a_narrower_interval_than_treating_the_arms_as_independent(
    correlated_pair, rng: np.random.Generator
):
    """The reason this function exists at all."""
    control, treatment = correlated_pair

    paired = paired_bootstrap(control, treatment, n_resamples=RESAMPLES, rng=rng)
    unpaired = bootstrap_diff(control, treatment, n_resamples=RESAMPLES, rng=rng)

    paired_width = paired.ci.high - paired.ci.low
    unpaired_width = unpaired.ci.high - unpaired.ci.low
    assert paired_width < unpaired_width / 5


def test_pairing_finds_an_effect_the_unpaired_interval_cannot(correlated_pair, rng):
    """Between-unit variance swamps a real 1.5 unit effect until the pairing removes it."""
    control, treatment = correlated_pair

    paired = paired_bootstrap(control, treatment, n_resamples=RESAMPLES, rng=rng)
    unpaired = bootstrap_diff(control, treatment, n_resamples=RESAMPLES, rng=rng)

    assert paired.ci.low > 0.0
    assert unpaired.ci.low < 0.0 < unpaired.ci.high


def test_the_estimate_is_treatment_minus_control(correlated_pair, rng):
    control, treatment = correlated_pair

    result = paired_bootstrap(control, treatment, n_resamples=RESAMPLES, rng=rng)

    assert result.estimate == pytest.approx(treatment.mean() - control.mean())


def test_a_statistic_other_than_the_mean_is_a_difference_of_statistics(rng):
    control = rng.normal(0.0, 1.0, 400)
    treatment = control + rng.normal(2.0, 1.0, 400)

    result = paired_bootstrap(
        control, treatment, statistic=np.median, n_resamples=RESAMPLES, rng=rng
    )

    assert result.test.endswith("(median)")
    assert result.estimate == pytest.approx(np.median(treatment) - np.median(control))


def test_the_false_positive_rate_sits_at_alpha_under_a_true_null(rng):
    """A/A on paired data: the same units, no effect, 2 000 times."""
    rejections = 0
    for _ in range(SIMULATIONS):
        unit = rng.normal(10.0, 5.0, 200)
        control = unit + rng.normal(0.0, 1.0, 200)
        treatment = unit + rng.normal(0.0, 1.0, 200)
        result = paired_bootstrap(control, treatment, n_resamples=400, rng=rng)
        rejections += int(result.is_significant(ALPHA))

    rate = rejections / SIMULATIONS
    assert abs(rate - ALPHA) < 4 * MONTE_CARLO_SIGMA, f"empirical alpha {rate:.4f}"


def test_the_interval_covers_a_known_effect_at_the_stated_rate(rng):
    """Coverage is the other half of the promise a 95% interval makes."""
    effect, trials = 2.0, 400
    covered = 0
    for _ in range(trials):
        unit = rng.normal(0.0, 8.0, 200)
        control = unit + rng.normal(0.0, 1.0, 200)
        treatment = unit + rng.normal(effect, 1.0, 200)
        result = paired_bootstrap(control, treatment, n_resamples=400, rng=rng)
        covered += int(result.ci.low <= effect <= result.ci.high)

    coverage = covered / trials
    sigma = (0.95 * 0.05 / trials) ** 0.5
    assert abs(coverage - 0.95) < 4 * sigma, f"empirical coverage {coverage:.3f}"


def test_mismatched_lengths_are_refused(rng):
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap(rng.normal(size=10), rng.normal(size=11))


def test_a_resample_count_too_small_to_mean_anything_is_refused(correlated_pair):
    control, treatment = correlated_pair

    with pytest.raises(ValueError, match="at least 100"):
        paired_bootstrap(control, treatment, n_resamples=50)


def test_results_are_reproducible_for_a_given_seed(correlated_pair):
    control, treatment = correlated_pair

    first = paired_bootstrap(
        control, treatment, n_resamples=RESAMPLES, rng=np.random.default_rng(7)
    )
    again = paired_bootstrap(
        control, treatment, n_resamples=RESAMPLES, rng=np.random.default_rng(7)
    )

    assert first == again


def test_the_pairing_assumption_travels_with_the_result(correlated_pair, rng):
    control, treatment = correlated_pair

    result = paired_bootstrap(control, treatment, n_resamples=RESAMPLES, rng=rng)

    assert any("same unit" in assumption for assumption in result.assumptions)
