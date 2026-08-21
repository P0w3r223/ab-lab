"""Cluster-robust variance, checked against an oracle and against reality.

``statsmodels`` is the oracle here and never the implementation: its
``cov_type="cluster"`` computes the CR1 sandwich, which is exactly what
``cluster.py`` implements by a shorter route. The agreement test compares
**standard errors**, which are convention-free, rather than p-values - the two
implementations disagree about degrees of freedom on purpose, and ADR 0008
records why comparing p-values first would produce a mysterious failure in the
fourth decimal.

The most informative test in the file is
``test_the_design_effect_predicts_the_variance_it_claims_to``: it checks the
formula against the *realised* variance of a sample mean over many draws, rather
than against a second formula.
"""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm

from ab_lab.cluster import (
    ClusteredSample,
    cluster_robust_t_test,
    design_effect,
    intraclass_correlation,
)
from ab_lab.results import TestResult as _TestResult
from ab_lab.simulate import clustered_normal_draw, poisson_cluster_size

ALPHA = 0.05


def _statsmodels_standard_error(control: ClusteredSample, treatment: ClusteredSample) -> float:
    """The oracle: an OLS of the metric on a treatment dummy, clustered."""
    values = np.concatenate([control.values, treatment.values])
    ids = np.concatenate([control.cluster_ids, treatment.cluster_ids])
    treated = np.concatenate([np.zeros(control.values.size), np.ones(treatment.values.size)])
    fit = sm.OLS(values, sm.add_constant(treated)).fit(
        cov_type="cluster", cov_kwds={"groups": ids}
    )
    return float(fit.bse[1])


def _arms(seed: int, sizes_control, sizes_treatment, icc=0.3, lift=0.0):
    rng = np.random.default_rng(seed)

    def arm(sizes, centre, first_id):
        between, within = np.sqrt(icc), np.sqrt(1.0 - icc)
        effects = rng.normal(0.0, between, len(sizes))
        ids = np.repeat(np.arange(len(sizes)) + first_id, sizes)
        values = np.repeat(effects, sizes) + rng.normal(centre, within, int(np.sum(sizes)))
        return ClusteredSample.from_arrays(values, ids)

    return arm(sizes_control, 0.0, 0), arm(sizes_treatment, lift, 10_000)


@pytest.mark.parametrize(
    ("name", "sizes"),
    [
        ("balanced", ([8] * 40, [8] * 40)),
        ("unbalanced", ([1, 2, 3, 4, 5] * 9, [7, 2, 11, 1, 4] * 9)),
        ("very unequal arms", ([6] * 30, [3] * 70)),
    ],
)
def test_the_standard_error_matches_the_statsmodels_cluster_oracle(name, sizes):
    control, treatment = _arms(11, *sizes, lift=0.25)
    result = cluster_robust_t_test(control, treatment)

    ours = abs(result.estimate / result.statistic)
    assert ours == pytest.approx(_statsmodels_standard_error(control, treatment), rel=1e-12)


def test_the_estimate_is_untouched_by_the_correction():
    """Clustering changes the variance of the difference, not the difference."""
    control, treatment = _arms(12, [5] * 30, [5] * 30, lift=0.4)
    assert cluster_robust_t_test(control, treatment).estimate == pytest.approx(
        treatment.values.mean() - control.values.mean()
    )


def test_the_intraclass_correlation_matches_hand_arithmetic():
    """Three clusters of two, chosen so every quantity can be checked by eye.

    Cluster means 2, 6, 10 around a grand mean of 6; within-cluster deviations
    are all +/-1. So MSB = 64/2 = 32, MSW = 6/3 = 2, the size term is 2, and
    the estimator gives (32 - 2) / (32 + 1*2) = 30/34.
    """
    sample = ClusteredSample.from_arrays(
        [1.0, 3.0, 5.0, 7.0, 9.0, 11.0], [0, 0, 1, 1, 2, 2]
    )
    assert intraclass_correlation(sample) == pytest.approx(30.0 / 34.0)


def test_identical_cluster_means_give_the_most_negative_estimate_possible():
    """Cluster means all equal, within-cluster spread large: MSB is exactly zero.

    The estimator then returns ``-MSW / ((m-1) * MSW)`` = ``-1/(m-1)``, which for
    pairs is -1 - the algebraic floor of the moment estimator, not a bug. It is
    returned rather than clipped to zero on purpose: a negative estimate says the
    rows within a unit are *less* alike than rows across units, which usually
    means the cluster was defined by the wrong column.
    """
    sample = ClusteredSample.from_arrays([1.0, 3.0, 1.0, 3.0, 1.0, 3.0], [0, 0, 1, 1, 2, 2])
    assert intraclass_correlation(sample) == pytest.approx(-1.0)


@pytest.mark.parametrize(
    ("sizes", "icc", "expected"),
    [
        (1, 0.3, 1.0),  # one row per unit: clustering that is not there is free
        ([10] * 5, 0.0, 1.0),  # uncorrelated rows: likewise
        ([10] * 5, 0.1, 1.9),  # the textbook 1 + (m-1)*rho
        ([10] * 5, 1.0, 10.0),  # perfectly correlated: n units, not n rows
    ],
)
def test_the_design_effect_matches_hand_arithmetic(sizes, icc, expected):
    assert design_effect(sizes, icc) == pytest.approx(expected)


def test_unequal_clusters_weight_the_large_ones():
    """The size-weighted mean, not the plain one: a few heavy users dominate."""
    plain_average_would_give = 1.0 + (5.5 - 1.0) * 0.2
    assert design_effect([1, 10], 0.2) == pytest.approx(1.0 + (101 / 11 - 1.0) * 0.2)
    assert design_effect([1, 10], 0.2) > plain_average_would_give


def test_the_design_effect_predicts_the_variance_it_claims_to():
    """Against realised variance, not against another formula.

    The claim is that the variance of a sample mean is inflated by the design
    effect. So draw many clustered samples, measure the variance of the mean
    across them, and compare. Nothing here re-uses the formula being tested.
    """
    n_clusters, size, icc, replications = 50, 8, 0.3, 4_000
    draw = clustered_normal_draw(n_clusters, size, icc, std_dev=1.0)
    rng = np.random.default_rng(2026)

    means = np.array([draw(rng)[0].values.mean() for _ in range(replications)])
    realised = float(means.var(ddof=1))

    n_observations = n_clusters * size
    predicted = design_effect([size] * n_clusters, icc) / n_observations
    # The sampling error of a variance estimate is var * sqrt(2/(R-1)).
    tolerance = 4.0 * predicted * np.sqrt(2.0 / (replications - 1))
    assert abs(realised - predicted) < tolerance


def test_the_result_reports_how_much_the_naive_interval_would_have_understated():
    control, treatment = _arms(13, [10] * 60, [10] * 60, icc=0.3)
    result = cluster_robust_t_test(control, treatment)

    assert isinstance(result, _TestResult)
    assert result.n_clusters == 120
    assert result.df == 118.0
    assert result.mean_cluster_size == pytest.approx(10.0)
    # 1 + 9*0.3 = 3.7 in expectation; a single sample is noisy about it.
    assert 2.0 < result.design_effect < 6.0
    assert result.effective_n < 1_200 / 2.0
    assert any("anti-conservative" in note for note in result.assumptions)


def test_unbalanced_clusters_are_drawn_and_analysed_without_special_casing():
    draw = clustered_normal_draw(80, poisson_cluster_size(6.0), icc=0.2)
    control, treatment = draw(np.random.default_rng(5))

    assert control.cluster_sizes.min() >= 1
    assert control.cluster_sizes.std() > 0.0
    result = cluster_robust_t_test(control, treatment)
    assert abs(result.estimate / result.statistic) == pytest.approx(
        _statsmodels_standard_error(control, treatment), rel=1e-12
    )


@pytest.mark.parametrize(
    ("values", "ids", "message"),
    [
        ([[1.0, 2.0]], [[0, 1]], "one-dimensional"),
        ([1.0, 2.0, 3.0], [0, 1], "shape"),
        ([1.0, np.nan], [0, 1], "NaN"),
        ([1.0, 2.0], [0.5, 1.5], "integers"),
    ],
)
def test_a_clustered_sample_refuses_data_it_cannot_interpret(values, ids, message):
    with pytest.raises(ValueError, match=message):
        ClusteredSample.from_arrays(values, ids)


def test_a_unit_cannot_be_in_both_arms():
    """Overlapping ids mean a paired design, and a different tool."""
    control = ClusteredSample.from_arrays([1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2])
    treatment = ClusteredSample.from_arrays([2.0, 3.0, 4.0, 5.0], [2, 2, 3, 3])
    with pytest.raises(ValueError, match="both arms"):
        cluster_robust_t_test(control, treatment)


def test_one_cluster_per_arm_is_refused_rather_than_estimated():
    control = ClusteredSample.from_arrays([1.0, 2.0, 3.0], [7, 7, 7])
    treatment = ClusteredSample.from_arrays([4.0, 5.0, 6.0], [8, 8, 8])
    with pytest.raises(ValueError, match="at least 2 clusters"):
        cluster_robust_t_test(control, treatment)


@pytest.mark.parametrize(
    ("values", "ids", "message"),
    [
        (np.array([1.0, np.nan]), np.array([0, 1]), "NaN"),
        (np.array([1.0, 2.0]), np.array([0.5, 1.5]), "integers"),
        (np.array([[1.0, 2.0]]), np.array([[0, 1]]), "one-dimensional"),
        (np.array([1.0, 2.0, 3.0]), np.array([0, 1]), "shape"),
    ],
)
def test_the_constructor_validates_as_well_as_the_classmethod(values, ids, message):
    """`from_arrays` is not the only way in, and for two releases it was the only
    way guarded.

    The dataclass constructor is public and exported. A NaN handed straight to
    it produced a NaN p-value, which compares False against alpha and reads as
    "not significant" - the exact failure `_validation.as_sample` names and
    `_checked_p_value` refuses. The class docstring said it validated at
    construction; now it does.
    """
    with pytest.raises(ValueError, match=message):
        ClusteredSample(values=values, cluster_ids=ids)
