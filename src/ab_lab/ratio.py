"""Metrics that are a ratio: click-through rate, orders per session, revenue per visit.

Most of the metrics people actually watch are ratios of two totals rather than
means of one column. Click-through rate is clicks over impressions; average order
value is revenue over orders. The randomisation unit is the *user*, but the
denominator counts something else, so the number of things being divided is
itself random - and that is what makes the ordinary standard error wrong.

The mistake this corrects is not the one :mod:`ab_lab.cluster` corrects, and the
two are easy to confuse. Clustering understates the variance because correlated
rows are counted as independent evidence. A ratio metric goes wrong for a second,
independent reason: ``Var(Y/X)`` is not ``Var(Y)/X**2``, because the denominator
varies too. Analysing a ratio as though the denominator were fixed can err in
*either* direction, and by how much depends on the correlation between numerator
and denominator - which for CTR-like metrics is strongly positive.

The fix is the delta method, and it turns out to be the same machinery. Writing
``R = sum(y) / sum(x)``, the linearised contribution of one observation is
``y - R*x``; the variance of ``R`` is the cluster-robust variance of the mean of
those contributions, divided by the squared mean denominator. So this module is a
wrapper around :func:`ab_lab.cluster.cluster_robust_variance` rather than a second
implementation of a sandwich - which is what ADR 0006 D7 arranged for, one
release early, by having that function take contributions rather than values.

**The estimand is the ratio of totals, not the mean of per-user ratios.** Those
are different numbers and the difference is not subtle: the ratio of totals
weights a user by their denominator, so a user with a hundred impressions counts
a hundred times as much as one with a single impression. That is usually what a
business means by "the click-through rate", and it is worth being sure before
reporting one as the other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from ._validation import Alternative, check_alpha, check_alternative
from .cluster import _MIN_CLUSTERS_PER_GROUP, _cr1_factor, cluster_robust_variance
from .results import ConfidenceInterval, RatioTestResult


@dataclass(frozen=True)
class RatioSample:
    """One arm of an experiment whose metric is a ratio of two totals.

    Three parallel arrays, bound together and validated once, for the reason
    :class:`~ab_lab.cluster.ClusteredSample` gives: passing them separately to a
    test makes the commonest typo - numerator and denominator swapped - return a
    number rather than an error.
    """

    numerator: NDArray[np.float64]
    denominator: NDArray[np.float64]
    cluster_ids: NDArray[np.int64]

    def __post_init__(self) -> None:
        """Validate on every path in, not only through :meth:`from_arrays`.

        The constructor is public, so validating in the classmethod alone left
        this class able to carry a NaN into a p-value that reads as "not
        significant".
        """
        top, bottom, units = self.numerator, self.denominator, self.cluster_ids
        if top.ndim != 1:
            raise ValueError(f"numerator must be one-dimensional, got shape {top.shape}")
        if bottom.shape != top.shape or units.shape != top.shape:
            raise ValueError(
                f"numerator, denominator and cluster_ids must have the same shape; got "
                f"{top.shape}, {bottom.shape} and {units.shape}"
            )
        if not (np.all(np.isfinite(top)) and np.all(np.isfinite(bottom))):
            raise ValueError("numerator or denominator contains NaN or infinite values")
        if np.any(bottom < 0.0):
            raise ValueError("denominator must be non-negative")
        if bottom.sum() <= 0.0:
            raise ValueError("denominator sums to zero: the ratio is undefined")
        if not np.issubdtype(units.dtype, np.integer):
            raise ValueError(f"cluster_ids must be integers, got dtype {units.dtype}")

    @classmethod
    def from_arrays(
        cls, numerator: ArrayLike, denominator: ArrayLike, cluster_ids: ArrayLike
    ) -> RatioSample:
        """Coerce anything array-like, then construct - which validates."""
        units = np.asarray(cluster_ids)
        return cls(
            numerator=np.asarray(numerator, dtype=np.float64),
            denominator=np.asarray(denominator, dtype=np.float64),
            cluster_ids=units.astype(np.int64) if np.issubdtype(units.dtype, np.integer) else units,
        )

    @property
    def n_clusters(self) -> int:
        return int(np.unique(self.cluster_ids).size)

    @property
    def ratio(self) -> float:
        """The metric itself: total numerator over total denominator."""
        return float(self.numerator.sum() / self.denominator.sum())

    @property
    def contributions(self) -> NDArray[np.float64]:
        """``y - R*x``, the linearisation the delta method runs on.

        These sum to exactly zero by construction, because ``R`` is defined as
        the value that makes them - so they need no further centring, unlike the
        residuals a mean's sandwich takes.
        """
        return self.numerator - self.ratio * self.denominator


def _ratio_variance(sample: RatioSample, n_clusters: int, n_observations: int) -> float:
    """Delta-method variance of a ratio of totals, robust to clustering.

    ``Var(R) = Var(mean(y - R*x)) / mean(x)**2``. The numerator is exactly the
    cluster-robust variance of a mean, which is why this is nine lines rather
    than a second sandwich implementation.
    """
    mean_denominator = float(sample.denominator.mean())
    if mean_denominator <= 0.0:
        raise ValueError("mean denominator is zero: the ratio is undefined")
    uncorrected = cluster_robust_variance(sample.contributions, sample.cluster_ids)
    corrected = _cr1_factor(n_clusters, n_observations) * uncorrected
    return corrected / mean_denominator**2


def ratio_metric_test(
    control: RatioSample,
    treatment: RatioSample,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> RatioTestResult:
    """Difference of two ratios, with a variance that accounts for both problems.

    Args:
        control: Numerator, denominator and unit label per observation.
        treatment: The same for the other arm. Units must not appear in both.

    The estimate is ``treatment.ratio - control.ratio`` on the metric's own
    scale. Both the randomisation unit's repeated rows and the randomness of the
    denominator are handled: the first by the sandwich, the second by the
    linearisation.

    Setting every denominator to 1 makes this exactly
    :func:`~ab_lab.cluster.cluster_robust_t_test`, which is not a coincidence but
    the definition - and is asserted as a test, because a special case that does
    not reduce is a sign the general case is wrong.
    """
    alpha = check_alpha(alpha)
    alternative = check_alternative(alternative)
    for sample, name in ((control, "control"), (treatment, "treatment")):
        if sample.n_clusters < _MIN_CLUSTERS_PER_GROUP:
            raise ValueError(
                f"{name} needs at least {_MIN_CLUSTERS_PER_GROUP} clusters, "
                f"got {sample.n_clusters}"
            )

    shared = np.intersect1d(control.cluster_ids, treatment.cluster_ids)
    if shared.size:
        raise ValueError(
            f"{shared.size} cluster id(s) appear in both arms, starting with {shared[0]}"
        )

    n_clusters = control.n_clusters + treatment.n_clusters
    n_observations = control.numerator.size + treatment.numerator.size

    variance = _ratio_variance(control, n_clusters, n_observations) + _ratio_variance(
        treatment, n_clusters, n_observations
    )
    if variance <= 0.0:
        raise ValueError("no variation between clusters: there is nothing to test against")

    estimate = treatment.ratio - control.ratio
    standard_error = math.sqrt(variance)
    df = float(n_clusters - 2)
    statistic = estimate / standard_error

    if alternative == "two-sided":
        p_value = float(2.0 * stats.t.sf(abs(statistic), df))
    elif alternative == "greater":
        p_value = float(stats.t.sf(statistic, df))
    else:
        p_value = float(stats.t.cdf(statistic, df))
    margin = float(stats.t.isf(alpha / 2.0, df) * standard_error)

    return RatioTestResult(
        test="ratio metric (delta method, CR1 sandwich)",
        estimate=float(estimate),
        ci=ConfidenceInterval(estimate - margin, estimate + margin, 1.0 - alpha),
        p_value=p_value,
        statistic=float(statistic),
        alternative=alternative,
        assumptions=(
            "the estimand is the ratio of totals, which weights a unit by its "
            "denominator - not the mean of per-unit ratios, which does not",
            "observations are independent across units; within a unit they may "
            "be correlated in any way",
            "the delta method is a first-order approximation: it needs the "
            "denominator to be comfortably away from zero, which a rate with "
            "few trials per unit is not",
            f"degrees of freedom are units minus two ({df:g}), not observations "
            f"minus two ({n_observations - 2})",
        ),
        control_ratio=control.ratio,
        treatment_ratio=treatment.ratio,
        relative_effect=(
            float(estimate / control.ratio) if control.ratio != 0.0 else math.nan
        ),
        n_clusters_control=control.n_clusters,
        n_clusters_treatment=treatment.n_clusters,
        df=df,
    )
