"""Repeated measurements per unit: when 5 000 sessions are not 5 000 observations.

Every interval in :mod:`ab_lab.analyze` assumes independent observations. Most
product metrics violate that. A user has several sessions, places several orders,
views several pages; those rows are correlated, and analysing them as though they
were independent understates the variance of the difference by a factor this
module measures directly and calls the *design effect*.

The consequence is not subtle. At an intraclass correlation of 0.30 with ten rows
per user, the design effect is ``1 + 9(0.30) = 3.7``, so the naive standard error
is ``sqrt(3.7) = 1.92`` times too small, so the naive test rejects whenever the
true statistic exceeds ``1.96 / 1.92 = 1.02`` - which under the null happens
about 31% of the time, against a nominal 5%. The simulation in
``tests/test_simulate.py`` measures 32.3% (±1.1) and puts the corrected test back
at 5.4% (±0.5) on identical draws. That is the same order of magnitude as the
peeking result this package leads with, arrived at by an entirely different
route (ADR 0006).

The correction is the cluster-robust ("sandwich") variance with the CR1
finite-sample factor, which treats each *cluster* rather than each row as the
unit that carries information. Its own failure mode is stated rather than hidden:
it is asymptotic in the number of clusters and anti-conservative below roughly
forty of them, so :class:`~ab_lab.results.ClusterTestResult` exposes the cluster
counts and carries the rule of thumb in its assumptions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from ._validation import Alternative, DesignAlternative, check_alpha, check_alternative
from .power import sample_size_for_mean
from .results import ClusteredSampleSizeResult, ClusterTestResult, ConfidenceInterval

# Fewer than two clusters in an arm leaves the between-cluster variance
# undefined: there is nothing for the sandwich to average over.
_MIN_CLUSTERS_PER_GROUP = 2

# Below this the sandwich estimator is known to be anti-conservative. Not
# enforced - the library never decides for the caller - but carried in the
# result's assumptions and exposed as `n_clusters` so it can be asserted.
_RELIABLE_CLUSTER_COUNT = 40


@dataclass(frozen=True)
class ClusteredSample:
    """One arm of an experiment, with each observation's unit recorded.

    Two arguments rather than four parallel arrays, and validated once at
    construction. The alternative - ``f(control, control_ids, treatment,
    treatment_ids)`` - has the property that its commonest typo, swapping the
    second and third arguments, returns a *believable wrong number*. This
    package raises rather than return a NaN from two constant arms and crashes
    rather than tally a NaN as "not significant"; a signature whose misuse is
    silent would undo that.
    """

    values: NDArray[np.float64]
    cluster_ids: NDArray[np.int64]

    @classmethod
    def from_arrays(cls, values: ArrayLike, cluster_ids: ArrayLike) -> ClusteredSample:
        """Validate and coerce. ``cluster_ids[i]`` names the unit ``values[i]`` came from."""
        observations = np.asarray(values, dtype=np.float64)
        units = np.asarray(cluster_ids)
        if observations.ndim != 1:
            raise ValueError(f"values must be one-dimensional, got shape {observations.shape}")
        if units.shape != observations.shape:
            raise ValueError(
                f"cluster_ids has shape {units.shape}, values has {observations.shape}"
            )
        if not np.all(np.isfinite(observations)):
            raise ValueError("values contains NaN or infinite values")
        if not np.issubdtype(units.dtype, np.integer):
            # Strings and floats both work as labels, but silently accepting a
            # float id invites 1.0000001 and 1.0 becoming two users.
            raise ValueError(f"cluster_ids must be integers, got dtype {units.dtype}")
        return cls(values=observations, cluster_ids=units.astype(np.int64))

    @property
    def n_clusters(self) -> int:
        return int(np.unique(self.cluster_ids).size)

    @property
    def cluster_sizes(self) -> NDArray[np.int64]:
        _, counts = np.unique(self.cluster_ids, return_counts=True)
        return counts.astype(np.int64)

    @property
    def mean_cluster_size(self) -> float:
        return float(self.values.size / self.n_clusters)

    @property
    def cluster_means(self) -> NDArray[np.float64]:
        """One value per unit - the aggregation a naive analysis skips."""
        _, inverse = np.unique(self.cluster_ids, return_inverse=True)
        totals = np.bincount(inverse, weights=self.values)
        return totals / np.bincount(inverse)


def _cluster_totals(
    contributions: NDArray[np.float64], cluster_ids: NDArray[np.int64]
) -> NDArray[np.float64]:
    _, inverse = np.unique(cluster_ids, return_inverse=True)
    return np.bincount(inverse, weights=contributions)


def cluster_robust_variance(
    contributions: NDArray[np.float64], cluster_ids: NDArray[np.int64]
) -> float:
    """Uncorrected sandwich variance of a mean, from per-cluster contributions.

    Takes *contributions* rather than values on purpose, and :mod:`ab_lab.ratio`
    is why: for a mean they are the residuals ``y - ybar``; for a ratio metric
    they are the linearised contributions ``y - R*x``. That foresight cost one
    parameter name in 0.3.0 and made the delta method an addition rather than a
    re-derivation (ADR 0006 D7). Named without an underscore for the same
    reason - it has a second module as a caller - but not re-exported at the
    package top level, because a caller wanting a variance wants a test.

    The finite-sample factor is not applied here, because it depends on the
    whole design rather than on one arm.
    """
    totals = _cluster_totals(contributions, cluster_ids)
    return float(np.sum(totals**2) / contributions.size**2)


def _cr1_factor(n_clusters: int, n_observations: int, n_parameters: int = 2) -> float:
    """The CR1 correction: ``G/(G-1) * (n-1)/(n-k)``.

    Chosen because it is exactly what ``statsmodels`` computes under
    ``cov_type="cluster"`` with its default settings, which makes an existing
    dev-extra dependency an exact oracle without importing it at runtime
    (ADR 0001, ADR 0008).
    """
    return (n_clusters / (n_clusters - 1.0)) * (
        (n_observations - 1.0) / (n_observations - n_parameters)
    )


def intraclass_correlation(sample: ClusteredSample) -> float:
    """Share of variance that lives *between* units rather than within them.

    The one-way ANOVA moment estimator. Zero means the rows are effectively
    independent and clustering costs nothing; one means every row from a unit is
    the same number and the sample is worth exactly its number of units.

    Can come out slightly negative when the between-cluster mean square falls
    below the within-cluster one, which is a real sampling outcome rather than an
    error; it is returned as-is rather than clipped, because a negative estimate
    is information about the sample and clipping it would hide a mis-specified
    cluster definition.
    """
    if sample.n_clusters < _MIN_CLUSTERS_PER_GROUP:
        raise ValueError(
            f"need at least {_MIN_CLUSTERS_PER_GROUP} clusters, got {sample.n_clusters}"
        )
    sizes = sample.cluster_sizes
    if np.all(sizes == 1):
        raise ValueError("every cluster has one observation: there is no within-cluster variance")

    n_total = sample.values.size
    n_clusters = sample.n_clusters
    grand_mean = float(sample.values.mean())
    means = sample.cluster_means

    between = float(np.sum(sizes * (means - grand_mean) ** 2)) / (n_clusters - 1)
    _, inverse = np.unique(sample.cluster_ids, return_inverse=True)
    within = float(np.sum((sample.values - means[inverse]) ** 2)) / (n_total - n_clusters)

    # The size that makes the moment estimator unbiased under unbalanced clusters.
    typical_size = (n_total - float(np.sum(sizes**2)) / n_total) / (n_clusters - 1)
    denominator = between + (typical_size - 1.0) * within
    if denominator == 0.0:
        return 0.0
    return (between - within) / denominator


def design_effect(cluster_sizes: ArrayLike | float, icc: float) -> float:
    """How many times larger clustering makes the variance of a mean.

    ``1 + (m - 1) * icc``, where ``m`` is the size-weighted mean cluster size
    ``sum(m_g^2) / sum(m_g)`` - not the plain average. The weighting is what
    makes the formula right for unbalanced clusters, and unbalanced is the case
    that matters: a handful of very heavy users move this number a long way.

    One observation per cluster, or an intraclass correlation of zero, gives
    exactly 1.0 - clustering that is not there costs nothing.
    """
    if not -1.0 <= icc <= 1.0:
        raise ValueError(f"icc must be in [-1, 1], got {icc}")
    sizes = np.atleast_1d(np.asarray(cluster_sizes, dtype=np.float64))
    if np.any(sizes < 1.0):
        raise ValueError("cluster sizes must be at least 1")
    weighted_mean = float(np.sum(sizes**2) / np.sum(sizes))
    return 1.0 + (weighted_mean - 1.0) * icc


def cluster_robust_t_test(
    control: ClusteredSample,
    treatment: ClusteredSample,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> ClusterTestResult:
    """Difference of means with a standard error that survives repeated measures.

    Args:
        control: All control observations, each labelled with the unit it came
            from. Units must not appear in both arms - that is a paired design,
            and :func:`~ab_lab.analyze.paired_bootstrap` is the tool for it.
        treatment: The same, for the treatment arm.

    The estimate is the ordinary difference of means; clustering changes its
    *variance*, not its value. Degrees of freedom are the number of clusters
    minus two, not the number of observations minus two, which is the whole
    point: a study of 5 000 sessions from 500 users has 498 degrees of freedom
    per arm, not 4 998.
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
            f"{shared.size} cluster id(s) appear in both arms, starting with "
            f"{shared[0]}: a unit assigned to both arms is a paired design, not "
            f"a clustered two-group one"
        )

    estimate = float(treatment.values.mean() - control.values.mean())
    n_observations = control.values.size + treatment.values.size
    n_clusters = control.n_clusters + treatment.n_clusters

    uncorrected = cluster_robust_variance(
        control.values - control.values.mean(), control.cluster_ids
    ) + cluster_robust_variance(
        treatment.values - treatment.values.mean(), treatment.cluster_ids
    )
    variance = _cr1_factor(n_clusters, n_observations) * uncorrected
    if variance <= 0.0:
        raise ValueError("no variation between clusters: there is nothing to test against")
    standard_error = math.sqrt(variance)

    independence_variance = (
        control.values.var(ddof=1) / control.values.size
        + treatment.values.var(ddof=1) / treatment.values.size
    )
    realised_design_effect = (
        variance / independence_variance if independence_variance > 0.0 else math.inf
    )

    df = float(n_clusters - 2)
    statistic = estimate / standard_error
    if alternative == "two-sided":
        p_value = float(2.0 * stats.t.sf(abs(statistic), df))
    elif alternative == "greater":
        p_value = float(stats.t.sf(statistic, df))
    else:
        p_value = float(stats.t.cdf(statistic, df))
    margin = float(stats.t.isf(alpha / 2.0, df) * standard_error)

    return ClusterTestResult(
        test="cluster-robust t-test (CR1 sandwich)",
        estimate=estimate,
        ci=ConfidenceInterval(estimate - margin, estimate + margin, 1.0 - alpha),
        p_value=p_value,
        statistic=float(statistic),
        alternative=alternative,
        assumptions=(
            "observations are independent *across* clusters; within a cluster "
            "they may be correlated in any way",
            "each unit belongs to exactly one arm",
            f"the sandwich estimator is asymptotic in the number of clusters and "
            f"is anti-conservative below about {_RELIABLE_CLUSTER_COUNT}; this "
            f"result has {n_clusters}",
            f"degrees of freedom are clusters minus two ({df:g}), not "
            f"observations minus two ({n_observations - 2})",
        ),
        n_clusters_control=control.n_clusters,
        n_clusters_treatment=treatment.n_clusters,
        mean_cluster_size=n_observations / n_clusters,
        design_effect=float(realised_design_effect),
        effective_n=float(n_observations / realised_design_effect),
        df=df,
    )


def sample_size_for_clustered_mean(
    mde: float,
    std_dev: float,
    icc: float,
    mean_cluster_size: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    alternative: DesignAlternative = "two-sided",
) -> ClusteredSampleSizeResult:
    """Units to recruit when each of them will contribute several observations.

    Clustering is not only an analysis problem, which is why it is sized here
    rather than in :mod:`ab_lab.power`: an experiment planned as though rows were
    independent is under-powered before it launches, and no amount of careful
    analysis afterwards recovers that.

    Args:
        mean_cluster_size: Expected observations per unit, from historical data.
        icc: Expected intraclass correlation, likewise. Both are forecasts, and
            the result says so - getting either wrong rescales the answer.

    The arithmetic is the independent sample size multiplied by the design
    effect, then divided into whole units. Nothing subtler is warranted: the
    design effect is exactly the factor by which the variance of the estimate
    grows, and sample size is inversely proportional to that variance.
    """
    if mean_cluster_size < 1.0:
        raise ValueError(f"mean_cluster_size must be at least 1, got {mean_cluster_size}")
    if not 0.0 <= icc <= 1.0:
        raise ValueError(f"icc must be in [0, 1], got {icc}")

    independent = sample_size_for_mean(
        mde, std_dev, alpha=alpha, power=power, ratio=ratio, alternative=alternative
    ).exact_per_group
    inflation = 1.0 + (mean_cluster_size - 1.0) * icc
    observations = independent * inflation
    n_clusters = math.ceil(observations / mean_cluster_size)

    return ClusteredSampleSizeResult(
        n_clusters_per_group=n_clusters,
        per_group=n_clusters * mean_cluster_size,
        independent_per_group=independent,
        design_effect=inflation,
        icc=icc,
        mean_cluster_size=mean_cluster_size,
        mde=mde,
        alpha=alpha,
        power=power,
        alternative=alternative,
        method="two-sample t-test with a design-effect inflation",
        assumptions=(
            f"the intraclass correlation really is about {icc:g}; it is a "
            f"forecast from historical data, and the answer scales with it",
            f"units contribute about {mean_cluster_size:g} observations each, on "
            f"average - unequal sizes make the effective figure larger, not smaller",
            "the analysis will use a cluster-robust standard error; sizing for "
            "clustering and then analysing without it wastes the extra traffic",
        ),
    )
