"""Simulation harness: checking that the statistics in this package are true.

Every method here is validated twice - once against a reference implementation
(``scipy``/``statsmodels``, in the test suite) and once against reality, by
running thousands of experiments where the answer is known:

* an **A/A** run has no true effect, so the rejection rate estimates the type I
  error and must land near the nominal alpha;
* an **A/B** run has a known effect, so the rejection rate estimates power and
  must land near what :mod:`ab_lab.power` promised;
* a **peeking** run applies a fixed-horizon test repeatedly, and shows the type
  I error climbing far above alpha - the failure this package exists to name.

The building blocks are two callables. A *draw* turns a random generator into
one experiment's data; a *p-value function* turns that data into a p-value.
Anything obeying those two signatures plugs into the same harness, which is why
the honest comparison between a fixed-horizon test and a sequential one costs
no extra code path.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from ._validation import check_alpha
from .analyze import proportion_test, welch_t_test
from .cluster import ClusteredSample, cluster_robust_t_test
from .cuped import CupedSample, cuped_t_test
from .multiplicity import Correction
from .ratio import RatioSample, ratio_metric_test
from .results import SimulationSummary, TestResult
from .sequential import msprt

Draw = Callable[[np.random.Generator], tuple[NDArray[np.float64], NDArray[np.float64]]]
PValueFn = Callable[[NDArray[np.float64], NDArray[np.float64]], float]

#: A clustered experiment cannot be expressed as two flat arrays without either
#: assuming balanced clusters or pre-aggregating - and pre-aggregating removes
#: the very analysis this is meant to catch. So the draw contract differs while
#: the verdict, `SimulationSummary`, stays the same (ADR 0006 D1).
ClusteredDraw = Callable[[np.random.Generator], tuple[ClusteredSample, ClusteredSample]]
ClusteredPValueFn = Callable[[ClusteredSample, ClusteredSample], float]

#: A metric suite: both arms are ``(n_metrics, n_per_group)``. One p-value per
#: metric comes back, and the *joint* outcome across them is what a family-wise
#: error rate is about - which is why running the harness K times cannot measure
#: it and this contract exists.
SuiteDraw = Callable[[np.random.Generator], tuple[NDArray[np.float64], NDArray[np.float64]]]
SuitePValueFn = Callable[[NDArray[np.float64], NDArray[np.float64]], tuple[float, ...]]

#: A ratio metric carries two columns per observation, so it needs its own draw
#: contract for the same reason the clustered one does. The tally and the summary
#: type stay shared (ADR 0006 D1).
RatioDraw = Callable[[np.random.Generator], tuple[RatioSample, RatioSample]]
RatioPValueFn = Callable[[RatioSample, RatioSample], float]

#: CUPED needs the *estimate* each analysis produced, not only its p-value,
#: because the failure it is checked against is a biased point estimate arriving
#: with a tighter interval. So this contract returns the whole result.
CupedDraw = Callable[[np.random.Generator], tuple[CupedSample, CupedSample]]
CupedTestFn = Callable[[CupedSample, CupedSample], TestResult]

#: What the harness should report as "the effect", given one experiment's data.
EstimateFn = Callable[[NDArray[np.float64], NDArray[np.float64]], float]


def mean_difference(
    control: NDArray[np.float64], treatment: NDArray[np.float64]
) -> float:
    """Treatment mean minus control mean - the default estimand.

    Made explicit and overridable because the runners used to hardcode it while
    accepting *any* ``p_value_fn``. That is harmless while every adapter here
    estimates a difference of means, and wrong the moment one does not:
    :func:`~ab_lab.analyze.mann_whitney` estimates P(treatment > control), and a
    summary pairing its p-value with a mean difference would be reporting two
    different quantities as though they were one.
    """
    return float(treatment.mean() - control.mean())


class _Tally:
    """The running counts behind every summary this module produces.

    One implementation on purpose. The three runners have to agree on what
    counts as a rejection, how the estimate is averaged, and which estimates
    feed the winner's-curse figure - otherwise the rows they produce are not
    comparable, and comparing a fixed-horizon test with a sequential one in a
    single table is the reason this module exists.
    """

    def __init__(self, n_experiments: int, nominal_alpha: float, label: str) -> None:
        self._n_experiments = n_experiments
        self._nominal_alpha = nominal_alpha
        self._label = label
        self._rejections = 0
        self._estimate_total = 0.0
        self._absolute_estimate_when_stopped = 0.0

    def record(self, estimate: float, rejected: bool) -> None:
        """Add one finished experiment."""
        self._estimate_total += estimate
        if rejected:
            self._rejections += 1
            self._absolute_estimate_when_stopped += abs(estimate)

    def summarise(self) -> SimulationSummary:
        return SimulationSummary(
            n_experiments=self._n_experiments,
            n_rejections=self._rejections,
            nominal_alpha=self._nominal_alpha,
            mean_estimate=self._estimate_total / self._n_experiments,
            label=self._label,
            mean_absolute_estimate_when_stopped=(
                self._absolute_estimate_when_stopped / self._rejections
                if self._rejections
                else None
            ),
        )


def binary_draw(
    n_per_group: int,
    baseline_rate: float,
    absolute_lift: float = 0.0,
) -> Draw:
    """Draw one conversion experiment: Bernoulli outcomes in both arms.

    ``absolute_lift`` is on the rate scale, so ``0.005`` moves a 5% baseline to
    5.5%. Zero makes it an A/A experiment.
    """
    if n_per_group < 2:
        raise ValueError(f"n_per_group must be at least 2, got {n_per_group}")
    treatment_rate = baseline_rate + absolute_lift
    for rate, name in ((baseline_rate, "baseline_rate"), (treatment_rate, "treatment rate")):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {rate}")

    def draw(rng: np.random.Generator) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        control = (rng.random(n_per_group) < baseline_rate).astype(np.float64)
        treatment = (rng.random(n_per_group) < treatment_rate).astype(np.float64)
        return control, treatment

    return draw


def normal_draw(
    n_per_group: int,
    mean: float = 0.0,
    std_dev: float = 1.0,
    absolute_lift: float = 0.0,
) -> Draw:
    """Draw one continuous-metric experiment from two normal distributions."""
    if n_per_group < 2:
        raise ValueError(f"n_per_group must be at least 2, got {n_per_group}")
    if std_dev <= 0.0:
        raise ValueError(f"std_dev must be positive, got {std_dev}")

    def draw(rng: np.random.Generator) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        control = rng.normal(mean, std_dev, n_per_group)
        treatment = rng.normal(mean + absolute_lift, std_dev, n_per_group)
        return control, treatment

    return draw


def lognormal_draw(
    n_per_group: int,
    mean_log: float = 0.0,
    sigma_log: float = 1.0,
    multiplicative_lift: float = 1.0,
) -> Draw:
    """Draw a heavy-tailed revenue-like metric.

    Useful for showing where the normal approximation strains: with a skewed
    metric the t-test's *actual* type I error drifts from its nominal alpha at
    small samples, which is the empirical case for reaching for the bootstrap.
    """
    if n_per_group < 2:
        raise ValueError(f"n_per_group must be at least 2, got {n_per_group}")
    if sigma_log <= 0.0:
        raise ValueError(f"sigma_log must be positive, got {sigma_log}")
    if multiplicative_lift <= 0.0:
        raise ValueError(f"multiplicative_lift must be positive, got {multiplicative_lift}")

    def draw(rng: np.random.Generator) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        control = rng.lognormal(mean_log, sigma_log, n_per_group)
        treatment = rng.lognormal(mean_log, sigma_log, n_per_group) * multiplicative_lift
        return control, treatment

    return draw


def poisson_cluster_size(
    mean_size: float, minimum: int = 1
) -> Callable[[np.random.Generator, int], NDArray[np.int64]]:
    """Cluster sizes that vary, because in practice they always do.

    Balanced clusters are the easy case: aggregating to unit means is exact
    there, so the sandwich estimator has nothing to prove. Unbalanced clusters
    are where it earns its place, and where a handful of very heavy users move
    the design effect a long way, so the harness makes them a first-class option
    rather than an afterthought.
    """
    if minimum < 1:
        raise ValueError(f"minimum must be at least 1, got {minimum}")
    if mean_size < minimum:
        raise ValueError(f"mean_size must be at least minimum={minimum}, got {mean_size}")

    def sizes(rng: np.random.Generator, n_clusters: int) -> NDArray[np.int64]:
        return (rng.poisson(mean_size - minimum, n_clusters) + minimum).astype(np.int64)

    return sizes


def clustered_normal_draw(
    n_clusters_per_group: int,
    cluster_size: int | Callable[[np.random.Generator, int], NDArray[np.int64]],
    icc: float,
    mean: float = 0.0,
    std_dev: float = 1.0,
    absolute_lift: float = 0.0,
) -> ClusteredDraw:
    """Draw an experiment whose rows repeat within units.

    The generative model is ``y_ij = mean + u_i + e_ij`` with
    ``u_i ~ N(0, icc * std_dev^2)`` and ``e_ij ~ N(0, (1 - icc) * std_dev^2)``,
    so the **total variance is held at ``std_dev**2`` whatever the intraclass
    correlation**. That is the point of fixing it: the resulting false positive
    curve is then attributable to dependence alone, and not confounded with a
    metric that simply got noisier.
    """
    if n_clusters_per_group < 2:
        raise ValueError(f"n_clusters_per_group must be at least 2, got {n_clusters_per_group}")
    if not 0.0 <= icc < 1.0:
        raise ValueError(f"icc must be in [0, 1), got {icc}")
    if std_dev <= 0.0:
        raise ValueError(f"std_dev must be positive, got {std_dev}")
    if not callable(cluster_size) and cluster_size < 1:
        raise ValueError(f"cluster_size must be at least 1, got {cluster_size}")

    between = np.sqrt(icc) * std_dev
    within = np.sqrt(1.0 - icc) * std_dev

    def one_arm(rng: np.random.Generator, centre: float, first_id: int) -> ClusteredSample:
        sizes = (
            cluster_size(rng, n_clusters_per_group)
            if callable(cluster_size)
            else np.full(n_clusters_per_group, cluster_size, dtype=np.int64)
        )
        unit_effects = rng.normal(0.0, between, n_clusters_per_group) if between else np.zeros(
            n_clusters_per_group
        )
        ids = np.repeat(np.arange(n_clusters_per_group, dtype=np.int64) + first_id, sizes)
        values = np.repeat(unit_effects, sizes) + rng.normal(centre, within, int(sizes.sum()))
        return ClusteredSample(values=values, cluster_ids=ids)

    def draw(rng: np.random.Generator) -> tuple[ClusteredSample, ClusteredSample]:
        control = one_arm(rng, mean, 0)
        treatment = one_arm(rng, mean + absolute_lift, n_clusters_per_group)
        return control, treatment

    return draw


def correlated_normal_suite_draw(
    n_per_group: int,
    n_metrics: int,
    correlation: float = 0.0,
    absolute_lifts: Sequence[float] | None = None,
    std_dev: float = 1.0,
) -> SuiteDraw:
    """Draw one experiment measured on several metrics at once.

    Args:
        correlation: Equicorrelation between metrics, in [0, 1). Real metric
            suites are correlated - sessions, clicks and revenue move together -
            and correlation is what makes Bonferroni conservative rather than
            merely blunt, so it is a parameter rather than an assumption.
        absolute_lifts: True effect per metric; ``None`` means a global null.
            A *partial* null - some entries zero, some not - is the only setting
            in which the false discovery rate and the family-wise rate differ,
            and therefore the only one where Benjamini-Hochberg can be told
            apart from Holm.

    Built from one shared factor plus per-metric noise, which gives exactly
    exchangeable correlation and costs one extra normal draw rather than a
    Cholesky factorisation per experiment.
    """
    if n_per_group < 2:
        raise ValueError(f"n_per_group must be at least 2, got {n_per_group}")
    if n_metrics < 1:
        raise ValueError(f"n_metrics must be at least 1, got {n_metrics}")
    if not 0.0 <= correlation < 1.0:
        raise ValueError(f"correlation must be in [0, 1), got {correlation}")
    if std_dev <= 0.0:
        raise ValueError(f"std_dev must be positive, got {std_dev}")

    lifts = np.zeros(n_metrics) if absolute_lifts is None else np.asarray(
        absolute_lifts, dtype=np.float64
    )
    if lifts.shape != (n_metrics,):
        raise ValueError(f"absolute_lifts must have {n_metrics} entries, got {lifts.shape}")

    shared, private = np.sqrt(correlation), np.sqrt(1.0 - correlation)

    def one_arm(rng: np.random.Generator, centres: NDArray[np.float64]) -> NDArray[np.float64]:
        common = rng.normal(0.0, 1.0, n_per_group)
        idiosyncratic = rng.normal(0.0, 1.0, (n_metrics, n_per_group))
        combined = shared * common + private * idiosyncratic
        return std_dev * combined + centres[:, None]

    def draw(rng: np.random.Generator) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return one_arm(rng, np.zeros(n_metrics)), one_arm(rng, lifts)

    return draw


def clustered_ratio_draw(
    n_clusters_per_group: int,
    trials_per_cluster: int | Callable[[np.random.Generator, int], NDArray[np.int64]],
    base_rate: float,
    rate_dispersion: float = 0.0,
    absolute_lift: float = 0.0,
    lift_scales_with_exposure: bool = False,
    rows_per_cluster: int = 1,
) -> RatioDraw:
    """Draw a click-through-rate-like metric: successes over trials, per unit.

    Args:
        trials_per_cluster: Impressions per user, as a constant or a callable
            drawing one count per unit. **A constant makes this a bad test**,
            and that is worth saying rather than leaving to be discovered: when
            every unit contributes the same denominator, the ratio of totals is
            *identically* the mean of per-unit ratios, so the two estimands
            coincide and nothing here has anything to do. Pass
            :func:`poisson_cluster_size` for the case that matters.
        rate_dispersion: Spread of each unit's own rate around ``base_rate``.
            Zero makes every user identical; positive values are what a real
            population looks like and are what the sandwich has to survive.
        lift_scales_with_exposure: Give a unit a lift proportional to its share
            of the mean exposure, so the treatment helps engaged users more than
            occasional ones. **This is the only setting in which the choice of
            estimand changes the answer**, and it is a parameter rather than an
            assumption because that fact is easy to state and easy to disbelieve:
            when a unit's rate is independent of how much it is exposed, the
            ratio of totals and the mean of per-unit ratios agree to the fourth
            decimal and the two analyses have the same power. Correlate the two
            and they answer different questions.
        rows_per_cluster: How many rows each unit's trials are split across. One
            row per unit - the default, and what every early version of this draw
            produced - leaves ``n_observations == n_clusters``, so the *sandwich*
            half of :func:`~ab_lab.ratio.ratio_metric_test` is never exercised:
            each cluster total is a single row and the estimator has nothing to
            aggregate. Above one, rows within a unit share that unit's rate and
            are therefore correlated, which is the case the cluster-robust part
            exists for.

    Each unit gets its own rate, then binomial trials at that rate - so outcomes
    are correlated within a unit and heterogeneous across units, which is the
    joint problem this metric has.
    """
    if n_clusters_per_group < 2:
        raise ValueError(f"n_clusters_per_group must be at least 2, got {n_clusters_per_group}")
    if not callable(trials_per_cluster) and trials_per_cluster < 1:
        raise ValueError(f"trials_per_cluster must be at least 1, got {trials_per_cluster}")
    if not 0.0 < base_rate < 1.0:
        raise ValueError(f"base_rate must be in (0, 1), got {base_rate}")
    if rate_dispersion < 0.0:
        raise ValueError(f"rate_dispersion must be non-negative, got {rate_dispersion}")
    if rows_per_cluster < 1:
        raise ValueError(f"rows_per_cluster must be at least 1, got {rows_per_cluster}")

    def one_arm(rng: np.random.Generator, lift: float, first_id: int) -> RatioSample:
        trials = (
            trials_per_cluster(rng, n_clusters_per_group)
            if callable(trials_per_cluster)
            else np.full(n_clusters_per_group, trials_per_cluster, dtype=np.int64)
        )
        drawn = (
            rng.normal(base_rate, rate_dispersion, n_clusters_per_group)
            if rate_dispersion
            else np.full(n_clusters_per_group, base_rate)
        )
        if lift and lift_scales_with_exposure:
            drawn = drawn + lift * trials / max(float(trials.mean()), 1e-12)
        elif lift:
            drawn = drawn + lift
        per_unit = np.clip(drawn, 1e-6, 1.0 - 1e-6)

        # Split each unit's trials across its rows as evenly as the integers
        # allow, then draw each row at that unit's own rate. Rows within a unit
        # are correlated through the shared rate, which is what the sandwich has
        # to survive.
        base, remainder = np.divmod(trials, rows_per_cluster)
        row_index = np.arange(rows_per_cluster)[None, :]
        row_trials = (base[:, None] + (row_index < remainder[:, None])).ravel()
        row_rates = np.repeat(per_unit, rows_per_cluster)
        successes = rng.binomial(row_trials, row_rates).astype(np.float64)
        ids = np.repeat(
            np.arange(n_clusters_per_group, dtype=np.int64) + first_id, rows_per_cluster
        )
        return RatioSample(
            numerator=successes,
            denominator=row_trials.astype(np.float64),
            cluster_ids=ids,
        )

    def draw(rng: np.random.Generator) -> tuple[RatioSample, RatioSample]:
        return (
            one_arm(rng, 0.0, 0),
            one_arm(rng, absolute_lift, n_clusters_per_group),
        )

    return draw


def ratio_p_value(control: RatioSample, treatment: RatioSample) -> float:
    """Adapter: the delta-method ratio test as a bare p-value."""
    return ratio_metric_test(control, treatment).p_value


def naive_ratio_p_value(control: RatioSample, treatment: RatioSample) -> float:
    """Adapter: the mistake - a Welch test on each unit's own ratio.

    What happens when a table already has a per-user rate column and somebody
    runs a t-test on it. Two things go wrong at once: the estimand becomes the
    mean of per-unit ratios rather than the ratio of totals, and every unit is
    weighted equally however little data it contributed.
    """
    return welch_t_test(
        control.numerator / control.denominator,
        treatment.numerator / treatment.denominator,
    ).p_value


def run_ratio_experiments(
    draw: RatioDraw,
    p_value_fn: RatioPValueFn,
    n_experiments: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
    label: str = "ratio metric",
) -> SimulationSummary:
    """Run many ratio-metric experiments, each analysed once at the end."""
    _check_run(n_experiments, alpha)
    tally = _Tally(n_experiments, alpha, label)

    for _ in range(n_experiments):
        control, treatment = draw(rng)
        estimate = treatment.ratio - control.ratio
        tally.record(estimate, _checked_p_value(p_value_fn, control, treatment) < alpha)

    return tally.summarise()


def covariate_draw(
    n_per_group: int,
    correlation: float,
    std_dev: float = 1.0,
    absolute_lift: float = 0.0,
    treatment_leaks_into_covariate: float = 0.0,
) -> CupedDraw:
    """Draw a metric alongside a covariate measured before the experiment.

    Args:
        correlation: Between the covariate and the metric. The variance CUPED
            can remove is its square, so this is the whole economics of the
            technique in one number.
        treatment_leaks_into_covariate: Fraction of the treatment effect that
            also shows up in the "pre-experiment" covariate. **Zero is the only
            honest setting**; anything else is the mistake CUPED invites, and it
            is a parameter so the mistake can be measured rather than described.
            A covariate the experiment touched sits on the causal path: adjusting
            for it subtracts part of the effect along with the noise, so the
            estimate shrinks toward zero and the experiment reports a null.

    The covariate is standard normal; the metric is that covariate scaled by the
    correlation plus independent noise, so the marginal variance is ``std_dev**2``
    whatever the correlation - the same discipline the clustered draw follows, and
    for the same reason.
    """
    if n_per_group < 2:
        raise ValueError(f"n_per_group must be at least 2, got {n_per_group}")
    if not -1.0 < correlation < 1.0:
        raise ValueError(f"correlation must be in (-1, 1), got {correlation}")
    if std_dev <= 0.0:
        raise ValueError(f"std_dev must be positive, got {std_dev}")

    explained, unexplained = correlation, np.sqrt(1.0 - correlation**2)

    def one_arm(rng: np.random.Generator, lift: float) -> CupedSample:
        covariate = rng.normal(0.0, 1.0, n_per_group)
        noise = rng.normal(0.0, 1.0, n_per_group)
        metric = std_dev * (explained * covariate + unexplained * noise) + lift
        observed_covariate = covariate + treatment_leaks_into_covariate * lift
        return CupedSample(metric=metric, covariate=observed_covariate)

    def draw(rng: np.random.Generator) -> tuple[CupedSample, CupedSample]:
        return one_arm(rng, 0.0), one_arm(rng, absolute_lift)

    return draw


def cuped_test(control: CupedSample, treatment: CupedSample) -> TestResult:
    """Adapter: the covariate-adjusted test."""
    return cuped_t_test(control, treatment)


def unadjusted_test(control: CupedSample, treatment: CupedSample) -> TestResult:
    """Adapter: the same experiment analysed without the covariate."""
    return welch_t_test(control.metric, treatment.metric)


def run_cuped_experiments(
    draw: CupedDraw,
    test_fn: CupedTestFn,
    n_experiments: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
    label: str = "covariate-adjusted",
) -> SimulationSummary:
    """Run many experiments, recording what each analysis *estimated*.

    The only runner that takes a whole result rather than a p-value, because the
    failure being checked here is a point estimate shrunk toward zero. A
    rejection rate alone cannot tell that apart from an experiment that simply
    had no effect to find, which is exactly what makes the mistake survivable.
    """
    _check_run(n_experiments, alpha)
    tally = _Tally(n_experiments, alpha, label)

    for _ in range(n_experiments):
        control, treatment = draw(rng)
        result = test_fn(control, treatment)
        p_value = float(result.p_value)
        if not np.isfinite(p_value):
            raise ValueError(f"{result.test} returned a non-finite p-value")
        tally.record(float(result.estimate), p_value < alpha)

    return tally.summarise()


def welch_p_value(control: NDArray[np.float64], treatment: NDArray[np.float64]) -> float:
    """Adapter: Welch's t-test as a bare p-value."""
    return welch_t_test(control, treatment).p_value


def welch_suite_p_values(
    control: NDArray[np.float64], treatment: NDArray[np.float64]
) -> tuple[float, ...]:
    """Adapter: one Welch p-value per metric.

    Deliberately loops over this package's own `welch_t_test` rather than
    calling a vectorised routine directly. The harness exists to measure what
    this package computes, and an adapter that quietly used a different code
    path would be measuring something else.
    """
    return tuple(
        welch_t_test(control[metric], treatment[metric]).p_value
        for metric in range(control.shape[0])
    )


def naive_welch_p_value(control: ClusteredSample, treatment: ClusteredSample) -> float:
    """Adapter: the broken baseline - every row treated as its own observation.

    Not a straw man. This is what an analysis does by default when the table has
    one row per session and nobody asked which user it belonged to.
    """
    return welch_t_test(control.values, treatment.values).p_value


def cluster_robust_p_value(control: ClusteredSample, treatment: ClusteredSample) -> float:
    """Adapter: the correction, on exactly the same data."""
    return cluster_robust_t_test(control, treatment).p_value


def proportion_p_value(control: NDArray[np.float64], treatment: NDArray[np.float64]) -> float:
    """Adapter: two-proportion z-test over 0/1 arrays."""
    return proportion_test(
        control_successes=int(control.sum()),
        control_total=int(control.size),
        treatment_successes=int(treatment.sum()),
        treatment_total=int(treatment.size),
    ).p_value


def msprt_p_value(tau: float) -> PValueFn:
    """Adapter factory: the always-valid mSPRT p-value at a given ``tau``."""

    def p_value_fn(control: NDArray[np.float64], treatment: NDArray[np.float64]) -> float:
        return msprt(control, treatment, tau=tau).p_value

    return p_value_fn


def run_experiments(
    draw: Draw,
    p_value_fn: PValueFn,
    n_experiments: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
    label: str = "single look",
    estimate_fn: EstimateFn | None = None,
) -> SimulationSummary:
    """Run many independent experiments, each analysed once at the end.

    This is the correct, non-peeking use of a fixed-horizon test: one decision,
    at the sample size the experiment was designed for.

    Args:
        estimate_fn: What to report as the effect. Defaults to
            :func:`mean_difference`; supply another when ``p_value_fn`` tests
            something other than a difference of means.
    """
    _check_run(n_experiments, alpha)
    estimate_of = mean_difference if estimate_fn is None else estimate_fn
    tally = _Tally(n_experiments, alpha, label)

    for _ in range(n_experiments):
        control, treatment = draw(rng)
        estimate = estimate_of(control, treatment)
        tally.record(estimate, _checked_p_value(p_value_fn, control, treatment) < alpha)

    return tally.summarise()


def run_clustered_experiments(
    draw: ClusteredDraw,
    p_value_fn: ClusteredPValueFn,
    n_experiments: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
    label: str = "clustered",
) -> SimulationSummary:
    """Run many clustered experiments, each analysed once at the end.

    Same tally, same summary and therefore the same table as the peeking runs -
    only the shape of one experiment's data differs. Pass
    :func:`naive_welch_p_value` to measure what ignoring the clustering costs,
    and :func:`cluster_robust_p_value` to measure the correction on identical
    draws.
    """
    _check_run(n_experiments, alpha)
    tally = _Tally(n_experiments, alpha, label)

    for _ in range(n_experiments):
        control, treatment = draw(rng)
        estimate = float(treatment.values.mean() - control.values.mean())
        tally.record(estimate, _checked_p_value(p_value_fn, control, treatment) < alpha)

    return tally.summarise()


def run_metric_suite(
    draw: SuiteDraw,
    p_value_fn: SuitePValueFn,
    n_experiments: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
    correction: Correction | None = None,
    is_null: Sequence[bool] | None = None,
    label: str = "metric suite",
) -> SimulationSummary:
    """Run many experiments measured on several metrics at once.

    Args:
        correction: ``None`` reads every metric at alpha, which is the mistake
            being measured. Pass :func:`~ab_lab.multiplicity.holm` or another
            entry from ``CORRECTIONS`` to measure the fix.
        is_null: Which metrics have no true effect. Required to measure a false
            discovery rate, because that needs to know which rejections were
            wrong. Omit it under a global null if only the family-wise rate is
            wanted.

    ``rejection_rate`` on the returned summary is the **family-wise** error rate:
    the share of experiments in which at least one comparison was rejected. That
    is a property of the joint outcome, which is why a metric suite needs its own
    runner rather than K passes of :func:`run_experiments`.
    """
    _check_run(n_experiments, alpha)
    if is_null is not None and not any(is_null):
        raise ValueError("is_null marks no metric as null: nothing could be a false discovery")

    tally = _Tally(n_experiments, alpha, label)
    n_comparisons: int | None = None
    false_discovery_total = 0.0
    family_wise_errors = 0

    for _ in range(n_experiments):
        control, treatment = draw(rng)
        p_values = _checked_suite_p_values(p_value_fn, control, treatment)

        if n_comparisons is None:
            n_comparisons = len(p_values)
            if is_null is not None and len(is_null) != n_comparisons:
                raise ValueError(
                    f"is_null has {len(is_null)} entries for {n_comparisons} metrics"
                )
        elif len(p_values) != n_comparisons:
            raise ValueError(
                f"the p-value function returned {len(p_values)} values after "
                f"returning {n_comparisons} on an earlier experiment"
            )

        if correction is None:
            rejected = tuple(value < alpha for value in p_values)
        else:
            rejected = correction(p_values, alpha).rejected

        n_rejected = sum(rejected)
        if is_null is not None and n_rejected:
            false_rejections = sum(
                1 for flag, null in zip(rejected, is_null, strict=True) if flag and null
            )
            # Two different things, and conflating them is why this runner needs
            # to know which metrics are null. The family-wise error rate counts
            # *experiments containing a mistake*; under a partial null the plain
            # rejection rate counts experiments containing anything at all, and
            # with real effects present that is close to 1 whatever the
            # correction does.
            if false_rejections:
                family_wise_errors += 1
            # And the false discovery *rate* is E[V/R] - a mean of per-experiment
            # ratios. A pooled sum(V)/sum(R) is a different quantity that merely
            # resembles it, so the ratio is accumulated per experiment.
            false_discovery_total += false_rejections / n_rejected

        estimate = float(np.mean(treatment.mean(axis=1) - control.mean(axis=1)))
        tally.record(estimate, n_rejected > 0)

    return replace(
        tally.summarise(),
        n_comparisons=n_comparisons,
        n_family_wise_errors=family_wise_errors if is_null is not None else None,
        mean_false_discovery_proportion=(
            false_discovery_total / n_experiments if is_null is not None else None
        ),
    )


def _checked_suite_p_values(
    p_value_fn: SuitePValueFn,
    control: NDArray[np.float64],
    treatment: NDArray[np.float64],
) -> tuple[float, ...]:
    """One p-value per metric, and none of them silently NaN."""
    p_values = tuple(float(value) for value in p_value_fn(control, treatment))
    if not p_values:
        raise ValueError("the p-value function returned no p-values")
    if not all(np.isfinite(value) for value in p_values):
        raise ValueError(
            f"the p-value function returned a non-finite value in {p_values} for "
            f"{control.shape[0]} metrics on {control.shape[1]} units per arm"
        )
    return p_values


def run_with_peeking(
    draw: Draw,
    p_value_fn: PValueFn,
    look_sizes: Sequence[int],
    n_experiments: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
    label: str = "peeking",
    estimate_fn: EstimateFn | None = None,
) -> SimulationSummary:
    """Run many experiments, stopping at the first look that reaches ``alpha``.

    Args:
        draw: Must produce at least ``max(look_sizes)`` observations per group;
            each look reads a prefix, which mimics data arriving over time.
        look_sizes: Cumulative sample sizes at which the results are checked -
            ``[200, 400, 600]`` is "peek every 200 users".

    With a fixed-horizon ``p_value_fn`` the resulting rejection rate is the
    inflated type I error. With a sequential one it stays at or below alpha,
    and the two runs are directly comparable because nothing else differs.
    """
    _check_run(n_experiments, alpha)
    if not look_sizes:
        raise ValueError("look_sizes must contain at least one look")
    if any(size < 2 for size in look_sizes):
        raise ValueError(f"every look needs at least 2 observations, got {tuple(look_sizes)}")
    if list(look_sizes) != sorted(look_sizes):
        raise ValueError(f"look_sizes must be increasing, got {tuple(look_sizes)}")

    estimate_of = mean_difference if estimate_fn is None else estimate_fn
    tally = _Tally(n_experiments, alpha, label)

    for _ in range(n_experiments):
        control, treatment = draw(rng)
        if control.size < max(look_sizes) or treatment.size < max(look_sizes):
            raise ValueError(
                f"draw produced {min(control.size, treatment.size)} observations per group "
                f"but the last look needs {max(look_sizes)}"
            )
        stopped_at = len(look_sizes) - 1
        stopped_early = False
        for index, size in enumerate(look_sizes):
            if _checked_p_value(p_value_fn, control[:size], treatment[:size]) < alpha:
                stopped_at = index
                stopped_early = True
                break
        # Report the effect as it stood when the experiment was stopped: this
        # is what an owner would have shipped, and it is biased upward exactly
        # because stopping happened on a large observed difference.
        final = look_sizes[stopped_at]
        tally.record(estimate_of(control[:final], treatment[:final]), stopped_early)

    return tally.summarise()


def peeking_curve(
    draw: Draw,
    p_value_fn: PValueFn,
    max_n: int,
    look_counts: Sequence[int],
    n_experiments: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> list[SimulationSummary]:
    """Type I error as a function of how many times the results are checked.

    Produces the headline curve: one look holds at alpha, and every additional
    look pushes the false positive rate further above it.
    """
    # Validated up front: a bad schedule at the end of the list would otherwise
    # surface only after the earlier, minutes-long cells had already run.
    if any(count < 1 for count in look_counts):
        raise ValueError(f"look counts must be positive, got {tuple(look_counts)}")
    schedules = [_even_looks(max_n, count) for count in look_counts]

    summaries: list[SimulationSummary] = []
    for count, look_sizes in zip(look_counts, schedules, strict=True):
        summaries.append(
            run_with_peeking(
                draw,
                p_value_fn,
                look_sizes,
                n_experiments,
                rng,
                alpha=alpha,
                label=f"{count} look(s)",
            )
        )
    return summaries


def _even_looks(max_n: int, count: int) -> list[int]:
    """Evenly spaced cumulative look sizes ending exactly at ``max_n``."""
    if max_n < 2 * count:
        raise ValueError(f"max_n={max_n} is too small for {count} looks of >= 2 observations")
    step = max_n / count
    return [int(round(step * (index + 1))) for index in range(count)]


def _checked_p_value(p_value_fn, control, treatment) -> float:  # noqa: ANN001
    """Call the p-value function and refuse to silently count a NaN.

    A non-finite p-value compares False against alpha, so it would be tallied
    as "no rejection" - and a harness that reports a 0% false positive rate
    because every p-value was NaN is worse than one that crashes.

    Untyped arguments on purpose: this guard is the one thing every runner must
    share, and the runners differ in what one experiment's data looks like.
    Duplicating it per shape is how one of them ends up without it.
    """
    p_value = float(p_value_fn(control, treatment))
    if not np.isfinite(p_value):
        raise ValueError(
            f"the p-value function returned {p_value} for a sample of "
            f"{_observation_count(control)} control and "
            f"{_observation_count(treatment)} treatment units"
        )
    return p_value


def _observation_count(sample) -> int:  # noqa: ANN001
    values = getattr(sample, "values", getattr(sample, "numerator", sample))
    return int(values.size)


def _check_run(n_experiments: int, alpha: float) -> None:
    if n_experiments < 1:
        raise ValueError(f"n_experiments must be positive, got {n_experiments}")
    check_alpha(alpha)
