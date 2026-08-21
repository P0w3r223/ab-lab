"""Immutable result types returned by the estimating functions in the package.

The library never prints and never plots: it returns data. Rendering lives in
``examples/`` and in ``sitegen/`` (see ADR 0004 and ADR 0007), so that every
number the package publishes can be asserted in a test.

The invariant is about *estimates*, not about every function: a procedure that
answers "what is the effect, and how sure am I?" returns a dataclass carrying
the caveats with the number. Functions returning a bare probability
(:func:`~ab_lab.power.power_t`), a standardised distance
(:func:`~ab_lab.power.cohens_h`) or a rescaling (:func:`~ab_lab.sequential.tau_from_mde`)
return a float, because there is nothing to attach.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: What a simulated rate is claimed to do. A fixed-horizon test promises its
#: type I error *equals* alpha; an anytime-valid one promises only *at most*.
Claim = Literal["equals", "at most"]

# How far an empirical rate may sit from its claim before the claim is in doubt.
# Two numbers because the claims differ. An equality claim can fail in either
# direction, so it gets 4 sigmas - roughly one false alarm per 16 000 checks,
# which keeps a fixed-seed suite quiet. An upper-bound claim can only fail
# upward, and a deliberately conservative procedure is *expected* to sit below
# its bound, so 3 sigmas of slack above the bound is the whole test.
SIGMAS_FOR_EQUALITY = 4.0
SIGMAS_FOR_UPPER_BOUND = 3.0


@dataclass(frozen=True)
class ConfidenceInterval:
    """A two-sided interval estimate for an effect."""

    low: float
    high: float
    level: float

    @property
    def excludes_zero(self) -> bool:
        """True when the interval is entirely above or entirely below zero."""
        return self.low > 0.0 or self.high < 0.0

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.low:.6g}, {self.high:.6g}] ({self.level:.0%})"


@dataclass(frozen=True)
class TestResult:
    """Outcome of a two-group hypothesis test.

    Attributes:
        test: Human-readable name of the procedure that produced this result.
        estimate: Point estimate of the effect (treatment minus control, unless
            the test's docstring states another scale).
        ci: Interval estimate for ``estimate``, or ``None`` when the procedure
            has no standard interval (e.g. Mann-Whitney).
        p_value: Two-sided or one-sided p-value, matching ``alternative``.
        statistic: The test statistic itself (t, z, U, ...).
        alternative: Which alternative hypothesis was tested.
        assumptions: What has to hold for this result to be valid. Carried with
            the number on purpose - an estimate without its assumptions is how
            experiments get misread.

    Every field here is deliberately non-default, and must stay that way:
    :class:`ClusterTestResult` subclasses this and appends *required* fields, so
    giving any field below a default would make that subclass a ``TypeError`` at
    class-definition time ("non-default argument follows default argument"). The
    house move of appending an optional field - which :class:`SimulationSummary`
    uses - is not available on this class.
    """

    test: str
    estimate: float
    ci: ConfidenceInterval | None
    p_value: float
    statistic: float
    alternative: str
    assumptions: tuple[str, ...]

    def is_significant(self, alpha: float = 0.05) -> bool:
        """True when the p-value falls below ``alpha``."""
        return self.p_value < alpha


@dataclass(frozen=True)
class ClusterTestResult(TestResult):
    """A two-group test whose standard error accounts for clustered units.

    The only subclass in this package, and the exception is deliberate. A
    cluster-robust result *is* a :class:`TestResult` - ``isinstance`` holds,
    ``.p_value`` works, every adapter in :mod:`ab_lab.simulate` takes it
    unchanged - but its guarantee is different, and the distinct type says so:
    the sandwich estimator is anti-conservative below roughly 40 clusters, which
    a Welch result never is. Composition would have cost ``.test_result.p_value``
    at every call site to hide that.

    Attributes:
        design_effect: The **realised** ratio of the cluster-robust variance to
            the variance an independence assumption would have given. This is the
            number that makes the finding concrete: at 3.7, an interval computed
            as though the rows were independent is a factor of sqrt(3.7) too
            narrow. It is measured from this data rather than derived from an
            estimated intraclass correlation.
        effective_n: Total observations divided by the design effect - how many
            genuinely independent observations this sample is worth.
        df: Degrees of freedom, ``n_clusters_control + n_clusters_treatment - 2``.
            The convention matters when comparing against another
            implementation; see ADR 0008.
    """

    n_clusters_control: int
    n_clusters_treatment: int
    mean_cluster_size: float
    design_effect: float
    effective_n: float
    df: float

    @property
    def n_clusters(self) -> int:
        """Clusters across both arms - what the guarantee actually depends on."""
        return self.n_clusters_control + self.n_clusters_treatment


@dataclass(frozen=True)
class SampleSizeResult:
    """Required sample size for a planned experiment.

    ``exact_per_group`` is the fractional solution of the power equation;
    ``per_group`` is that value rounded up, which is what you actually ship.
    """

    per_group: int
    exact_per_group: float
    total: int
    mde: float
    alpha: float
    power: float
    alternative: str
    method: str

    @property
    def rounding_slack(self) -> float:
        """How many units of sample size the ceiling added."""
        return self.per_group - self.exact_per_group


@dataclass(frozen=True)
class CupedResult(TestResult):
    """A test run on a metric with its pre-experiment noise subtracted out.

    Attributes:
        theta: The coefficient applied to the centred covariate. Reported because
            it is the one number that says whether the covariate was worth using
            at all, and because a wildly different theta between two runs of the
            same experiment means the covariate is unstable.
        correlation: Pooled correlation between metric and covariate. The
            variance reduction is its square, so 0.7 buys about half.
        variance_reduction: The **realised** share of variance removed, measured
            on this data rather than predicted from the correlation. Squaring the
            correlation gives the expectation; this gives what happened.
        unadjusted_estimate: The same difference without the adjustment. It
            should be close to ``estimate`` - CUPED changes the precision, not
            the answer - and a gap between them is evidence that randomisation
            did not balance the covariate.
    """

    theta: float
    correlation: float
    variance_reduction: float
    unadjusted_estimate: float

    @property
    def effective_sample_multiplier(self) -> float:
        """How many times the sample the adjustment is worth.

        A 50% variance reduction is worth twice the traffic, because the variance
        of a mean falls as 1/n. This is the number to put in front of anyone
        deciding whether the covariate is worth the pipeline it needs.
        """
        remaining = 1.0 - self.variance_reduction
        return 1.0 / remaining if remaining > 0.0 else float("inf")


@dataclass(frozen=True)
class RatioTestResult(TestResult):
    """A difference of two ratios of totals.

    Carries both arms' ratios rather than only their difference, because a ratio
    metric is the case where the absolute difference is least readable on its
    own: 0.004 means one thing on a 2% click-through rate and another on a 40%
    one. ``relative_effect`` is the difference over the control ratio - the
    "+20%" a business actually discusses - and it is reported beside the
    absolute figure rather than instead of it, since a relative lift on a tiny
    baseline is how small effects get oversold.
    """

    control_ratio: float
    treatment_ratio: float
    relative_effect: float
    n_clusters_control: int
    n_clusters_treatment: int
    df: float

    @property
    def n_clusters(self) -> int:
        return self.n_clusters_control + self.n_clusters_treatment


@dataclass(frozen=True)
class ClusteredSampleSizeResult:
    """Required sample size when each unit contributes several observations.

    Deliberately *not* a subclass of :class:`SampleSizeResult`, unlike
    :class:`ClusterTestResult` which does subclass :class:`TestResult`. The
    inheritance there buys something concrete - every adapter in
    :mod:`ab_lab.simulate` consumes a ``TestResult`` and keeps working. Nothing
    consumes a ``SampleSizeResult`` polymorphically, so subclassing here would
    add a hierarchy for the look of consistency and gain nothing.

    Attributes:
        n_clusters_per_group: The number that actually has to be recruited.
        per_group: Observations per arm, ``n_clusters_per_group * mean_cluster_size``.
        independent_per_group: What the same design would have needed if every
            observation were its own unit - the number an experiment gets sized
            at by mistake.
        design_effect: The factor between the two.
    """

    n_clusters_per_group: int
    per_group: float
    independent_per_group: float
    design_effect: float
    icc: float
    mean_cluster_size: float
    mde: float
    alpha: float
    power: float
    alternative: str
    method: str
    assumptions: tuple[str, ...]

    @property
    def extra_units_clustering_costs(self) -> float:
        """How many more observations per arm the clustering is charging."""
        return self.per_group - self.independent_per_group


@dataclass(frozen=True)
class MdeResult:
    """The smallest effect a planned experiment can actually see.

    The answer to the question worth asking before launching - not "is this
    enough traffic?" but "what is the smallest effect this traffic can detect at
    all?".

    ``mde`` is a positive magnitude on the metric's own scale. It is returned
    inside a result rather than as a bare float because the number is meaningless
    without the design that produced it: the same 5 000 units per arm give a
    different answer at 80% power than at 90%, and - for a proportion - a
    different answer for a drop than for a lift.

    Attributes:
        mde: Smallest detectable effect, as a positive magnitude.
        n_per_group: The sample size it was solved for.
        direction: ``"increase"``, ``"decrease"``, or ``"either"`` for a metric
            whose scale is symmetric (a mean, where a lift and a drop of the same
            size cost the same).
        assumptions: What has to hold for this number to mean what it says.
    """

    mde: float
    n_per_group: float
    alpha: float
    power: float
    alternative: str
    direction: str
    method: str
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class SrmResult:
    """Sample ratio mismatch check on the observed group allocation."""

    observed: tuple[int, ...]
    expected: tuple[float, ...]
    statistic: float
    p_value: float
    alpha: float

    @property
    def is_mismatch(self) -> bool:
        """True when the allocation deviates more than randomisation explains.

        A mismatch invalidates the experiment itself, not just its effect size:
        it means units were not assigned the way the design assumed.
        """
        return self.p_value < self.alpha


@dataclass(frozen=True)
class SequentialResult:
    """One look at a running experiment under a sequential test.

    ``p_value`` is an *always-valid* p-value: it may be compared to alpha at
    every look without inflating the type I error, which is exactly what a
    fixed-horizon p-value may not do.

    ``likelihood_ratio`` saturates at ``math.inf`` under overwhelming evidence;
    ``log_likelihood_ratio`` is the same quantity on a scale that stays finite,
    and is the one to read when the magnitude matters.
    """

    n_control: int
    n_treatment: int
    estimate: float
    likelihood_ratio: float
    log_likelihood_ratio: float
    p_value: float
    alpha: float
    tau: float

    @property
    def should_stop(self) -> bool:
        """True when the evidence crosses the boundary and the test may stop."""
        return self.p_value < self.alpha


@dataclass(frozen=True)
class MultipleComparisonResult:
    """One family of tests, and which of them survive the correction.

    ``error_rate_controlled`` is the field that matters, and it is a string
    rather than a flag because the two guarantees are not stronger and weaker
    versions of one thing. Controlling the family-wise error rate bounds the
    probability of **any** false positive in the family. Controlling the false
    discovery rate bounds the expected **share** of the rejections that are
    false, and says nothing about whether there is one. Reading a
    Benjamini-Hochberg result as though it were Holm is the usual way this goes
    wrong, so the promise travels with the numbers.

    Attributes:
        adjusted_p_values: Comparable to alpha directly. Returned rather than a
            bare mask so the family can be re-thresholded without re-running,
            and because that is what makes the comparison against a reference
            implementation exact.
        labels: What the family consists of. Optional, but naming the members is
            the whole discipline: a correction is meaningless until "the family"
            is defined, and the definition is a decision, not a computation.
    """

    method: str
    alpha: float
    p_values: tuple[float, ...]
    adjusted_p_values: tuple[float, ...]
    rejected: tuple[bool, ...]
    error_rate_controlled: str
    assumptions: tuple[str, ...]
    labels: tuple[str, ...] | None = None

    @property
    def n_comparisons(self) -> int:
        return len(self.p_values)

    @property
    def n_rejected(self) -> int:
        return sum(self.rejected)

    @property
    def any_rejected(self) -> bool:
        return any(self.rejected)

    def for_label(self, label: str) -> tuple[float, bool]:
        """The adjusted p-value and verdict for one named member of the family."""
        if self.labels is None:
            raise ValueError("this result has no labels; pass labels= to the correction")
        if label not in self.labels:
            raise KeyError(f"{label!r} is not in this family: {self.labels}")
        index = self.labels.index(label)
        return self.adjusted_p_values[index], self.rejected[index]


@dataclass(frozen=True)
class SimulationSummary:
    """Empirical behaviour of a procedure over many simulated experiments.

    For A/A runs ``rejection_rate`` estimates the type I error; for A/B runs it
    estimates statistical power.
    """

    n_experiments: int
    n_rejections: int
    nominal_alpha: float
    mean_estimate: float
    label: str
    # Mean |effect| among the experiments that were declared significant. The
    # winner's curse lives here: a *signed* average cancels out under A/A, so
    # it cannot show that stopping early inflates what gets reported.
    mean_absolute_estimate_when_stopped: float | None = None
    # Set only by a metric-suite run. `n_rejections` then counts experiments with
    # *at least one* rejection, so `rejection_rate` is the family-wise error rate;
    # the false discovery rate needs its own accumulator because it is a mean of
    # per-experiment ratios and cannot be recovered from totals.
    n_comparisons: int | None = None
    n_family_wise_errors: int | None = None
    mean_false_discovery_proportion: float | None = None

    @property
    def rejection_rate(self) -> float:
        """Share of simulated experiments declared significant.

        For a metric suite this is the share of experiments in which *anything*
        was rejected. Under a global null that is the family-wise error rate;
        under a partial null it is not, because true rejections count towards it
        too - see :attr:`family_wise_error_rate`.
        """
        return self.n_rejections / self.n_experiments

    @property
    def family_wise_error_rate(self) -> float:
        """Share of experiments containing at least one **false** rejection.

        The quantity Bonferroni and Holm actually promise to bound. It differs
        from :attr:`rejection_rate` exactly when some metrics have a real effect,
        which is the setting where Benjamini-Hochberg can be told apart from
        Holm - so conflating the two makes that comparison unmeasurable.
        """
        if self.n_family_wise_errors is None:
            raise ValueError(
                "this summary does not know which comparisons were null; pass "
                "is_null= to the metric-suite runner to measure a false positive rate"
            )
        return self.n_family_wise_errors / self.n_experiments

    def family_wise_monte_carlo_error(self) -> float:
        """Standard error of :attr:`family_wise_error_rate`."""
        rate = self.family_wise_error_rate
        return (rate * (1.0 - rate) / self.n_experiments) ** 0.5

    @property
    def monte_carlo_error(self) -> float:
        """Standard error of ``rejection_rate`` - the noise floor of the run.

        Comparing an empirical rate to a target without this number is how
        simulation studies get over-read.
        """
        rate = self.rejection_rate
        return (rate * (1.0 - rate) / self.n_experiments) ** 0.5

    def agrees_with(
        self,
        expected: float,
        claim: Claim = "equals",
        sigmas: float | None = None,
    ) -> bool:
        """Does the empirical rate support the claim the procedure makes?

        Args:
            expected: The rate the theory promises.
            claim: ``"equals"`` for a fixed-horizon procedure, whose type I error
                is supposed to *be* alpha; ``"at most"`` for an anytime-valid one,
                which only promises not to exceed it. Judging the second by the
                first's standard reports correct, deliberately conservative
                behaviour as a failure - which is exactly what the first version
                of the validation table did.
            sigmas: Override the default tolerance. Defaults to
                :data:`SIGMAS_FOR_EQUALITY` or :data:`SIGMAS_FOR_UPPER_BOUND`
                according to ``claim``, so that no caller has to name a bare
                number to get the standard behaviour.

        The comparison is in multiples of :attr:`monte_carlo_error` rather than
        against a fixed threshold, so the answer does not silently depend on how
        many experiments were run.
        """
        if claim == "equals":
            tolerance = SIGMAS_FOR_EQUALITY if sigmas is None else sigmas
            return abs(self.rejection_rate - expected) < tolerance * self.monte_carlo_error
        if claim == "at most":
            tolerance = SIGMAS_FOR_UPPER_BOUND if sigmas is None else sigmas
            return self.rejection_rate <= expected + tolerance * self.monte_carlo_error
        raise ValueError(f"claim must be 'equals' or 'at most', got {claim!r}")
