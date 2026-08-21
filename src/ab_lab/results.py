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

    @property
    def rejection_rate(self) -> float:
        """Share of simulated experiments declared significant."""
        return self.n_rejections / self.n_experiments

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
