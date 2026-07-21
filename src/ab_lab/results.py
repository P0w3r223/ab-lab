"""Immutable result types returned by every public function in the package.

The library never prints and never plots: it returns data. Reporting lives in
notebooks and in :mod:`ab_lab.report`-style user code, so that every number can
be asserted in a test.
"""

from __future__ import annotations

from dataclasses import dataclass


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
    """

    n_control: int
    n_treatment: int
    estimate: float
    likelihood_ratio: float
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
