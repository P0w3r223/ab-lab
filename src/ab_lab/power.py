"""Experiment design: power, minimum detectable effect and sample size.

Everything here is built on one standardised-effect core and two distributions:

* :func:`power_z` - normal approximation, used for proportions;
* :func:`power_t` - exact noncentral-t power, used for means.

The practitioner-facing helpers (:func:`sample_size_for_mean`,
:func:`sample_size_for_proportion`) only translate domain quantities into a
standardised effect size and then invert the corresponding power function
numerically. Keeping the core separate is what makes the whole module
checkable against ``statsmodels`` in a handful of tests.

Sign conventions: effect sizes are used as absolute values, so a lift and a
drop of the same magnitude need the same sample size. ``alternative`` is either
``"two-sided"`` or ``"one-sided"``; the one-sided variant spends the full alpha
in a single tail.
"""

from __future__ import annotations

import math
from typing import Literal

from scipy import optimize, stats

from .results import SampleSizeResult

Alternative = Literal["two-sided", "one-sided"]

# The power equation is solved numerically inside these bounds. The lower bound
# keeps the t-distribution's degrees of freedom positive; the upper bound turns
# "this effect is too small to measure" into an error instead of a silent
# multi-year experiment.
_MIN_N_PER_GROUP = 2.0
_MAX_N_PER_GROUP = 1e9

# Two-sided power includes rejections in the direction *opposite* to the true
# effect. That term dies off faster than exponentially, and SciPy's noncentral
# t returns NaN for parts of the region where it is already below 1e-15. It is
# therefore evaluated only while an upper bound on it can still move the
# result - see _far_tail_bound.
_NEGLIGIBLE_FAR_TAIL = 1e-16


def _tail_alpha(alpha: float, alternative: Alternative) -> float:
    """Alpha spent in the tail the alternative points at."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if alternative == "two-sided":
        return alpha / 2.0
    if alternative == "one-sided":
        return alpha
    raise ValueError(f"alternative must be 'two-sided' or 'one-sided', got {alternative!r}")


def _effective_n(n_per_group: float, ratio: float) -> float:
    """Harmonic-style effective sample size of a two-group comparison.

    With ``n1`` control and ``n2 = ratio * n1`` treatment units, the variance of
    the difference scales with ``1/n1 + 1/n2``; its reciprocal is the single
    number that drives power.
    """
    if n_per_group < _MIN_N_PER_GROUP:
        raise ValueError(f"n_per_group must be >= {_MIN_N_PER_GROUP}, got {n_per_group}")
    if ratio <= 0.0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    n_treatment = n_per_group * ratio
    return 1.0 / (1.0 / n_per_group + 1.0 / n_treatment)


def cohens_h(baseline_rate: float, variant_rate: float) -> float:
    """Cohen's h - the arcsine-transformed distance between two proportions.

    The transform stabilises the variance of a proportion, so a single
    standardised effect covers both a 1% -> 1.2% and a 40% -> 44% comparison
    without a separate variance term.
    """
    for rate in (baseline_rate, variant_rate):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"rates must be in [0, 1], got {rate}")
    return 2.0 * math.asin(math.sqrt(variant_rate)) - 2.0 * math.asin(math.sqrt(baseline_rate))


def power_z(
    effect_size: float,
    n_per_group: float,
    alpha: float = 0.05,
    ratio: float = 1.0,
    alternative: Alternative = "two-sided",
) -> float:
    """Power of a two-sample z-test for a standardised ``effect_size``.

    Normal approximation: assumes the test statistic is normal under both the
    null and the alternative, which holds for proportions at the sample sizes
    online experiments run at (rule of thumb: at least ~10 successes and ~10
    failures expected per group).
    """
    tail = _tail_alpha(alpha, alternative)
    critical = stats.norm.isf(tail)
    shift = abs(effect_size) * math.sqrt(_effective_n(n_per_group, ratio))
    power = stats.norm.sf(critical - shift)
    if alternative == "two-sided":
        # The far tail: rejections in the direction opposite to the true effect.
        # Negligible for large effects, but dropping it biases power downward.
        power += stats.norm.cdf(-critical - shift)
    return float(power)


def power_t(
    effect_size: float,
    n_per_group: float,
    alpha: float = 0.05,
    ratio: float = 1.0,
    alternative: Alternative = "two-sided",
) -> float:
    """Power of Student's two-sample t-test for a standardised ``effect_size``.

    Exact under normality: uses the noncentral t-distribution rather than the
    normal approximation, so it stays honest at the small sample sizes where
    estimating the variance actually costs power.
    """
    tail = _tail_alpha(alpha, alternative)
    n_treatment = n_per_group * ratio
    df = n_per_group + n_treatment - 2.0
    if df <= 0.0:
        raise ValueError("need more than one unit per group for a t-test")
    noncentrality = abs(effect_size) * math.sqrt(_effective_n(n_per_group, ratio))
    critical = stats.t.isf(tail, df)
    power = float(stats.nct.sf(critical, df, noncentrality))
    far_tail_matters = _far_tail_bound(critical, noncentrality) > _NEGLIGIBLE_FAR_TAIL
    if alternative == "two-sided" and far_tail_matters:
        far_tail = float(stats.nct.cdf(-critical, df, noncentrality))
        if not math.isfinite(far_tail):
            raise ArithmeticError(
                f"noncentral t returned a non-finite lower tail at df={df:g}, "
                f"noncentrality={noncentrality:g}"
            )
        power += far_tail
    return power


def _far_tail_bound(critical: float, noncentrality: float) -> float:
    """Upper bound on the wrong-direction rejection probability.

    Power is a number of order one, so a term below double precision's relative
    resolution cannot change it. The normal tail is a cheap screen for exactly
    that: both distributions decay in ``critical + noncentrality``, and the
    threshold sits far enough above the point where the two differ.
    """
    return float(stats.norm.cdf(-critical - noncentrality))


def _solve_sample_size(
    power_fn,
    effect_size: float,
    alpha: float,
    power: float,
    ratio: float,
    alternative: Alternative,
) -> float:
    """Invert a power function: smallest ``n_per_group`` reaching ``power``.

    Power is monotone in n, so a bracketed root finder is both sufficient and
    robust - no gradients, no starting guess to tune.
    """
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}")
    if effect_size == 0.0:
        raise ValueError("effect_size must be non-zero: a null effect needs infinite sample")

    def gap(n: float) -> float:
        return power_fn(effect_size, n, alpha=alpha, ratio=ratio, alternative=alternative) - power

    if gap(_MAX_N_PER_GROUP) < 0.0:
        raise ValueError(
            f"effect_size={effect_size:g} needs more than {_MAX_N_PER_GROUP:g} units per group "
            f"for power={power:g}; the effect is too small to measure at this alpha"
        )
    if gap(_MIN_N_PER_GROUP) >= 0.0:
        return _MIN_N_PER_GROUP
    return float(optimize.brentq(gap, _MIN_N_PER_GROUP, _MAX_N_PER_GROUP, xtol=1e-8, rtol=1e-10))


def sample_size_for_mean(
    mde: float,
    std_dev: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    alternative: Alternative = "two-sided",
) -> SampleSizeResult:
    """Units per group needed to detect an absolute difference of ``mde``.

    Args:
        mde: Minimum detectable effect on the metric's own scale (e.g. 2.50 PLN
            of average order value).
        std_dev: Pooled standard deviation of the metric, from historical data.
        ratio: Treatment-to-control size ratio; 1.0 means an even split.

    The standardised effect is ``mde / std_dev`` (Cohen's d) and power is the
    exact noncentral-t power.
    """
    if std_dev <= 0.0:
        raise ValueError(f"std_dev must be positive, got {std_dev}")
    effect_size = mde / std_dev
    exact = _solve_sample_size(power_t, effect_size, alpha, power, ratio, alternative)
    per_group = math.ceil(exact)
    return SampleSizeResult(
        per_group=per_group,
        exact_per_group=exact,
        total=per_group + math.ceil(exact * ratio),
        mde=mde,
        alpha=alpha,
        power=power,
        alternative=alternative,
        method="two-sample t-test (noncentral t)",
    )


def sample_size_for_proportion(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    alternative: Alternative = "two-sided",
) -> SampleSizeResult:
    """Units per group needed to detect an absolute lift of ``mde`` in a rate.

    Args:
        baseline_rate: Current conversion rate, e.g. ``0.05``.
        mde: Absolute change worth detecting, in percentage points expressed as
            a fraction: ``0.005`` means 5.0% -> 5.5%, not 5% -> 5.025%.

    Sample size is driven by Cohen's h rather than by the raw difference, so
    the required n depends on where on the 0-1 scale the baseline sits. A
    proportion's variance peaks at 0.5, so the *same* 0.5pp lift is far more
    expensive to detect on a 50% baseline than on a 1% one - a fact worth
    knowing before promising a timeline for a high-conversion funnel step.
    """
    variant_rate = baseline_rate + mde
    if not 0.0 <= variant_rate <= 1.0:
        raise ValueError(
            f"baseline_rate + mde must stay in [0, 1], got {baseline_rate} + {mde}"
        )
    effect_size = cohens_h(baseline_rate, variant_rate)
    exact = _solve_sample_size(power_z, effect_size, alpha, power, ratio, alternative)
    per_group = math.ceil(exact)
    return SampleSizeResult(
        per_group=per_group,
        exact_per_group=exact,
        total=per_group + math.ceil(exact * ratio),
        mde=mde,
        alpha=alpha,
        power=power,
        alternative=alternative,
        method="two-proportion z-test (Cohen's h)",
    )


def mde_for_mean(
    n_per_group: float,
    std_dev: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    alternative: Alternative = "two-sided",
) -> float:
    """Smallest absolute mean difference detectable with the sample you have.

    The question every experiment owner should ask before launching: not "is
    this enough traffic?" but "what is the smallest effect this traffic can
    see at all?".
    """
    if std_dev <= 0.0:
        raise ValueError(f"std_dev must be positive, got {std_dev}")

    def gap(effect_size: float) -> float:
        return power_t(effect_size, n_per_group, alpha, ratio, alternative) - power

    effect_size = optimize.brentq(gap, 1e-9, 1e3, xtol=1e-10, rtol=1e-12)
    return float(effect_size * std_dev)


def mde_for_proportion(
    n_per_group: float,
    baseline_rate: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    alternative: Alternative = "two-sided",
) -> float:
    """Smallest absolute lift in a conversion rate detectable at ``n_per_group``.

    Returned on the rate scale (``0.004`` means 0.4 percentage points), found by
    inverting :func:`sample_size_for_proportion` numerically because Cohen's h
    has no closed-form inverse in terms of an absolute lift.
    """
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError(f"baseline_rate must be in (0, 1), got {baseline_rate}")

    def gap(mde: float) -> float:
        effect_size = cohens_h(baseline_rate, baseline_rate + mde)
        return power_z(effect_size, n_per_group, alpha, ratio, alternative) - power

    upper = (1.0 - baseline_rate) * (1.0 - 1e-9)
    if gap(upper) < 0.0:
        raise ValueError(
            f"no detectable lift at n_per_group={n_per_group:g} for "
            f"baseline_rate={baseline_rate:g}: even a jump to 100% stays under-powered"
        )
    return float(optimize.brentq(gap, 1e-12, upper, xtol=1e-12, rtol=1e-12))
