"""Analysis of a finished two-group experiment.

Four procedures, each with the assumptions that make it the right choice:

* :func:`welch_t_test` - difference of means, unequal variances (the default
  for revenue-like metrics);
* :func:`proportion_test` - difference of conversion rates;
* :func:`mann_whitney` - stochastic dominance, no distributional assumption,
  but it does not estimate a difference of means;
* :func:`bootstrap_diff` - difference of *any* statistic, at the cost of
  resampling and of a lower bound on the p-value.

Every function takes control first and treatment second, and every estimate is
``treatment - control``. Confidence intervals are always two-sided at level
``1 - alpha`` even when a one-sided test is requested: a one-sided interval
tends to be read as a two-sided one, which overstates precision.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from .results import ConfidenceInterval, TestResult

Alternative = Literal["two-sided", "less", "greater"]

# Bootstrap resampling is vectorised in chunks so that memory stays bounded
# regardless of sample size and resample count.
_MAX_CHUNK_ELEMENTS = 4_000_000


def _as_sample(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Validate one arm of the experiment and return it as a float array."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size < 2:
        raise ValueError(f"{name} needs at least 2 observations, got {array.size}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _check_alternative(alternative: str) -> Alternative:
    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError(
            f"alternative must be 'two-sided', 'less' or 'greater', got {alternative!r}"
        )
    return alternative  # type: ignore[return-value]


def _check_alpha(alpha: float) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return alpha


def welch_t_test(
    control: ArrayLike,
    treatment: ArrayLike,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> TestResult:
    """Welch's t-test on the difference of means.

    Welch rather than Student is the default on purpose: equal variances are an
    assumption you rarely get to check and almost never get for free in an
    experiment that changes user behaviour. The cost of Welch when variances
    *are* equal is a fraction of a degree of freedom.
    """
    alpha = _check_alpha(alpha)
    alternative = _check_alternative(alternative)
    control_sample = _as_sample(control, "control")
    treatment_sample = _as_sample(treatment, "treatment")

    result = stats.ttest_ind(
        treatment_sample, control_sample, equal_var=False, alternative=alternative
    )
    estimate = float(treatment_sample.mean() - control_sample.mean())

    # Welch-Satterthwaite: the standard error and the degrees of freedom both
    # come from the per-group variances, which is what makes the test robust.
    var_control = control_sample.var(ddof=1) / control_sample.size
    var_treatment = treatment_sample.var(ddof=1) / treatment_sample.size
    standard_error = float(np.sqrt(var_control + var_treatment))
    df = (var_control + var_treatment) ** 2 / (
        var_control**2 / (control_sample.size - 1)
        + var_treatment**2 / (treatment_sample.size - 1)
    )
    margin = float(stats.t.isf(alpha / 2.0, df) * standard_error)

    return TestResult(
        test="Welch's t-test",
        estimate=estimate,
        ci=ConfidenceInterval(estimate - margin, estimate + margin, 1.0 - alpha),
        p_value=float(result.pvalue),
        statistic=float(result.statistic),
        alternative=alternative,
        assumptions=(
            "observations are independent within and across groups",
            "group means are approximately normal (CLT: fine for large n, "
            "fragile for heavy-tailed metrics like revenue at small n)",
            "variances may differ between groups",
        ),
    )


def proportion_test(
    control_successes: int,
    control_total: int,
    treatment_successes: int,
    treatment_total: int,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> TestResult:
    """Two-proportion z-test on the absolute difference of conversion rates.

    The p-value uses the *pooled* variance (valid under the null, which is what
    a p-value is computed under) while the confidence interval uses the
    *unpooled* variance (valid under the observed rates, which is what an
    interval describes). They can therefore disagree at the margin - a real
    property of the procedure, not a bug, and worth knowing before someone
    points at a p of 0.049 next to an interval that touches zero.
    """
    alpha = _check_alpha(alpha)
    alternative = _check_alternative(alternative)
    for total, name in ((control_total, "control_total"), (treatment_total, "treatment_total")):
        if total <= 0:
            raise ValueError(f"{name} must be positive, got {total}")
    for successes, total, name in (
        (control_successes, control_total, "control"),
        (treatment_successes, treatment_total, "treatment"),
    ):
        if not 0 <= successes <= total:
            raise ValueError(
                f"{name}_successes must be in [0, {name}_total], got {successes} of {total}"
            )

    rate_control = control_successes / control_total
    rate_treatment = treatment_successes / treatment_total
    estimate = rate_treatment - rate_control

    pooled_rate = (control_successes + treatment_successes) / (control_total + treatment_total)
    pooled_se = np.sqrt(
        pooled_rate * (1.0 - pooled_rate) * (1.0 / control_total + 1.0 / treatment_total)
    )
    if pooled_se == 0.0:
        raise ValueError("no variation in either group: every unit converted, or none did")
    z_statistic = float(estimate / pooled_se)

    if alternative == "two-sided":
        p_value = float(2.0 * stats.norm.sf(abs(z_statistic)))
    elif alternative == "greater":
        p_value = float(stats.norm.sf(z_statistic))
    else:
        p_value = float(stats.norm.cdf(z_statistic))

    unpooled_se = float(
        np.sqrt(
            rate_control * (1.0 - rate_control) / control_total
            + rate_treatment * (1.0 - rate_treatment) / treatment_total
        )
    )
    margin = float(stats.norm.isf(alpha / 2.0) * unpooled_se)

    return TestResult(
        test="two-proportion z-test",
        estimate=estimate,
        ci=ConfidenceInterval(estimate - margin, estimate + margin, 1.0 - alpha),
        p_value=p_value,
        statistic=z_statistic,
        alternative=alternative,
        assumptions=(
            "each unit contributes one independent Bernoulli outcome",
            "normal approximation holds (rule of thumb: >= 10 successes and "
            ">= 10 failures expected in each group)",
            "p-value uses the pooled variance, the interval the unpooled one",
        ),
    )


def mann_whitney(
    control: ArrayLike,
    treatment: ArrayLike,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> TestResult:
    """Mann-Whitney U test for stochastic dominance.

    Note what the estimate is: the probability that a random treatment unit
    exceeds a random control unit (0.5 under the null), *not* a difference of
    means. Reaching for this test because a metric is skewed and then reporting
    the result as "revenue went up by X" is the standard way to misuse it - if
    the business cares about the total, the mean is the estimand and the
    bootstrap is the tool.
    """
    alpha = _check_alpha(alpha)
    alternative = _check_alternative(alternative)
    control_sample = _as_sample(control, "control")
    treatment_sample = _as_sample(treatment, "treatment")

    result = stats.mannwhitneyu(treatment_sample, control_sample, alternative=alternative)
    probability_of_superiority = float(
        result.statistic / (control_sample.size * treatment_sample.size)
    )

    return TestResult(
        test="Mann-Whitney U",
        estimate=probability_of_superiority,
        ci=None,
        p_value=float(result.pvalue),
        statistic=float(result.statistic),
        alternative=alternative,
        assumptions=(
            "observations are independent",
            "estimand is P(treatment > control), not a difference of means",
            "ties are handled by the normal approximation with a tie correction",
        ),
    )


def bootstrap_diff(
    control: ArrayLike,
    treatment: ArrayLike,
    statistic: Callable[..., NDArray[np.float64]] = np.mean,
    alpha: float = 0.05,
    n_resamples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> TestResult:
    """Percentile bootstrap for the difference of an arbitrary statistic.

    Args:
        statistic: Any callable with numpy's ``(array, axis=...)`` signature -
            ``np.mean``, ``np.median``, or a custom metric such as a trimmed
            mean. Applied per resample, per group.
        n_resamples: Number of bootstrap resamples. This bounds the resolution
            of the p-value: with ``B`` resamples nothing below ``2/(B+1)`` can
            be reported, so 10 000 resamples cannot produce a p under ~0.0002.

    The p-value is the achieved significance level implied by inverting the
    percentile interval, not a separately derived test. The percentile method
    is used rather than BCa deliberately: it is the one whose failure mode
    (bias with strongly skewed statistics at small n) is easy to state and to
    demonstrate in a simulation.
    """
    alpha = _check_alpha(alpha)
    control_sample = _as_sample(control, "control")
    treatment_sample = _as_sample(treatment, "treatment")
    if n_resamples < 100:
        raise ValueError(f"n_resamples must be at least 100, got {n_resamples}")
    generator = np.random.default_rng() if rng is None else rng

    observed = float(statistic(treatment_sample) - statistic(control_sample))
    differences = _bootstrap_differences(
        control_sample, treatment_sample, statistic, n_resamples, generator
    )

    low, high = np.quantile(differences, [alpha / 2.0, 1.0 - alpha / 2.0])
    # Two-sided achieved significance level, floored at the resolution the
    # resample count can actually support.
    tail = min(
        float(np.mean(differences <= 0.0)),
        float(np.mean(differences >= 0.0)),
    )
    p_value = max(2.0 * tail, 2.0 / (n_resamples + 1.0))

    return TestResult(
        test=f"percentile bootstrap ({getattr(statistic, '__name__', 'statistic')})",
        estimate=observed,
        ci=ConfidenceInterval(float(low), float(high), 1.0 - alpha),
        p_value=min(p_value, 1.0),
        statistic=observed / float(differences.std(ddof=1)) if differences.std(ddof=1) else 0.0,
        alternative="two-sided",
        assumptions=(
            "observations are independent and identically distributed per group",
            "the sample is large enough for its empirical distribution to stand "
            "in for the population one",
            f"p-value resolution is bounded below by 2/(n_resamples+1) = "
            f"{2.0 / (n_resamples + 1.0):.2g}",
        ),
    )


def _bootstrap_differences(
    control: NDArray[np.float64],
    treatment: NDArray[np.float64],
    statistic: Callable[..., NDArray[np.float64]],
    n_resamples: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Bootstrap distribution of ``statistic(treatment) - statistic(control)``.

    Resamples are drawn in chunks: fully vectorising 10 000 resamples of a
    large sample would allocate gigabytes, while looping one resample at a time
    is an order of magnitude slower than it needs to be.
    """
    widest_group = max(control.size, treatment.size)
    chunk_size = max(1, min(n_resamples, _MAX_CHUNK_ELEMENTS // widest_group))
    chunks: list[NDArray[np.float64]] = []

    remaining = n_resamples
    while remaining > 0:
        size = min(chunk_size, remaining)
        control_draws = rng.choice(control, size=(size, control.size), replace=True)
        treatment_draws = rng.choice(treatment, size=(size, treatment.size), replace=True)
        chunks.append(
            np.asarray(statistic(treatment_draws, axis=1))
            - np.asarray(statistic(control_draws, axis=1))
        )
        remaining -= size

    return np.concatenate(chunks)
