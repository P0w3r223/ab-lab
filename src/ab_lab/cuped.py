"""CUPED: the same experiment, on a fraction of the traffic.

Every other module here is about a test that lies. This one is about a test that
is honest and expensive. If a metric was already measurable *before* the
experiment started - last month's revenue for the same user, last week's session
count - then most of what makes the metric noisy is a property of the user rather
than of the treatment, and it can be subtracted out before anything is compared.

The construction is one line. With a pre-experiment covariate ``X`` and
``theta = Cov(Y, X) / Var(X)``, analyse ``Y - theta * (X - mean(X))`` instead of
``Y``. Subtracting a constant times a *pre-treatment* quantity cannot move the
expected difference between arms, because randomisation made ``X`` balanced - so
the estimate is unbiased for the same effect. What changes is the variance:

    Var(Y_adjusted) = Var(Y) * (1 - correlation**2)

A covariate correlated 0.7 with the metric removes about half the variance, which
is the same as doubling the sample. That is the entire appeal, and it is why the
technique is worth more to a real experimentation programme than any correction
in this package: the corrections stop you being wrong, this one lets you afford
to be right.

**The trap has one shape and it is worth stating precisely.** The covariate must
be measured before the treatment could possibly have influenced it. A covariate
the experiment touched is on the causal path: subtracting it removes part of the
effect along with the noise, and the estimate shrinks toward zero.

Measured here, on a true effect of 0.10 with a correlation of 0.7: with half the
effect leaking into the covariate the estimate comes back at 0.065 - **35% too
small** - and with all of it, 0.030, **70% too small**. The rejection rate falls
from 80% to 13% along the way.

So the failure does not look like success, and saying it did would be the easy
wrong story. It looks like a *null result*: a smaller effect, reported with a
respectable-looking interval, from an experiment that has never been more
precise. An experiment that concludes "no effect" when there was one is not
obviously the cheaper mistake, and nothing inside the numbers flags it -
which is why :mod:`ab_lab.simulate` measures it rather than describing it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import Alternative, as_sample, check_alpha, check_alternative
from .analyze import welch_t_test
from .results import CupedResult

# Below this the adjustment is not worth the explanation it costs: a correlation
# of 0.1 removes 1% of the variance. Not enforced - the result reports the
# realised reduction and the caller decides - but named so the docstring can be
# concrete about when this is a waste of everyone's time.
NEGLIGIBLE_CORRELATION = 0.1


@dataclass(frozen=True)
class CupedSample:
    """One arm: the metric, and a covariate measured before the experiment began.

    Bound together and validated once, for the reason the other paired types in
    this package exist - two arrays passed separately can be passed in the wrong
    order, and the result is a number rather than an error.
    """

    metric: NDArray[np.float64]
    covariate: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate on every path in, not only through :meth:`from_arrays`."""
        as_sample(self.metric, "metric")
        as_sample(self.covariate, "covariate")
        if self.metric.shape != self.covariate.shape:
            raise ValueError(
                f"metric has shape {self.metric.shape}, covariate has "
                f"{self.covariate.shape}; they must describe the same units"
            )

    @classmethod
    def from_arrays(cls, metric: ArrayLike, covariate: ArrayLike) -> CupedSample:
        """Coerce anything array-like, then construct - which validates.

        ``covariate[i]`` describes the same unit as ``metric[i]``.
        """
        return cls(
            metric=np.asarray(metric, dtype=np.float64),
            covariate=np.asarray(covariate, dtype=np.float64),
        )


def cuped_theta(control: CupedSample, treatment: CupedSample) -> float:
    """The coefficient that removes as much variance as the covariate can explain.

    Estimated on the two arms **pooled**. Pooling is the point: a per-arm theta
    would be fitted partly to the treatment effect, which is the thing being
    measured, and would bias the estimate toward whatever the fit happened to
    absorb.
    """
    metric = np.concatenate([control.metric, treatment.metric])
    covariate = np.concatenate([control.covariate, treatment.covariate])
    covariate_variance = float(covariate.var(ddof=1))
    if covariate_variance <= 0.0:
        raise ValueError("the covariate is constant: it can explain no variance")
    return float(np.cov(metric, covariate, ddof=1)[0, 1] / covariate_variance)


def cuped_t_test(
    control: CupedSample,
    treatment: CupedSample,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> CupedResult:
    """Welch's t-test on the covariate-adjusted metric.

    Args:
        control: Metric and pre-experiment covariate for the control arm.
        treatment: The same for the treatment arm.

    The estimate is on the metric's own scale and means the same thing as the
    unadjusted one - :attr:`~ab_lab.results.CupedResult.unadjusted_estimate` is
    reported beside it so the two can be compared, and they should be close. If
    they are not, the covariate was probably not balanced across arms, which
    randomisation was supposed to guarantee and a sample ratio mismatch would
    explain.

    What should differ is the interval, and by a factor the result reports.
    """
    alpha = check_alpha(alpha)
    alternative = check_alternative(alternative)

    theta = cuped_theta(control, treatment)
    covariate_mean = float(
        np.concatenate([control.covariate, treatment.covariate]).mean()
    )

    adjusted_control = control.metric - theta * (control.covariate - covariate_mean)
    adjusted_treatment = treatment.metric - theta * (treatment.covariate - covariate_mean)
    result = welch_t_test(adjusted_control, adjusted_treatment, alpha, alternative)

    pooled_metric = np.concatenate([control.metric, treatment.metric])
    pooled_adjusted = np.concatenate([adjusted_control, adjusted_treatment])
    before = float(pooled_metric.var(ddof=1))
    after = float(pooled_adjusted.var(ddof=1))
    reduction = 1.0 - after / before if before > 0.0 else 0.0

    correlation = float(
        np.corrcoef(pooled_metric, np.concatenate([control.covariate, treatment.covariate]))[
            0, 1
        ]
    )

    return CupedResult(
        test="CUPED-adjusted Welch's t-test",
        estimate=result.estimate,
        ci=result.ci,
        p_value=result.p_value,
        statistic=result.statistic,
        alternative=alternative,
        assumptions=(
            "the covariate was measured BEFORE the treatment could influence it - "
            "a covariate the experiment touched shrinks the estimate toward zero, "
            "and the result looks like a clean null rather than like an error",
            "theta is estimated on the pooled arms, so it cannot be fitted to the "
            "treatment effect",
            "randomisation balanced the covariate across arms; check the sample "
            "ratio if the adjusted and unadjusted estimates disagree",
            # Spelled out rather than fetched by running the unadjusted test a
            # second time. That doubled the work on every call, and it could
            # raise where the adjusted test succeeds: a metric constant within
            # each arm but differing between them leaves the *adjusted* arms with
            # variance, so this function returns a result while the second call
            # was reporting "both groups are constant" about it.
            "observations are independent within and across groups",
            "group means are approximately normal (CLT: fine for large n, "
            "fragile for heavy-tailed metrics like revenue at small n)",
            "variances may differ between groups",
        ),
        theta=theta,
        correlation=correlation,
        variance_reduction=reduction,
        unadjusted_estimate=float(treatment.metric.mean() - control.metric.mean()),
    )
