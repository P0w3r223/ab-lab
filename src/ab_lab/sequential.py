"""Sequential testing: looking at results early without paying for it.

A fixed-horizon p-value is only valid at the sample size the experiment was
designed for. Checking it daily and stopping at the first p below 0.05 turns a
nominal 5% false positive rate into roughly 30% (see
:mod:`ab_lab.simulate`) - the single most common way A/B programmes fool
themselves.

This module implements one correction, the **mixture sequential probability
ratio test (mSPRT)**, rather than three shallow ones. The reasoning is in
``docs/decisions/0003-msprt-over-group-sequential.md``; the short version is
that mSPRT needs no pre-committed number of looks and no pre-committed horizon,
which is what an always-on experimentation platform actually looks like.

How it works, in one paragraph: instead of asking "how extreme is this
difference if the true effect is zero?", the mSPRT asks "how much more likely
is this data under *some* non-zero effect than under no effect?", averaging
over a normal prior on the effect with standard deviation ``tau``. That ratio
is a martingale under the null, so Ville's inequality bounds the probability
that it *ever* exceeds ``1/alpha`` by ``alpha`` - across every look, taken at
any time, however many. The price is a choice of ``tau``: the test is most
powerful near effects of that size and loses power far from it.
"""

from __future__ import annotations

import math
import sys

from numpy.typing import ArrayLike

from ._validation import as_sample as _as_sample
from ._validation import check_alpha as _check_alpha
from .results import SequentialResult

# Above this, exp() overflows a float. The likelihood ratio grows like
# exp(z^2 / 2), so it leaves float range at |z| ~ 37.7 - reachable on a large
# experiment with a real effect, which is precisely when the test should be
# stopping rather than raising. Everything is therefore computed in log space.
_MAX_LOG = math.log(sys.float_info.max)


def log_mixture_likelihood_ratio(estimate: float, variance: float, tau: float) -> float:
    """Log likelihood ratio of the effect estimate under a N(0, tau^2) mixture.

    Args:
        estimate: Observed difference of means, treatment minus control.
        variance: Variance of that difference at the current sample size.
        tau: Standard deviation of the prior on the true effect.

    The closed form follows from the fact that mixing a normal likelihood with
    a normal prior just widens the variance::

        LR = sqrt(V / (V + tau^2)) * exp(estimate^2 * tau^2 / (2 * V * (V + tau^2)))

    The square-root factor is the price of the mixture - early on, when ``V``
    is large, it holds the ratio down and stops the test from firing on noise.
    This function returns the logarithm of that expression, which stays finite
    for any evidence a real experiment can produce.
    """
    if variance <= 0.0:
        raise ValueError(f"variance must be positive, got {variance}")
    if tau <= 0.0:
        raise ValueError(f"tau must be positive, got {tau}")
    widened = variance + tau**2
    exponent = (estimate**2 * tau**2) / (2.0 * variance * widened)
    return 0.5 * math.log(variance / widened) + exponent


def mixture_likelihood_ratio(estimate: float, variance: float, tau: float) -> float:
    """The mixture likelihood ratio itself.

    Returns ``math.inf`` once the evidence exceeds float range - overwhelming
    evidence is still a usable answer, whereas an exception at the moment of
    stopping is not. Use :func:`log_mixture_likelihood_ratio` when the
    magnitude matters.
    """
    log_ratio = log_mixture_likelihood_ratio(estimate, variance, tau)
    return math.exp(log_ratio) if log_ratio < _MAX_LOG else math.inf


def always_valid_p_value(estimate: float, variance: float, tau: float) -> float:
    """Anytime-valid p-value: the reciprocal of the mixture likelihood ratio.

    Capped at 1.0, and computed as ``exp(-log LR)`` so that overwhelming
    evidence underflows to 0.0 rather than overflowing. Unlike a fixed-horizon
    p-value this may be compared to alpha at every look; the guarantee is on the
    *whole sequence* of looks, not on a single pre-specified one.
    """
    return min(1.0, math.exp(-log_mixture_likelihood_ratio(estimate, variance, tau)))


def msprt(
    control: ArrayLike,
    treatment: ArrayLike,
    tau: float,
    alpha: float = 0.05,
) -> SequentialResult:
    """Evaluate the mSPRT on the data collected so far.

    Args:
        control: All control observations up to this look.
        treatment: All treatment observations up to this look.
        tau: Prior standard deviation of the effect. A defensible default is
            the minimum effect worth detecting - see :func:`tau_from_mde`.
        alpha: Decision threshold, valid at every look.

    Variances are plugged in from the sample rather than assumed known. That
    makes the guarantee asymptotic rather than exact; at the sample sizes where
    anyone peeks daily (thousands of units) the difference is immaterial, and
    :mod:`ab_lab.simulate` checks it empirically instead of taking it on faith.
    """
    control_sample = _as_sample(control, "control")
    treatment_sample = _as_sample(treatment, "treatment")
    _check_alpha(alpha)

    estimate = float(treatment_sample.mean() - control_sample.mean())
    variance = float(
        control_sample.var(ddof=1) / control_sample.size
        + treatment_sample.var(ddof=1) / treatment_sample.size
    )
    if variance <= 0.0:
        raise ValueError("both groups are constant: there is no variance to test against")

    log_ratio = log_mixture_likelihood_ratio(estimate, variance, tau)
    return SequentialResult(
        n_control=int(control_sample.size),
        n_treatment=int(treatment_sample.size),
        estimate=estimate,
        likelihood_ratio=math.exp(log_ratio) if log_ratio < _MAX_LOG else math.inf,
        log_likelihood_ratio=log_ratio,
        p_value=min(1.0, math.exp(-log_ratio)),
        alpha=alpha,
        tau=tau,
    )


def tau_from_mde(mde: float) -> float:
    """Pick the mixture width from the effect the experiment was sized for.

    Setting ``tau`` to the minimum detectable effect centres the test's power
    where the decision actually matters. Any positive value keeps the type I
    error guarantee - ``tau`` trades power between small and large effects, it
    does not trade away validity.
    """
    if mde <= 0.0:
        raise ValueError(f"mde must be positive, got {mde}")
    return float(mde)


class SequentialMonitor:
    """Stateful monitor for an experiment that is checked repeatedly.

    Keeps the running minimum of the always-valid p-value, so that a decision
    once taken is not un-taken by later noise::

        monitor = SequentialMonitor(tau=0.01, alpha=0.05)
        for day in days:
            result = monitor.look(control_so_far, treatment_so_far)
            if result.should_stop:
                break
    """

    def __init__(self, tau: float, alpha: float = 0.05) -> None:
        if tau <= 0.0:
            raise ValueError(f"tau must be positive, got {tau}")
        _check_alpha(alpha)
        self.tau = tau
        self.alpha = alpha
        self._looks: list[SequentialResult] = []
        self._best_p_value = 1.0

    @property
    def looks(self) -> tuple[SequentialResult, ...]:
        """Every look taken so far, in order."""
        return tuple(self._looks)

    @property
    def stopped(self) -> bool:
        """True once any look has crossed the boundary."""
        return self._best_p_value < self.alpha

    def look(self, control: ArrayLike, treatment: ArrayLike) -> SequentialResult:
        """Take one look at the cumulative data and record it."""
        result = msprt(control, treatment, tau=self.tau, alpha=self.alpha)
        self._best_p_value = min(self._best_p_value, result.p_value)
        monotone = SequentialResult(
            n_control=result.n_control,
            n_treatment=result.n_treatment,
            estimate=result.estimate,
            likelihood_ratio=result.likelihood_ratio,
            log_likelihood_ratio=result.log_likelihood_ratio,
            p_value=self._best_p_value,
            alpha=self.alpha,
            tau=self.tau,
        )
        self._looks.append(monotone)
        return monotone
