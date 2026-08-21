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

import numpy as np
from numpy.typing import NDArray

from ._validation import check_alpha
from .analyze import proportion_test, welch_t_test
from .results import SimulationSummary
from .sequential import msprt

Draw = Callable[[np.random.Generator], tuple[NDArray[np.float64], NDArray[np.float64]]]
PValueFn = Callable[[NDArray[np.float64], NDArray[np.float64]], float]

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


def welch_p_value(control: NDArray[np.float64], treatment: NDArray[np.float64]) -> float:
    """Adapter: Welch's t-test as a bare p-value."""
    return welch_t_test(control, treatment).p_value


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


def _checked_p_value(
    p_value_fn: PValueFn,
    control: NDArray[np.float64],
    treatment: NDArray[np.float64],
) -> float:
    """Call the p-value function and refuse to silently count a NaN.

    A non-finite p-value compares False against alpha, so it would be tallied
    as "no rejection" - and a harness that reports a 0% false positive rate
    because every p-value was NaN is worse than one that crashes.
    """
    p_value = float(p_value_fn(control, treatment))
    if not np.isfinite(p_value):
        raise ValueError(
            f"the p-value function returned {p_value} for a sample of "
            f"{control.size} control and {treatment.size} treatment units"
        )
    return p_value


def _check_run(n_experiments: int, alpha: float) -> None:
    if n_experiments < 1:
        raise ValueError(f"n_experiments must be positive, got {n_experiments}")
    check_alpha(alpha)
