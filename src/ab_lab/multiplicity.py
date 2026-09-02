"""Testing one experiment against many metrics, without pretending it is one test.

An experiment rarely has one metric. It has a success metric, three secondary
ones, and a dozen guardrails, and each is read at alpha = 0.05. Under a global
null with ten independent metrics, the probability that at least one comes back
significant is ``1 - 0.95**10 = 40.13%`` - measured at 40.23% (±0.90) over 3 000
runs. That is the same arithmetic as peeking, run across metrics instead of
across time.

Three corrections, controlling **two different things**, and the difference is
the reason all three are here rather than one:

* :func:`bonferroni` and :func:`holm` control the *family-wise error rate* - the
  probability of **any** false positive in the family. Holm dominates Bonferroni
  uniformly, at the same guarantee and with no independence assumption, so it is
  the default worth reaching for.
* :func:`benjamini_hochberg` controls the *false discovery rate* - the expected
  **share** of rejections that are false. It rejects more, and it is not a
  better Holm: it permits false positives in exchange for finding more true
  ones. Under a complete null the two coincide, which is exactly why the
  demonstration in :mod:`ab_lab.simulate` uses a partial null instead.

Every function takes bare p-values rather than result objects. Keeping this
module ignorant of where a p-value came from is what lets it accept Welch, the
proportion test, the bootstrap and the mSPRT alike.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from ._validation import check_alpha
from .results import MultipleComparisonResult

FAMILY_WISE = "family-wise error rate"
FALSE_DISCOVERY = "false discovery rate"

_FAMILY_ASSUMPTION = (
    "the family is exactly the comparisons passed in; correcting a subset and "
    "reading the rest uncorrected controls nothing"
)
_DIRECTION_ASSUMPTION = (
    "guardrail metrics tested one-sided do not belong in the same family as "
    "two-sided success metrics - they are a different question"
)


def _check_family(
    p_values: Sequence[float], labels: Sequence[str] | None
) -> tuple[np.ndarray, tuple[str, ...] | None]:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"p_values must be a non-empty 1-D sequence, got {p_values!r}")
    if not np.all(np.isfinite(values)):
        raise ValueError("p_values contains NaN or infinite values")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"p_values must all be in [0, 1], got {p_values!r}")
    if labels is None:
        return values, None
    if len(labels) != values.size:
        raise ValueError(
            f"got {len(labels)} labels for {values.size} p-values; every member of "
            f"the family has to be named or none of them"
        )
    return values, tuple(labels)


def _result(
    method: str,
    controlled: str,
    values: np.ndarray,
    adjusted: np.ndarray,
    alpha: float,
    labels: tuple[str, ...] | None,
    extra: tuple[str, ...],
) -> MultipleComparisonResult:
    return MultipleComparisonResult(
        method=method,
        alpha=alpha,
        p_values=tuple(float(value) for value in values),
        adjusted_p_values=tuple(float(value) for value in adjusted),
        # `<` rather than `<=`, matching `TestResult.is_significant` and every
        # other decision in this package. Reference implementations commonly use
        # `<=`; the two differ only on an exact tie, which continuous p-values do
        # not produce, and internal consistency is worth more here.
        rejected=tuple(bool(value) for value in adjusted < alpha),
        error_rate_controlled=controlled,
        assumptions=extra,
        labels=labels,
    )


def bonferroni(
    p_values: Sequence[float],
    alpha: float = 0.05,
    labels: Sequence[str] | None = None,
) -> MultipleComparisonResult:
    """Multiply every p-value by the family size. The one everyone knows.

    Valid under any dependence structure, and correspondingly blunt: with
    correlated metrics it spends far less than alpha and loses real effects. It
    is here as the baseline the others are measured against - :func:`holm`
    rejects everything this does and sometimes more, at the identical guarantee,
    so there is no situation in which Bonferroni is the better choice.
    """
    alpha = check_alpha(alpha)
    values, names = _check_family(p_values, labels)
    adjusted = np.minimum(1.0, values * values.size)
    return _result(
        "Bonferroni",
        FAMILY_WISE,
        values,
        adjusted,
        alpha,
        names,
        (
            _FAMILY_ASSUMPTION,
            "valid under any dependence between the tests",
            "uniformly dominated by Holm: same guarantee, never fewer rejections",
            _DIRECTION_ASSUMPTION,
        ),
    )


def holm(
    p_values: Sequence[float],
    alpha: float = 0.05,
    labels: Sequence[str] | None = None,
) -> MultipleComparisonResult:
    """Holm's step-down procedure: Bonferroni's guarantee, more power.

    Sort the p-values, multiply the smallest by the family size, the next by one
    less, and so on. The running maximum is what enforces monotonicity - without
    it a later, larger p-value could be adjusted below an earlier one, and the
    procedure would reject a weaker result while rejecting nothing stronger.
    That step is the classic implementation bug in this procedure, so it has its
    own hand-arithmetic test.
    """
    alpha = check_alpha(alpha)
    values, names = _check_family(p_values, labels)
    order = np.argsort(values, kind="stable")
    n_tests = values.size

    scaled = values[order] * (n_tests - np.arange(n_tests))
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(scaled))
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted

    return _result(
        "Holm",
        FAMILY_WISE,
        values,
        adjusted,
        alpha,
        names,
        (
            _FAMILY_ASSUMPTION,
            "valid under any dependence between the tests",
            "step-down: a comparison can only be rejected if every stronger one was",
            _DIRECTION_ASSUMPTION,
        ),
    )


def benjamini_hochberg(
    p_values: Sequence[float],
    alpha: float = 0.05,
    labels: Sequence[str] | None = None,
) -> MultipleComparisonResult:
    """Control the expected share of rejections that are false.

    A different promise from Holm's, not a better one. Under a partial null this
    rejects more, and some of those extra rejections are wrong on purpose: the
    guarantee is that the *proportion* of mistakes stays bounded, not that there
    are none. If the cost of shipping one wrong conclusion is high, this is the
    wrong tool however many true effects it finds.

    The step-up direction, and the running minimum from the largest p-value
    down, are what make it valid; getting the direction wrong yields something
    that looks plausible and controls nothing.
    """
    alpha = check_alpha(alpha)
    values, names = _check_family(p_values, labels)
    order = np.argsort(values, kind="stable")
    n_tests = values.size

    ranks = np.arange(1, n_tests + 1)
    scaled = values[order] * n_tests / ranks
    adjusted_sorted = np.minimum(1.0, np.minimum.accumulate(scaled[::-1])[::-1])
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted

    return _result(
        "Benjamini-Hochberg",
        FALSE_DISCOVERY,
        values,
        adjusted,
        alpha,
        names,
        (
            _FAMILY_ASSUMPTION,
            "controls the expected share of false rejections, not the chance of "
            "there being one: the family-wise error rate here exceeds alpha by design",
            "valid under independence or positive dependence between the tests",
            _DIRECTION_ASSUMPTION,
        ),
    )


Correction = Callable[..., MultipleComparisonResult]

#: Every correction by name, so the simulation harness and the examples can
#: parametrise over them without a second dispatcher whose docstring would have
#: to state two incompatible guarantees at once.
CORRECTIONS: Mapping[str, Correction] = {
    "bonferroni": bonferroni,
    "holm": holm,
    "benjamini-hochberg": benjamini_hochberg,
}
