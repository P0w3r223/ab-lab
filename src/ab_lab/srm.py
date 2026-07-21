"""Sample ratio mismatch (SRM) - the first check to run on any experiment.

An SRM means the observed split between groups is further from the intended
one than randomisation can explain. It is not a small problem: if assignment is
broken, so is the comparison, and the effect estimate is not worth reading. The
usual causes are mundane - a redirect that drops users, bot filtering applied
to one arm, an assignment bug on a particular platform - and all of them
correlate with user behaviour, so they bias the metric in unknown directions.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats

from .results import SrmResult

# SRM is checked on every experiment, and a false alarm costs an investigation.
# The conventional threshold is far stricter than 0.05 for that reason: at
# 0.05, one experiment in twenty would be flagged for no reason at all.
DEFAULT_SRM_ALPHA = 0.001


def check_srm(
    observed: Sequence[int],
    expected_ratios: Sequence[float] | None = None,
    alpha: float = DEFAULT_SRM_ALPHA,
) -> SrmResult:
    """Chi-square goodness-of-fit test on the group allocation.

    Args:
        observed: Unit counts per group, in the same order as ``expected_ratios``.
        expected_ratios: Intended allocation, e.g. ``(0.5, 0.5)`` or
            ``(0.9, 0.1)``. Ratios need not sum to one - they are normalised.
            Defaults to an even split across the groups given.
        alpha: Significance threshold; see :data:`DEFAULT_SRM_ALPHA` for why the
            default is 0.001 rather than 0.05.

    Returns:
        An :class:`~ab_lab.results.SrmResult`; check ``is_mismatch``.
    """
    counts = np.asarray(observed, dtype=np.float64)
    if counts.ndim != 1 or counts.size < 2:
        raise ValueError(f"observed must be a 1-D sequence of >= 2 counts, got {observed!r}")
    if np.any(counts < 0):
        raise ValueError(f"observed counts must be non-negative, got {observed!r}")
    total = counts.sum()
    if total <= 0:
        raise ValueError("observed counts are all zero: nothing to check")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    if expected_ratios is None:
        ratios = np.full(counts.size, 1.0 / counts.size)
    else:
        ratios = np.asarray(expected_ratios, dtype=np.float64)
        if ratios.shape != counts.shape:
            raise ValueError(
                f"expected_ratios has shape {ratios.shape}, observed has {counts.shape}"
            )
        if np.any(ratios <= 0):
            raise ValueError(f"expected_ratios must all be positive, got {expected_ratios!r}")
        ratios = ratios / ratios.sum()

    expected = ratios * total
    if np.any(expected < 5.0):
        raise ValueError(
            "the chi-square approximation needs at least 5 expected units per group; "
            f"got expected counts {tuple(round(value, 2) for value in expected)}"
        )

    statistic = float(np.sum((counts - expected) ** 2 / expected))
    p_value = float(stats.chi2.sf(statistic, df=counts.size - 1))

    return SrmResult(
        observed=tuple(int(count) for count in counts),
        expected=tuple(float(value) for value in expected),
        statistic=statistic,
        p_value=p_value,
        alpha=alpha,
    )
