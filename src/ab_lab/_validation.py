"""Argument checks shared by every module that takes a sample or an alpha.

Private on purpose: these are not part of the package's surface, they are the
guards behind it. They live here because the alternative is what this module
replaced - the same ``alpha must be in (0, 1)`` check written out in six places
and the same twelve-line sample validator written out in two, which is one
edit away from two modules disagreeing about what a valid argument is.

The messages are deliberately unchanged from the copies they replace: several
tests match on them, and an error message is part of the interface a caller
programs against.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: Analysis vocabulary, following SciPy: a p-value depends on *which* direction.
Alternative = Literal["two-sided", "less", "greater"]

#: Design vocabulary: a sample size does not depend on which direction.
DesignAlternative = Literal["two-sided", "one-sided"]


def check_alpha(alpha: float) -> float:
    """Reject a significance level outside the open unit interval."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return alpha


def check_power(power: float) -> float:
    """Reject a target power the power equation has no solution for.

    Without this the failure surfaces from deep inside the root finder as
    "f(a) and f(b) must have different signs", which says nothing useful.
    """
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}")
    return power


def check_alternative(alternative: str) -> Alternative:
    """Validate an *analysis* alternative."""
    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError(
            f"alternative must be 'two-sided', 'less' or 'greater', got {alternative!r}"
        )
    return alternative  # type: ignore[return-value]


def as_sample(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Validate one arm of an experiment and return it as a float array.

    Rejects the three shapes that would otherwise produce a number rather than
    an error: more than one dimension, fewer than two observations (no variance
    to estimate), and non-finite values (which propagate silently into a NaN
    p-value that compares False against alpha and reads as "not significant").
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size < 2:
        raise ValueError(f"{name} needs at least 2 observations, got {array.size}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array
