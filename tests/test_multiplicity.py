"""Three corrections, two guarantees, one oracle.

``statsmodels.stats.multitest.multipletests`` is the reference for all three
procedures, and the comparison is on **adjusted p-values**, which are exactly
defined - not on a rejection mask, where this package's ``<`` and the reference's
``<=`` differ on an exact tie. The mask is compared separately, on random
p-values, where a tie has probability zero.

The hand-arithmetic cases are chosen so every step is visible. Holm's running
maximum and Benjamini-Hochberg's running minimum are the two places these
procedures are implemented wrongly in practice - forget either and the result
still looks plausible - so each has a fixture that fails without it.
"""

from __future__ import annotations

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from ab_lab.multiplicity import (
    CORRECTIONS,
    benjamini_hochberg,
    bonferroni,
    holm,
)

ALPHA = 0.05

#: statsmodels' name for each of our procedures.
ORACLE_METHOD = {
    "bonferroni": "bonferroni",
    "holm": "holm",
    "benjamini-hochberg": "fdr_bh",
}


@pytest.mark.parametrize("name", sorted(CORRECTIONS))
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_adjusted_p_values_match_the_statsmodels_oracle(name, seed):
    p_values = np.random.default_rng(seed).uniform(0.0, 1.0, 12)
    ours = CORRECTIONS[name](p_values, ALPHA)

    reject, adjusted, _, _ = multipletests(p_values, alpha=ALPHA, method=ORACLE_METHOD[name])
    assert np.allclose(ours.adjusted_p_values, adjusted, rtol=0, atol=1e-12)
    # Random uniforms never tie, so the two rejection conventions agree here.
    assert list(ours.rejected) == list(reject)


def test_bonferroni_is_the_family_size_and_nothing_more():
    result = bonferroni([0.01, 0.02, 0.5, 0.9])
    assert result.adjusted_p_values == pytest.approx((0.04, 0.08, 1.0, 1.0))
    assert result.rejected == (True, False, False, False)


def test_holm_enforces_monotonicity_with_a_running_maximum():
    """Without the running maximum the fourth entry would come back below the third.

    Raw: 0.001, 0.012, 0.030, 0.040, 0.200. Scaled by 5, 4, 3, 2, 1 that is
    0.005, 0.048, 0.090, 0.080, 0.200 - and 0.080 < 0.090, so a weaker result
    would be reported as more significant than a stronger one. The cumulative
    maximum lifts it to 0.090.
    """
    result = holm([0.001, 0.012, 0.030, 0.040, 0.200])
    assert result.adjusted_p_values == pytest.approx((0.005, 0.048, 0.09, 0.09, 0.2))
    assert result.n_rejected == 2


def test_benjamini_hochberg_steps_up_with_a_running_minimum():
    """Same p-values, scaled by K/rank and made monotone from the largest down."""
    result = benjamini_hochberg([0.001, 0.012, 0.030, 0.040, 0.200])
    assert result.adjusted_p_values == pytest.approx((0.005, 0.03, 0.05, 0.05, 0.2))
    assert result.n_rejected == 3


def test_holm_never_rejects_less_than_bonferroni():
    """The uniform domination that makes Bonferroni the wrong default."""
    p_values = np.random.default_rng(9).uniform(0.0, 0.2, 40)
    strict = bonferroni(p_values, ALPHA)
    better = holm(p_values, ALPHA)

    assert better.n_rejected >= strict.n_rejected
    assert all(
        theirs or not ours
        for ours, theirs in zip(strict.rejected, better.rejected, strict=True)
    )


def test_the_order_of_the_family_does_not_change_any_verdict():
    """Adjusted p-values follow their own p-value, not their position."""
    p_values = [0.04, 0.001, 0.2, 0.012]
    shuffled = [0.2, 0.012, 0.04, 0.001]
    forward = dict(zip(p_values, holm(p_values).adjusted_p_values, strict=True))
    backward = dict(zip(shuffled, holm(shuffled).adjusted_p_values, strict=True))
    assert forward == pytest.approx(backward)


def test_a_family_of_one_is_no_correction_at_all():
    for correction in CORRECTIONS.values():
        assert correction([0.03]).adjusted_p_values == pytest.approx((0.03,))


def test_the_two_guarantees_are_named_and_are_different():
    p_values = [0.01, 0.02, 0.03]
    assert bonferroni(p_values).error_rate_controlled == holm(p_values).error_rate_controlled
    assert (
        benjamini_hochberg(p_values).error_rate_controlled
        != holm(p_values).error_rate_controlled
    )


def test_a_named_family_can_be_read_back_by_name():
    result = holm([0.001, 0.4, 0.02], labels=["signups", "latency", "revenue"])
    adjusted, rejected = result.for_label("signups")
    assert adjusted == pytest.approx(0.003)
    assert rejected

    with pytest.raises(KeyError, match="churn"):
        result.for_label("churn")


def test_an_unnamed_family_says_so_rather_than_guessing():
    with pytest.raises(ValueError, match="no labels"):
        holm([0.01, 0.02]).for_label("anything")


@pytest.mark.parametrize(
    ("p_values", "labels", "message"),
    [
        ([], None, "non-empty"),
        ([0.5, np.nan], None, "NaN"),
        ([0.5, 1.5], None, r"\[0, 1\]"),
        ([0.5, 0.2], ["only one"], "every member of the family"),
    ],
)
def test_a_family_it_cannot_interpret_is_refused(p_values, labels, message):
    with pytest.raises(ValueError, match=message):
        holm(p_values, labels=labels)
