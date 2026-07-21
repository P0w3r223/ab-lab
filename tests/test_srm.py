"""SRM detection: it must fire on real allocation bugs and stay quiet otherwise."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from ab_lab.srm import DEFAULT_SRM_ALPHA, check_srm


def test_a_clean_even_split_is_not_flagged():
    result = check_srm([10_042, 9_958])
    assert not result.is_mismatch
    assert result.p_value > 0.1
    assert result.expected == (10_000.0, 10_000.0)


def test_statistic_matches_the_chi_square_definition():
    result = check_srm([5_100, 4_900])
    expected = (100.0**2) / 5_000.0 + (100.0**2) / 5_000.0
    assert result.statistic == pytest.approx(expected)
    assert result.p_value == pytest.approx(stats.chi2.sf(expected, df=1))


def test_a_realistic_redirect_bug_is_caught():
    """A 2% shortfall in one arm reads as rounding and is not: at 200k units
    randomisation practically never produces it."""
    result = check_srm([100_000, 98_000])
    assert result.is_mismatch
    assert result.p_value < 1e-4


def test_uneven_designs_are_judged_against_their_own_target():
    balanced_against_ninety_ten = check_srm([90_100, 9_900], expected_ratios=(0.9, 0.1))
    assert not balanced_against_ninety_ten.is_mismatch
    # The same counts are a gross mismatch if the design promised an even split.
    assert check_srm([90_100, 9_900]).is_mismatch


def test_expected_ratios_need_not_be_normalised():
    from_fractions = check_srm([6_000, 4_000], expected_ratios=(0.6, 0.4))
    from_weights = check_srm([6_000, 4_000], expected_ratios=(3.0, 2.0))
    assert from_fractions == from_weights


def test_more_than_two_arms_are_supported():
    result = check_srm([3_300, 3_350, 3_350])
    assert not result.is_mismatch
    assert len(result.expected) == 3


def test_the_default_threshold_is_strict_on_purpose():
    assert DEFAULT_SRM_ALPHA == 0.001
    # A deviation that a 0.05 test would flag is left alone by the default.
    borderline = check_srm([10_200, 9_800])
    assert borderline.p_value < 0.05
    assert not borderline.is_mismatch
    assert check_srm([10_200, 9_800], alpha=0.05).is_mismatch


def test_false_alarm_rate_matches_the_threshold():
    """Simulation check: on honest 50/50 randomisation the test must fire at
    its nominal rate, not more."""
    rng = np.random.default_rng(11)
    n_experiments, n_units = 4_000, 20_000
    alpha = 0.01
    flagged = 0
    for _ in range(n_experiments):
        in_control = int(rng.binomial(n_units, 0.5))
        if check_srm([in_control, n_units - in_control], alpha=alpha).is_mismatch:
            flagged += 1

    rate = flagged / n_experiments
    monte_carlo_error = (alpha * (1 - alpha) / n_experiments) ** 0.5
    assert abs(rate - alpha) < 4 * monte_carlo_error


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (([100],), "1-D sequence of >= 2 counts"),
        (([-1, 100],), "must be non-negative"),
        (([0, 0],), "nothing to check"),
        (([2, 3],), "at least 5 expected units"),
    ],
)
def test_malformed_input_is_rejected(args, message):
    with pytest.raises(ValueError, match=message):
        check_srm(*args)


def test_mismatched_ratio_length_is_rejected():
    with pytest.raises(ValueError, match="expected_ratios has shape"):
        check_srm([100, 100], expected_ratios=(0.3, 0.3, 0.4))


def test_zero_expected_ratio_is_rejected():
    with pytest.raises(ValueError, match="expected_ratios must all be positive"):
        check_srm([100, 100], expected_ratios=(1.0, 0.0))
