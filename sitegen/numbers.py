"""How a measured rate becomes text - in one place, for every surface.

The page, the README and the scripts' stdout all print the same figures. When
each owned its own formatting they drifted, and the proof is in this
repository's history: one validation row existed as ``14,745`` in the script
that computed it, ``14 745`` in the README and ``14,745`` under different
wording on the page.

So there is one function per shape of number, and no renderer is allowed a
format string of its own.
"""

from __future__ import annotations

from ab_lab.results import SimulationSummary


def percent(value: float, places: int = 1) -> str:
    """A rate as a percentage, for prose and for tables of rates."""
    return f"{value:.{places}%}"


def rate_with_error(summary: SimulationSummary) -> str:
    """A measured rate with its Monte Carlo error attached.

    Never without it. The error is the noise floor of the run itself, and a
    published rate that hides it invites exactly the over-reading this package
    is about - which is how the previous version of the page came to print the
    peeking figures bare while the README printed them with a tolerance.
    """
    return f"{percent(summary.rejection_rate)} (±{percent(summary.monte_carlo_error)})"


def proportion(value: float) -> str:
    """A probability on the 0-1 scale, as the validation table shows it."""
    return f"{value:.4f}"


def error(summary: SimulationSummary) -> str:
    """The Monte Carlo error alone, on the 0-1 scale."""
    return f"±{summary.monte_carlo_error:.4f}"


def integer(value: float) -> str:
    """A count, grouped the way every surface groups counts: U+202F.

    `0007` §5 clause 8 of the portfolio page specification. Written as an escape
    rather than as the character, because U+0020 and U+202F are one string in a
    diff, a terminal and a `grep` -- and this module exists because three surfaces
    once disagreed about a number nobody could see them disagreeing about.
    """
    return f"{value:,.0f}".replace(",", "\u202f")
