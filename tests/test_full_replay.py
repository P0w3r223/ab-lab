"""The weekly drift check: every recorded finding, re-measured at full size.

Marked ``slow`` and therefore skipped by ``pytest``; ``.github/workflows/refresh.yml``
runs ``pytest -m slow`` once a week and on demand. This is the guard the cheap
ones in ``test_record.py`` are a smoke test for - they replay two cells at a
tenth of their size, which finds a broken method and not a drift of two points.

The comparison is between **rates, within Monte Carlo error** - never bytes.
numpy guarantees stream stability for ``RandomState`` and not for ``Generator``,
so a legitimate numpy release may move the stream even at a fixed seed. A
byte-exact weekly job would go red for a reason that is not a defect, and a job
that cries wolf is a job nobody reads.

A failure here does not mean the page is wrong. It means the record predates a
change in behaviour, and the response is to find the change, then re-record and
rebuild - in that order.
"""

from __future__ import annotations

import numpy as np
import pytest

from ab_lab.results import SimulationSummary
from examples import peeking_pitfalls, validation_table
from sitegen import record

pytestmark = pytest.mark.slow

#: Two independent runs of the same cell, each with its own error bar. Four
#: combined sigmas is roughly one false alarm per 16 000 comparisons, and this
#: job makes about twenty a week.
SIGMAS = 4.0


def _combined_error(first: SimulationSummary, second: SimulationSummary) -> float:
    return (first.monte_carlo_error**2 + second.monte_carlo_error**2) ** 0.5


def _assert_agrees(label: str, recorded: SimulationSummary, fresh: SimulationSummary) -> None:
    combined = _combined_error(recorded, fresh)
    distance = abs(recorded.rejection_rate - fresh.rejection_rate)
    sigmas = distance / combined if combined else float("inf")
    assert distance <= SIGMAS * combined, (
        f"{label}: recorded {recorded.rejection_rate:.4f}, re-measured "
        f"{fresh.rejection_rate:.4f}, {sigmas:.1f} combined sigmas apart. "
        f"Find what changed before re-recording."
    )


def test_the_recorded_peeking_curve_still_reproduces():
    evidence = record.load().findings["peeking"]
    fixed_horizon, sequential = peeking_pitfalls.run()

    for role, fresh_series in (("naive", fixed_horizon), ("corrected", sequential)):
        recorded = evidence.series_by_role(role)
        for looks, cell, fresh in zip(
            evidence.x_values, recorded.cells, fresh_series, strict=True
        ):
            _assert_agrees(f"peeking/{role} at {looks:g} look(s)", cell.summary, fresh)


def test_every_recorded_validation_row_still_reproduces():
    evidence = record.load().validation
    fresh_rows = validation_table.build_rows(np.random.default_rng(validation_table.SEED))

    assert [row.scenario for row in evidence.rows] == [row.scenario for row in fresh_rows], (
        "the validation table's rows changed; re-record rather than compare"
    )
    for recorded, fresh in zip(evidence.rows, fresh_rows, strict=True):
        _assert_agrees(recorded.scenario, recorded.summary, fresh.summary)
        assert fresh.summary.agrees_with(recorded.expected, recorded.claim), (
            f"{recorded.scenario}: the fresh run no longer supports its own claim"
        )
