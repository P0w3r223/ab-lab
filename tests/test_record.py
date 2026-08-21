"""The committed evidence still describes the code that produced it.

Two guards with very different costs and very different promises.

**G2** is free and exact: the design recorded beside the counts must match the
constants the scripts would use today. It catches the cheapest way for the page
to become a lie - someone edits ``N_EXPERIMENTS`` or the look schedule, the page
keeps quoting yesterday's run, and nothing looks wrong.

**G4** is a smoke test and is described as one rather than oversold. It replays
two cells at roughly a tenth of their recorded size and asks whether the
recorded rate is still credible. At that size its own Monte Carlo error near
p = 0.25 is about 0.02, so the band is wide: it catches "the mSPRT stopped being
anytime-valid", not a drift of two points. The real check is the scheduled full
re-run in ``.github/workflows/refresh.yml``.

**What is deliberately not replayed**, since a silent cap reads as coverage:
only the endpoints of the peeking curve and the first two validation rows. The
14 745-per-arm sample-size row and the ten-look mSPRT row are the expensive ones
and are left to the weekly job.
"""

from __future__ import annotations

import numpy as np
import pytest

from ab_lab.power import sample_size_for_proportion
from ab_lab.sequential import tau_from_mde
from ab_lab.simulate import (
    msprt_p_value,
    normal_draw,
    peeking_curve,
    run_experiments,
    welch_p_value,
)
from examples import peeking_pitfalls, validation_table
from sitegen import record

#: Replays are small, so the band has to be generous or the suite becomes flaky.
SIGMAS = 4.0
REPLAY_EXPERIMENTS = 400
REPLAY_SEED = 4242


@pytest.fixture(scope="module")
def evidence() -> record.Record:
    return record.load()


def test_the_recorded_peeking_design_is_the_one_the_script_would_run(evidence):
    """G2: constants and evidence agree, or the page is quoting an old run."""
    design = evidence.findings["peeking"].design

    assert design["seed"] == peeking_pitfalls.SEED
    assert design["alpha"] == peeking_pitfalls.ALPHA
    assert design["n_experiments"] == peeking_pitfalls.N_EXPERIMENTS
    assert design["n_per_group"] == peeking_pitfalls.MAX_N
    assert design["look_counts"] == peeking_pitfalls.LOOK_COUNTS

    for series in evidence.findings["peeking"].series:
        assert [cell.x for cell in series.cells] == peeking_pitfalls.LOOK_COUNTS
        assert all(
            cell.summary.n_experiments == peeking_pitfalls.N_EXPERIMENTS
            for cell in series.cells
        )


def test_the_recorded_validation_design_is_the_one_the_script_would_run(evidence):
    design = evidence.validation.design

    assert design["seed"] == validation_table.SEED
    assert design["alpha"] == validation_table.ALPHA
    assert design["n_experiments"] == validation_table.N_EXPERIMENTS


def test_the_sample_size_row_still_names_the_size_the_design_solves_for(evidence):
    """The row's own label carries a number, and that number is closed-form.

    Cheap to re-derive and worth re-deriving: if `sample_size_for_proportion`
    ever moves, the recorded scenario string is the first place the page would
    quietly disagree with the package.
    """
    solved = sample_size_for_proportion(0.10, 0.01, alpha=0.05, power=0.8)
    scenarios = [row.scenario for row in evidence.validation.rows]
    assert any(f"n = {solved.per_group:,}/arm" in scenario for scenario in scenarios)


def _credible(recorded, replayed, sigmas: float = SIGMAS) -> bool:
    """Do two independent runs of the same cell agree, given both error bars?"""
    combined = (recorded.monte_carlo_error**2 + replayed.monte_carlo_error**2) ** 0.5
    return abs(recorded.rejection_rate - replayed.rejection_rate) <= sigmas * combined


@pytest.mark.parametrize("role", ["naive", "corrected"])
def test_a_small_replay_still_finds_the_recorded_peeking_rates(evidence, role):
    """G4: a gross change in behaviour since the record was made."""
    finding = evidence.findings["peeking"]
    endpoints = [int(finding.x_values[0]), int(finding.x_values[-1])]
    p_value_fn = (
        welch_p_value if role == "naive" else msprt_p_value(tau=tau_from_mde(0.1))
    )

    replayed = peeking_curve(
        normal_draw(n_per_group=peeking_pitfalls.MAX_N, mean=0.0, std_dev=1.0),
        p_value_fn,
        peeking_pitfalls.MAX_N,
        endpoints,
        REPLAY_EXPERIMENTS,
        np.random.default_rng(REPLAY_SEED),
        alpha=peeking_pitfalls.ALPHA,
    )

    series = finding.series_by_role(role)
    recorded = {int(cell.x): cell.summary for cell in series.cells}
    for looks, summary in zip(endpoints, replayed, strict=True):
        assert _credible(recorded[looks], summary), (
            f"{role} at {looks} look(s): recorded "
            f"{recorded[looks].rejection_rate:.3f}, replay "
            f"{summary.rejection_rate:.3f} - the record may predate a change in behaviour"
        )


def test_a_small_replay_still_finds_the_recorded_welch_type_i_error(evidence):
    """The cheapest validation row, replayed. See the module docstring for what
    is left to the weekly job and why."""
    recorded = evidence.validation.rows[0].summary
    replayed = run_experiments(
        draw=normal_draw(n_per_group=500, mean=10.0, std_dev=3.0),
        p_value_fn=welch_p_value,
        n_experiments=1_000,
        rng=np.random.default_rng(REPLAY_SEED + 1),
        alpha=validation_table.ALPHA,
    )
    assert _credible(recorded, replayed)
