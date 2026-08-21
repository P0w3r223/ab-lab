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
from examples import peeking_pitfalls, three_inflations, validation_table
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


#: Which module's constants each recording script owns.
RECORDING_SCRIPTS = {
    "examples/peeking_pitfalls.py": peeking_pitfalls,
    "examples/validation_table.py": validation_table,
    "examples/three_inflations.py": three_inflations,
}


def test_every_recorded_finding_names_a_script_that_still_exists(evidence):
    """G2, widened. It used to check peeking and validation by name.

    `clustering` and `multiplicity` - two of the three sections on the page -
    had no design check at all, so every constant in `three_inflations.py` could
    be edited without a single test noticing the page now quoted an older run.
    Parametrising over the record means a fourth finding is covered the day it
    is added rather than the day someone remembers.
    """
    for finding in evidence.findings.values():
        script = finding.recorded["script"]
        assert script in RECORDING_SCRIPTS, f"{finding.key} names an unknown script: {script}"
        assert (record.ROOT / script).exists(), f"{script} is recorded but missing"


def test_every_recorded_finding_agrees_with_its_script_constants(evidence):
    for finding in evidence.findings.values():
        module = RECORDING_SCRIPTS[finding.recorded["script"]]
        assert finding.design["seed"] == module.SEED, f"{finding.key}: seed drifted"
        assert finding.design["alpha"] == module.ALPHA, f"{finding.key}: alpha drifted"
        assert finding.nominal == module.ALPHA


def test_the_three_inflations_findings_match_that_script_exactly(evidence):
    clustering = evidence.findings["clustering"]
    assert clustering.design["n_experiments"] == three_inflations.CLUSTER_EXPERIMENTS
    assert clustering.design["rows_per_user"] == three_inflations.ROWS_PER_USER
    assert clustering.design["icc"] == three_inflations.ICC
    assert clustering.design["n_clusters_per_group"] == three_inflations.CLUSTERS_PER_ARM
    assert [int(x) for x in clustering.x_values] == three_inflations.ROWS_PER_USER

    suite = evidence.findings["multiplicity"]
    assert suite.design["n_experiments"] == three_inflations.SUITE_EXPERIMENTS
    assert suite.design["metric_counts"] == three_inflations.METRIC_COUNTS
    assert suite.design["n_per_group"] == three_inflations.UNITS_PER_ARM
    assert [int(x) for x in suite.x_values] == three_inflations.METRIC_COUNTS
