"""Regenerate the validation table in the README.

The answer to "how do you know your code computes the right thing?": run each
method on thousands of experiments where the truth is known and compare the
empirical rejection rate to the rate the theory promises.

Usage::

    python examples/validation_table.py             # print the table
    python examples/validation_table.py --record    # also write the evidence

``--record`` writes the validation section of ``docs/data/findings.json``, which
the published page and the README's table are built from (ADR 0007).
"""

from __future__ import annotations

import argparse
import datetime
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ab_lab.power import power_t, sample_size_for_proportion
from ab_lab.results import Claim, SimulationSummary
from ab_lab.sequential import tau_from_mde
from ab_lab.simulate import (
    binary_draw,
    msprt_p_value,
    normal_draw,
    proportion_p_value,
    run_experiments,
    run_with_peeking,
    welch_p_value,
)

# Running a file puts *its own* directory on the path, not the project root, so
# `python examples/...` cannot see the site generator that owns the table
# formatting. pytest gets this from `pythonpath` in pyproject; a script has to
# say it out loud, and it is kept to one import so the noqa stays local.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sitegen import markdown, record  # noqa: E402

ALPHA = 0.05
N_EXPERIMENTS = 10_000
SEED = 20260721


@dataclass(frozen=True)
class Row:
    """One falsifiable claim, and the simulation that tests it.

    ``claim`` matters: a fixed-horizon test promises its type I error *equals*
    alpha, while an anytime-valid test promises only that it stays *at most*
    alpha. Judging the second by the first's standard would flag correct,
    deliberately conservative behaviour as a failure.
    """

    scenario: str
    expected: float
    summary: SimulationSummary
    claim: Claim = "equals"

    def as_payload(self) -> dict:
        """One row as evidence: the claim, and the counts that test it.

        No rate and no verdict is stored. Both are derived at render time by
        ``SimulationSummary``, which is also what the test suite asserts
        against - so the page cannot call a row "pass" that the suite would
        not.
        """
        return {
            "scenario": self.scenario,
            "expected": self.expected,
            "claim": self.claim,
            **record.counts_of(self.summary),
        }


def build_rows(rng: np.random.Generator) -> list[Row]:
    """Each row is one falsifiable claim the package makes about itself."""
    rows: list[Row] = []

    rows.append(
        Row(
            "Welch t-test, A/A (no effect)",
            ALPHA,
            run_experiments(
                normal_draw(n_per_group=500, mean=10.0, std_dev=3.0),
                welch_p_value,
                N_EXPERIMENTS,
                rng,
                alpha=ALPHA,
            ),
        )
    )
    rows.append(
        Row(
            "Two-proportion z-test, A/A (no effect)",
            ALPHA,
            run_experiments(
                binary_draw(n_per_group=3_000, baseline_rate=0.10),
                proportion_p_value,
                N_EXPERIMENTS,
                rng,
                alpha=ALPHA,
            ),
        )
    )
    rows.append(
        Row(
            "Welch t-test, A/B (d = 0.2, n = 400)",
            power_t(0.2, 400, alpha=ALPHA),
            run_experiments(
                normal_draw(n_per_group=400, mean=0.0, std_dev=4.0, absolute_lift=0.8),
                welch_p_value,
                N_EXPERIMENTS,
                rng,
                alpha=ALPHA,
            ),
        )
    )

    design = sample_size_for_proportion(0.10, 0.01, alpha=ALPHA, power=0.8)
    rows.append(
        Row(
            f"Sample size solved for 80% power (n = {design.per_group:,}/arm)",
            0.80,
            run_experiments(
                binary_draw(design.per_group, 0.10, absolute_lift=0.01),
                proportion_p_value,
                N_EXPERIMENTS,
                rng,
                alpha=ALPHA,
            ),
        )
    )
    rows.append(
        Row(
            "mSPRT, A/A with 10 looks (anytime-valid)",
            ALPHA,
            run_with_peeking(
                normal_draw(n_per_group=2_000, mean=0.0, std_dev=1.0),
                msprt_p_value(tau=tau_from_mde(0.1)),
                list(range(200, 2_001, 200)),
                N_EXPERIMENTS,
                rng,
                alpha=ALPHA,
            ),
            claim="at most",
        )
    )
    return rows


def payload(rows: list[Row], recorded_on: str) -> dict:
    """The validation section as evidence: the design, and one entry per claim."""
    counts = {row.summary.n_experiments for row in rows}
    if len(counts) != 1:
        raise ValueError(f"rows disagree on the experiment count: {sorted(counts)}")
    return {
        "design": {"seed": SEED, "alpha": ALPHA, "n_experiments": counts.pop()},
        "recorded": record.provenance("examples/validation_table.py", recorded_on),
        "rows": [row.as_payload() for row in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the package against a known truth.")
    parser.add_argument(
        "--record",
        action="store_true",
        help="write docs/data/findings.json, the evidence the page is built from",
    )
    arguments = parser.parse_args()

    # The tables carry "±" and "≤". A Windows console defaults to a code page
    # that cannot encode either, and the fix is to make this stream capable
    # rather than to keep a second, ASCII-only formatting of the same numbers.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rng = np.random.default_rng(SEED)
    section = payload(build_rows(rng), datetime.date.today().isoformat())

    # Rendered through the same code as the page, so stdout, the README and the
    # published table cannot disagree about a number or how it is formatted.
    validation = record.validation_of(section)
    print(markdown.validation_preamble(validation) + "\n")
    print(markdown.validation_table(validation))

    if arguments.record:
        record.write_section("validation", section)
        print(f"\nRecorded to {record.RECORD_PATH.relative_to(record.ROOT)}")
        print("Rebuild the page with: python -m sitegen.build")


if __name__ == "__main__":
    main()
