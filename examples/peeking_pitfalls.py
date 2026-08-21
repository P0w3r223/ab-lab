"""The peeking problem, measured: type I error against the number of looks.

Checking a fixed-horizon test repeatedly and stopping at the first significant
result does not "get to the answer faster" - it changes the test. This script
quantifies by how much, and shows that the same schedule of looks is harmless
under a sequential test.

Usage::

    python examples/peeking_pitfalls.py             # print the table
    python examples/peeking_pitfalls.py --record    # also write the evidence

``--record`` writes ``docs/data/findings.json``, which is what the published
page and the README's tables are built from (ADR 0007). The chart is no longer
written here: it is inline SVG generated from that record, because a raster
image cannot follow the reader's colour scheme and its numbers cannot be
checked against anything.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np

from ab_lab.results import SimulationSummary
from ab_lab.sequential import tau_from_mde
from ab_lab.simulate import msprt_p_value, normal_draw, peeking_curve, welch_p_value

# Running a file puts *its own* directory on the path, not the project root, so
# `python examples/...` cannot see the site generator that owns the table
# formatting. pytest gets this from `pythonpath` in pyproject; a script has to
# say it out loud, and it is kept to one import so the noqa stays local.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sitegen import markdown, record  # noqa: E402

ALPHA = 0.05
MAX_N = 2_000
LOOK_COUNTS = [1, 2, 3, 5, 7, 10, 14, 20]
N_EXPERIMENTS = 4_000
SEED = 20260721

KEY = "peeking"
TITLE = "The cost of looking early"
QUESTION = (
    "A fixed-horizon p-value is valid at the sample size the experiment was designed "
    "for, and nowhere else. These are A/A experiments - there is no effect to find - "
    "analysed with the same data and the same schedule of looks under two decision "
    "rules: the ordinary t-test read repeatedly, and a sequential test whose guarantee "
    "holds at every look."
)
X_LABEL = "Times the results are checked"
NAIVE_NAME = "Welch t-test (fixed horizon)"
CORRECTED_NAME = "mSPRT (anytime-valid)"


def run() -> tuple[list[SimulationSummary], list[SimulationSummary]]:
    """Same data-generating process, same looks, two decision rules."""
    draw = normal_draw(n_per_group=MAX_N, mean=0.0, std_dev=1.0)

    fixed_horizon = peeking_curve(
        draw, welch_p_value, MAX_N, LOOK_COUNTS, N_EXPERIMENTS,
        np.random.default_rng(SEED), alpha=ALPHA,
    )
    sequential = peeking_curve(
        draw, msprt_p_value(tau=tau_from_mde(0.1)), MAX_N, LOOK_COUNTS, N_EXPERIMENTS,
        np.random.default_rng(SEED), alpha=ALPHA,
    )
    return fixed_horizon, sequential


def payload(
    fixed_horizon: list[SimulationSummary],
    sequential: list[SimulationSummary],
    recorded_on: str,
) -> dict:
    """The finding as counts plus the design that produced them.

    No rate is stored. Everything the page prints as a percentage is derived
    from ``n_rejections / n_experiments`` at render time, by the same dataclass
    the test suite asserts against.
    """
    return {
        "title": TITLE,
        "question": QUESTION,
        "x_label": X_LABEL,
        "nominal_alpha": ALPHA,
        "design": {
            "seed": SEED,
            "alpha": ALPHA,
            "n_experiments": N_EXPERIMENTS,
            "n_per_group": MAX_N,
            "look_counts": list(LOOK_COUNTS),
            "draw": "normal(mean=0, std_dev=1), no true effect",
        },
        "recorded": record.provenance("examples/peeking_pitfalls.py", recorded_on),
        "series": [
            {
                "name": name,
                "role": role,
                "cells": [
                    {"x": looks, **record.counts_of(summary)}
                    for looks, summary in zip(LOOK_COUNTS, summaries, strict=True)
                ],
            }
            for name, role, summaries in (
                (NAIVE_NAME, "naive", fixed_horizon),
                (CORRECTED_NAME, "corrected", sequential),
            )
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure what peeking costs.")
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

    fixed_horizon, sequential = run()
    today = datetime.date.today().isoformat()
    section = payload(fixed_horizon, sequential, today)

    # Rendered through the same code as the page, so stdout, the README and the
    # published table cannot disagree about a number or how it is formatted.
    finding = record.finding_of(KEY, section)
    print(markdown.finding_preamble(finding).replace("**", "") + "\n")
    print(markdown.finding_table(finding))

    if arguments.record:
        record.write_finding(KEY, section)
        print(f"\nRecorded to {record.RECORD_PATH.relative_to(record.ROOT)}")
        print("Rebuild the page with: python -m sitegen.build")


if __name__ == "__main__":
    main()
