"""One claim, measured three ways: what turns a 5% test into something else.

A nominal false positive rate is not a property of a test. It is a property of a
test *plus* how many decisions are taken from it. Peeking takes many decisions in
time; clustering takes many correlated rows and counts them as independent
evidence; a metric suite takes many decisions in parallel. Each independently
turns 5% into 25-40%, each has a named correction, and each is measured here on
experiments where there is no effect to find.

The peeking curve lives in ``peeking_pitfalls.py`` because it was the first and
has its own narrative. This script records the other two and prints all three
together.

Usage::

    python examples/three_inflations.py             # print the tables
    python examples/three_inflations.py --record    # also write the evidence
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np

from ab_lab.multiplicity import CORRECTIONS
from ab_lab.results import SimulationSummary
from ab_lab.simulate import (
    cluster_robust_p_value,
    clustered_normal_draw,
    correlated_normal_suite_draw,
    naive_welch_p_value,
    run_clustered_experiments,
    run_metric_suite,
    welch_suite_p_values,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sitegen import markdown, record  # noqa: E402

ALPHA = 0.05
SEED = 20260821

# --- Finding 2: repeated measurements per user -------------------------------
CLUSTER_KEY = "clustering"
ROWS_PER_USER = [1, 2, 5, 10, 20]
CLUSTERS_PER_ARM = 200
ICC = 0.30
CLUSTER_EXPERIMENTS = 2_000

# --- Finding 3: many metrics at once -----------------------------------------
SUITE_KEY = "multiplicity"
METRIC_COUNTS = [1, 2, 5, 10, 20]
UNITS_PER_ARM = 400
SUITE_EXPERIMENTS = 1_500


def run_clustering() -> tuple[list[SimulationSummary], list[SimulationSummary]]:
    """Same draws, two analyses, across increasingly repetitive users."""
    naive: list[SimulationSummary] = []
    robust: list[SimulationSummary] = []
    for rows in ROWS_PER_USER:
        draw = clustered_normal_draw(CLUSTERS_PER_ARM, rows, ICC, std_dev=1.0)
        for p_value_fn, into in ((naive_welch_p_value, naive), (cluster_robust_p_value, robust)):
            into.append(
                run_clustered_experiments(
                    draw,
                    p_value_fn,
                    CLUSTER_EXPERIMENTS,
                    np.random.default_rng(SEED + rows),
                    alpha=ALPHA,
                    label=f"{rows} rows per user",
                )
            )
    return naive, robust


def run_multiplicity() -> tuple[list[SimulationSummary], list[SimulationSummary]]:
    """Same draws, corrected and not, across increasingly large metric suites."""
    uncorrected: list[SimulationSummary] = []
    corrected: list[SimulationSummary] = []
    for metrics in METRIC_COUNTS:
        draw = correlated_normal_suite_draw(UNITS_PER_ARM, metrics, correlation=0.0)
        for correction, into in ((None, uncorrected), (CORRECTIONS["holm"], corrected)):
            into.append(
                run_metric_suite(
                    draw,
                    welch_suite_p_values,
                    SUITE_EXPERIMENTS,
                    np.random.default_rng(SEED + metrics),
                    alpha=ALPHA,
                    correction=correction,
                    label=f"{metrics} metrics",
                )
            )
    return uncorrected, corrected


def _payload(
    title: str,
    question: str,
    x_label: str,
    x_values: list[int],
    design: dict,
    script: str,
    recorded_on: str,
    naive_name: str,
    naive: list[SimulationSummary],
    corrected_name: str,
    corrected: list[SimulationSummary],
) -> dict:
    return {
        "title": title,
        "question": question,
        "x_label": x_label,
        "nominal_alpha": ALPHA,
        "design": design,
        "recorded": record.provenance(script, recorded_on),
        "series": [
            {
                "name": name,
                "role": role,
                "cells": [
                    {"x": x, **record.counts_of(summary)}
                    for x, summary in zip(x_values, summaries, strict=True)
                ],
            }
            for name, role, summaries in (
                (naive_name, "naive", naive),
                (corrected_name, "corrected", corrected),
            )
        ],
    }


def clustering_payload(naive, robust, recorded_on: str) -> dict:
    return _payload(
        title="The cost of counting a user more than once",
        question=(
            "A/A experiments with no effect anywhere, where each user contributes "
            "several rows that move together. Analysed row by row, every extra row "
            "looks like extra evidence and the variance of the difference is "
            "understated; analysed with a cluster-robust standard error, on the same "
            "draws, the rate holds. One row per user is the control case, where the "
            "correction must cost nothing at all."
        ),
        x_label="Rows per user",
        x_values=ROWS_PER_USER,
        design={
            "seed": SEED,
            "alpha": ALPHA,
            "n_experiments": CLUSTER_EXPERIMENTS,
            "n_clusters_per_group": CLUSTERS_PER_ARM,
            "icc": ICC,
            "rows_per_user": list(ROWS_PER_USER),
            "draw": "normal(0, 1) split between user and row, total variance fixed",
        },
        script="examples/three_inflations.py",
        recorded_on=recorded_on,
        naive_name="Every row an observation",
        naive=naive,
        corrected_name="Cluster-robust standard error",
        corrected=robust,
    )


def multiplicity_payload(uncorrected, corrected, recorded_on: str) -> dict:
    return _payload(
        title="The cost of asking more than one question",
        question=(
            "A/A experiments measured on several independent metrics at once, each "
            "read at alpha. The chance that at least one comes back significant is "
            "1 - (1 - alpha) to the power of the family size - the same arithmetic as "
            "peeking, run across metrics instead of across time. Holm holds the "
            "family-wise rate at alpha whatever the family size."
        ),
        x_label="Metrics measured at once",
        x_values=METRIC_COUNTS,
        design={
            "seed": SEED,
            "alpha": ALPHA,
            "n_experiments": SUITE_EXPERIMENTS,
            "n_per_group": UNITS_PER_ARM,
            "correlation": 0.0,
            "metric_counts": list(METRIC_COUNTS),
            "draw": "independent normal(0, 1) metrics, no true effect",
        },
        script="examples/three_inflations.py",
        recorded_on=recorded_on,
        naive_name="Each metric read at alpha",
        naive=uncorrected,
        corrected_name="Holm",
        corrected=corrected,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure two more ways a 5% test is not 5%.")
    parser.add_argument(
        "--record",
        action="store_true",
        help="write docs/data/findings.json, the evidence the page is built from",
    )
    arguments = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    today = datetime.date.today().isoformat()
    sections = {
        CLUSTER_KEY: clustering_payload(*run_clustering(), today),
        SUITE_KEY: multiplicity_payload(*run_multiplicity(), today),
    }

    for key, section in sections.items():
        finding = record.finding_of(key, section)
        print(f"\n## {finding.title}\n")
        print(markdown.finding_table(finding))
        if arguments.record:
            record.write_finding(key, section)

    if arguments.record:
        print(f"\nRecorded to {record.RECORD_PATH.relative_to(record.ROOT)}")
        print("Rebuild the page with: python -m sitegen.build")


if __name__ == "__main__":
    main()
