"""Regenerate the validation table in the README.

The answer to "how do you know your code computes the right thing?": run each
method on thousands of experiments where the truth is known and compare the
empirical rejection rate to the rate the theory promises.

Usage::

    python examples/validation_table.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ab_lab.power import power_t, sample_size_for_proportion
from ab_lab.results import SimulationSummary
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
    claim: str = "equals"

    def as_markdown(self) -> str:
        rate = self.summary.rejection_rate
        error = self.summary.monte_carlo_error
        if self.claim == "equals":
            target = f"= {self.expected:.4f}"
            passed = abs(rate - self.expected) < 4.0 * error
        elif self.claim == "at most":
            target = f"<= {self.expected:.4f}"
            passed = rate <= self.expected + 3.0 * error
        else:
            raise ValueError(f"claim must be 'equals' or 'at most', got {self.claim!r}")
        return (
            f"| {self.scenario} | {target} | {rate:.4f} | +/-{error:.4f} "
            f"| {'pass' if passed else 'CHECK'} |"
        )


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
                N_EXPERIMENTS // 2,
                rng,
                alpha=ALPHA,
            ),
            claim="at most",
        )
    )
    return rows


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = build_rows(rng)

    print(f"Validation run: {N_EXPERIMENTS:,} simulated experiments per row, seed {SEED}.\n")
    print("| Scenario | Claim | Empirical | MC error | Verdict |")
    print("|---|---|---|---|---|")
    for row in rows:
        print(row.as_markdown())


if __name__ == "__main__":
    main()
