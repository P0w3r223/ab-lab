"""The peeking problem, measured: type I error against the number of looks.

Checking a fixed-horizon test repeatedly and stopping at the first significant
result does not "get to the answer faster" - it changes the test. This script
quantifies by how much, and shows that the same schedule of looks is harmless
under a sequential test.

Usage::

    python examples/peeking_pitfalls.py            # table only
    python examples/peeking_pitfalls.py --plot     # also writes the chart
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ab_lab.results import SimulationSummary
from ab_lab.sequential import tau_from_mde
from ab_lab.simulate import msprt_p_value, normal_draw, peeking_curve, welch_p_value

ALPHA = 0.05
MAX_N = 2_000
LOOK_COUNTS = [1, 2, 3, 5, 7, 10, 14, 20]
N_EXPERIMENTS = 4_000
SEED = 20260721
CHART_PATH = Path(__file__).resolve().parents[1] / "docs" / "images" / "peeking.png"

# --- Chart styling: clean matplotlib aligned with the portfolio page palette. ---
_ACCENT = "#2563eb"
_CRITICAL = "#d03b3b"
_CHART_STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
    "grid.color": "#e3e7ee", "grid.linewidth": 0.9,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": "#1c2430",
    "axes.titlepad": 12, "axes.labelcolor": "#667085", "axes.labelsize": 10.5,
    "text.color": "#1c2430", "xtick.color": "#667085", "ytick.color": "#667085",
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "font.size": 10.5,
    "legend.frameon": False, "legend.fontsize": 9.5,
}


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


def print_table(
    fixed_horizon: list[SimulationSummary], sequential: list[SimulationSummary]
) -> None:
    print(
        f"A/A experiments: no true effect, alpha = {ALPHA}, {N_EXPERIMENTS:,} runs per cell, "
        f"{MAX_N:,} units per arm, seed {SEED}.\n"
    )
    print("| Looks | Welch t-test (fixed horizon) | mSPRT (anytime-valid) |")
    print("|---|---|---|")
    for looks, naive, safe in zip(LOOK_COUNTS, fixed_horizon, sequential, strict=True):
        print(
            f"| {looks} | {naive.rejection_rate:.1%} "
            f"(+/-{naive.monte_carlo_error:.1%}) | {safe.rejection_rate:.1%} "
            f"(+/-{safe.monte_carlo_error:.1%}) |"
        )


def write_chart(
    fixed_horizon: list[SimulationSummary], sequential: list[SimulationSummary]
) -> None:
    """Save the chart. Imported lazily so the table runs without matplotlib."""
    try:
        import matplotlib
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "--plot needs matplotlib; install the dev extra: pip install -e \".[dev]\""
        ) from error

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(_CHART_STYLE)
    figure, axes = plt.subplots(figsize=(8.0, 4.5))
    axes.plot(
        LOOK_COUNTS, [summary.rejection_rate for summary in fixed_horizon],
        marker="o", markersize=6, lw=2.2, color=_CRITICAL,
        label="Welch t-test, checked repeatedly",
    )
    axes.plot(
        LOOK_COUNTS, [summary.rejection_rate for summary in sequential],
        marker="s", markersize=6, lw=2.2, color=_ACCENT,
        label="mSPRT (anytime-valid)",
    )
    axes.axhline(ALPHA, linestyle="--", linewidth=1.2, color="#898781",
                 label=f"nominal alpha = {ALPHA}")

    axes.set_xlabel("Number of times the results are checked")
    axes.set_ylabel("False positive rate (A/A experiments)")
    axes.set_title("Peeking turns a 5% test into something else")
    axes.set_ylim(0.0, max(summary.rejection_rate for summary in fixed_horizon) * 1.2)
    axes.legend()
    figure.tight_layout()

    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(CHART_PATH, dpi=150)
    print(f"\nChart written to {CHART_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="also write the chart (matplotlib)")
    arguments = parser.parse_args()

    fixed_horizon, sequential = run()
    print_table(fixed_horizon, sequential)
    if arguments.plot:
        write_chart(fixed_horizon, sequential)


if __name__ == "__main__":
    main()
