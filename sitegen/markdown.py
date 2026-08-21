"""Markdown renderings: the README's tables and the scripts' stdout.

The same data the page renders, in the format GitHub shows. Both call
:mod:`sitegen.numbers` for every figure, so the two surfaces cannot disagree
about a rate, and the README's tables are rewritten by the generator between
fences rather than pasted by hand.
"""

from __future__ import annotations

from ab_lab.results import SimulationSummary

from .numbers import error, integer, percent, proportion, rate_with_error
from .record import Finding, Validation

#: The generator replaces whatever sits between these markers. They are HTML
#: comments so GitHub does not render them.
FENCE_OPEN = "<!-- generated:{name} -->"
FENCE_CLOSE = "<!-- /generated:{name} -->"


def fence(name: str, body: str) -> str:
    return f"{FENCE_OPEN.format(name=name)}\n{body}\n{FENCE_CLOSE.format(name=name)}"


def finding_preamble(finding: Finding) -> str:
    design = finding.design
    return (
        f"A/A experiments - **no true effect at all** - with "
        f"{integer(design['n_per_group'])} units per arm, alpha "
        f"{design['alpha']}, {integer(design['n_experiments'])} runs per cell:"
    )


def finding_table(finding: Finding) -> str:
    """One finding as a Markdown table: x, the naive rate, the corrected rate."""
    naive = finding.series_by_role("naive")
    corrected = finding.series_by_role("corrected")
    worst = max(naive.rates)

    header = f"| {finding.x_label} | {naive.name} | {corrected.name} |"
    rows = [header, "|---|---|---|"]
    for x, naive_cell, corrected_cell in zip(
        finding.x_values, naive.cells, corrected.cells, strict=True
    ):
        # The worst cell is the headline, and bolding it in the source keeps the
        # emphasis with the number rather than with a row index that can shift.
        naive_text = rate_with_error(naive_cell.summary)
        if naive_cell.summary.rejection_rate == worst:
            naive_text = _bold_rate(naive_cell.summary)
        rows.append(f"| {x:g} | {naive_text} | {rate_with_error(corrected_cell.summary)} |")
    return "\n".join(rows)


def _bold_rate(summary: SimulationSummary) -> str:
    return f"**{percent(summary.rejection_rate)}** (±{percent(summary.monte_carlo_error)})"


def mechanisms_table(findings: list[Finding]) -> str:
    """All the findings in one table: each mechanism at its worst, and corrected.

    This is the claim in five columns. Generated rather than written, because a
    summary of numbers that live elsewhere is precisely the thing that drifts -
    and it drifted here once already, in a table the README used to keep by hand.
    """
    rows = [
        "| At its worst | Nominal | Measured | Correction | After |",
        "|---|---|---|---|---|",
    ]
    for finding in findings:
        naive = finding.series_by_role("naive")
        corrected = finding.series_by_role("corrected")
        worst_index = naive.rates.index(max(naive.rates))
        where = f"{finding.x_values[worst_index]:g} {finding.x_label.lower()}"
        rows.append(
            f"| {where} | {percent(finding.nominal, 0)} "
            f"| **{percent(naive.rates[worst_index])}** | {corrected.name} "
            f"| {percent(corrected.rates[worst_index])} |"
        )
    return "\n".join(rows)


def validation_table(validation: Validation) -> str:
    """Each row is a falsifiable claim, and the verdict comes from the summary."""
    rows = [
        "| Scenario | Claim | Empirical | MC error | Verdict |",
        "|---|---|---|---|---|",
    ]
    for row in validation.rows:
        symbol = "=" if row.claim == "equals" else "≤"
        rows.append(
            f"| {row.scenario} | {symbol} {proportion(row.expected)} "
            f"| {proportion(row.summary.rejection_rate)} | {error(row.summary)} "
            f"| {'pass' if row.passed else 'CHECK'} |"
        )
    return "\n".join(rows)


def validation_preamble(validation: Validation) -> str:
    design = validation.design
    return (
        f"{integer(design['n_experiments'])} simulated experiments per row, seed "
        f"{design['seed']}. \"MC error\" is the standard error of the empirical rate - "
        f"the noise floor of the run itself:"
    )


def replace_fences(text: str, blocks: dict[str, str]) -> str:
    """Swap each named fenced region for freshly rendered content.

    A missing fence is an error rather than a silent no-op: the failure mode
    worth guarding against is a README that keeps a stale table because the
    marker was renamed, which looks exactly like a README that was updated.
    """
    for name, body in blocks.items():
        opening = FENCE_OPEN.format(name=name)
        closing = FENCE_CLOSE.format(name=name)
        start = text.find(opening)
        end = text.find(closing)
        if start == -1 or end == -1:
            raise ValueError(f"README has no '{name}' fence to fill")
        text = text[:start] + fence(name, body) + text[end + len(closing) :]
    return text
