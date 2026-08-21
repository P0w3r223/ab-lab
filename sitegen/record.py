"""The committed evidence the published page is built from.

The page's figures come from Monte Carlo runs costing minutes, so the generator
cannot recompute them - it would put a multi-minute simulation in front of every
pull request on two Python versions. It reads them from
``docs/data/findings.json`` instead, which the example scripts write with
``--record``.

**The record stores counts, never rates.** ``n_experiments`` and
``n_rejections`` go in; :meth:`SimulationSummary.rejection_rate` and
:meth:`SimulationSummary.monte_carlo_error` come back out. Neither the page nor
the README computes either of them, so neither can disagree with the test suite
about what a rate is - which is the failure this whole arrangement exists to
prevent, and one this repository had already committed once, with a single table
row appearing in three different formats.

The file is published (see ADR 0007): a reader can download the counts and check
any figure on the page without running anything, which is the page's argument
made literal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ab_lab.results import Claim, SimulationSummary

ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "docs" / "data" / "findings.json"

#: Bumped when the shape below changes in a way a reader would notice.
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Cell:
    """One simulated cell: a position on the x axis and what it measured."""

    x: float
    summary: SimulationSummary


@dataclass(frozen=True)
class Series:
    """One decision rule, measured across the x axis.

    ``role`` is ``"naive"`` for the procedure that breaks, ``"corrected"`` for
    the one that does not. The renderer styles by role rather than by name, so
    a finding cannot accidentally colour its correction as its failure.
    """

    name: str
    role: str
    cells: tuple[Cell, ...]

    @property
    def rates(self) -> tuple[float, ...]:
        return tuple(cell.summary.rejection_rate for cell in self.cells)


@dataclass(frozen=True)
class Finding:
    """One mechanism that turns a nominal alpha into something larger."""

    key: str
    title: str
    question: str
    x_label: str
    nominal: float
    design: dict[str, Any]
    recorded: dict[str, Any]
    series: tuple[Series, ...]

    def series_by_role(self, role: str) -> Series:
        for series in self.series:
            if series.role == role:
                return series
        raise KeyError(f"finding {self.key!r} has no {role!r} series")

    @property
    def x_values(self) -> tuple[float, ...]:
        return tuple(cell.x for cell in self.series[0].cells)

    @property
    def worst_rate(self) -> float:
        """The naive procedure at its worst - the finding's headline number."""
        return max(self.series_by_role("naive").rates)


@dataclass(frozen=True)
class ValidationRow:
    """One falsifiable claim the package makes about itself."""

    scenario: str
    expected: float
    claim: Claim
    summary: SimulationSummary

    @property
    def passed(self) -> bool:
        return self.summary.agrees_with(self.expected, self.claim)


@dataclass(frozen=True)
class Validation:
    design: dict[str, Any]
    recorded: dict[str, Any]
    rows: tuple[ValidationRow, ...]


@dataclass(frozen=True)
class Record:
    """Everything the page is allowed to know."""

    findings: dict[str, Finding]
    validation: Validation | None


def counts_of(summary: SimulationSummary) -> dict[str, Any]:
    """The part of a summary that is evidence rather than arithmetic."""
    return {
        "n_experiments": summary.n_experiments,
        "n_rejections": summary.n_rejections,
        "mean_estimate": summary.mean_estimate,
        "mean_absolute_estimate_when_stopped": (
            summary.mean_absolute_estimate_when_stopped
        ),
    }


def summary_of(counts: dict[str, Any], nominal_alpha: float, label: str) -> SimulationSummary:
    """Rebuild a summary from counts, so the rates are derived in one place."""
    return SimulationSummary(
        n_experiments=counts["n_experiments"],
        n_rejections=counts["n_rejections"],
        nominal_alpha=nominal_alpha,
        mean_estimate=counts["mean_estimate"],
        label=label,
        mean_absolute_estimate_when_stopped=counts["mean_absolute_estimate_when_stopped"],
    )


def provenance(script: str, recorded_on: str) -> dict[str, Any]:
    """Where a section's numbers came from.

    ``recorded_on`` is passed in rather than read from the clock here, so that
    the only impurity in the pipeline sits in the script a human runs.
    """
    import platform

    import numpy

    import ab_lab

    return {
        "script": script,
        "recorded_on": recorded_on,
        "ab_lab_version": ab_lab.__version__,
        "numpy_version": numpy.__version__,
        "python_version": platform.python_version(),
    }


def write_finding(key: str, payload: dict[str, Any], path: Path = RECORD_PATH) -> None:
    """Add or replace one finding, leaving its siblings alone.

    Three scripts will eventually write into ``findings``; recording the peeking
    curve must not delete the clustering one.
    """
    existing: dict[str, Any] = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8")).get("findings", {})
    write_section("findings", {**existing, key: payload}, path)


def write_section(section: str, payload: dict[str, Any], path: Path = RECORD_PATH) -> None:
    """Replace one section of the record, leaving the others untouched.

    Two scripts write into one file and neither owns all of it, so recording the
    peeking curve must not silently delete the validation table.
    """
    document: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    if path.exists():
        document = json.loads(path.read_text(encoding="utf-8"))
        document["schema_version"] = SCHEMA_VERSION
    document[section] = payload

    path.parent.mkdir(parents=True, exist_ok=True)
    serialised = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False)
    path.write_text(serialised + "\n", encoding="utf-8", newline="\n")


def load(path: Path = RECORD_PATH) -> Record:
    """Read the record, or explain precisely how to create it."""
    if not path.exists():
        raise FileNotFoundError(
            f"no evidence at {path}. Record it with:\n"
            f"    python examples/peeking_pitfalls.py --record\n"
            f"    python examples/validation_table.py --record"
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path} is schema version {document.get('schema_version')}, "
            f"this generator reads version {SCHEMA_VERSION}"
        )

    findings = {
        key: finding_of(key, raw) for key, raw in sorted(document.get("findings", {}).items())
    }
    raw_validation = document.get("validation")
    return Record(
        findings=findings,
        validation=validation_of(raw_validation) if raw_validation else None,
    )


def finding_of(key: str, raw: dict[str, Any]) -> Finding:
    nominal = raw["nominal_alpha"]
    return Finding(
        key=key,
        title=raw["title"],
        question=raw["question"],
        x_label=raw["x_label"],
        nominal=nominal,
        design=raw["design"],
        recorded=raw["recorded"],
        series=tuple(
            Series(
                name=series["name"],
                role=series["role"],
                cells=tuple(
                    Cell(
                        x=cell["x"],
                        summary=summary_of(cell, nominal, f"{series['name']} @ {cell['x']}"),
                    )
                    for cell in series["cells"]
                ),
            )
            for series in raw["series"]
        ),
    )


def validation_of(raw: dict[str, Any]) -> Validation:
    return Validation(
        design=raw["design"],
        recorded=raw["recorded"],
        rows=tuple(
            ValidationRow(
                scenario=row["scenario"],
                expected=row["expected"],
                claim=row["claim"],
                summary=summary_of(row, raw["design"]["alpha"], row["scenario"]),
            )
            for row in raw["rows"]
        ),
    )
