"""Build every generated artefact from the committed record.

Deliberately a pure function of files on disk. ``build()`` touches no clock, no
``git``, no network and no random generator, so the staleness test is one
byte-equality assertion with no exemptions - the reference implementation this
pattern came from has to strip a commit hash out of its page before comparing,
and not needing that is worth the constraint.

The import budget of ADR 0007 is enforced by ``tests/test_site_committed.py``,
which runs this module in a subprocess and asserts that ``ab_lab.simulate``
never entered ``sys.modules``. Cheap closed-form facts may be computed here;
anything Monte Carlo comes from the record.

Usage::

    python -m sitegen.build            # write the artefacts
    python -m sitegen.build --check    # fail if the committed ones are stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import markdown, page
from .charts import chart
from .record import ROOT, Record, load

PAGE_PATH = ROOT / "docs" / "index.html"
README_PATH = ROOT / "README.md"
CHART_LIGHT = ROOT / "docs" / "images" / "peeking-light.svg"
CHART_DARK = ROOT / "docs" / "images" / "peeking-dark.svg"


def readme(record: Record, current: str) -> str:
    """The README with its generated tables refreshed in place.

    Only the fenced regions move. Every paragraph around them is written by
    hand and reconciled with the page by a human reading both - which is the
    part of the presentation standard that cannot be automated, so it is not
    pretended otherwise here.
    """
    peeking = record.findings["peeking"]
    blocks = {
        "mechanisms": markdown.mechanisms_table(page.ordered_findings(record)),
        "peeking-preamble": markdown.finding_preamble(peeking),
        "peeking": markdown.finding_table(peeking),
    }
    if record.validation is not None:
        blocks["validation-preamble"] = markdown.validation_preamble(record.validation)
        blocks["validation"] = markdown.validation_table(record.validation)
    return markdown.replace_fences(current, blocks)


def build(record: Record | None = None) -> dict[Path, str]:
    """Every generated file, as a mapping from path to intended content."""
    record = load() if record is None else record
    peeking = record.findings["peeking"]
    return {
        PAGE_PATH: page.render(record),
        CHART_LIGHT: chart(peeking, theme="light") + "\n",
        CHART_DARK: chart(peeking, theme="dark") + "\n",
        README_PATH: readme(record, README_PATH.read_text(encoding="utf-8")),
    }


def write(artefacts: dict[Path, str]) -> list[Path]:
    """Write with explicit LF, whatever the platform would otherwise do."""
    changed = []
    for path, content in artefacts.items():
        previous = path.read_text(encoding="utf-8") if path.exists() else None
        if previous == content:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        changed.append(path)
    return changed


def stale(artefacts: dict[Path, str]) -> list[Path]:
    """Committed files that no longer match what the generator produces."""
    return [
        path
        for path, content in artefacts.items()
        if not path.exists() or path.read_text(encoding="utf-8") != content
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the published page from the record.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit non-zero if a committed artefact is stale",
    )
    arguments = parser.parse_args(argv)

    artefacts = build()
    if arguments.check:
        outdated = stale(artefacts)
        for path in outdated:
            print(f"stale: {path.relative_to(ROOT)}")
        return 1 if outdated else 0

    for path in write(artefacts):
        print(f"wrote: {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
