"""The published page is generated, so it can be stale - and it is a page of numbers.

Two guards, both cheap enough to run on every pull request.

**G1** compares the committed bytes against what the generator produces. Unlike
the reference implementation this pattern came from, there is nothing to exempt:
``build()`` reads a JSON file and formats strings, touching no clock, no ``git``,
no network and no random generator, so a byte-equality assertion needs no
allowance for a commit hash or a timestamp.

**G3** enforces the import budget instead of documenting it. ``sitegen`` may use
the closed-form parts of ``ab_lab``; it may not simulate. The obvious version of
this guard - monkeypatch the runners in-process - does not work, twice over: a
``from ... import`` at module top binds the name before any fixture runs, and
patching ``numpy.random.Generator`` is a no-op because ``default_rng``
constructs the C-level type directly and never looks up the module attribute.
So the build runs in a subprocess, with ``default_rng`` itself replaced, and the
proof is that ``ab_lab.simulate`` never entered ``sys.modules``.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from sitegen import build
from sitegen.record import ROOT

# Run the build with the one function that can start a simulation replaced, then
# report which parts of the package were imported at all.
#
# The patch goes in *after* the imports, and that ordering is load-bearing rather
# than cosmetic: scipy calls `np.random.default_rng` itself at import time, while
# building the example blocks in its distribution docstrings. Patching first
# turns this guard into an assertion about scipy's internals. What the guard is
# for is the build *call*, and import-time purity is covered by the separate
# assertion that `ab_lab.simulate` never arrived.
_PURITY_PROBE = """
import json
import sys

import numpy.random

import sitegen.build


def _forbidden(*args, **kwargs):
    raise AssertionError("sitegen.build constructed a random generator")


numpy.random.default_rng = _forbidden
sitegen.build.build()
print(json.dumps(sorted(name for name in sys.modules if name.startswith("ab_lab"))))
"""


@pytest.mark.parametrize("path", sorted(build.build(), key=str), ids=lambda path: path.name)
def test_the_committed_artefact_is_what_the_generator_produces(path):
    """G1: a hand-edited page, README table or chart turns the pull request red."""
    expected = build.build()[path]
    assert path.exists(), f"{path.relative_to(ROOT)} is missing; run `python -m sitegen.build`"
    actual = path.read_text(encoding="utf-8")
    assert actual == expected, (
        f"{path.relative_to(ROOT)} is stale or was edited by hand. "
        f"Rebuild it with `python -m sitegen.build`."
    )


def test_the_generator_does_not_simulate():
    """G3: the page is built from the record, and the record only."""
    completed = subprocess.run(
        [sys.executable, "-c", _PURITY_PROBE],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    imported = json.loads(completed.stdout)
    assert "ab_lab.simulate" not in imported, (
        "sitegen imported the simulation harness. The page must be built from "
        "docs/data/findings.json, not from a run: recomputing it would put a "
        "multi-minute Monte Carlo in front of every pull request."
    )


def test_the_check_flag_agrees_with_the_committed_tree():
    """`--check` is what a future CI step would call; it must match G1."""
    assert build.main(["--check"]) == 0
