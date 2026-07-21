# 0004 - the demonstrations are scripts, not notebooks

Date: 2026-07-21
Status: accepted
Author: P0w3r223 + Claude

---

## Context

The peeking result and the validation table are the two things a reader should
see first. Both are produced by running thousands of simulations, and both end
in a table and a chart. The obvious home for that is a Jupyter notebook.

## Options

1. **Notebook.** Renders on GitHub with outputs inline, no setup for the reader,
   and it is the format this kind of analysis is usually shipped in. Its outputs
   are stored in the file, so a committed notebook can disagree with the code it
   claims to run, and its diffs are unreadable.
2. **Plain scripts under `examples/`.** Reviewable diffs, importable in tests,
   and no way for the stored output to drift from the code. The reader has to
   run something, or trust the numbers pasted into the README.

## Decision

Scripts (`examples/validation_table.py`, `examples/peeking_pitfalls.py`), which
print markdown-ready tables and optionally write the chart. Their output is
pasted into the README, and the chart is committed under `docs/images/`.

The deciding argument is what these particular artefacts are *for*. They are the
project's evidence, and evidence that can silently disagree with the code is not
evidence. Every number in the README is regenerable with one command, and the
numeric claims the README makes about the API are also assertions in `tests/`.

## Consequences

- A reader who wants the analysis without running it gets the tables and the
  chart in the README - the common path is covered.
- The scripts are linted and importable, so they cannot rot the way a notebook
  quietly does.
- Cost: no inline narrative around the code the way a notebook interleaves it.
  The narrative lives in the README and in the module docstrings instead.
- Roadmap item 3 (the e-commerce case study) is the one artefact where a
  notebook genuinely fits - it is exposition, not evidence - and this decision
  does not rule that out.
