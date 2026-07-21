# CLAUDE.md - ab-lab

Guidance for Claude Code (and any contributor) working in this repo.

## What this project is

Portfolio project **P2**. A Python package for designing and analysing A/B
experiments - power and sample size, the standard two-group tests, bootstrap
intervals, sample ratio mismatch, and a sequential test that makes early looks
legitimate. Deliberately not a wrapper: the methods are implemented here and
`statsmodels` appears only as a test oracle (see `docs/decisions/`).

## Architecture

```
src/ab_lab/
  results.py     # frozen dataclasses: every public function returns one of these
  power.py       # design: power_z / power_t core, sample size and MDE on top
  analyze.py     # post-hoc: Welch, two-proportion z, Mann-Whitney, bootstrap
  srm.py         # sample ratio mismatch (chi-square on the allocation)
  sequential.py  # mSPRT: anytime-valid p-values, SequentialMonitor
  simulate.py    # draws + p-value adapters + the A/A / A/B / peeking harness
examples/        # scripts that regenerate the README's tables and chart
docs/decisions/  # ADRs
```

Data flows one way: `power` is used before an experiment, `srm` the moment data
arrives, `sequential` during, `analyze` after. `simulate` depends on the others
and nothing depends on it.

## Commands

```bash
python -m venv .venv && source .venv/Scripts/activate
pip install -e ".[dev]"
pytest                                          # full suite
ruff check .                                    # lint
python examples/validation_table.py             # regenerate the validation table
python examples/peeking_pitfalls.py --plot      # regenerate the peeking table + chart
```

## Code rules

- **Typed, documented, no magic numbers.** Type hints on every public function;
  thresholds live in named module constants with a comment explaining the value.
- **The library never prints and never plots.** It returns frozen dataclasses.
  Rendering belongs in `examples/` and in user code.
- **Assumptions ship with the estimate.** Every `TestResult` carries an
  `assumptions` tuple. A number without its caveats is how experiments get
  misread.
- **Randomness is injected, never global.** Functions take a
  `numpy.random.Generator`; no module-level seeding.
- **Every new method needs both kinds of test**: (a) agreement with a reference
  implementation or hand arithmetic, (b) a simulation where the true answer is
  known. Simulation assertions are expressed in multiples of the Monte Carlo
  error, never as bare thresholds.

## Working rules

- Plan before code; discuss an architecture change before writing it.
- Small commits, Conventional Commits, one PR per session, with a "why".
- Explain the statistics before implementing them - assumptions, formula,
  failure modes. Any concept in this repo has to be defensible out loud.
- Non-trivial choices become an ADR in `docs/decisions/`: context, options,
  decision, consequences.

## What not to do

- Do not add a runtime dependency on `statsmodels`, or on any A/B library - it
  would defeat the purpose of the package (ADR 0001).
- Do not change a public signature or a returned dataclass without asking.
- Do not weaken a simulation test to make it pass. A drifting empirical rate is
  a finding, not a flaky test.
- Do not commit generated artefacts other than `docs/images/`.
