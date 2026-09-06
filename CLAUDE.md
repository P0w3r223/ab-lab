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
  results.py     # frozen dataclasses: every estimating function returns one
  _validation.py # private: the shared alpha / power / sample argument checks
  power.py       # design: power_z / power_t core, sample size and MDE on top
  analyze.py     # post-hoc: Welch, two-proportion z, Mann-Whitney, bootstrap
  srm.py         # sample ratio mismatch (chi-square on the allocation)
  sequential.py  # mSPRT: anytime-valid p-values, SequentialMonitor
  cluster.py     # repeated measurements per unit: CR1 sandwich, design effect
  multiplicity.py # many metrics: Bonferroni, Holm, Benjamini-Hochberg
  ratio.py       # ratios of two totals: the delta method over cluster.py
  cuped.py       # variance reduction from a pre-experiment covariate
  simulate.py    # draws + p-value adapters + the A/A / A/B / peeking harness
sitegen/         # renders the page, the README tables and the SVG charts
  record.py      #   the committed evidence: counts in, SimulationSummary out
  charts.py      #   one chart primitive, page mode and standalone mode
  numbers.py     #   the one place a figure is formatted - it exists because `14,745` and
                 #   `14 745` once shipped for the same number from two call sites
  markdown.py    #   the README tables
  page.py        #   the HTML document
  theme.py       #   the portfolio's head block and colour palette, vendored rather than
                 #   shared; a recorded debt, and the file the page-spec work touches
  build.py       #   pure function of files on disk; never simulates
examples/        # scripts that measure, then --record the evidence
docs/index.html  # generated, guarded byte for byte by tests/test_site_committed.py
docs/data/       # the committed evidence the page is built from
docs/decisions/  # ADRs
CHANGELOG.md     # what changed; the ADRs say why
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
python examples/validation_table.py             # print the validation table
python examples/peeking_pitfalls.py             # print the peeking table
python examples/three_inflations.py             # print the other two
python examples/ecommerce_case_study.py         # one experiment, end to end

# Re-measure and republish. The scripts cost minutes; the build costs milliseconds.
python examples/peeking_pitfalls.py --record    # rewrite docs/data/findings.json
python examples/validation_table.py --record
python examples/three_inflations.py --record
python -m sitegen.build                         # page, README tables and charts
python -m sitegen.build --check                 # fail if a committed artefact is stale
```

## Code rules

- **Typed, documented, no magic numbers.** Type hints on every public function;
  thresholds live in named module constants with a comment explaining the value.
- **The library never prints and never plots.** It returns frozen dataclasses.
  The rule's subject is `src/ab_lab/`: rendering belongs in `sitegen/`, which
  owns every table and chart the project publishes, and the scripts in
  `examples/` print through it rather than formatting anything themselves.
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
- Do not commit generated artefacts other than `docs/images/`, `docs/index.html`
  and `docs/data/`, each of which is guarded by a test asserting that the
  committed bytes are what the generator produces (ADR 0007). The rule exists to
  keep *unverifiable* artefacts out; the evidence record is its opposite - it is
  what makes the page checkable. Never hand-edit any of them, including the
  fenced `<!-- generated:... -->` regions of `README.md`.

## The published page

`docs/index.html` is one of twelve surfaces held to a single specification: ten house colour tokens
with pinned per-theme values, a dark override, six card-metadata tags, a profile back-link, a
result-shaped `h1`, and — since S4 — the rule that **every figure the surface prints is a figure
a committed artifact prints**, never a rounding and never a re-derivation. The spec is
`docs/audit/0007_divergence-and-the-page-spec.md` §5 in the private portfolio index, and
`tools/pagespec` there sweeps all twelve from the submodule working trees on every push.

`tests/test_site_committed.py` is **not** the local half of that carrier. It asserts byte-equality
with the generator and the import budget — that the committed page is what `sitegen` produces and
that `sitegen` never simulates — and it asserts no clause at all. So clause conformance here is
carried by the index checker alone: a token change belongs in `sitegen/theme.py`, and the only
place it is checked is one directory up.

## Code intelligence

Two indexes exist over this repo, and which one is reachable depends on where the session started:

- `.codegraph/` — the `codegraph_explore` MCP tool, or `codegraph explore "<question>"` from a
  shell. Returns the relevant symbols' verbatim source plus the call paths between them, so it
  usually answers a "how does X work" or "what calls Y" question in one call. The CLI ships as
  `codegraph.cmd`, so from Git Bash it needs the extension — bare `codegraph` resolves only
  where PATHEXT applies.
- `.code-review-graph/` — its MCP server is declared in **this repository's** `.mcp.json`, so it
  loads when Claude Code runs with this directory as the working directory, and is simply absent
  when the session started in the private portfolio index one level up. When its tools are
  missing the CLI still works: `uvx code-review-graph <command>`.

**Neither index has a hook**, so both are only as fresh as the last manual update — and a graph
that predates the work you are looking at will answer confidently about code that is gone.
`codegraph.cmd status` reports the index's age; `codegraph.cmd sync` brings it forward, and
`uvx code-review-graph update` does the same for the other. Check before trusting either on a
question about recent changes.

Grep, Glob and Read stay correct whenever the question is about text rather than structure, or
when neither index is available.
