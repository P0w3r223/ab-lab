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
  simulate.py    # draws + p-value adapters + the A/A / A/B / peeking harness
sitegen/         # renders the page, the README tables and the SVG charts
  record.py      #   the committed evidence: counts in, SimulationSummary out
  charts.py      #   one chart primitive, page mode and standalone mode
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

# Re-measure and republish. The scripts cost minutes; the build costs milliseconds.
python examples/peeking_pitfalls.py --record    # rewrite docs/data/findings.json
python examples/validation_table.py --record
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

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes_tool` or `query_graph_tool` instead of Grep
- **Understanding impact**: `get_impact_radius_tool` instead of manually tracing imports
- **Code review**: `detect_changes_tool` + `get_review_context_tool` instead of reading entire files
- **Finding relationships**: `query_graph_tool` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview_tool` + `list_communities_tool`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
| ------ | ---------- |
| `detect_changes_tool` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context_tool` | Need source snippets for review — token-efficient |
| `get_impact_radius_tool` | Understanding blast radius of a change |
| `get_affected_flows_tool` | Finding which execution paths are impacted |
| `query_graph_tool` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes_tool` | Finding functions/classes by name or keyword |
| `get_architecture_overview_tool` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. No hooks installed — run `code-review-graph update` after code changes.
2. Use `detect_changes_tool` for code review.
3. Use `get_affected_flows_tool` to understand impact.
4. Use `query_graph_tool` pattern="tests_for" to check coverage.
