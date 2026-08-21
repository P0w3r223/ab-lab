# Changelog

Notable changes to `ab-lab`. The reasoning behind each decision lives in
[`docs/decisions/`](docs/decisions/); this file records only what changed.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project follows [semantic versioning](https://semver.org/).

## [0.3.0] — 2026-08-21

**A 5% test is only 5% if you look once, count each user once, and test one
metric.** Three mechanisms, measured on A/A data where there is no effect to
find, each with the correction that puts the rate back:

| At its worst | Nominal | Measured | Correction | After |
|---|---|---|---|---|
| 20 times the results are checked | 5% | **25.3%** | mSPRT | 1.2% |
| 20 rows per user | 5% | **44.0%** | cluster-robust SE | 5.6% |
| 20 metrics measured at once | 5% | **65.7%** | Holm | 4.9% |

Three independent routes to one conclusion. The derivations for the second and
third were written into [ADR 0006](docs/decisions/0006-three-mechanisms-of-alpha-inflation.md)
*before their code existed*, so the simulation had something falsifiable to
contradict; at the ten-unit setting they landed 1.4 and 0.5 Monte Carlo sigmas
from the measurements.

### Added — multiplicity control

[ADR 0009](docs/decisions/0009-multiplicity-control.md). The third mechanism,
and the last one this release needs.

- `ab_lab.multiplicity`: `bonferroni`, `holm`, `benjamini_hochberg` and a
  `CORRECTIONS` registry, returning a `MultipleComparisonResult` that carries
  **adjusted p-values** and names which error rate it controls. Three functions
  rather than one dispatcher because there are two guarantees, not one: a single
  docstring would have to state both at once.
- Measured on ten independent metrics: read as ten separate tests, a true null
  rejects **39.4%** (±1.6) of the time; with Holm, **5.2%** (±0.7). The exact
  value is `1 - 0.95¹⁰ = 40.13%`, so the test asserts that number rather than
  "greater than alpha" — a directional assertion would pass for a badly broken
  harness.
- Adjusted p-values match `statsmodels.stats.multitest.multipletests` to 1e-12
  for all three procedures. Holm's running maximum and Benjamini-Hochberg's
  running minimum each have a hand-arithmetic fixture, because omitting either
  leaves a result that still looks plausible.
- `SimulationSummary` gained `n_comparisons`, `n_family_wise_errors` and
  `mean_false_discovery_proportion`, plus a `family_wise_error_rate` property.
  Three fields, not the two the design predicted: under a *partial* null —
  the only setting where Benjamini-Hochberg differs from Holm — `rejection_rate`
  counts experiments where anything was rejected, including correctly, so it is
  not a false positive rate at all.

### Added — cluster-robust variance

[ADR 0008](docs/decisions/0008-cluster-robust-variance.md). The second of the
three mechanisms in ADR 0006, and the limitation the README had listed first.

- `ab_lab.cluster`: `ClusteredSample`, `cluster_robust_t_test`,
  `intraclass_correlation`, `design_effect`, `sample_size_for_clustered_mean`.
  The test returns a `ClusterTestResult`, which carries the **realised** design
  effect and the effective sample size — the numbers that make the finding
  concrete ("your 5 000 sessions are worth 1 350 independent observations").
- Measured on A/A data with ten rows per user at an intraclass correlation of
  0.30: the row-level analysis rejects **32.3%** (±1.1) of the time against a
  nominal 5%, and the same draws with a cluster-robust standard error reject
  **5.4%** (±0.5). The derivation predicted 30.8% before the code existed.
- The standard error matches `statsmodels`' `cov_type="cluster"` to 1e-12
  relative on balanced clusters, unbalanced clusters and unequal arms.
  `design_effect` is checked against the *realised* variance of a sample mean
  over 4 000 draws rather than against a second formula.
- `ab_lab.simulate` gained `clustered_normal_draw`, `poisson_cluster_size`,
  `naive_welch_p_value`, `cluster_robust_p_value` and
  `run_clustered_experiments`. The draw contract differs because a clustered
  experiment cannot be two flat arrays without either assuming balanced clusters
  or pre-aggregating — and pre-aggregating removes the analysis being caught.
  The tally and the summary type stay shared, so all three mechanisms land in
  one table.

### Added — the page is generated

[ADR 0007](docs/decisions/0007-the-page-is-generated.md). `docs/index.html` was
hand-written, in a package whose argument is that every table it publishes is
regenerable, and one validation row already existed in three different formats
across the script, the README and the page.

- `sitegen/` renders the page, the README's fenced tables and the charts from
  `docs/data/findings.json`, which `examples/*.py --record` writes. The record
  stores **counts, never rates**: `SimulationSummary` derives every percentage,
  so no surface can disagree with the test suite about what a rate is.
- The chart is inline SVG that follows `prefers-color-scheme`, replacing a PNG
  that could not. `docs/images/peeking.png` is deleted and the README references
  two generated SVGs through `<picture>`.
- Four guards on every pull request: the committed bytes match the generator;
  the recorded design matches the scripts' constants; the generator provably
  never simulates (a subprocess asserts `ab_lab.simulate` never loaded); and two
  cells are replayed at a tenth of size. A weekly workflow re-measures
  everything at full size and compares rates within Monte Carlo error.
- The page now carries a `meta description`, Open Graph and Twitter tags, an
  inline icon, dark mode, and **zero external-origin resources** — it previously
  fetched Google Fonts. The README links the live page from its first screen,
  which it had never done.

### Changed

- `examples/*.py` gained `--record` and lost `--plot`; they print through the
  same renderer as the page rather than formatting tables themselves, which
  amends [ADR 0004](docs/decisions/0004-scripts-instead-of-notebooks.md) on
  where the formatter lives. Every previously documented command still works.
- `matplotlib` left the `dev` extra; the `--plot` flag was its only user.

### Changed — breaking

- `mde_for_mean` and `mde_for_proportion` return an `MdeResult` instead of a
  bare `float`. They estimate an effect, and an effect without the alpha, power
  and direction it was solved at is exactly what this package argues against.
  The magnitude is `.mde`. There is deliberately **no** implicit conversion to
  `float`: a caller who wants the bare number names it, and a caller sizing a
  guardrail in the other direction has to negate `.mde` explicitly rather than
  negate a result object by accident.

### Added

- `py.typed`, so a consumer's type checker can finally see the annotations that
  were always there (PEP 561).
- `SimulationSummary.agrees_with(expected, claim)` — the one implementation of
  "does this empirical rate support the claim", previously written out in the
  test suite and again in `examples/validation_table.py`. `claim="equals"` for a
  fixed-horizon procedure, `claim="at most"` for an anytime-valid one, with the
  tolerance for each as a named constant rather than a number at the call site.
- `estimate_fn` on `run_experiments` and `run_with_peeking`, defaulting to the
  new `mean_difference`. Both runners previously reported a difference of means
  regardless of what the supplied `p_value_fn` actually estimated — harmless
  while every adapter estimated one, wrong for anything else.
- `[project.urls]`, trove classifiers, and `monte-carlo` in the keywords.

### Changed

- The package version has one source: `ab_lab.__version__`, which
  `pyproject.toml` now reads dynamically instead of restating.
- A bootstrap called without `rng=` gains an assumption saying so. The default
  is convenient and returns a different p-value on every run; that consequence
  now travels with the result like every other caveat in this package.
- Internal: argument validation moved to a private `_validation` module. The
  `alpha must be in (0, 1)` check existed in six places and the sample validator
  in two. Messages are unchanged.
- Internal: the three simulation runners share one `_Tally` rather than each
  re-implementing the count-and-summarise loop. Verified not to move the random
  stream — both example scripts produce byte-identical output.

## [0.2.0] — 2026-07-22

- `paired_bootstrap`: a bootstrap that resamples units rather than arms, for a
  difference measured on the same units
  ([ADR 0005](docs/decisions/0005-paired-bootstrap.md)).

## [0.1.0] — 2026-07-21

- First release: power and sample size, Welch, two-proportion z, Mann-Whitney,
  percentile bootstrap, sample ratio mismatch, mSPRT sequential testing, and the
  simulation harness that validates all of them.
