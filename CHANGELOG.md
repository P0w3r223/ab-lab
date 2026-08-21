# Changelog

Notable changes to `ab-lab`. The reasoning behind each decision lives in
[`docs/decisions/`](docs/decisions/); this file records only what changed.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project follows [semantic versioning](https://semver.org/).

## [Unreleased] — 0.3.0.dev0

Foundations for the 0.3.0 release described in
[ADR 0006](docs/decisions/0006-three-mechanisms-of-alpha-inflation.md). No
statistical method changed; every published number is unchanged.

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
