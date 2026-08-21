# ADR 0009 — Three corrections because there are two guarantees, not one

Date: 2026-08-21
Status: accepted
Author: P0w3r223
Related to: [0006](0006-three-mechanisms-of-alpha-inflation.md), ADR 0001, ADR 0003

---

## Context

The third mechanism in ADR 0006, and the fourth item on the README's roadmap. An
experiment rarely has one metric: it has a success metric, some secondaries and a
handful of guardrails, each read at alpha = 0.05. Under a global null with ten
independent metrics the chance that at least one comes back significant is
exactly `1 − 0.95¹⁰ = 40.13%` — the same arithmetic as peeking, run across
metrics instead of across time.

Measured by the implementation on 1 000 experiments of ten independent metrics:

| | Family-wise rejection rate |
|---|---|
| ten metrics read as ten separate tests | **39.4%** (±1.6) |
| the same, with Holm | **5.2%** (±0.7) |

The exact value is 40.13%; the measurement sits 0.5 Monte Carlo sigmas from it,
and the corrected arm 0.3 sigmas from alpha.

---

## Decision

### D1 — Three named functions, not one dispatcher, and why that is not a contradiction

ADR 0003 chose *one* sequential correction rather than three shallow ones, and
ADR 0008 cited that precedent to pick a single cluster-robust estimator. This ADR
ships three corrections. The precedent still holds, because it is about
implementing one thing well rather than three things badly — and CR0, CR1 and CR2
are three ways to estimate *the same quantity*, whereas these three corrections
control **two genuinely different error rates**:

| Procedure | Bounds | Reading |
|---|---|---|
| Bonferroni | family-wise error rate | probability of **any** false positive in the family |
| Holm | family-wise error rate | same guarantee, uniformly more rejections |
| Benjamini-Hochberg | false discovery rate | expected **share** of rejections that are false |

Shipping only one would make the release unable to state the distinction it
exists to teach. A single `adjust(p_values, method=...)` dispatcher was rejected
for the same reason: one docstring would have to state two incompatible promises
at once. The `CORRECTIONS` registry gives the harness and the examples the
parametrisation a dispatcher would have offered, without that cost.

Benjamini-Yekutieli is scope creep. Bonferroni is kept despite being uniformly
dominated by Holm, because it is the procedure everyone has heard of and the one
a reader will be comparing against.

### D2 — Bare p-values in, not result objects

`holm(p_values, alpha, labels)` rather than `holm(results)`. Keeping the module
ignorant of where a p-value came from is what lets it accept Welch, the
two-proportion z-test, the bootstrap, the mSPRT and the cluster-robust test
alike, and it is what makes the `multipletests` agreement test direct.

`labels` is optional but is not decoration. A multiplicity correction is
meaningless until the family is *named*, and naming it is a decision rather than
a computation. Forcing the caller to enumerate the members is the pedagogy; one
assumption string states that guardrail metrics tested one-sided are a different
question and do not belong in the same family.

The result carries **adjusted p-values**, not a bare mask, so a family can be
re-thresholded without re-running and so the oracle comparison is exact.

### D3 — `<` rather than `<=`, and how the oracle test copes

Reference implementations reject when the adjusted p-value is **at most** alpha.
Everything else in this package uses `<`, including `TestResult.is_significant`
and every simulation runner. Internal consistency wins: the two differ only on an
exact tie, which continuous p-values do not produce.

The agreement test therefore compares **adjusted p-values** — exactly defined, to
1e-12 — and compares rejection masks separately on random uniform draws, where a
tie has probability zero. Stated here because a future reader finding `<` next to
a statsmodels comparison will otherwise assume it is a bug.

### D4 — The two implementation bugs, each with its own fixture

Both procedures are step procedures whose monotonicity enforcement is easy to
omit, and omitting it leaves a result that still looks plausible.

* **Holm** needs a running *maximum* over the step-down sequence. Without it,
  p-values `0.001, 0.012, 0.030, 0.040, 0.200` scaled by `5, 4, 3, 2, 1` give
  `0.005, 0.048, 0.090, 0.080, 0.200` — and `0.080 < 0.090` means a weaker
  result is reported as more significant than a stronger one.
* **Benjamini-Hochberg** needs a running *minimum* taken from the largest
  p-value downwards. The direction is the whole procedure; reversed, it controls
  nothing and reports numbers in a believable range.

### D5 — What the summary had to gain, and a mistake worth recording

ADR 0006 approved two optional fields on `SimulationSummary` and deferred them to
this PR, on the grounds that the right field set is settled by the demonstration
that consumes it. That deferral paid, and then the first attempt still got it
wrong.

The plan was `n_comparisons` plus a false-discovery accumulator. Both are needed.
What the plan missed is that under a **partial** null — which is the only setting
where Benjamini-Hochberg differs from Holm — `rejection_rate` is not a false
positive rate at all. It counts experiments in which *anything* was rejected, and
with five real effects present at 85% power that is very nearly every experiment
whatever the correction does. The first version of the test asserted
`rejection_rate ≤ alpha` for Holm and failed at 1000/1000, which is the correct
answer to a meaningless question.

So the summary gains three optional fields, not two:

| Field | Why it cannot be derived from the others |
|---|---|
| `n_comparisons` | family size, not recoverable from counts |
| `n_family_wise_errors` | experiments containing at least one **false** rejection — the quantity Bonferroni and Holm bound |
| `mean_false_discovery_proportion` | `E[V/R]`, a mean of per-experiment ratios; a pooled `ΣV/ΣR` is a different functional that merely resembles it |

All three default to `None`, so existing construction is unaffected, and the
runner sets them only when told which metrics are null — because nothing else can
know.

---

## The two mandatory tests

| Method | Agreement | Simulation |
|---|---|---|
| all three | `statsmodels.stats.multitest.multipletests` on adjusted p-values to 1e-12, across three random families of twelve, plus the rejection mask on the same draws | — |
| `holm` | Hand arithmetic on the five-value fixture above, including the running maximum. Plus: Holm rejects a superset of Bonferroni's rejections on 40 random p-values, and the family's *order* changes no verdict. | Global null, K = 10 independent: the uncorrected family-wise rate lands within four Monte Carlo sigmas of the exact 0.4013 — a falsifiable claim, not "greater than alpha". Holm lands at or below alpha. |
| `benjamini_hochberg` | Hand arithmetic including the running minimum | **Partial null, five real effects**: its false discovery proportion stays at or below alpha while its *family-wise* rate exceeds alpha by more than four sigmas, and Holm's does not. Under a complete null the two rates coincide and this comparison is unmeasurable, which is why the fixture is partial. |
| `bonferroni` | as above | Marked `slow` and run weekly: it holds under independence, and under correlation of 0.8 it spends measurably *less* than alpha — the power it gives away for validity under any dependence. |

The correlated draw is built from one shared factor plus per-metric noise, which
gives exactly exchangeable correlation for the cost of one extra normal draw
rather than a Cholesky factorisation per experiment.

---

## Consequences

* The README's third limitation bullet and fourth roadmap item become false and
  are struck in the same commit.
* `SimulationSummary` gains three optional fields and one property that raises
  rather than return a misleading number when the truth was never supplied.
* The suite demonstrations cost about ten Welch tests per experiment, so they are
  sized for the sharpest claim in the file and no larger, and the second
  Bonferroni confirmation is deferred to the weekly job rather than run on every
  pull request. That trade is stated rather than made silently.
* **Revisit when** a metric suite needs per-comparison power, or when the family
  spans procedures with different alternatives — the assumption string currently
  refuses that combination rather than handling it.
