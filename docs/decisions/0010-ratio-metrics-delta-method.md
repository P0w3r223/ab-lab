# ADR 0010 — Ratio metrics by the delta method, and the demonstration that took three tries

Date: 2026-08-21
Status: accepted
Author: P0w3r223
Related to: [0006](0006-three-mechanisms-of-alpha-inflation.md) D7, [0008](0008-cluster-robust-variance.md)

---

## Context

Most metrics people actually watch are ratios of two totals rather than means of
one column: clicks over impressions, orders over sessions, revenue over visits.
The randomisation unit is the user, but the denominator counts something else, so
the number of things being divided is itself random.

That is a second, independent problem from the one ADR 0008 solved. Clustering
understates the variance because correlated rows are counted as independent
evidence. A ratio goes wrong because `Var(Y/X)` is not `Var(Y)/X²` when the
denominator varies, and the error can go in either direction depending on the
correlation between numerator and denominator.

ADR 0006 D7 deferred this to 0.4 on purpose — it is not a fourth mechanism of
alpha inflation, and folding it into that release would have made the headline
about nothing — while arranging for it in advance: the cluster-robust variance
helper was written to take **linearised contributions** rather than values,
against exactly this case.

## Decision

### D1 — The delta method, as a wrapper and not a second sandwich

Writing `R = ΣY / ΣX`, the linearised contribution of one observation is
`y − R·x`, and `Var(R) = Var(mean(y − R·x)) / mean(x)²`. The numerator is the
cluster-robust variance of a mean, so `ratio.py` is nine lines of arithmetic
around `cluster.cluster_robust_variance` rather than a second implementation of
a sandwich. The foresight in ADR 0006 D7 cost one parameter name and it paid
exactly as described.

The helper lost its leading underscore because it now has a second module as a
caller. It is still not re-exported at the package top level: a caller who wants
a variance wants a test.

### D2 — `RatioSample`, for the reason `ClusteredSample` exists

Three parallel arrays — numerator, denominator, unit label — bound into one
argument and validated once. The alternative passes six arrays to a two-arm test,
whose commonest typo is numerator and denominator swapped, and which then returns
a number rather than an error.

### D3 — The estimand is the ratio of totals, and it is written on the result

The ratio of totals weights a unit by its denominator; the mean of per-unit
ratios does not. Those are different numbers and the gap is not subtle: one user
clicking 1 of 1 and another clicking 10 of 100 give a ratio of totals of 10.9%
and a mean of per-user rates of 55%. Both are defensible; only one is the
click-through rate.

`RatioTestResult` therefore carries both arms' ratios and a `relative_effect`
alongside the absolute difference, because 0.004 means one thing on a 2% baseline
and another on a 40% one — and the relative figure is reported *beside* the
absolute one rather than instead of it, since a big relative lift on a tiny
baseline is how small effects get oversold.

### D4 — The oracle is the reduction, not another library

Nothing in the dev extra implements a delta-method ratio test, so ADR 0001's
"agree with a reference implementation" is not available in its usual form. What
is available is stronger than a third-party check: **setting every denominator to
one turns a ratio of totals into a mean**, and the result must then be
`cluster_robust_t_test` — which is itself verified against `statsmodels` to 1e-12
in ADR 0008.

Measured: the estimate, the p-value and the degrees of freedom are bit-identical,
and the test statistic differs by one unit in the last place, because the ratio
path divides by a mean denominator of exactly 1.0 and that is one extra
floating-point operation. The test asserts equality on the first three and a
1e-12 relative tolerance on the fourth, and says why.

A special case that fails to reduce is the clearest possible sign that the
general case is wrong somewhere.

---

## The demonstration was wrong twice before it was right

Worth recording, because the two wrong versions were both plausible and both
would have shipped a claim this package could not support.

**First attempt.** A/A on a click-through metric, comparing the delta method
against a t-test on per-user rates, expecting the naive one to inflate alpha.
Measured: 4.85% against 4.85% — *identical*. The draw gave every user the same
number of impressions, and with a constant denominator the ratio of totals **is**
the mean of per-unit ratios. The demonstration had removed the very thing it was
demonstrating.

**Second attempt.** Unequal impressions per user, drawn from a Poisson. Measured:
validity 5.20% against 4.95%, power 43.2% against 43.3% — still no difference,
and the two estimands agreed to the fourth decimal (0.19458 against 0.19485).
That is not a bug either. When a unit's rate is independent of how much it is
exposed, the two estimands coincide in expectation and weighting by exposure buys
nothing.

**Third attempt, and the finding.** The two analyses diverge only when **exposure
predicts the rate** — when engaged users respond differently from occasional
ones, which is the realistic case and the reason anyone argues about this.

| Scenario | Delta method | t-test on per-unit rates |
|---|---|---|
| A/A — validity | 5.05% (±0.49) | 4.95% (±0.49) |
| A/B, lift equal for everyone | 51.5% (±1.1) | 49.8% (±1.1) |
| A/B, lift proportional to exposure | **58.4%** (±1.1) | 49.7% (±1.1) |

With the lift landing on engaged users the business metric moves +0.04405 while
the average of per-user rates moves +0.03950 — an 11% difference in the reported
effect — and the analysis targeting the business metric finds it 5.5 combined
sigmas more often.

So the honest claim is *not* "the naive analysis is invalid". It is valid. It
answers a different question, and whether that matters is an empirical property
of the population rather than a fact about the method. Both halves are asserted
as tests, including the one that says the two agree, because asserting a
difference that is not there is how a package ends up publishing something false.

`lift_scales_with_exposure` is a parameter on the draw rather than a fixed
property of it, precisely so that both halves are reachable.

---

## The two mandatory tests

| Method | Agreement | Simulation |
|---|---|---|
| `ratio_metric_test` | Exact reduction to `cluster_robust_t_test` at a denominator of one (estimate, p-value and dof bit-identical; statistic to 1e-12) | A/A with unequal exposure holds at alpha within four Monte Carlo sigmas |
| the estimand | Hand arithmetic on two users: 11/101 against 55% | — |
| `contributions` | Sum to exactly zero by construction, since `R` is defined as the value that makes them | — |
| the choice of estimand | — | Both halves: equal power when exposure does not predict the rate, and materially more power for the delta method when it does |

---

## Consequences

* The README's roadmap loses its ratio-metric entry.
* `simulate` gains a third draw contract. The tally and the summary type stay
  shared, which is still ADR 0006 D1.
* `RatioTestResult` is the second subclass of `TestResult`, on the same grounds
  as the first: the simulation adapters consume a `TestResult` and keep working.
* **Revisit when** the denominator can be zero for a unit that still belongs in
  the analysis — a user with no impressions has no rate, and the current code
  requires only that the *total* denominator be positive, which is the weaker
  condition. The delta method's first-order approximation also degrades when the
  denominator is small, and that is stated in the result's assumptions rather
  than guarded, because where the boundary sits depends on the metric.
