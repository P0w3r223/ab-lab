# ADR 0008 — CR1 sandwich for repeated measurements, and a sample it cannot be handed wrong

Date: 2026-08-21
Status: accepted
Author: P0w3r223
Related to: [0006](0006-three-mechanisms-of-alpha-inflation.md), ADR 0001, ADR 0003

---

## Context

The package's own README named non-independence as its first limitation, and
ADR 0006 made it the second of three mechanisms by which a nominal 5% stops
being 5%. This ADR records how it is implemented.

The failure it corrects is ordinary and expensive. A metrics table has one row
per session; the analysis has one row per observation; nobody asks how many rows
came from the same user. Those rows are correlated, so the variance of the
difference is understated by the *design effect* `1 + (m̄ − 1)ρ`, and the test
rejects far more often than it promises.

Measured by the implementation, on A/A data with ten rows per user at ρ = 0.30,
200 clusters per arm, 2 000 runs:

| Analysis of identical draws | Rejection rate |
|---|---|
| every row treated as its own observation | **32.3%** (±1.1) |
| cluster-robust standard error | **5.4%** (±0.5) |

The derivation in ADR 0006 says 30.8%; the measurement sits 1.4 Monte Carlo
sigmas from it, and the corrected arm sits 0.8 sigmas from alpha. This lands
alongside 25.3% from peeking — a number reached by a completely different route,
which is what makes ADR 0006's table an argument rather than a list.

---

## Decision

### D1 — CR1, and only CR1

ADR 0003 set the precedent for the sequential module: one correction implemented
properly, not three shallow ones.

| Option | Why not |
|---|---|
| CR0 — the plain sandwich | Simplest to derive out loud, materially anti-conservative below ~50 clusters, and **no oracle**: nothing in the dev extra computes it. |
| **CR1 — CR0 × `G/(G−1) · (n−1)/(n−k)`** ✅ | Exactly what `statsmodels` computes under `cov_type="cluster"` with its defaults, so an existing test-only dependency becomes an *exact* oracle. Industry standard. |
| CR2 / Bell-McCaffrey | Better in small samples; no oracle in the dev extra; needs a leverage computation; over-engineered for a two-group mean difference. |
| Cluster bootstrap | Symmetric with ADR 0005's "a different resampling unit deserves its own function", and slow inside a 2 000-run simulation. Recorded as a v0.4 candidate: it would complete a three-way bootstrap family — independent, paired, clustered. |

Why the "one, not three" precedent binds here but not in `multiplicity.py`:
CR0, CR1 and CR2 estimate *the same quantity* with different finite-sample
corrections, so shipping all three would be three ways to say one thing. The
three multiplicity corrections control two genuinely different error rates, and
the release exists partly to show that they differ.

### D2 — The sandwich is implemented by its reduced form, not by fitting an OLS

For a two-group design with clusters nested inside arms — which is what
user-level randomisation gives — the sandwich variance of the treatment
coefficient collapses. With `u_i = y_i − ȳ_arm` and `S_g = Σ_{i∈g} u_i`:

```
V[β₁] = c · ( Σ_{g ∈ control} S_g² / n_c²  +  Σ_{g ∈ treatment} S_g² / n_t² )
```

Derived by carrying `(X'X)⁻¹ · meat · (X'X)⁻¹` through with dummy coding, then
**checked numerically against `statsmodels` rather than trusted**: agreement to
1e-12 relative on balanced clusters, unbalanced clusters, and arms of very
different sizes. The alternative — building a design matrix and running an OLS
inside the library — would have meant implementing linear regression to compute
a difference of two means.

The private helper takes **linearised contributions**, not values. For a mean
they are the residuals; for a ratio metric `R = ΣY/ΣX` they are `y − R·x`, so
the v0.4 ratio-metric test is a wrapper rather than a re-derivation
(ADR 0006 D7). This costs nothing today.

### D3 — Degrees of freedom, and why the oracle test compares standard errors

Three implementations, three answers, and an earlier draft of ADR 0006 got this
wrong in a way worth recording:

| Quantity | Value in the probe |
|---|---|
| `statsmodels` `fit.df_resid` | `n − k` = 798 |
| `statsmodels` `df_resid_inference` | `G − 1` = 99 |
| `statsmodels` `use_t` | **`False`** — the p-value comes from the *normal* |
| this package | `G_c + G_t − 2` = 98, with a t distribution |

So `statsmodels` does not use a t distribution at all by default, and its
p-value matches `2Φ(−|t|)` to twelve decimals. The agreement test therefore
compares the **standard error**, which is convention-free and matched exactly,
and never takes `fit.pvalues` on trust. Comparing p-values first would have
produced a mystifying failure in the fourth decimal.

The two-sample convention `G_c + G_t − 2` is chosen because it is the one the
rest of this package uses, and because it is the honest statement of what the
test knows: a study of 5 000 sessions from 500 users has 498 degrees of freedom
per arm, not 4 998.

### D4 — `ClusteredSample`, not four parallel arrays

`cluster_robust_t_test(control, treatment)` where each argument carries its own
unit labels, rather than `f(control, control_ids, treatment, treatment_ids)`.

The deciding argument is this package's own ethic rather than tidiness. The
four-argument form has the property that its commonest typo — swapping
arguments two and three — returns a **believable wrong number**. `welch_t_test`
already raises rather than return a NaN from two constant arms, and the
simulation harness crashes rather than tally a NaN as "not significant"; a
signature whose misuse is silent would undo exactly that care.

Construction validates once: matching shapes, finite values, and integer labels.
Float labels are refused rather than coerced, because `1.0` and `1.0000001`
becoming two users is the kind of bug that shows up as a suspiciously small
design effect and is never traced back.

Overlapping ids across arms are refused too, with a message naming the offending
id. A unit in both arms is a paired design, and `paired_bootstrap` is the tool.

### D5 — What is deliberately not decided for the caller

The sandwich estimator is asymptotic in the number of clusters and
anti-conservative below roughly forty. The library **does not warn**: it never
prints. Instead `n_clusters` is exposed on the result and the rule of thumb
travels in `assumptions`, so a caller can assert on it and a reader of the
result can see it. Fewer than two clusters in an arm *does* raise, because there
the between-cluster variance is not merely unreliable but undefined.

`intraclass_correlation` can return a negative estimate and it is returned
unclipped. It is a real sampling outcome, and at `m = 2` the estimator's floor is
exactly −1; clipping it to zero would hide the commonest cause, which is a
cluster defined by the wrong column.

### D6 — Sizing lives here too, and does not inherit

Clustering is not only an analysis concern: an experiment planned as though rows
were independent is under-powered before it launches, and no careful analysis
afterwards recovers that. So `sample_size_for_clustered_mean` lives in
`cluster.py`, which owns the concept end to end — estimate the correlation,
derive the design effect, size the experiment, analyse it.

`ClusteredSampleSizeResult` is **not** a subclass of `SampleSizeResult`, while
`ClusterTestResult` **is** a subclass of `TestResult`. That looks inconsistent
and is not. The inheritance buys something concrete in the second case: every
adapter in `ab_lab.simulate` consumes a `TestResult`, and subclassing keeps them
all working unchanged. Nothing consumes a `SampleSizeResult` polymorphically, so
subclassing there would add a hierarchy for the appearance of symmetry and gain
nothing.

---

## The two mandatory tests

| Method | Agreement | Simulation |
|---|---|---|
| `cluster_robust_t_test` | `statsmodels` OLS with `cov_type="cluster"`, standard errors to 1e-12 relative, across balanced clusters, unbalanced clusters and unequal arms | A/A at ρ = 0.30, m = 10, 200 clusters per arm: the naive analysis rejects more than six Monte Carlo sigmas above alpha and lands within four sigmas of the derived 0.308; the robust one lands within four sigmas of alpha. Also checked at one row per unit, where the correction must cost nothing, and with Poisson-distributed cluster sizes. |
| `intraclass_correlation` | Hand arithmetic on a six-observation fixture, chosen so every mean square is checkable by eye: 30/34. Plus the algebraic floor at identical cluster means. | Requested ρ ∈ {0, 0.05, 0.2, 0.5} recovered from the draw within Monte Carlo error. |
| `design_effect` | Hand arithmetic: `m = 1 → 1.0`, `ρ = 0 → 1.0`, `m = 10, ρ = 0.1 → 1.9`, `ρ = 1 → m`. Plus the size-weighted mean, which for sizes {1, 10} is 101/11 and not 5.5. | **Against realised variance rather than a second formula**: the variance of a sample mean over 4 000 clustered draws matches the predicted `DE/n` within four times the sampling error of a variance estimate. |
| `sample_size_for_clustered_mean` | — | Size it, simulate it, and count: 80% power within four sigmas. And the counterpart — the same experiment sized as though rows were independent comes out measurably under-powered, which is the error this function exists to prevent. |

The generative model states `y_ij = μ + u_i + e_ij` with the **total variance
held at `std_dev²` whatever the intraclass correlation**. Fixing it is what makes
the false-positive curve attributable to dependence alone, rather than
confounded with a metric that simply got noisier.

---

## Consequences

* The README's first limitation bullet and first roadmap item become false and
  are struck in the release commit.
* Nine public names are added, one of them the package's only use of dataclass
  inheritance, documented at the class.
* `simulate.py` gains a second draw contract. The tally, the summary type and
  therefore the published table stay shared, which is the whole of ADR 0006 D1.
* **Revisit when** the cluster bootstrap or ratio metrics land: the private
  variance helper already takes the shape they need, and neither should require
  touching the estimator itself.
