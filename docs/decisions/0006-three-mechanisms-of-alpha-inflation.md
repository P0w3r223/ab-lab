# ADR 0006 — v0.3 is one claim measured three ways, not two new features

Date: 2026-08-21
Status: accepted
Author: Piotr Cząstkiewicz
Related to: ADR 0001, ADR 0003, ADR 0004, [0007](0007-the-page-is-generated.md)

---

## Context

The README names four of its own limitations. Two of them — non-independent units
and no multiple-comparison control — are also roadmap items 1 and 4. Shipped as
two unrelated functions they are two more procedures in a package that already
has nine, and a reader has no reason to remember either.

Shipped together they are the generalisation of the result the package already
leads with:

> A nominal 5% false positive rate is not a property of a test. It is a property
> of a test **plus** how many decisions are taken from it. Peeking takes many
> decisions in time. Clustering takes many correlated observations and counts
> them as independent decisions' worth of evidence. A metric suite takes many
> decisions in parallel.

Each mechanism independently converts 5% into 25–40%; each has a named,
implementable correction; and each is measurable in the same harness, reported
through the same summary object, and rendered in the same table.

| Mechanism | Nominal | Actual | Correction | After |
|---|---|---|---|---|
| Peeking, 20 looks | 5% | **25.3%** | mSPRT | 1.2% |
| Clustering, ICC 0.30, 10 rows/user | 5% | **32.3%** (±1.1) | cluster-robust SE | 5.4% (±0.5) |
| Metric suite, 10 independent metrics | 5% | **39.4%** (±1.6) | Holm | 5.2% (±0.7) |

Row one comes from `examples/peeking_pitfalls.py` at 4 000 runs per cell. Rows
two and three were derived arithmetically while writing this ADR and then
**measured in a throwaway pilot before any of this was accepted**, precisely so
the design could be contradicted before it was built:

| Claim | Derived | Measured | Distance |
|---|---|---|---|
| naive Welch on clustered rows, A/A | 0.3082 | 0.3060 ± 0.0084 | 0.3σ |
| the same experiment analysed per user | 0.0500 | 0.0503 ± 0.0040 | 0.1σ |
| uncorrected family-wise rate, K = 10 | 0.4013 | 0.4073 ± 0.0078 | 0.8σ |

Every row has since been **superseded by the implementation**, which is what the
headline table above now quotes. The pilot used Welch on cluster means as its
corrected arm — exact only for balanced clusters — where the shipped code uses
the CR1 sandwich, and it ran the metric suite at a different size. The shipped
numbers:

| Claim | Derived | Shipped code | Distance |
|---|---|---|---|
| naive Welch on clustered rows, A/A | 0.3082 | 0.3230 ± 0.0105 | 1.4σ |
| the same, cluster-robust | 0.0500 | 0.0540 ± 0.0051 | 0.8σ |
| uncorrected family-wise rate, K = 10 | 0.4013 | 0.3940 ± 0.0155 | 0.5σ |
| the same, Holm | ≤ 0.0500 | 0.0520 ± 0.0070 | 0.3σ |

Same conclusions, measured by the code that actually runs. The pilot rows are
kept above because a prediction is only evidence if it stays visible after the
measurement arrives.

Derivations, kept because the *agreement* is the evidence and a lone measurement
is not: design effect `1 + (m−1)ρ = 1 + 9(0.30) = 3.7`, so the naive standard
error is understated by `√3.7 = 1.9235`, so the naive test rejects whenever
`|z_true| > 1.96 / 1.9235 = 1.019`, i.e. `2Φ(−1.019) = 0.3082`; and
`1 − 0.95¹⁰ = 0.4013`.

The pilot is not a substitute for the tests specified below — it used Welch on
cluster means as the corrected arm, which is exact only for balanced clusters,
and it is the unbalanced case that motivates CR1 in the first place. What it
establishes is that the headline table is not going to change shape once the
real implementation lands, which is what would otherwise put PR 6 at risk of
rewriting PR 3.

---

## Decision

Two new modules, `cluster.py` and `multiplicity.py`, one shared tally in the
simulation harness, one summary type, one table. The load-bearing choices follow.

### D1 — One harness, or one per mechanism?

The three mechanisms differ precisely in *what varies*: time, dependence
structure, family size. A single input contract would therefore have to lie
about at least one of them.

| Option | Consequence |
|---|---|
| Force-fit the existing `Draw`/`PValueFn` | Positional cluster ids work only for **balanced** clusters — and unbalanced clusters are exactly where the naive variance is worst and where aggregating to user means stops being exact. A metric suite run K times cannot measure family-wise error at all, because that needs the *joint* outcome per experiment. |
| Pre-aggregate clustered draws to user means | Destroys the demonstration: the broken session-level analysis becomes unrepresentable. |
| Generic harness over two type parameters | Genuinely one code path, but it rewrites the code that produces every published number, and `Draw`/`PValueFn` are public. |
| **Shared tally, three entry points** ✅ | One implementation of "count rejections, compute the Monte Carlo error, build a `SimulationSummary`" — the part that must be identical for the three rows to be comparable — with an honest input type per mechanism. |

What must be identical across the three is not the input but the **verdict**.
One `SimulationSummary`, one private `_Tally`, three draw contracts.

**Non-negotiable acceptance condition.** Extracting `_Tally` must not change the
number or the order of `rng` calls. `pytest` green is not sufficient evidence;
both example scripts must produce byte-identical output to `main`. If that
cannot be achieved, the extraction is abandoned and the new runners get their
own tally. The published numbers outrank the removal of duplication.

The condition is cheap insurance rather than a schedule risk, and the difference
matters for planning. In `simulate.py` the generator is consumed only by
`draw(rng)`; neither the tally, nor `_checked_p_value`, nor any shipped
`PValueFn` touches it, so the extraction cannot move the stream unless
`estimate_fn` does. Nor does either PR 1 extraction have test blast radius: no
test and no example references any private helper in this package —
`_as_sample`, `_check_alpha`, `_check_alternative`, `_checked_p_value`,
`_even_looks`, `_tail_alpha` are all called only from within `src/`.

### D2 — What shape is clustered data?

`ClusteredSample(values, cluster_ids)`, a frozen dataclass validated once at
construction, rather than four positional arrays.

The deciding argument is this package's own ethic, not tidiness.
`welch_t_test` raises rather than return NaN from two constant arms;
`_checked_p_value` crashes rather than tally a NaN as "no rejection". A
four-argument signature whose commonest typo — swapping arguments two and
three — returns a *believable wrong number* is the same class of failure those
guards exist to prevent.

It lives in `cluster.py`, not `results.py`: `results.py`'s stated remit is what
public functions **return**, and this is what they accept.

### D3 — What does the cluster test return?

`ClusterTestResult(TestResult)` — the only inheritance in the codebase, and the
exception is documented at the class.

A plain `TestResult` has nowhere to put `design_effect`, `n_clusters_*`,
`mean_cluster_size` or `effective_n` — and the realised design effect *is* the
pedagogical payload ("your 5 000 sessions are worth 1 350 independent
observations"). Composition would cost `.test_result.p_value` at every call site
and break every adapter in `simulate.py`. Inheritance keeps `isinstance` and
`.p_value` working while the distinct type states that the guarantee differs: a
cluster-robust standard error is anti-conservative below roughly 40 clusters,
which a Welch result never is.

Every `TestResult` field is non-default, so appending required fields is legal
and unambiguous — verified, not assumed.

**The trap this creates, and the comment that defuses it.** D6 makes "append an
optional field with a default" the house move for evolving a result dataclass.
Applied to `TestResult` later, it would turn `ClusterTestResult` into a
`TypeError` at class-definition time — *non-default argument follows default
argument* — reproduced to confirm. A note at `TestResult` saying that a subclass
adds required fields, so this class may not gain a defaulted one, costs a line
and prevents a confusing failure in a file nobody was editing.

### D4 — Which cluster-robust estimator?

CR1 — the CR0 sandwich with the `G/(G−1) · (n−1)/(n−k)` finite-sample factor.

ADR 0003 set the precedent: one correction implemented properly, not three
shallow ones. CR1 is chosen because it is exactly what
`statsmodels.regression.linear_model.OLS(...).fit(cov_type="cluster")` computes
under its default `use_correction=True` — an **exact oracle already present in
the `dev` extra**, satisfying ADR 0001 without circularity, because scipy does
not implement this. CR2 / Bell-McCaffrey behaves better in small samples and has
no oracle here; the cluster bootstrap is a second estimator where the precedent
says pick one, and is noted as a v0.4 candidate that would complete a
three-way bootstrap family (independent / paired / clustered).

Three details that will otherwise cost a debugging session each:

1. The variance is computed over **linearised per-cluster contributions**, via a
   private `_cluster_robust_variance(contributions, cluster_ids)` to which
   `cluster_robust_t_test` passes the values themselves. This costs nothing now
   and makes the ratio-metric delta method (D7) an addition rather than a
   re-derivation.
2. **The oracle does not use a t-distribution at all**, and an earlier draft of
   this ADR got that wrong. Measured against statsmodels 0.14 on this repo's own
   virtualenv: `fit.df_resid` is `n − k` (798 in the probe), `df_resid_inference`
   is `G − 1` (99), and **`use_t` is `False`** — so the reported p-value comes
   from the *normal* distribution, matching `2Φ(−|t|)` to twelve decimals and the
   t-with-`df_resid` value to only four. The two-sample convention wanted here is
   a t with `G_c + G_t − 2`, which is a third answer again.

   The agreement test therefore compares the **standard error**, which is
   convention-free and matched exactly (`0.123063724989` from statsmodels against
   `0.123063724989` hand-rolled; CR0 differs in the third digit, which is what
   makes this a discriminating test rather than a tautology). Only then does it
   compare a p-value, and it does so either against the normal or with
   `use_t=True` and a matched dof — never by taking `fit.pvalues` on trust.
3. Fewer than two clusters per arm raises. Between two and forty nothing is
   printed — the library never prints — so the rule of thumb travels as an
   `assumptions` string and `n_clusters_*` is exposed for the caller to assert.

### D5 — Multiple-comparison API

Three named functions, `bonferroni` / `holm` / `benjamini_hochberg`, plus a
`CORRECTIONS` registry so the harness and the examples can parametrise over them.
A single `adjust(p_values, method=...)` dispatcher would force one docstring to
state two mutually incompatible guarantees.

Input is `Sequence[float]` plus optional `labels`, **not** `Sequence[TestResult]`.
Keeping the module ignorant of where a p-value came from is what lets it accept
Welch, the proportion test, the bootstrap and the mSPRT alike, and is what makes
the `multipletests` agreement test direct.

`labels` is not decoration. A multiplicity correction is meaningless until the
*family* is named; forcing the caller to enumerate its members is the pedagogy.
One assumption string states that guardrail metrics tested one-sided do not
belong in the same family as success metrics.

Why these three: Bonferroni is the one everyone knows and the one to beat; Holm
dominates it uniformly at the same family-wise guarantee with no independence
assumption, so it is the recommended default; Benjamini-Hochberg exists to make
the point that FDR is a *different* promise, not a better one.
Benjamini-Yekutieli is scope creep.

**Why ADR 0003's "one, not three" binds D4 and not this.** That precedent
rejects three shallow implementations of *the same guarantee* — three ways to
correct for peeking, none understood. Bonferroni and Holm control the family-wise
rate; Benjamini-Hochberg controls the false discovery rate. Shipping only one
would make the release unable to state the distinction it exists to teach, and
the partial-null simulation below is what turns that distinction from an
assertion into a measurement. Two estimators of the same cluster variance would
have no such demonstration to justify them, which is why D4 picks one.

**scipy already ships Benjamini-Hochberg** as `stats.false_discovery_control`,
present since 1.11 — this package's declared runtime floor, verified on 1.18. It
does not change the decision, because ADR 0001's position is that implementing
these methods is the point, but it is worth naming for two reasons: it is a
second non-circular oracle for that one function, and it is the single place in
this release where "we would be a thin wrapper" is a fair question. Better
answered here than discovered in review.

`MultipleComparisonResult` returns **adjusted p-values**, not a bare rejection
mask — it lets a caller re-threshold later, and it is what the oracle returns.

### D6 — Does `SimulationSummary` stretch to a metric suite?

Partly, and the gap is real. For a suite, `n_rejections` counts experiments with
at least one rejection, so `rejection_rate` is the empirical family-wise error
rate and `monte_carlo_error` remains correct. But `mean_estimate` is hardwired to
a difference of means, which on a `(n_metrics, n)` array is meaningless, and the
per-comparison error rate has nowhere to go — so the BH-versus-Holm
demonstration cannot be made at all.

**Approved** (this changes a returned dataclass, which the project rules require
asking about): two optional fields with defaults, `n_comparisons` and
`n_false_rejections`, plus an `estimate_fn` parameter on the runners. This is
exactly the move `mean_absolute_estimate_when_stopped` already made in the same
file, it is backward compatible positionally and by keyword, and it keeps one
summary type and therefore one table.

**The suite fields land in PR 5, not PR 1.** The approval is for the change, not
for its timing, and implementation showed why the timing matters: the obvious
pair of counts cannot express what Benjamini-Hochberg actually controls.
`rejection_rate` gives the family-wise rate for free, but the false discovery
rate is `E[V/R]` — a mean of per-experiment ratios — and a pooled `ΣV/ΣR` is a
different functional that merely resembles it. Which fields are right is settled
by the demonstration that consumes them, so they are defined in the PR that
writes them and can be tested the day they appear. PR 1 adds only what PR 1 can
test: `agrees_with` and `estimate_fn`.

`estimate_fn` also closes a latent bug: the runners report an estimate computed
as a difference of means regardless of what the supplied `p_value_fn` actually
estimates. Harmless while only mean-difference adapters exist — wrong the moment
`mann_whitney`, whose estimand is P(T > C), or a suite adapter is passed.

**Also approved:** `SimulationSummary.agrees_with(...)`. The rule deciding
whether an empirical rate agrees with a claim is currently implemented twice — in
`tests/test_simulate.py` and in `examples/validation_table.py` — and the page
renderer of ADR 0007 would be the third. Pure, returns a bool, prints nothing.

**The multiplier is per claim, not a parameter with one default.** The two
existing copies use *different* tolerances: 4 sigmas for "equals", 3 for "at
most". A single `agrees_with(expected, claim, sigmas)` would push a bare number
to every call site, which the no-magic-numbers rule exists to stop. The
tolerances become named module constants keyed by claim, and `sigmas` stays an
override for the caller who needs one.

### D7 — Ratio metrics are v0.4

For a ratio `R = ΣY / ΣX` under user-level randomisation, the delta-method
variance is the cluster-robust variance of the linearised contribution
`y_ij − R·x_ij`. Given D4, `ratio_metric_test` is a thin wrapper — which is
precisely the argument for *not* shipping it here.

It is not a fourth mechanism of alpha inflation. It is a wrong-variance-for-a-
ratio story, and adding it makes the release about four things and the headline
about none, while introducing a third estimand and doubling the new-method test
matrix in the same release that lands two new modules. It belongs with roadmap
item 3, the e-commerce case study, where conversion and average order value are
compared together and the ratio question arises from the narrative.

The entire foresight cost is one private signature. No speculative code.

**Second explicit non-goal: compounding.** Peeking on clustered data, sequential
testing across a metric suite. The thesis is that the three mechanisms are
independent, not that they compose. Demonstrating composition needs cluster-aware
prefix slicing — `run_with_peeking` slices by row, and a prefix would cut a user
in half. v0.4 candidate.

### D8 — Module boundaries

Two new modules, matching the existing one-module-per-question layout.
`multiplicity.py` rather than `corrections.py`, because it names the problem the
way `srm.py` and `sequential.py` do. `cluster.py` owns the concept end to end —
estimate ICC, derive the design effect, size the experiment, analyse it — which
is also why extending `analyze.py` in place was rejected: clustering is not only
an analysis concern, so it would have had to be split across two files anyway,
and `analyze.py` would have reached ~600 lines mixing four independent-unit
procedures with one non-independent one.

`cluster.py` imports **both** `Alternative` aliases, because it spans design
(`"two-sided" | "one-sided"`) and analysis (`"two-sided" | "less" | "greater"`).
That vocabulary split is already documented in the package docstring.

```
src/ab_lab/
  _validation.py  # new, private: _as_sample (verbatim in analyze.py and
                  #   sequential.py) and the alpha check (in analyze, power,
                  #   sequential x2, simulate and srm - six sites, not two)
  cluster.py      # new
  multiplicity.py # new
  results.py      # + ClusterTestResult, ClusteredSampleSizeResult,
                  #   MultipleComparisonResult, MdeResult; SimulationSummary
                  #   gains two optional fields and agrees_with
  simulate.py     # + clustered and suite draws, two runners, _Tally, estimate_fn
examples/
  three_inflations.py   # new: the one table, three mechanisms
```

### Generative model for the clustered draw

`y_ij = μ + u_i + e_ij` with `u_i ~ N(0, σ_u²)`, `e_ij ~ N(0, σ_e²)`,
`ρ = σ_u² / (σ_u² + σ_e²)`, and **`σ_u² + σ_e² = std_dev²` held fixed**.

Fixing the total variance is what makes the resulting false-positive curve
attributable to dependence alone rather than confounded with a larger metric
variance. Cluster sizes are injectable via a callable, so the unbalanced case —
where the naive analysis is worst and where aggregation to user means stops being
exact — is first-class rather than an afterthought.

---

## The two mandatory tests per method

The project rule is agreement with a reference implementation or hand arithmetic,
**and** a simulation where the truth is known, with assertions in multiples of
the Monte Carlo error.

| Method | Agreement | Simulation |
|---|---|---|
| `cluster_robust_t_test` | statsmodels OLS with `cov_type="cluster"` — standard error to ~10 decimals, p-value under matched dof (D4). Hand check: with balanced clusters the CR1 standard error equals the Welch-on-cluster-means one up to the known factor. | A/A clustered, ICC 0.3, m = 10, G = 200/arm. Two assertions in one test: naive rate exceeds α by more than 6 Monte Carlo sigmas (target ≈ 0.31), robust rate within 4 sigmas of α. |
| `intraclass_correlation` | Hand arithmetic on a three-cluster fixture (one-way ANOVA moment estimator) — more defensible out loud than a MixedLM fit. | Generated at ρ ∈ {0, 0.05, 0.2, 0.5}; mean estimate within k standard errors of truth. |
| `design_effect` | Hand arithmetic: `m = 1 → 1.0`; `ρ = 0 → 1.0`; `m = 10, ρ = 0.1 → 1.9`. | **The best test in the release**: the realised variance of the sample mean over many clustered draws matches the predicted design effect within Monte Carlo error. It validates the formula against actual variance rather than against a second formula. |
| `sample_size_for_clustered_mean` | — | Design, then simulate, then assert 80% power within 4 sigmas — mirroring the existing sample-size row in the validation table. |
| the three corrections | `statsmodels.stats.multitest.multipletests` — exact match on adjusted p-values **and** the rejection mask. Plus hand arithmetic on three p-values, because Holm's step-down monotonicity enforcement is the classic bug. | Global null, K = 10 independent: uncorrected family-wise rate within 4 sigmas of the **exact** 0.4013 — a sharp falsifiable claim, not "greater than alpha". Corrected ≤ α + 3σ. Correlated suite at ρ = 0.8: Bonferroni falls *below* α by more than 3σ, which is the conservatism being paid for. |
| Benjamini-Hochberg, specifically | — | **Partial null — five true nulls, five real effects.** Under the *complete* null FDR equals FWER and BH is indistinguishable from Holm, so the complete-null run demonstrates nothing about it. Only the partial null shows BH's empirical FDR ≤ α while its family-wise rate exceeds α. |
| the new draws | Requested ICC recovered by `intraclass_correlation`, and requested correlation matrix recovered, both within Monte Carlo error; marginal variance equals `std_dev²` regardless. | Covered by the rows above. |
| the new runners | Guards mirroring `_checked_p_value`: a NaN p-value raises, a suite p-value function returning other than K values raises, fewer than two clusters per arm raises. | — |

**Test placement**, written down because two files could each reasonably claim
these tests and nothing currently says which wins: agreement tests go in
`tests/test_cluster.py` and `tests/test_multiplicity.py`; cross-cutting
*demonstration* simulations go in `tests/test_simulate.py`, where the peeking
demonstrations already live. (An earlier draft justified this by claiming the
convention "already broke once" when `test_paired_bootstrap.py` split from
`test_analyze.py`. It did not: `git show --stat` of that commit does not touch
`test_analyze.py`. A new module's tests went into a new file. The rule stands on
its own; the precedent was invented.)

---

## Approved changes to public API and project rules

The project rules forbid changing a public signature or a returned dataclass
without asking. Four changes were asked about and approved on 2026-08-21:

1. `SimulationSummary` gains two optional fields and `agrees_with` (D6).
2. `mde_for_mean` and `mde_for_proportion` return `MdeResult` instead of a bare
   `float`.

   The reason is **not** that they are the only public functions returning a bare
   number — an earlier draft said so and it is false: `power_z`, `power_t`,
   `cohens_h`, `always_valid_p_value` and `tau_from_mde` all return bare floats
   and will continue to. The reason is that those five return a *probability* or
   a *transform*, whereas these two return an **estimate**, and an estimate
   carrying no alpha, no power, no direction and no assumptions is exactly what
   "assumptions ship with the estimate" forbids. The architecture claim in the
   README is narrowed to match, in the same commit.

   This is the release's only source-breaking change, and it breaks four sites
   PR 1 must carry with it: `tests/test_power.py:115` and `:123` (the returned
   value is fed straight back into a sample-size call), `:194-195` (two returned
   values compared, then negated — unary minus on a dataclass), and
   `README.md:99`, which prints the value and annotates it `# 0.0174`. That
   README line sits outside ADR 0007's generated fences, so no guard catches it.
3. `ClusterTestResult` inherits from `TestResult` (D3).
4. The rule "do not commit generated artefacts other than `docs/images/`" is
   amended in ADR 0007. It lands with that ADR's implementation, not before —
   a rule describing a generator that does not yet exist is worse than the rule
   it replaced.

---

## Consequences

* The README's limitation list and roadmap become false the moment `cluster.py`
  merges. Both are struck in the same release, and it is a merge-blocking
  checklist item rather than a follow-up.

  **The two lists are numbered differently and an earlier draft conflated them.**
  The limitation bullets are: 1 independent units, 2 mSPRT conservative,
  **3 no multiple-comparison correction**, 4 bootstrap p-value floor, 5 normal
  approximations, 6 sample sizes versus online calculators. The roadmap items
  are: **1 cluster-robust variance**, 2 CUPED, 3 e-commerce case study,
  **4 multiple-comparison control**. So the release strikes bullets **1 and 3**
  and roadmap items **1 and 4**. Struck by the wrong index, PR 6 would delete the
  bootstrap-floor caveat, which stays true, and leave standing the
  multiple-comparison one, which the release makes false — the precise failure
  the checklist item exists to prevent.
* The package gains a second and third headline result, which is what ADR 0007's
  page is designed around. The two ADRs are one release.
* Nine public names are added. `CHANGELOG.md` is introduced with this release,
  because ADRs record decisions and not changes.
* `simulate` remains the one module with no re-exported surface while its
  docstring lists it alongside the other four. Decided in the packaging commit:
  export the harness or drop it from the list — not left as it is once the
  surface doubles.

---

## Implementation plan

| PR | Contents |
|---|---|
| 1 | `py.typed` and package-data, single-source version, `[project.urls]` and classifiers, `_validation.py` extraction, `_Tally` extraction under the byte-identical condition, `MdeResult`, `agrees_with` and `estimate_fn` (the suite fields move to PR 5 — see D6), reproducibility caveat on the bootstrap defaults, `CHANGELOG.md` |
| 2 | ADR 0007's generator, proven on the one finding that already exists |
| 3 | `cluster.py`, `ClusterTestResult`, the clustered draws and runner, ADR 0008, both tests |
| 4 | `sample_size_for_clustered_mean` — foldable into PR 3 if the session allows |
| 5 | `multiplicity.py`, the suite draws and runner, ADR 0009, both tests |
| 6 | `examples/three_inflations.py`, re-recorded evidence, the page and README on three findings, strike **limitation bullets 1 and 3** and **roadmap items 1 and 4**, release 0.3.0 |

PR 1 changes no statistics and exists to de-risk everything after it.
