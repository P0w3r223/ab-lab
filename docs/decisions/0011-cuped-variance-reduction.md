# ADR 0011 — CUPED, and the failure that looks like a null result

Date: 2026-08-21
Status: accepted
Author: P0w3r223
Related to: [0006](0006-three-mechanisms-of-alpha-inflation.md), [0010](0010-ratio-metrics-delta-method.md)

---

## Context

Every other method in this package is about a test that lies. This one is about a
test that is honest and expensive.

If a metric was already measurable *before* the experiment — last month's revenue
for the same user, last week's session count — then most of what makes it noisy
is a property of the user rather than of the treatment, and it can be subtracted
out before anything is compared. With `theta = Cov(Y, X) / Var(X)`, analysing
`Y − theta·(X − mean(X))` leaves the expected difference between arms unchanged,
because randomisation made `X` balanced, and multiplies the variance by
`1 − correlation²`.

Measured, on 4 000 units per arm:

| Correlation | Variance removed | Promise (ρ²) | Worth this much sample |
|---|---|---|---|
| 0.3 | 0.0917 | 0.09 | ×1.10 |
| 0.5 | 0.2527 | 0.25 | ×1.34 |
| 0.7 | 0.4928 | 0.49 | **×1.97** |
| 0.9 | 0.8113 | 0.81 | ×5.30 |

A covariate correlated 0.7 with the metric is worth doubling the traffic. That is
why this is worth more to a real experimentation programme than any correction in
the 0.3.0 release: those stop you being wrong, this one lets you afford to be
right.

At the same sample size and a true effect of 0.10, power goes from 51.5% to
**91.0%** at a correlation of 0.8 — and the estimate does not move (+0.1003
against a true 0.10).

## Decision

### D1 — Theta is estimated on the pooled arms

A per-arm theta would be fitted partly to the treatment effect, which is the
thing being measured. Pooling is not a convenience; it is what keeps the estimate
unbiased.

### D2 — The realised variance reduction is reported, not the predicted one

`variance_reduction` is measured on the data in hand rather than computed as the
squared correlation. Squaring the correlation gives the expectation; the result
should say what happened. `effective_sample_multiplier` turns that into the
number a person deciding whether to build a covariate pipeline actually wants:
"this is worth twice the traffic".

`unadjusted_estimate` travels alongside, because the two should agree — CUPED
changes the precision, not the answer — and a gap between them is evidence that
randomisation did not balance the covariate, which is a sample-ratio question
rather than a CUPED one.

### D3 — The trap, and the version of it that is wrong

The covariate must be measured before the treatment could influence it. A
covariate the experiment touched sits on the causal path, so adjusting for it
subtracts part of the effect along with the noise.

**An earlier draft of this ADR and of the module docstring described the
consequence as "the estimate is biased toward zero while the interval gets
tighter, so everything looks like it is working". The measurement says
otherwise, and the difference matters.** On a true effect of 0.10 with a
correlation of 0.7:

| Effect leaking into the covariate | Estimate | Bias | Rejection rate |
|---|---|---|---|
| none | +0.0998 | −0.2% | 80.4% |
| half | +0.0648 | **−35.2%** | 44.0% |
| all of it | +0.0298 | **−70.2%** | 12.7% |

The rejection rate *collapses*. The failure does not look like success — it looks
like a **null result**: a smaller effect, reported with a respectable-looking
interval, from an experiment that has never been more precise. That is arguably
the more survivable mistake, because nobody investigates an experiment that found
nothing, and an organisation that concludes "no effect" when there was one pays
for it silently.

Saying the tempting thing would have been a claim this package's own harness
contradicts. `treatment_leaks_into_covariate` is a parameter on the draw so the
mistake can be measured rather than described, and both the loose bias bound and
the collapsed rejection rate are asserted as tests.

### D4 — No oracle, and none needed

Nothing in the dev extra implements CUPED, and ADR 0001's usual arrangement does
not apply. What is available is better than a library check because the quantity
is elementary: `theta` is a covariance over a variance and is compared against
numpy directly, and the adjusted difference is
`(Ȳ_t − Ȳ_c) − theta·(X̄_t − X̄_c)` by construction and is compared the same way.

The claim that needs simulating is not the arithmetic but the *promise*: that the
variance falls by `1 − ρ²`. That is checked against realised variance and against
realised power, never against a restatement of the formula.

---

## The two mandatory tests

| Method | Agreement | Simulation |
|---|---|---|
| `cuped_theta` | numpy covariance over variance, to 1e-12 | — |
| `cuped_t_test` | the adjusted difference equals the hand formula to 1e-12 | Variance reduction matches ρ² within 0.02 at ρ ∈ {0.3, 0.5, 0.7, 0.9}; A/A holds at alpha; a useless covariate is a no-op rather than a penalty; a correlated one buys power more than ten combined sigmas' worth without moving the estimate |
| the trap | — | Half the effect leaking costs at least 0.02 of a 0.10 effect, all of it at least 0.05, and the rejection rate falls from above 0.7 to below 0.3 |

The bias bounds are deliberately loose. The point is the direction and the order
of magnitude; a tight bound would turn this into a regression test for a number
nobody should rely on.

---

## Consequences

* The README's roadmap loses its CUPED entry, leaving only the case study.
* `CupedResult` is the third subclass of `TestResult`, on the same grounds as the
  first two.
* `simulate` gains a fourth draw contract and its only runner that consumes a
  whole result rather than a p-value — necessary, because a rejection rate cannot
  tell a shrunken estimate apart from an experiment that found nothing.
* **Revisit when** the covariate is itself clustered or a ratio. CUPED composes
  with both in principle; nothing here has measured that, and the module says so
  rather than implying the composition is free.
