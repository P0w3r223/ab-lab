# 0002 - Cohen's h for conversion-rate sample size

Date: 2026-07-21
Status: accepted
Author: Piotr Cząstkiewicz
Related to: [0001](0001-no-statsmodels-at-runtime.md)

---

## Context

Sizing an experiment on a conversion rate needs a variance, and a proportion's
variance depends on the proportion itself. Two conventions are in wide use and
they do not give the same number.

## Options

1. **Absolute-difference formula with pooled or unpooled variance.** The one in
   most online A/B calculators:
   `n = (z_alpha + z_beta)^2 * (p1(1-p1) + p2(1-p2)) / delta^2`.
   Reads naturally on the rate scale; there is no single agreed variant
   (pooled vs unpooled changes the answer by a few percent).
2. **Cohen's h**, the arcsine-transformed distance
   `h = 2*asin(sqrt(p2)) - 2*asin(sqrt(p1))`, fed into the standardised normal
   power equation. Variance-stabilising, so one power function covers both means
   and proportions.

## Decision

Cohen's h, with the public API still taking a baseline rate and an **absolute**
lift (`sample_size_for_proportion(0.05, 0.005)` = 5.0% -> 5.5%), so callers
never have to touch the transform.

Reasons, in order of weight:

1. One standardised core (`power_z`, `power_t`) serves every design function.
   The alternative needs a separate variance term per metric type, which is more
   surface for an error to hide in.
2. It matches `statsmodels.stats.power.NormalIndPower` exactly, which makes the
   validation in ADR 0001 sharp rather than approximate.
3. The two approaches differ by only a few percent of required sample at the
   rates online experiments actually run at, and that gap is smaller than the
   uncertainty in the baseline rate that goes into the calculation anyway.

## Consequences

- The docstring has to state the transform, because the required sample size is
  *not* a function of the absolute lift alone: the same 0.5pp lift is roughly
  20x more expensive to detect on a 50% baseline than on a 1% one. That surprise
  is real statistics, not an artefact - it is the variance of a proportion.
- `mde_for_proportion` has no closed form (Cohen's h is not invertible in terms
  of an absolute lift) and is solved numerically with `brentq`. Round-trip tests
  pin it to `sample_size_for_proportion` at 1e-6 relative.
- Cohen's h is **not symmetric** around a baseline, so a 1pp drop and a 1pp lift
  are different experiments: from a 2% baseline, 2 254 against 3 789 units per
  arm. `mde_for_proportion` therefore takes a `direction` argument, and the
  module docstring says so - sizing a guardrail metric as if the two were the
  same would over-buy traffic by two thirds.
- A user comparing our number against an online calculator may see a few percent
  difference. Named in the README's limitations, with the reason.
