# 0001 - statsmodels is a test oracle, not a dependency

Date: 2026-07-21
Status: accepted
Author: P0w3r223

---

## Context

Every method in this package already exists in `statsmodels`: power analysis,
the two-proportion z-test, Welch's t-test. Depending on it would make the
package a thin wrapper, and shipping a wrapper proves nothing about
understanding the statistics inside.

At the same time, "I implemented it myself" is worth less than nothing if the
implementation is subtly wrong - and hand-rolled statistics are wrong far more
often than they look.

## Options

1. **Depend on statsmodels at runtime.** Fastest, certainly correct, and
   removes the reason for the package to exist.
2. **Implement everything, test nothing against a reference.** Independent, and
   unverifiable: a sign error in the noncentrality parameter would survive.
3. **Implement everything, use statsmodels as a test-only oracle.** The formulas
   are ours; correctness is pinned to an established implementation in CI.

## Decision

Option 3. `numpy` and `scipy` are runtime dependencies (distributions and root
finding are not the interesting part); `statsmodels` sits in the `dev` extra and
appears only under `tests/`.

Where a reference comparison would be circular - `scipy.stats.ttest_ind` backs
our own Welch p-value - the test compares against a *different* implementation
instead (`statsmodels.stats.weightstats.CompareMeans`) or against hand
arithmetic. The mSPRT closed form has no reference implementation at all, so it
is checked against numerical integration of the mixture it claims to equal.

## Consequences

- Any drift between our formulas and the reference fails CI, at four decimals.
- Users install two packages, not five.
- The methods statsmodels does *not* cover (mSPRT, the simulation harness,
  SRM) carry their own validation: simulated worlds with a known answer, with
  every assertion expressed in multiples of the Monte Carlo error.
- Cost: the test suite is slower than the library, and dominated by simulation.
  Accepted - the simulations are the evidence.
