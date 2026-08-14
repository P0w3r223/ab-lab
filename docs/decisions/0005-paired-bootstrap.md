# ADR 0005 — A separate paired bootstrap, resampling units rather than arms

Date: 2026-07-22
Status: accepted
Author: P0w3r223
Related to: ADR 0001, `analyze.bootstrap_diff`

---

## Context

`bootstrap_diff` resamples the two arms independently, which is right for an ordinary
experiment: control and treatment contain different users, and the two samples are
independent by construction.

A different shape of comparison kept turning up: the same units measured twice. The same
rows scored by a champion and a challenger model, the same days measured before and after a
change, the same users seeing both variants. There the between-unit variance is shared, and
resampling the arms independently counts it twice instead of letting it cancel. On a
realistic example — units with a standard deviation of 30 and a real within-unit effect of
1.5 — the independent interval spans zero while the paired one does not. The effect is
there; the unpaired procedure simply cannot see it.

The immediate consumer was the `mlops-car-price` promotion gate, where champion and
challenger score the *same* frozen holdout rows and the question is whether the difference
in mean absolute error is more than noise.

## Options

1. **Let callers use `bootstrap_diff` anyway.** Conservative — the interval is too wide, not
   too narrow — so nothing is *wrong*, only weak. It also quietly violates the function's
   own stated assumption that the groups are independent.
2. **Ask callers to bootstrap the per-unit differences themselves.** Correct for the mean,
   and a trap for anything else: bootstrapping `treatment - control` and taking a median is
   not the difference of medians.
3. **A dedicated function that resamples unit indices** and applies them to both arms.

## Decision

Option 3: `paired_bootstrap(control, treatment, statistic=np.mean, ...)`. One set of indices
is drawn per resample and used for both arms, so a unit that happens to be extreme enters
both sides at once. The statistic is applied to each arm and the difference is taken —
matching what `bootstrap_diff` means by "difference of a statistic", and what people mean
when they name a median.

## Consequences

- **Callers must state the shape of their data.** Passing unpaired arms to the paired
  function is now an error when the lengths differ, and a silent modelling mistake when they
  happen to match — the assumption travels in the result, which is the mitigation this
  library already relies on elsewhere.
- **The gain is real but bounded.** Pairing removes between-unit variance; it does not
  rescue a sample too small for the bootstrap in the first place. That caveat is in the
  docstring rather than implied.
- **Two bootstrap functions to keep in step.** They share the percentile method, the
  achieved-significance-level p-value and the resolution floor at `2/(B+1)`, so a change to
  one is a change to both.
- **Validated the same way as everything else here**: a constant shift is recovered exactly
  (the interval collapses to a point), the A/A false positive rate sits at alpha within four
  Monte Carlo sigmas over 2 000 simulations, and a 95% interval covers a known effect 95% of
  the time.
