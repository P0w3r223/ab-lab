# 0003 - mSPRT rather than group-sequential boundaries

Date: 2026-07-21
Status: accepted
Author: P0w3r223 + Claude

---

## Context

Experiment owners look at running results. Telling them not to is not a
solution; the package needs one correction that makes early looks legitimate.

## Options

1. **O'Brien-Fleming / Pocock group-sequential boundaries.** The clinical-trial
   standard. Requires committing in advance to the number of looks and the
   final sample size; the boundaries are a function of that schedule.
2. **Lan-DeMets alpha spending.** Relaxes the fixed schedule, but still needs
   the information fraction - i.e. the planned horizon - to spend against.
3. **mSPRT (mixture sequential probability ratio test).** No schedule and no
   horizon: the likelihood ratio against a normal mixture over effect sizes is
   a martingale under the null, and Ville's inequality bounds the probability
   that it ever crosses `1/alpha` by `alpha`.
4. **All three, shallowly.** More surface, less understanding of any of them.

## Decision

mSPRT alone (option 3).

The deciding argument is which failure mode fits the setting. Group-sequential
methods are exact when the schedule is honoured, and their guarantee lapses
exactly when it is not - and a dashboard refreshed whenever someone feels like
it is the definition of not honouring a schedule. mSPRT's guarantee holds for
looks taken at arbitrary times, in arbitrary number, chosen after seeing the
data. That is what continuous monitoring actually is.

The price is a tuning parameter, `tau`, the width of the mixture over effects.
It trades *power* between small and large effects; it never trades away
validity, which is checked empirically rather than asserted.

## Consequences

- The test is conservative: measured type I error under ten looks lands near
  1.2% against a nominal 5%. That is the guarantee working (`<= alpha`, not
  `= alpha`) - and it cost a fix in `examples/validation_table.py`, whose first
  version judged the mSPRT row by an equality criterion and flagged correct
  behaviour as a failure.
- Conservatism means longer experiments than a correctly-executed
  group-sequential design. Named as a limitation in the README rather than
  hidden.
- `tau` has to be chosen. `tau_from_mde` makes the defensible default explicit:
  centre the mixture on the effect the experiment was sized for.
- Variances are plugged in from the sample, so the guarantee is asymptotic. At
  the sample sizes where anyone peeks daily this is immaterial - and it is
  verified by simulation instead of assumed.
