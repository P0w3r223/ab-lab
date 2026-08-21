"""One checkout experiment, analysed end to end, ending in a decision.

Every other script here demonstrates a method. This one demonstrates the order
the methods go in, and what to do when they disagree - which is the situation a
real experiment is usually in.

The scenario is the commonest genuine dilemma in e-commerce testing: a change to
the checkout that makes **more** people buy **cheaper** things. Conversion goes
up, average order value goes down, and the only question anyone actually asked -
does this make more money? - is answered by a metric noisier than either of them.

The script does not decide by counting significant results. It decides by asking
which metric the decision depends on, and whether the experiment could have
resolved it. Usually it could not, and saying so is the finding.

Usage::

    python examples/ecommerce_case_study.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ab_lab.analyze import proportion_test
from ab_lab.cuped import CupedSample, cuped_t_test
from ab_lab.multiplicity import holm
from ab_lab.power import sample_size_for_mean, sample_size_for_proportion
from ab_lab.ratio import RatioSample, ratio_metric_test
from ab_lab.srm import check_srm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sitegen.numbers import integer, percent  # noqa: E402

SEED = 20260821
USERS_PER_ARM = 40_000

# The world the experiment is run in. These are the *truth*, which no analysis
# gets to see - they exist so that every conclusion below can be graded.
BASELINE_CONVERSION = 0.050
TRUE_CONVERSION_LIFT = 0.006
BASELINE_ORDER_VALUE = 120.0
TRUE_ORDER_VALUE_RATIO = 0.94
ORDER_VALUE_SIGMA = 0.75

# Declared before launch, not after the result: the smallest revenue-per-user
# change the business would act on. Sizing against the *observed* effect instead
# is how post-hoc power analysis gets invented.
WORTH_ACTING_ON = 0.50

ALPHA = 0.05
GUARDRAIL_ALPHA = 0.001


def simulate_experiment(rng: np.random.Generator) -> dict:
    """Draw one experiment. The only place in this script that knows the truth."""

    def arm(conversion: float, value_multiplier: float, first_id: int) -> dict:
        # A user's spending tendency, which exists before the experiment does and
        # drives both months. Writing it this way is the whole point: the first
        # draft of this script built last month's revenue *out of* this month's,
        # which makes the covariate carry the treatment effect. CUPED then
        # subtracted the effect along with the noise and reported +0.078 where
        # the raw difference was +0.50 - exactly the failure ADR 0011 describes,
        # produced here by accident within an hour of implementing the warning.
        tendency = rng.normal(0.0, 1.0, USERS_PER_ARM)

        # Last month. A function of the user and of nothing the experiment did,
        # because it had not happened yet.
        prior_revenue = np.exp(
            np.log(8.0) + 0.9 * tendency + 0.6 * rng.normal(0.0, 1.0, USERS_PER_ARM)
        )

        # Heavier spenders convert more often and buy dearer things. The
        # recentring keeps the arm's average conversion at `conversion`, so the
        # treatment effect stays the number declared at the top of the file.
        propensity = np.clip(
            conversion * np.exp(0.45 * tendency - 0.45**2 / 2), 0.0, 0.95
        )
        converted = rng.random(USERS_PER_ARM) < propensity
        orders = np.where(converted, rng.integers(1, 4, USERS_PER_ARM), 0)

        total_orders = int(orders.sum())
        owner = np.repeat(np.arange(USERS_PER_ARM), orders)
        order_values = (
            rng.lognormal(
                np.log(BASELINE_ORDER_VALUE) - ORDER_VALUE_SIGMA**2 / 2,
                ORDER_VALUE_SIGMA,
                total_orders,
            )
            * np.exp(0.35 * tendency[owner] - 0.35**2 / 2)
            * value_multiplier
        )
        revenue = np.bincount(owner, weights=order_values, minlength=USERS_PER_ARM)

        return {
            "converted": converted,
            "orders": orders.astype(np.float64),
            "order_values": order_values,
            "order_owner": owner + first_id,
            "revenue": revenue,
            "prior_revenue": prior_revenue,
        }

    return {
        "control": arm(BASELINE_CONVERSION, 1.0, 0),
        "treatment": arm(
            BASELINE_CONVERSION + TRUE_CONVERSION_LIFT, TRUE_ORDER_VALUE_RATIO, 10**7
        ),
    }


def _heading(text: str) -> None:
    print(f"\n{text}\n{'-' * len(text)}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rng = np.random.default_rng(SEED)
    data = simulate_experiment(rng)
    control, treatment = data["control"], data["treatment"]

    _heading("1. Before launching: what could this experiment have seen?")
    design = sample_size_for_proportion(
        BASELINE_CONVERSION, TRUE_CONVERSION_LIFT, alpha=ALPHA, power=0.8
    )
    print(
        f"  Conversion, {percent(BASELINE_CONVERSION, 1)} baseline, "
        f"{TRUE_CONVERSION_LIFT * 100:.1f}pp lift: needs "
        f"{integer(design.per_group)} per arm. We have {integer(USERS_PER_ARM)}."
    )
    revenue_std = float(np.concatenate([control["revenue"], treatment["revenue"]]).std(ddof=1))
    revenue_design = sample_size_for_mean(
        WORTH_ACTING_ON, revenue_std, alpha=ALPHA, power=0.8
    )
    print(
        f"  Revenue per user, to see the {WORTH_ACTING_ON:.2f} that would be worth acting "
        f"on, on a\n  standard deviation of {revenue_std:.1f}: needs "
        f"{integer(revenue_design.per_group)} per arm."
    )
    print("  The metric that decides this is the one we cannot afford. Note it now,")
    print("  not after the result disappoints.")

    _heading("2. On arrival: was assignment even correct?")
    srm = check_srm([USERS_PER_ARM, USERS_PER_ARM], alpha=GUARDRAIL_ALPHA)
    print(f"  Allocation {srm.observed}, p = {srm.p_value:.3f}, mismatch: {srm.is_mismatch}")

    _heading("3. Conversion: did more people buy?")
    conversion = proportion_test(
        int(control["converted"].sum()),
        USERS_PER_ARM,
        int(treatment["converted"].sum()),
        USERS_PER_ARM,
        alpha=ALPHA,
    )
    print(
        f"  {percent(control['converted'].mean(), 2)} -> "
        f"{percent(treatment['converted'].mean(), 2)}  "
        f"({conversion.estimate * 100:+.2f}pp), p = {conversion.p_value:.4f}"
    )

    _heading("4. Average order value: did they spend as much when they did?")
    order_value = ratio_metric_test(
        RatioSample.from_arrays(
            control["order_values"],
            np.ones_like(control["order_values"]),
            control["order_owner"],
        ),
        RatioSample.from_arrays(
            treatment["order_values"],
            np.ones_like(treatment["order_values"]),
            treatment["order_owner"],
        ),
        alpha=ALPHA,
    )
    print(
        f"  {order_value.control_ratio:.2f} -> {order_value.treatment_ratio:.2f}  "
        f"({order_value.estimate:+.2f}, {percent(order_value.relative_effect, 1)}), "
        f"p = {order_value.p_value:.4f}"
    )
    print(f"  Interval: {order_value.ci}")
    print("  Orders are clustered by user, so that interval is the cluster-robust one.")

    _heading("5. Revenue per user: the only metric the decision depends on")
    revenue = cuped_t_test(
        CupedSample.from_arrays(control["revenue"], control["prior_revenue"]),
        CupedSample.from_arrays(treatment["revenue"], treatment["prior_revenue"]),
        alpha=ALPHA,
    )
    print(
        f"  {control['revenue'].mean():.2f} -> {treatment['revenue'].mean():.2f}  "
        f"({revenue.estimate:+.3f}), p = {revenue.p_value:.4f}"
    )
    print(
        f"  Last month's spending correlates {revenue.correlation:.2f} with this "
        f"month's and removed\n  {percent(revenue.variance_reduction, 1)} of the "
        f"variance - worth x{revenue.effective_sample_multiplier:.2f} the traffic."
    )
    print(
        f"  Unadjusted, the same difference is {revenue.unadjusted_estimate:+.3f}. "
        f"These should agree;\n  a gap would mean the covariate was not balanced, "
        f"which is a randomisation question."
    )
    print(f"  Interval: {revenue.ci}")

    _heading("6. Three metrics, asked at once")
    family = holm(
        [conversion.p_value, order_value.p_value, revenue.p_value],
        alpha=ALPHA,
        labels=["conversion", "average order value", "revenue per user"],
    )
    for label in family.labels or ():
        adjusted, rejected = family.for_label(label)
        print(f"  {label:<22} adjusted p = {adjusted:.4f}  {'reject' if rejected else '-'}")
    print(f"  Controlling: {family.error_rate_controlled}")

    _heading("The decision")
    decisive_p = family.for_label("revenue per user")[0]
    lower, upper = revenue.ci.low, revenue.ci.high
    print(
        f"  More people bought, and they spent less when they did. Both of those are\n"
        f"  resolved. Whether the change makes money is not: revenue per user lands at\n"
        f"  {revenue.estimate:+.3f} with an interval of [{lower:.3f}, {upper:.3f}] and an\n"
        f"  adjusted p of {decisive_p:.4f}."
    )
    print(
        f"\n  In money, across an arm this size, that interval spans "
        f"{lower * USERS_PER_ARM:+,.0f} to {upper * USERS_PER_ARM:+,.0f}.\n"
        f"  The experiment was never sized to resolve it: step 1 said "
        f"{integer(revenue_design.per_group)} users per\n  arm were needed to see "
        f"{WORTH_ACTING_ON:.2f}, and it had {integer(USERS_PER_ARM)}."
    )
    print(
        f"\n  CUPED bought almost nothing here - {percent(revenue.variance_reduction, 1)} of "
        f"the variance, worth\n  x{revenue.effective_sample_multiplier:.2f} the traffic - and "
        f"the reason is worth knowing rather than hiding.\n  Revenue per user is about 95% "
        f"zeros, so most of its variance is the conversion\n  lottery, and last month's "
        f"spending does not predict who wins it. A covariate helps\n  in proportion to what "
        f"it explains, and on a metric this sparse that is little. It\n  pays properly on "
        f"continuous metrics: ADR 0011 measures that case at up to five\n  times the traffic."
    )
    print(
        "\n  Ship / do not ship is the wrong question to put to this data. The answer is"
    )
    print("  that the experiment cannot answer it, which was knowable before launch and")
    print("  is the one conclusion here that does not depend on a p-value.")
    print(
        f"\n  (Truth, for grading: conversion {TRUE_CONVERSION_LIFT * 100:+.1f}pp and order "
        f"value x{TRUE_ORDER_VALUE_RATIO:.2f} are\n  both real, and revenue per user really "
        f"did rise. Every measured effect lands\n  inside its interval; the third one's "
        f"interval is simply too wide to read.)"
    )


if __name__ == "__main__":
    main()
