"""ab-lab: designing and analysing A/B experiments, with the assumptions attached.

The package is organised by the question you are asking:

* :mod:`ab_lab.power` - before the experiment: how many units do I need?
* :mod:`ab_lab.srm` - the moment data arrives: was assignment even correct?
* :mod:`ab_lab.analyze` - after the experiment: what is the effect, and how sure am I?
* :mod:`ab_lab.sequential` - during the experiment: may I look at this yet?
* :mod:`ab_lab.cluster` - when a unit appears more than once: is 5 000 sessions
  really 5 000 observations?
* :mod:`ab_lab.multiplicity` - when one experiment has many metrics: which
  of these results survive being asked all at once?
* :mod:`ab_lab.ratio` - when the metric is a ratio of two totals: what is the
  variance of a number whose denominator moves too?
* :mod:`ab_lab.cuped` - before *and* after: can last month's data buy this
  experiment a smaller sample?
* :mod:`ab_lab.simulate` - underneath all of it: does this code actually work?

One vocabulary difference to know about: design functions take
``alternative="two-sided"`` or ``"one-sided"`` (a sample size does not depend on
*which* direction), while the analysis functions follow SciPy and take
``"two-sided"``, ``"less"`` or ``"greater"`` (a p-value does).

:mod:`ab_lab.simulate` is deliberately not re-exported here. It is a laboratory
rather than an API - its callables exist to be composed into an experiment about
this package's own behaviour - so it is reached as ``from ab_lab.simulate import
...``, which keeps the top-level surface the set of things you would use *on your
own data*.
"""

from .analyze import (
    bootstrap_diff,
    mann_whitney,
    paired_bootstrap,
    proportion_test,
    welch_t_test,
)
from .cluster import (
    ClusteredSample,
    cluster_robust_t_test,
    design_effect,
    intraclass_correlation,
    sample_size_for_clustered_mean,
)
from .cuped import CupedSample, cuped_t_test, cuped_theta
from .multiplicity import (
    CORRECTIONS,
    benjamini_hochberg,
    bonferroni,
    holm,
)
from .power import (
    cohens_h,
    mde_for_mean,
    mde_for_proportion,
    power_t,
    power_z,
    sample_size_for_mean,
    sample_size_for_proportion,
)
from .ratio import RatioSample, ratio_metric_test
from .results import (
    ClusteredSampleSizeResult,
    ClusterTestResult,
    ConfidenceInterval,
    CupedResult,
    MdeResult,
    MultipleComparisonResult,
    RatioTestResult,
    SampleSizeResult,
    SequentialResult,
    SimulationSummary,
    SrmResult,
    TestResult,
)
from .sequential import SequentialMonitor, always_valid_p_value, msprt, tau_from_mde
from .srm import check_srm

#: Single source of truth: ``pyproject.toml`` reads this attribute rather than
#: restating the number, so the two cannot drift at the release where it matters.
__version__ = "0.4.1"

__all__ = [
    "CORRECTIONS",
    "ClusterTestResult",
    "ClusteredSample",
    "ClusteredSampleSizeResult",
    "ConfidenceInterval",
    "CupedResult",
    "CupedSample",
    "MdeResult",
    "MultipleComparisonResult",
    "RatioSample",
    "RatioTestResult",
    "SampleSizeResult",
    "SequentialMonitor",
    "SequentialResult",
    "SimulationSummary",
    "SrmResult",
    "TestResult",
    "always_valid_p_value",
    "benjamini_hochberg",
    "bonferroni",
    "bootstrap_diff",
    "check_srm",
    "cluster_robust_t_test",
    "cohens_h",
    "cuped_t_test",
    "cuped_theta",
    "design_effect",
    "holm",
    "intraclass_correlation",
    "mann_whitney",
    "paired_bootstrap",
    "mde_for_mean",
    "mde_for_proportion",
    "msprt",
    "power_t",
    "power_z",
    "proportion_test",
    "ratio_metric_test",
    "sample_size_for_clustered_mean",
    "sample_size_for_mean",
    "sample_size_for_proportion",
    "tau_from_mde",
    "welch_t_test",
]
