"""ab-lab: designing and analysing A/B experiments, with the assumptions attached.

The package is organised by the question you are asking:

* :mod:`ab_lab.power` - before the experiment: how many units do I need?
* :mod:`ab_lab.srm` - the moment data arrives: was assignment even correct?
* :mod:`ab_lab.analyze` - after the experiment: what is the effect, and how sure am I?
* :mod:`ab_lab.sequential` - during the experiment: may I look at this yet?
* :mod:`ab_lab.simulate` - underneath all of it: does this code actually work?
"""

from .analyze import bootstrap_diff, mann_whitney, proportion_test, welch_t_test
from .power import (
    cohens_h,
    mde_for_mean,
    mde_for_proportion,
    power_t,
    power_z,
    sample_size_for_mean,
    sample_size_for_proportion,
)
from .results import (
    ConfidenceInterval,
    SampleSizeResult,
    SequentialResult,
    SimulationSummary,
    SrmResult,
    TestResult,
)
from .sequential import SequentialMonitor, always_valid_p_value, msprt, tau_from_mde
from .srm import check_srm

__version__ = "0.1.0"

__all__ = [
    "ConfidenceInterval",
    "SampleSizeResult",
    "SequentialMonitor",
    "SequentialResult",
    "SimulationSummary",
    "SrmResult",
    "TestResult",
    "always_valid_p_value",
    "bootstrap_diff",
    "check_srm",
    "cohens_h",
    "mann_whitney",
    "mde_for_mean",
    "mde_for_proportion",
    "msprt",
    "power_t",
    "power_z",
    "proportion_test",
    "sample_size_for_mean",
    "sample_size_for_proportion",
    "tau_from_mde",
    "welch_t_test",
]
