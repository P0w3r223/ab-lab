"""The package's advertised surface has to exist.

`__all__` is documentation that the interpreter can check, and for two releases
nobody checked it: 0.4.0 shipped with `CupedResult`, `RatioSample` and
`ratio_metric_test` listed and never imported, so `from ab_lab import *` raised
`AttributeError` on a released version while `from ab_lab.ratio import ...`
worked, which makes the failure look arbitrary rather than like a missing line.

`ruff` does not catch this - F822 does not fire inside `__init__.py` - and no
other test imports from the top level, so nothing did. These three assertions
cost microseconds and close that gap permanently.
"""

from __future__ import annotations

import importlib

import ab_lab


def test_every_advertised_name_exists():
    missing = sorted(name for name in ab_lab.__all__ if not hasattr(ab_lab, name))
    assert not missing, f"__all__ advertises names the package never imports: {missing}"


def test_a_star_import_succeeds():
    """The one thing `__all__` is actually for."""
    namespace: dict[str, object] = {}
    exec("from ab_lab import *", namespace)  # noqa: S102 - the assertion is the point
    for name in ab_lab.__all__:
        assert name in namespace, f"{name} is in __all__ but not exported by a star import"


def test_all_is_sorted_and_free_of_duplicates():
    """Two names appeared twice in earlier edits; sorting makes that visible."""
    assert len(ab_lab.__all__) == len(set(ab_lab.__all__))


def test_every_module_named_in_the_package_docstring_imports():
    """The docstring lists the modules by the question each answers.

    A module named there and absent - or renamed - is a broken map, and the
    docstring is the first thing a reader meets.
    """
    named = [
        line.split("`")[1]
        for line in (ab_lab.__doc__ or "").splitlines()
        if line.strip().startswith("* :mod:")
    ]
    assert named, "the package docstring no longer lists its modules"
    for module in named:
        importlib.import_module(module)
