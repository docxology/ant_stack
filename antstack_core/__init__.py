"""Top-level package metadata and public namespace for Ant Stack."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Iterable

__version__ = "2.0.0"
__author__ = "Daniel Ari Friedman"
__email__ = "daniel@activeinference.institute"

from . import analysis, architecture, cohereants, figures, mathematics, orchestration, publishing

__all__ = [
    "analysis",
    "architecture",
    "check_runtime_dependencies",
    "cohereants",
    "figures",
    "mathematics",
    "orchestration",
    "publishing",
]


def check_runtime_dependencies(
    packages: Iterable[str] = ("matplotlib", "numpy", "yaml"),
) -> tuple[str, ...]:
    """Return optional runtime dependencies that are not importable.

    The package import itself is intentionally side-effect free. Call this
    helper from diagnostics, setup checks, or CLI validation when an explicit
    dependency report is needed.
    """
    missing: list[str] = []
    for package in packages:
        try:
            if find_spec(package) is None:
                missing.append(package)
        except (ImportError, AttributeError, ValueError):
            missing.append(package)
    return tuple(missing)
