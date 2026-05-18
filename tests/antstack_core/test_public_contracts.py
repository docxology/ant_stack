"""Public API contract tests for antstack_core namespaces."""

from __future__ import annotations

import importlib


PUBLIC_MODULES = (
    "antstack_core",
    "antstack_core.analysis",
    "antstack_core.architecture",
    "antstack_core.cohereants",
    "antstack_core.figures",
    "antstack_core.mathematics",
    "antstack_core.publishing",
    "antstack_core.orchestration",
    "antstack_core.cli",
    "antstack_core.cli.build",
    "antstack_core.cli.ce",
    "antstack_core.cli.run_all",
)


def test_public_exports_resolve_to_attributes() -> None:
    """Every exported public name should resolve on its module."""
    for module_name in PUBLIC_MODULES:
        module = importlib.import_module(module_name)
        exports = getattr(module, "__all__", ())
        assert exports, f"{module_name} should define a discoverable __all__"
        missing = [name for name in exports if not hasattr(module, name)]
        assert missing == [], f"{module_name} exports missing names: {missing}"


def test_top_level_runtime_dependency_check_is_explicit() -> None:
    """Dependency diagnostics should be callable, not an import-time side effect."""
    antstack_core = importlib.import_module("antstack_core")
    assert callable(antstack_core.check_runtime_dependencies)
    assert antstack_core.check_runtime_dependencies(("sys",)) == ()
