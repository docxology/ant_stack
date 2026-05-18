"""CLI wrapper contract tests."""

from __future__ import annotations


def test_build_entrypoint_loads_canonical_main() -> None:
    """The antstack-build wrapper should resolve the build pipeline main."""
    from antstack_core.cli import build

    assert callable(build._load_build_main())
    assert callable(build.main)


def test_complexity_entrypoint_loads_runner_contract() -> None:
    """The antstack-ce wrapper should expose the paper runner contract."""
    from antstack_core.cli import ce

    runner = ce._load_runner()
    assert callable(runner.main)
    assert callable(runner.run_manifest)
    assert callable(ce.run_manifest)
    assert callable(ce.main)


def test_run_all_entrypoint_delegates_to_orchestration() -> None:
    """The run-all wrapper should expose package-owned orchestration."""
    from antstack_core.cli import run_all

    assert callable(run_all.run_all)
    assert callable(run_all.main)
