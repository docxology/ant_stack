"""Contracts for the complexity energetics paper runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from antstack_core.analysis import ExperimentManifest, brain_workload_closed_form


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = PROJECT_ROOT / "papers" / "complexity_energetics" / "src" / "runner.py"
MANIFEST_PATH = PROJECT_ROOT / "papers" / "complexity_energetics" / "manifest.example.yaml"


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("antstack_ce_runner_contract", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scaling_workloads_honor_closed_form_manifest_mode() -> None:
    """Scaling sweeps should use deterministic closed-form loads when configured."""
    runner = _load_runner_module()
    manifest = ExperimentManifest.load(MANIFEST_PATH)
    params = runner._workload_params(manifest, "brain")

    load_from_runner = runner._load_for_manifest_mode(manifest, "brain", 0.25, params)
    expected_load = brain_workload_closed_form(0.25, params)

    assert load_from_runner == expected_load
