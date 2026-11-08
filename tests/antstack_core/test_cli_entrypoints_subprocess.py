"""Real CLI validation rejects bad inputs and never produces data in validation mode."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run(script, *args):
    return subprocess.run([sys.executable, "-m", "antstack_core.cli." + script, *map(str, args)],
                          cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)


def test_ce_validation_has_no_output_side_effects(tmp_path):
    output = tmp_path / "unused"
    result = _run("ce", PROJECT_ROOT / "papers/complexity_energetics/manifest.example.yaml", "--validate-only", "--out", output)
    assert result.returncode == 0, result.stderr
    assert not output.exists()


def test_run_all_validation_has_no_output_side_effects(tmp_path):
    output = tmp_path / "unused"
    result = _run("run_all", "--config", PROJECT_ROOT / "configs/run_all_antstack.example.yaml",
                  "--out", output, "--run-id", "validation-only", "--validate-only")
    assert result.returncode == 0, result.stderr
    assert not output.exists()


def test_ce_rejects_zero_repeats_without_producing_an_output(tmp_path):
    manifest = tmp_path / "invalid.yaml"
    manifest.write_text("experiment_name: boundary\nseed: 7\nworkloads:\n  brain:\n    name: brain\n    duration_s: 0.01\n    repeats: 0\n    mode: closed_form\n")
    output = tmp_path / "out"
    result = _run("ce", manifest, "--out", output, "--validate-only")
    assert result.returncode != 0
    assert not output.exists()
