"""Tests for canonical run-all Ant Stack orchestration."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
import yaml

from antstack_core.orchestration import (
    RUN_SUBDIRECTORIES,
    ArtifactRecord,
    RunAllConfig,
    checksum_file,
    create_run_layout,
    parse_task_names,
    run_all,
    write_manifest,
)


def _write_minimal_ce_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "ce_manifest.yaml"
    manifest.write_text(
        textwrap.dedent(
            """
            experiment_name: run-all-test
            seed: 7
            coefficients:
              flops_pj: 1.0
              sram_pj_per_byte: 0.1
              dram_pj_per_byte: 20.0
              spike_aj: 1.0
              baseline_w: 0.05
              body_per_joint_w: 1.0
              body_sensor_w_per_channel: 0.001
            workloads:
              body:
                name: body
                duration_s: 0.01
                repeats: 2
                mode: closed_form
                params: {J: 6, C: 4, S: 12, hz: 50, contact_solver: pgs}
              brain:
                name: brain
                duration_s: 0.01
                repeats: 2
                mode: closed_form
                params: {K: 32, N_KC: 512, rho: 0.05, H: 16, hz: 50}
              mind:
                name: mind
                duration_s: 0.01
                repeats: 2
                mode: closed_form
                params: {B: 2, H_p: 4, hz: 50, state_dim: 8, action_dim: 3}
            scaling:
              analyses:
                - workload: body
                  param: J
                  values: [4, 6]
                  description: body joint sweep
                - workload: brain
                  param: K
                  values: [16, 32]
                  description: brain channel sweep
                - workload: mind
                  param: H_p
                  values: [3, 4]
                  description: mind horizon sweep
            meter:
              meter_type: null
            mass_kg: 0.02
            distance_m: 1.0
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _write_run_config(
    tmp_path: Path,
    *,
    run_id: str = "test-run",
    paper_projection: bool = False,
    output_root: Path | None = None,
) -> Path:
    output_root = output_root or (tmp_path / "outputs")
    config = {
        "run": {
            "run_id": run_id,
            "seed": 7,
            "output_root": str(output_root),
            "clean_run_dir": True,
            "log_level": "INFO",
        },
        "tasks": {
            "data": True,
            "statistics": True,
            "visualizations": True,
            "animations": True,
            "reports": True,
            "papers": True,
            "validation": True,
        },
        "complexity_energetics": {
            "manifest": str(_write_minimal_ce_manifest(tmp_path)),
            "paper_projection": paper_projection,
            "paper_root": str(tmp_path / "paper_projection"),
            "build_pdf": False,
            "pdf_sources": [],
        },
        "visualization": {
            "output_formats": ["png"],
            "dpi": 80,
            "figure_size": [4.0, 3.0],
            "statistical_annotations": True,
            "animation_fps": 2,
            "animation_duration_s": 0.5,
            "animation_formats": ["gif", "html"],
        },
        "validation": {
            "required_artifact_types": ["csv", "json", "markdown", "image", "animation", "log", "provenance"],
            "fail_on_missing": True,
            "fail_on_empty": True,
            "deterministic_checksum_checks": True,
        },
    }
    config_path = tmp_path / "run_all.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_config_defaults_and_validation_errors(tmp_path: Path) -> None:
    config_path = _write_run_config(tmp_path)
    config = RunAllConfig.from_file(config_path)

    assert config.run.run_id == "test-run"
    assert config.tasks.enabled() == (
        "data",
        "statistics",
        "visualizations",
        "animations",
        "reports",
        "papers",
        "validation",
    )

    bad_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    bad_config["complexity_energetics"]["manifest"] = str(tmp_path / "missing.yaml")
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(yaml.safe_dump(bad_config), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest not found"):
        RunAllConfig.from_file(bad_path)


def test_output_directory_layout_creation(tmp_path: Path) -> None:
    config = RunAllConfig.from_file(_write_run_config(tmp_path))
    run_dir, layout = create_run_layout(config)

    assert run_dir == tmp_path / "outputs" / "test-run"
    for rel in RUN_SUBDIRECTORIES:
        assert layout[rel].is_dir()


def test_task_selection_parsing() -> None:
    assert parse_task_names("data,statistics,animations") == ("data", "statistics", "animations")
    with pytest.raises(ValueError, match="Unknown task"):
        parse_task_names("data,unknown")


def test_manifest_checksum_inventory(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text('{"ok": true}\n', encoding="utf-8")
    record = ArtifactRecord(
        artifact_type="json",
        path="artifact.json",
        absolute_path=str(artifact_path),
        producing_task="test",
        source_config="test",
        checksum=checksum_file(artifact_path),
        bytes=artifact_path.stat().st_size,
    )
    config = RunAllConfig.from_file(_write_run_config(tmp_path))
    manifest_path = write_manifest(tmp_path / "manifest.json", config, [record])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["artifact_count"] == 1
    assert payload["artifacts"][0]["checksum"] == checksum_file(artifact_path)


def test_validate_only_creates_no_run_artifacts(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    result = run_all(_write_run_config(tmp_path, output_root=output_root), validate_only=True)

    assert result.validate_only is True
    assert result.artifacts == ()
    assert not (output_root / "test-run").exists()


def test_run_all_generates_canonical_outputs(tmp_path: Path) -> None:
    result = run_all(_write_run_config(tmp_path), run_id="integration-run")
    run_dir = tmp_path / "outputs" / "integration-run"

    for rel in RUN_SUBDIRECTORIES:
        assert (run_dir / rel).is_dir()

    expected_files = [
        run_dir / "data/derived/complexity_energetics_results.csv",
        run_dir / "statistics/workload_summary.json",
        run_dir / "statistics/scaling_summary.json",
        run_dir / "statistics/bootstrap_intervals.csv",
        run_dir / "statistics/validation_summary.json",
        run_dir / "visualizations/static/workload_energy_summary.png",
        run_dir / "visualizations/animations/energy_components.gif",
        run_dir / "visualizations/animations/scaling_sweep.gif",
        run_dir / "visualizations/animations/preview.html",
        run_dir / "reports/run_summary.md",
        run_dir / "papers/paper_artifacts.md",
        run_dir / "logs/run_all_antstack.log",
        run_dir / "provenance/provenance.json",
        run_dir / "provenance/output_inventory.json",
        run_dir / "manifest.json",
    ]
    for path in expected_files:
        assert path.is_file(), path
        assert path.stat().st_size > 0, path

    artifact_types = {artifact.artifact_type for artifact in result.artifacts}
    assert {"csv", "json", "markdown", "image", "animation", "log", "provenance"} <= artifact_types

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "integration-run"
    assert manifest["artifact_count"] == len(manifest["artifacts"])
    assert all(item["checksum"] for item in manifest["artifacts"])

    provenance = json.loads((run_dir / "provenance/provenance.json").read_text(encoding="utf-8"))
    assert str((run_dir / "manifest.json").resolve()) in provenance["output_paths"]
    assert str((run_dir / "provenance/output_inventory.json").resolve()) in provenance["output_paths"]


def test_paper_projection_can_write_compatibility_files(tmp_path: Path) -> None:
    config_path = _write_run_config(tmp_path, paper_projection=True, run_id="projection-run")
    run_all(config_path, tasks="data")
    projection_root = tmp_path / "paper_projection"

    assert (projection_root / "out/results.csv").is_file()
    assert (projection_root / "out/summary.json").is_file()
    assert (projection_root / "Generated.md").is_file()
