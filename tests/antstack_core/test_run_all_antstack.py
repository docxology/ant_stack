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
    assert "manifest.json" in provenance["output_paths"]
    assert "provenance/output_inventory.json" in provenance["output_paths"]


def test_paper_projection_can_write_compatibility_files(tmp_path: Path) -> None:
    config_path = _write_run_config(tmp_path, paper_projection=True, run_id="projection-run")
    run_all(config_path, tasks="data")
    projection_root = tmp_path / "paper_projection"

    assert (projection_root / "out/results.csv").is_file()
    assert (projection_root / "out/summary.json").is_file()
    assert (projection_root / "Generated.md").is_file()


@pytest.mark.parametrize("section,key", [("run", "clean_run_dir"), ("tasks", "data"),
                                          ("complexity_energetics", "paper_projection"),
                                          ("complexity_energetics", "build_pdf"),
                                          ("validation", "fail_on_missing")])
def test_quoted_boolean_is_rejected_before_outputs(tmp_path, section, key):
    path = _write_run_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw[section][key] = "false"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="boolean"):
        run_all(path, tasks="")
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("run_id", [".", "..", "a.b", "a/b", "a\\b", "", "-unsafe"])
def test_public_layout_rejects_unsafe_run_id(tmp_path, run_id):
    from dataclasses import replace
    config = RunAllConfig.from_file(_write_run_config(tmp_path))
    config = replace(config, run=replace(config.run, run_id=run_id))
    with pytest.raises(ValueError, match="run_id"):
        create_run_layout(config)
    assert not (tmp_path / "outputs").exists()


def test_stable_symlink_and_source_overlap_reject_before_mutation(tmp_path):
    path = _write_run_config(tmp_path)
    victim = tmp_path / "victim"
    victim.mkdir()
    (tmp_path / "outputs").symlink_to(victim, target_is_directory=True)
    with pytest.raises(ValueError, match="ordinary"):
        run_all(path, tasks="")
    assert not list(victim.iterdir())
    (tmp_path / "outputs").unlink()
    raw = yaml.safe_load(path.read_text())
    raw["run"]["output_root"] = str(tmp_path)
    raw["run"]["run_id"] = "ce_manifest"
    raw["complexity_energetics"]["manifest"] = str(tmp_path / "ce_manifest" / "input.yaml")
    source_dir = tmp_path / "ce_manifest"
    source_dir.mkdir()
    (source_dir / "input.yaml").write_bytes((tmp_path / "ce_manifest.yaml").read_bytes())
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="overlap"):
        run_all(path, tasks="")
    assert list(source_dir.iterdir()) == [source_dir / "input.yaml"]


def test_unowned_existing_run_is_preserved(tmp_path):
    path = _write_run_config(tmp_path)
    run = tmp_path / "outputs/test-run"
    run.mkdir(parents=True)
    guide = run / "README.md"
    guide.write_text("unowned data")
    with pytest.raises(ValueError, match="unowned"):
        run_all(path, tasks="")
    assert guide.read_text() == "unowned data"
    assert list(run.iterdir()) == [guide]


def test_completed_log_is_closed_and_caller_logging_restored(tmp_path):
    import logging
    root = logging.getLogger()
    before_handlers, before_level = tuple(root.handlers), root.level
    result = run_all(_write_run_config(tmp_path), tasks="")
    log = result.run_dir / "logs/run_all_antstack.log"
    identity = checksum_file(log)
    root.warning("after completed run")
    assert checksum_file(log) == identity
    assert tuple(root.handlers) == before_handlers
    assert root.level == before_level
    for record in result.artifacts:
        assert checksum_file(result.run_dir / record.path) == record.checksum
        assert (result.run_dir / record.path).stat().st_size == record.bytes


def test_failed_owned_rerun_preserves_completed_tree_and_projection(tmp_path):
    path = _write_run_config(tmp_path, paper_projection=True)
    first = run_all(path, tasks="data")
    old_manifest = first.manifest_path.read_bytes()
    projected = tmp_path / "paper_projection/out/results.csv"
    old_csv = projected.read_bytes()
    raw = yaml.safe_load(path.read_text())
    raw["validation"]["required_artifact_types"] = ["impossible"]
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(RuntimeError, match="validation failed"):
        run_all(path, tasks="data,validation")
    assert first.manifest_path.read_bytes() == old_manifest
    assert projected.read_bytes() == old_csv
    for record in first.artifacts:
        assert checksum_file(first.run_dir / record.path) == record.checksum


def test_owned_rerun_archives_prior_complete_tree(tmp_path):
    path = _write_run_config(tmp_path)
    first = run_all(path, tasks="")
    previous = first.manifest_path.read_bytes()
    second = run_all(path, tasks="")
    archives = list(second.run_dir.parent.glob(".test-run.archive-*"))
    assert len(archives) == 1
    assert (archives[0] / "manifest.json").read_bytes() == previous
    assert second.manifest_path.is_file()


@pytest.mark.parametrize("mutation", ["delete", "truncate", "change"])
def test_validation_checks_current_artifact_bytes(tmp_path, mutation):
    import importlib
    module = importlib.import_module("antstack_core.orchestration.run_all")
    config = RunAllConfig.from_file(_write_run_config(tmp_path))
    root, layout = create_run_layout(config)
    ctx = module.RunContext(config, root, layout)
    artifact = root / "data/derived/example.csv"
    artifact.write_text("data\n1\n")
    module._record_artifact(ctx, artifact, "csv", "test", "fixture")
    if mutation == "delete":
        artifact.unlink()
    else:
        artifact.write_text("" if mutation == "truncate" else "data\n2\n")
    with pytest.raises((ValueError, RuntimeError)):
        module._task_validation(ctx)


def test_every_reader_consumes_frozen_manifest(tmp_path):
    import importlib
    module = importlib.import_module("antstack_core.orchestration.run_all")
    config = RunAllConfig.from_file(_write_run_config(tmp_path))
    root, layout = create_run_layout(config)
    ctx = module.RunContext(config, root, layout)
    module._write_initial_inputs(ctx)
    original = Path(config.complexity_energetics.manifest)
    raw = yaml.safe_load(original.read_text())
    raw["seed"] = 999
    raw["scaling"]["analyses"] = [{"workload": "brain", "param": "K", "values": [8, 16]}]
    original.write_text(yaml.safe_dump(raw))
    module._task_data(ctx)
    assert ctx.ce_summary["seed"] == 7
    assert {curve["workload"] for curve in module._scaling_curves(ctx)} == {"body", "brain", "mind"}
    assert checksum_file(ctx.manifest_snapshot) == next(record.checksum for record in ctx.artifacts if record.path.endswith("complexity_energetics_manifest.yaml"))


def test_html_only_contains_real_embedded_animation(tmp_path):
    path = _write_run_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["visualization"]["animation_formats"] = ["html"]
    path.write_text(yaml.safe_dump(raw))
    result = run_all(path, tasks="animations")
    html = result.run_dir / "visualizations/animations/preview.html"
    assert "data:image/gif;base64,R0lGOD" in html.read_text()
    assert (html.parent / "energy_components.gif").stat().st_size > 0


def test_mp4_missing_tool_fails_before_outputs(tmp_path, monkeypatch):
    import importlib
    module = importlib.import_module("antstack_core.orchestration.run_all")
    path = _write_run_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["visualization"]["animation_formats"] = ["mp4"]
    path.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="ffmpeg"):
        run_all(path, tasks="animations")
    assert not (tmp_path / "outputs").exists()


def test_paper_note_is_not_pdf_completion_evidence(tmp_path):
    path = _write_run_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["validation"]["required_artifact_types"] = ["paper"]
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(RuntimeError, match="paper"):
        run_all(path, tasks="papers,validation")
    assert not (tmp_path / "outputs/test-run").exists()


def test_requested_mp4_is_real_or_preflight_fails(tmp_path):
    import shutil
    path = _write_run_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["visualization"]["animation_formats"] = ["mp4"]
    path.write_text(yaml.safe_dump(raw))
    if shutil.which("ffmpeg") is None:
        with pytest.raises(ValueError, match="ffmpeg"):
            run_all(path, tasks="animations")
        assert not (tmp_path / "outputs").exists()
    else:
        result = run_all(path, tasks="animations")
        media = result.run_dir / "visualizations/animations/energy_components.mp4"
        assert media.stat().st_size > 32
        assert b"ftyp" in media.read_bytes()[:32]
        assert not (media.parent / "mp4_writer_status.json").exists()


def test_requested_pdf_is_real_or_preflight_fails(tmp_path):
    import shutil
    path = _write_run_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["complexity_energetics"]["build_pdf"] = True
    path.write_text(yaml.safe_dump(raw))
    if any(shutil.which(name) is None for name in ("pandoc", "xelatex")):
        with pytest.raises(ValueError, match="PDF requires"):
            run_all(path, tasks="papers")
        assert not (tmp_path / "outputs").exists()
    else:
        result = run_all(path, tasks="papers")
        pdf = result.run_dir / "papers/complexity_energetics.pdf"
        assert pdf.read_bytes().startswith(b"%PDF-")
        assert any(record.path == "papers/complexity_energetics.pdf" and record.artifact_type == "paper" for record in result.artifacts)


def test_projection_failure_restores_prior_completed_run(tmp_path, monkeypatch):
    import importlib
    module = importlib.import_module("antstack_core.orchestration.run_all")
    path = _write_run_config(tmp_path, paper_projection=True)
    first = run_all(path, tasks="data")
    old_manifest = first.manifest_path.read_bytes()
    paper = tmp_path / "paper_projection"
    old_csv = (paper / "out/results.csv").read_bytes()
    original = module.os.replace
    # Fail once, so rollback of the old canonical tree remains available.
    failed = False
    def fail_once(source, destination):
        nonlocal failed
        if not failed and Path(destination) == first.run_dir and ".stage-" in Path(source).name:
            failed = True
            raise OSError("canonical commit failure")
        return original(source, destination)
    monkeypatch.setattr(module.os, "replace", fail_once)
    with pytest.raises(OSError, match="commit failure"):
        run_all(path, tasks="data")
    assert first.manifest_path.read_bytes() == old_manifest
    assert (paper / "out/results.csv").read_bytes() == old_csv



def test_error_log_level_can_publish_verified_immutable_log(tmp_path):
    config_path = _write_run_config(tmp_path)
    result = run_all(config_path, tasks='', log_level='ERROR')
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest['complete'] is True
    record = next(item for item in manifest['artifacts'] if item['artifact_type'] == 'log')
    log = result.run_dir / record['path']
    from hashlib import sha256
    assert sha256(log.read_bytes()).hexdigest() == record['checksum']
