"""Contracts for the complexity energetics paper runner."""

from __future__ import annotations

from pathlib import Path
import json
import re

import pytest
import yaml

from antstack_core.analysis import ExperimentManifest, brain_workload_closed_form
from antstack_core.orchestration import complexity_energetics as runner


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "papers" / "complexity_energetics" / "manifest.example.yaml"




def test_scaling_workloads_honor_closed_form_manifest_mode() -> None:
    """Scaling sweeps should use deterministic closed-form loads when configured."""
    manifest = ExperimentManifest.load(MANIFEST_PATH)
    params = runner._workload_params(manifest, "brain")

    load_from_runner = runner._load_for_manifest_mode(manifest, "brain", 0.25, params)
    expected_load = brain_workload_closed_form(0.25, params)

    assert load_from_runner == expected_load




def _manifest(tmp_path, *, names=("body", "brain", "mind")):
    params = {"body": {"J": 6, "C": 4, "S": 12, "hz": 100, "contact_solver": "pgs"},
              "brain": {"K": 16, "N_KC": 32, "rho": 0.1, "H": 8, "hz": 100},
              "mind": {"B": 2, "H_p": 3, "state_dim": 4, "action_dim": 2, "hz": 100}}
    raw = {"experiment_name": "producer-contract", "seed": 7,
           "coefficients": {"baseline_w": 0.05},
           "workloads": {name: {"name": name, "duration_s": 0.01, "repeats": 2,
                                "mode": "closed_form", "params": params[name]} for name in names},
           "statistics": {"confidence_level": 0.9, "bootstrap_samples": 50},
           "meter": {"meter_type": "null"}}
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path, raw


@pytest.mark.parametrize("case", ["unknown", "mode", "repeats", "duration", "meter", "solver", "connectivity", "quoted_bool"])
def test_semantic_rejection_precedes_output_creation(tmp_path, case):
    path, raw = _manifest(tmp_path)
    if case == "unknown":
        raw["workloads"]["ghost"] = raw["workloads"]["brain"]
    elif case == "mode":
        raw["workloads"]["brain"]["mode"] = "invented"
    elif case == "repeats":
        raw["workloads"]["brain"]["repeats"] = 0
    elif case == "duration":
        raw["workloads"]["body"]["duration_s"] = float("nan")
    elif case == "meter":
        raw["meter"]["meter_type"] = "invented"
    elif case == "solver":
        raw["workloads"]["body"]["params"]["contact_solver"] = "lcp"
    elif case == "connectivity":
        raw["workloads"]["brain"]["params"]["connectivity_pattern"] = "biological"
    else:
        raw["workloads"]["mind"]["params"]["hierarchical"] = "false"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "artifact"
    with pytest.raises(ValueError):
        runner.run_manifest(str(path), str(output))
    assert not output.exists()


def test_direct_outputs_are_isolated_normalized_and_immutable(tmp_path):
    path, _ = _manifest(tmp_path, names=("brain",))
    first = runner.run_manifest(str(path), str(tmp_path / "A") + "/")
    second = runner.run_manifest(str(path), str(tmp_path / "B"))
    for result in (first, second):
        root = Path(result.output_dir)
        assert Path(result.generated_markdown_path).parent == root
        assert all(Path(output).is_relative_to(root) for output in result.output_paths)
        assert all(Path(output).is_file() for output in result.output_paths)
        assert len(result.output_paths) == len(set(result.output_paths))
    saved = Path(first.csv_path).read_bytes()
    with pytest.raises(FileExistsError):
        runner.run_manifest(str(path), first.output_dir)
    assert Path(first.csv_path).read_bytes() == saved
    assert not (tmp_path / "assets").exists()


def test_requested_repeat_failure_is_not_partial_success(tmp_path, monkeypatch):
    path, _ = _manifest(tmp_path)
    original = runner._load_for_manifest_mode
    def fail_mind(manifest, workload, duration_s, params):
        if workload == "mind":
            raise RuntimeError("required repeat failed")
        return original(manifest, workload, duration_s, params)
    monkeypatch.setattr(runner, "_load_for_manifest_mode", fail_mind)
    output = tmp_path / "artifacts"
    with pytest.raises(RuntimeError, match="required repeat failed"):
        runner.run_manifest(str(path), str(output))
    assert not output.exists()


def test_accounting_includes_mind_and_shared_idle_once(tmp_path):
    path, _ = _manifest(tmp_path)
    result = runner.run_manifest(str(path), str(tmp_path / "artifacts"))
    assert {(row["workload"], row["repeat"]) for row in result.rows} == {(name, repeat) for name in ("body", "brain", "mind") for repeat in range(2)}
    for row in result.rows:
        assert row["energy_est_j"] == pytest.approx(row["compute_energy_j"] + row["isolated_idle_energy_j"])
        assert row["measured_energy_j"] is None
        assert row["measurement_status"] == "not_measured"
        assert row["record_kind"] == "analytical"
    assert next(row["compute_energy_j"] for row in result.rows if row["workload"] == "mind") > 0
    text = Path(result.generated_markdown_path).read_text()
    table = text.split("| Component |", 1)[1].split("## Table", 1)[0]
    values = {name.strip(): float(value.strip()) for name, value in re.findall(r"^\| ([^|]+) \| ([\d.eE+-]+) \|$", table, re.M)}
    total = values.pop("Modeled system total")
    assert total == pytest.approx(sum(values.values()), rel=1e-6)
    assert values["Shared idle"] == pytest.approx(0.05*0.01*1000)
    assert values["mind"] > 0
    assert "Unavailable:" in text and "locomotion" in text
    assert "90% CI Low" in text
    assert "|---|---:|---:|---:|---:|" in text
    for link in re.findall(r"\]\(([^)]+)\)", text):
        assert not Path(link).is_absolute()
        assert (Path(result.generated_markdown_path).parent / link).is_file()


def test_projection_preserves_guides_and_removes_only_owned_obsolete(tmp_path):
    path, _ = _manifest(tmp_path)
    first = runner.run_manifest(str(path), str(tmp_path / "first"))
    paper = tmp_path / "paper"
    (paper / "assets").mkdir(parents=True)
    (paper / "README.md").write_text("paper guide")
    (paper / "assets/README.md").write_text("asset guide")
    (paper / "assets/unrelated.png").write_bytes(b"unowned")
    runner.project_result(first, paper)
    assert (paper / "assets/body_split.png").is_file()
    path, _ = _manifest(tmp_path, names=("brain",))
    second = runner.run_manifest(str(path), str(tmp_path / "second"))
    runner.project_result(second, paper)
    assert not (paper / "assets/body_split.png").exists()
    assert (paper / "README.md").read_text() == "paper guide"
    assert (paper / "assets/README.md").read_text() == "asset guide"
    assert (paper / "assets/unrelated.png").read_bytes() == b"unowned"
    text = (paper / "Generated.md").read_text()
    for link in re.findall(r"\]\(([^)]+)\)", text):
        assert (paper / link).is_file()
    assert list(tmp_path.glob(".paper.archive-*"))


def test_projection_rejects_link_without_touching_previous(tmp_path):
    path, _ = _manifest(tmp_path, names=("brain",))
    result = runner.run_manifest(str(path), str(tmp_path / "fresh"))
    paper = tmp_path / "paper"
    paper.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    (paper / "assets").symlink_to(victim, target_is_directory=True)
    with pytest.raises(ValueError, match="ordinary"):
        runner.project_result(result, paper)
    assert not list(victim.iterdir())



