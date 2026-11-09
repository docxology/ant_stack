"""End-to-end key-number consumers of the corrected enhanced producer."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from antstack_core.analysis.ce_tables_and_numbers import generate_key_numbers
from antstack_core.analysis.energy import EnergyCoefficients
from antstack_core.analysis.key_numbers import KeyNumbersLoader, KeyNumbersManager
from antstack_core.analysis.key_numbers_updater import PaperKeyNumbersUpdater


def test_fresh_producer_reaches_checkout_default_with_exact_context(tmp_path, monkeypatch):
    supplied = generate_key_numbers()
    root = tmp_path / "papers/complexity_energetics/generated_content"
    root.mkdir(parents=True)
    path = root / "key_numbers.json"
    path.write_text(json.dumps(supplied), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    loader = KeyNumbersLoader()
    assert loader.load().to_dict() == supplied
    coefficients = EnergyCoefficients(**loader.load().assumed_energy_coefficients)
    assert coefficients.to_dict() == supplied["assumed_energy_coefficients"]
    for component in ("body", "brain", "mind", "total"):
        assert loader.get_energy_value(component, "mJ") == supplied["per_decision_energy"][f"{component}_mj"]
        assert loader.get_energy_value(component, "J") == supplied["per_decision_energy"][f"{component}_mj"] / 1000
    manager = KeyNumbersManager(loader)
    status = manager.validate_key_numbers()
    assert status["valid"]
    assert status["provenance"] == supplied["provenance"]
    assert status["assumed_energy_coefficients"] == supplied["assumed_energy_coefficients"]
    assert not status["empirically_validated"]
    assert not status["canonical_base_evidence"]
    assert not status["source_freshness_verified"]
    section = tmp_path / "Results.md"
    section.write_text("{{key_numbers.provenance}}\nBody {{key_numbers.energy.body:.9f}} mJ")
    updater = PaperKeyNumbersUpdater(project_root=tmp_path)
    assert updater.update_results(section)
    assert updater.key_manager.update_paper_section(section)
    assert section.read_text() == supplied["provenance"] + "\nBody " + format(supplied["per_decision_energy"]["body_mj"], ".9f") + " mJ"




@pytest.mark.parametrize("historical", [
    {},
    {"per_decision_energy": {"body": 1.2, "brain": 0.05, "mind": 0.02, "total": 1.27}},
    {"per_decision_energy": {"body_mj": 0.5, "brain_mj": 2, "mind_mj": 1.5, "total_mj": 4},
     "computational_load": {"body_flops": 100000, "brain_flops": 500000, "mind_flops": 200000}},
])
def test_historical_numerical_subsets_cannot_certify_current_model(tmp_path, historical):
    path = tmp_path / "historical.json"
    path.write_text(json.dumps(historical))
    manager = KeyNumbersManager(path)
    assert not manager.validate_key_numbers()["valid"]
    with pytest.raises(ValueError):
        manager.loader.get_energy_value("body")


def test_canonical_projection_is_not_an_enhanced_input(tmp_path):
    # A base canonical run uses its own manifest/accounting contract. A plausible
    # numerical field name is not authority to substitute that family here.
    path = tmp_path / "canonical.json"
    path.write_text(json.dumps({"model_family": "base", "compute_energy_j": {"body": 1}}))
    with pytest.raises(ValueError):
        KeyNumbersLoader(path).load()


def test_stale_preferred_file_is_not_replaced_by_a_loose_fallback(tmp_path, monkeypatch):
    root = tmp_path / "papers/complexity_energetics/generated_content"
    root.mkdir(parents=True)
    (root / "key_numbers.json").write_text(json.dumps({"per_decision_energy": {"body_mj": 0.5}}))
    (tmp_path / "key_numbers.json").write_text(json.dumps(generate_key_numbers()))
    monkeypatch.chdir(tmp_path)
    manager = KeyNumbersManager(KeyNumbersLoader())
    assert not manager.validate_key_numbers()["valid"]
    with pytest.raises(ValueError):
        manager.generate_report()


@pytest.mark.parametrize("mode,placeholder,bucket", [
    ("--validate", "{{key_numbers.scaling.brain.exponent}}", "Missing"),
    ("--validate", "{{key_numbers.energy.body:not-a-format}}", "Invalid"),
    ("--validate", "{{key_numbers.scaling.body.exponent}}", "Invalid"),
    ("update", "{{key_numbers.scaling.brain.exponent}}", None),
    ("update", "{{key_numbers.energy.body:not-a-format}}", None),
    ("update", "{{key_numbers.scaling.body.exponent}}", None),
])
def test_actual_updater_cli_reports_failed_placeholders_without_changing_section(tmp_path, mode, placeholder, bucket):
    supplied = generate_key_numbers()
    if placeholder == "{{key_numbers.scaling.body.exponent}}":
        supplied["scaling_exponents"].update(
            body_energy=None, body_r_squared=None, body_regime="unavailable",
        )
    paper = tmp_path / "papers/complexity_energetics"
    generated = paper / "generated_content"
    generated.mkdir(parents=True)
    (generated / "key_numbers.json").write_text(json.dumps(supplied))
    section = paper / "Results.md"
    original = "Literal 123.45; {{key_numbers.energy.body:.9f}}\n" + placeholder
    section.write_text(original)
    wrapper = Path(__file__).resolve().parents[2] / "scripts/common_pipeline/update_paper_key_numbers.py"
    command = [sys.executable, "-B", str(wrapper), "--section", "results"]
    if mode == "--validate":
        command.append(mode)
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    if bucket:
        assert f"{bucket}: 1" in result.stdout
        assert placeholder.removeprefix("{{key_numbers.").removesuffix("}}") in result.stdout
    else:
        assert "Key-number operation failed:" in result.stdout
    assert section.read_text() == original


def test_actual_updater_cli_accepts_coherent_caller_input_and_exact_formats(tmp_path):
    supplied = generate_key_numbers()
    paper = tmp_path / "papers/complexity_energetics"
    generated = paper / "generated_content"
    generated.mkdir(parents=True)
    (generated / "key_numbers.json").write_text(json.dumps(supplied))
    section = paper / "Results.md"
    section.write_text("Literal 123.45; {{key_numbers.energy.body:.9f}}")
    wrapper = Path(__file__).resolve().parents[2] / "scripts/common_pipeline/update_paper_key_numbers.py"
    command = [sys.executable, "-B", str(wrapper), "--section", "results"]
    validated = subprocess.run([*command, "--validate"], cwd=tmp_path, capture_output=True, text=True)
    assert validated.returncode == 0, validated.stdout + validated.stderr
    updated = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert updated.returncode == 0, updated.stdout + updated.stderr
    assert section.read_text() == "Literal 123.45; {{key_numbers.per_decision_energy.body_mj:.9f}}"
    rendered = KeyNumbersManager(generated / "key_numbers.json").replace_placeholders(section.read_text())
    assert rendered == "Literal 123.45; " + format(supplied["per_decision_energy"]["body_mj"], ".9f")


@pytest.mark.parametrize("arguments", [
    ["--validate"],
    ["--validate", "--section", "results"],
    ["--section", "results"],
    [],
])
def test_actual_updater_cli_missing_section_inputs_fail(tmp_path, arguments):
    wrapper = Path(__file__).resolve().parents[2] / "scripts/common_pipeline/update_paper_key_numbers.py"
    result = subprocess.run(
        [sys.executable, "-B", str(wrapper), *arguments],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "section" in result.stdout.lower()


@pytest.mark.parametrize("payload", [None, "not JSON"])
def test_actual_updater_cli_missing_or_invalid_metric_input_fails(tmp_path, payload):
    paper = tmp_path / "papers/complexity_energetics"
    generated = paper / "generated_content"
    generated.mkdir(parents=True)
    if payload is not None:
        (generated / "key_numbers.json").write_text(payload)
    section = paper / "Results.md"
    original = "{{key_numbers.energy.body:.9f}}"
    section.write_text(original)
    wrapper = Path(__file__).resolve().parents[2] / "scripts/common_pipeline/update_paper_key_numbers.py"
    result = subprocess.run(
        [sys.executable, "-B", str(wrapper), "--validate", "--section", "results"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode != 0
    if payload is None:
        assert "Key-number operation failed:" in result.stdout
    else:
        assert "Invalid: 1" in result.stdout
    assert section.read_text() == original
