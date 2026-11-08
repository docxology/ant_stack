"""Consumer-visible enhanced-input boundaries; no empirical certification."""

import json
import os

import pytest

from antstack_core.analysis.ce_tables_and_numbers import generate_key_numbers
from antstack_core.analysis.key_numbers import (
    KeyNumbersData, KeyNumbersLoader, KeyNumbersManager,
    get_key_numbers_loader, get_key_numbers_manager,
)
from antstack_core.analysis.key_numbers_updater import PaperKeyNumbersUpdater


@pytest.fixture
def supplied():
    return generate_key_numbers()


@pytest.fixture
def source(tmp_path, supplied):
    path = tmp_path / "enhanced.json"
    path.write_text(json.dumps(supplied), encoding="utf-8")
    return path


def test_round_trip_and_manager_edit_retain_model_context(source, supplied):
    manager = KeyNumbersManager(source)
    restored = KeyNumbersData.from_dict(manager.loader.load().to_dict())
    assert restored.to_dict() == supplied
    # A coherent edit must not strip provenance or coefficients on disk.
    manager.update_value("scaling_exponents.body_regime", "descriptive caller annotation")
    written = json.loads(source.read_text())
    assert written["provenance"] == supplied["provenance"]
    assert written["assumed_energy_coefficients"] == supplied["assumed_energy_coefficients"]
    assert manager.get_value("provenance") == supplied["provenance"]
    assert manager.get_value("assumed_energy_coefficients") == supplied["assumed_energy_coefficients"]
    assert manager.get_scaling_analysis()["provenance"] == supplied["provenance"]
    report = manager.generate_report()
    assert supplied["provenance"] in report
    assert '"assumed_energy_coefficients"' in report
    summary = manager.loader.get_summary()
    assert summary["total_energy_mj"] == supplied["per_decision_energy"]["total_mj"]
    assert summary["assumed_energy_coefficients"] == supplied["assumed_energy_coefficients"]
    validation = manager.validate_key_numbers()
    assert validation["valid"]
    assert validation["validation_scope"] == "enhanced_model_input_coherence"
    for evidence in (summary, validation, manager.get_scaling_analysis()):
        assert evidence["empirically_validated"] is False
        assert evidence["canonical_base_evidence"] is False
        assert evidence["source_freshness_verified"] is False


@pytest.mark.parametrize("missing", [
    "schema_version", "model_family", "provenance", "assumed_energy_coefficients", "per_decision_energy",
    "computational_load", "scaling_exponents", "system_parameters",
])
def test_historical_or_partial_input_never_certifies(source, supplied, missing):
    del supplied[missing]
    source.write_text(json.dumps(supplied))
    manager = KeyNumbersManager(source)
    with pytest.raises(ValueError):
        manager.loader.load()
    valid, errors = manager.validate_data()
    assert not valid and errors
    assert manager.validate_key_numbers()["empirically_validated"] is False


@pytest.mark.parametrize("section,key,value", [
    ("per_decision_energy", "total_mj", 1),
    ("per_decision_energy", "body_mj", float("nan")),
    ("per_decision_energy", "brain_mj", True),
    ("computational_load", "mind_flops", -1),
    ("system_parameters", "control_frequency_hz", 0),
    ("system_parameters", "decision_period_ms", 20),
    ("system_parameters", "baseline_power_mw", 0),
    ("system_parameters", "total_power_w", 0),
    ("system_parameters", "total_power_boundary", "whole-device measured power"),
    ("assumed_energy_coefficients", "baseline_w", 0),
    ("scaling_exponents", "body_r_squared", 1.5),
])
def test_malformed_or_incoherent_input_is_not_rendered(source, supplied, section, key, value):
    supplied[section][key] = value
    source.write_text(json.dumps(supplied))
    manager = KeyNumbersManager(source)
    assert manager.validate_key_numbers()["valid"] is False
    with pytest.raises(ValueError):
        manager.replace_placeholders("{{key_numbers.energy.body:.6f}}")


def test_algebraically_consistent_energy_cannot_ignore_idle(source, supplied):
    energy = supplied["per_decision_energy"]
    energy["body_mj"] = 0
    energy["total_mj"] = energy["brain_mj"] + energy["mind_mj"]
    supplied["system_parameters"]["total_power_w"] = energy["total_mj"] / 10
    source.write_text(json.dumps(supplied))
    assert KeyNumbersManager(source).validate_key_numbers()["valid"] is False


@pytest.mark.parametrize("payload", ["{}", "[]", "null", "not JSON"])
def test_missing_or_malformed_evidence_fails_honestly(source, payload):
    source.write_text(payload)
    manager = KeyNumbersManager(source)
    assert manager.validate_data()[0] is False
    with pytest.raises(ValueError):
        manager.loader.load()
    source.unlink()
    assert manager.validate_key_numbers()["valid"] is False
    with pytest.raises(FileNotFoundError):
        manager.loader.load()


def test_cached_input_cannot_hide_same_mtime_corruption(source, supplied):
    loader = KeyNumbersLoader(source)
    loader.load()
    before = source.stat()
    supplied["per_decision_energy"]["total_mj"] *= 2
    source.write_text(json.dumps(supplied))
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError):
        loader.load()


def test_returned_mutation_cannot_relabel_backing_input(source, supplied):
    loader = KeyNumbersLoader(source)
    data = loader.load()
    data.assumed_energy_coefficients["baseline_w"] = 999
    data.per_decision_energy["total_mj"] = 0
    assert loader.load().to_dict() == supplied


def test_rejected_edit_preserves_file_and_coherent_edit_is_atomic(source, supplied):
    manager = KeyNumbersManager(source)
    original = source.read_bytes()
    with pytest.raises(ValueError):
        manager.update_value("per_decision_energy.body_mj", 999)
    assert source.read_bytes() == original
    with pytest.raises(KeyError):
        manager.update_multiple_values({"scaling_exponents.body_regime": "changed", "absent.path": 1})
    assert source.read_bytes() == original
    energy = supplied["per_decision_energy"]
    delta = 1
    manager.update_multiple_values({
        "per_decision_energy.body_mj": energy["body_mj"] + delta,
        "per_decision_energy.total_mj": energy["total_mj"] + delta,
        "system_parameters.total_power_w": (energy["total_mj"] + delta) / 10,
    })
    assert manager.calculate_total_energy() == energy["total_mj"] + delta
    assert manager.validate_data() == (True, [])


def test_unavailable_fit_and_absent_module_do_not_become_zero(source, supplied):
    supplied["scaling_exponents"] = {
        "body_energy": None, "body_r_squared": None, "body_regime": "unavailable",
    }
    source.write_text(json.dumps(supplied))
    manager = KeyNumbersManager(source)
    assert manager.get_value("scaling_exponents.body_energy") is None
    assert manager.loader.get_scaling_info("body")["exponent"] is None
    with pytest.raises(ValueError):
        manager.replace_placeholders("{{key_numbers.scaling.body.exponent}}")
    for operation in (
        lambda: manager.loader.get_scaling_info("brain"),
        lambda: manager.loader.get_computational_load("body", "spikes"),
        lambda: manager.get_formatted_value("absent.path", ".2f"),
    ):
        with pytest.raises(KeyError):
            operation()


def test_default_does_not_silently_pick_loose_or_canonical_files(tmp_path, supplied, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "key_numbers.json").write_text(json.dumps(supplied))
    loose = tmp_path / "generated_content"
    loose.mkdir()
    (loose / "key_numbers.json").write_text(json.dumps(supplied))
    for constructor in (KeyNumbersLoader, get_key_numbers_loader, get_key_numbers_manager):
        with pytest.raises(FileNotFoundError):
            constructor()


def test_default_is_bound_to_caller_checkout_not_a_previous_project(tmp_path, supplied, monkeypatch):
    monkeypatch.chdir(tmp_path)
    enhanced = tmp_path / "papers/complexity_energetics/generated_content"
    enhanced.mkdir(parents=True)
    path = enhanced / "key_numbers.json"
    path.write_text(json.dumps(supplied))
    assert get_key_numbers_loader().load().to_dict() == supplied
    other = tmp_path / "other-project"
    other.mkdir()
    monkeypatch.chdir(other)
    with pytest.raises(FileNotFoundError):
        get_key_numbers_manager()


def test_updater_uses_owned_project_and_never_rewrites_literals(tmp_path, source, supplied):
    updater = PaperKeyNumbersUpdater(tmp_path, json_path=source)
    literal = "Body R^2=0.91; Brain R^2=0.72; 2.0 ± 0.1 mJ; 123 FLOPs."
    section = tmp_path / "Results.md"
    section.write_text(literal + " Body {{key_numbers.scaling.body.exponent:.4f}}")
    assert updater.update_results(section)
    assert section.read_text() == literal + " Body {{key_numbers.scaling_exponents.body_energy:.4f}}"
    expected = format(supplied["scaling_exponents"]["body_energy"], ".4f")
    assert updater.key_manager.replace_placeholders(section.read_text()) == literal + " Body " + expected
    section.write_text("{{key_numbers.scaling.brain.exponent}}")
    result = updater.validate_key_numbers_placeholders(section)
    assert not result["valid"] and result["missing"]
    original = section.read_bytes()
    with pytest.raises(KeyError):
        updater.update_results(section)
    assert section.read_bytes() == original


@pytest.mark.parametrize("section,key", [
    ("per_decision_energy", "mind_mj"),
    ("per_decision_energy", "total_mj"),
    ("computational_load", "brain_flops"),
    ("assumed_energy_coefficients", "spike_aj"),
    ("system_parameters", "total_power_boundary"),
])
def test_context_cannot_turn_a_missing_required_metric_into_success(source, supplied, section, key):
    del supplied[section][key]
    source.write_text(json.dumps(supplied))
    manager = KeyNumbersManager(source)
    assert not manager.validate_key_numbers()["valid"]
    with pytest.raises(ValueError):
        manager.calculate_total_energy()


def test_mutated_data_cannot_serialize_or_expose_invalid_context(supplied):
    data = KeyNumbersData.from_dict(supplied)
    data.assumed_energy_coefficients.clear()
    with pytest.raises(ValueError):
        data.to_dict()
    with pytest.raises(ValueError):
        data.model_context()


@pytest.mark.parametrize("key,value", [("schema_version", 2), ("schema_version", True), ("model_family", "canonical_base")])
def test_unsupported_machine_schema_cannot_certify_coherence(source, supplied, key, value):
    supplied[key] = value
    source.write_text(json.dumps(supplied))
    assert KeyNumbersManager(source).validate_key_numbers()["valid"] is False
