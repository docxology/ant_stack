"""End-to-end key-number consumers of the corrected enhanced producer."""

import json
from pathlib import Path

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
