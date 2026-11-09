"""Real-format, isolated-model and complete-family publication regressions."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from antstack_core.analysis import EnergyCoefficients
from antstack_core.analysis import ce_manuscript_figures as manuscript
from antstack_core.analysis import ce_results_figures as results
from antstack_core.publishing.provenance import file_identity


def _args(output, image_format="png"):
    return SimpleNamespace(output=str(output), format=image_format)


def _bytes(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def _assert_inventory(root, manifest_name, count, image_format):
    manifest = json.loads((root / manifest_name).read_text())
    assert manifest["complete"] is True
    assert manifest["generation_info"]["format"] == image_format
    figures = manifest["figures"]
    assert len(figures) == count
    assert set(figures.values()) == set(manifest["figure_models"])
    inventory = json.loads((root / "provenance/output_inventory.json").read_text())
    assert inventory["artifact_path_base"] == "family_root"
    paths = [artifact["path"] for artifact in inventory["artifacts"]]
    assert len(paths) == len(set(paths))
    assert set(paths) == set(_bytes(root)) - {"provenance/output_inventory.json"}
    for artifact in inventory["artifacts"]:
        assert not Path(artifact["path"]).is_absolute()
        data = (root / artifact["path"]).read_bytes()
        assert artifact["bytes"] == len(data)
        assert artifact["checksum"] == hashlib.sha256(data).hexdigest()
    provenance = json.loads((root / "provenance/provenance.json").read_text())
    assert set(provenance["output_paths"]) == set(_bytes(root))
    implementation = provenance["source_state"]["generating_package"]["files"]
    for module in (manuscript, results):
        relative = "analysis/" + Path(module.__file__).name
        assert implementation[relative] == file_identity(module.__file__)
    return manifest


@pytest.mark.parametrize("image_format,signature", [("png", b"\x89PNG"), ("svg", b"<?xml"), ("pdf", b"%PDF-")])
def test_manuscript_complete_family_has_requested_real_format(tmp_path, image_format, signature):
    root = tmp_path / "manuscript"
    manuscript.run(_args(root, image_format))
    manifest = _assert_inventory(root, "figure_manifest.json", 15, image_format)
    for name in manifest["figures"].values():
        assert Path(name).suffix == "." + image_format
        assert (root / name).read_bytes().startswith(signature)


def test_results_family_keeps_distinct_models_and_cross_suite_bytes(tmp_path):
    root = tmp_path / "results"
    root.mkdir()  # An explicitly selected empty root is also supported.
    results.run(_args(root))
    _assert_inventory(root, "results_figure_model.json", 13, "png")
    previous = _bytes(root)
    with pytest.raises(ValueError, match="absent or empty"):
        manuscript.run(_args(root, "svg"))
    assert _bytes(root) == previous


def test_results_refuses_completed_manuscript_family(tmp_path):
    root = tmp_path / "manuscript"
    manuscript.run(_args(root))
    previous = _bytes(root)
    with pytest.raises(ValueError, match="absent or empty"):
        results.run(_args(root))
    assert _bytes(root) == previous


@pytest.mark.parametrize("module", [manuscript, results])
@pytest.mark.parametrize("kind", ["occupied", "hardlinked-metadata", "symlink", "linked-ancestor", "git", "runtime", "node_modules", "package"])
def test_refused_destinations_never_mutate_prior_bytes(tmp_path, module, kind):
    sentinel = tmp_path / "prior.json"
    sentinel.write_bytes(b'{"prior":"must survive"}\n')
    root = tmp_path / "family"
    if kind == "occupied":
        root.mkdir()
        (root / "body_energy_complexity.png").write_bytes(b"historical figure")
    elif kind == "hardlinked-metadata":
        root.mkdir()
        os.link(sentinel, root / "figure_manifest.json")
    elif kind == "symlink":
        target = tmp_path / "target"
        target.mkdir()
        root.symlink_to(target, target_is_directory=True)
    elif kind == "linked-ancestor":
        target = tmp_path / "target"
        target.mkdir()
        root.symlink_to(target, target_is_directory=True)
        root = root / "child"
    elif kind == "git":
        root = tmp_path / ".git" / "family"
    elif kind == "runtime":
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        (runtime / "pyvenv.cfg").write_text("home = controlled fixture\n")
        root = runtime / "family"
    elif kind == "node_modules":
        root = tmp_path / "node_modules" / "family"
    else:
        root = Path(manuscript.__file__).parents[1] / "refused-family-fixture"
    previous = _bytes(tmp_path)
    with pytest.raises(ValueError):
        module.run(_args(root))
    assert _bytes(tmp_path) == previous
    assert sentinel.read_bytes() == b'{"prior":"must survive"}\n'
    if kind == "package":
        assert not root.exists()
    assert not list(tmp_path.glob(".*.stage-*"))


@pytest.mark.parametrize("module", [manuscript, results])
@pytest.mark.parametrize("existing_empty", [False, True])
def test_failure_after_real_figures_never_publishes_partial_family(tmp_path, monkeypatch, module, existing_empty):
    root = tmp_path / "family"
    if existing_empty:
        root.mkdir()
        root.chmod(0o750)
    previous_mode = root.stat().st_mode if existing_empty else None

    def fail_late(stage, *args, **kwargs):
        assert (stage / "body_joint_energy_scaling.png").read_bytes().startswith(b"\x89PNG")
        assert len(list(stage.glob("*.png"))) >= 11
        raise RuntimeError("deterministic late-producer failure")

    monkeypatch.setattr(module, "generate_contact_solver_figures", fail_late)
    with pytest.raises(RuntimeError, match="late-producer failure"):
        module.run(_args(root))
    if existing_empty:
        assert list(root.iterdir()) == []
        assert root.stat().st_mode == previous_mode
    else:
        assert not root.exists()
    assert not list(tmp_path.glob(".*.stage-*"))


def test_nonfinite_metadata_failure_keeps_destination_absent(tmp_path, monkeypatch):
    root = tmp_path / "family"
    monkeypatch.setattr(manuscript, "generate_complexity_overview_data", lambda: {"invalid": float("nan")})
    with pytest.raises(ValueError):
        manuscript.run(_args(root))
    assert not root.exists()
    assert not list(tmp_path.glob(".*.stage-*"))


def test_failed_final_rename_keeps_empty_destination_and_discards_complete_stage(tmp_path, monkeypatch):
    root = tmp_path / "family"
    root.mkdir()
    root.chmod(0o750)
    previous_mode = root.stat().st_mode
    real_replace = manuscript.os.replace

    def refuse_family(source, destination):
        if Path(destination) == root:
            stage = Path(source)
            _assert_inventory(stage, "results_figure_model.json", 13, "png")
            raise OSError("deterministic final-publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(manuscript.os, "replace", refuse_family)
    with pytest.raises(OSError, match="final-publication failure"):
        results.run(_args(root))
    assert list(root.iterdir()) == []
    assert root.stat().st_mode == previous_mode
    assert not list(tmp_path.glob(".*.stage-*"))


def test_unsupported_format_fails_before_creating_any_output(tmp_path):
    root = tmp_path / "family"
    with pytest.raises(ValueError, match="Unsupported figure format"):
        manuscript.run(_args(root, "jpeg"))
    assert list(tmp_path.iterdir()) == []


def test_sparse_only_results_energy_is_not_silently_full_brain(tmp_path, monkeypatch):
    real_line = results.line_plot
    captured = {}

    def retain_real_plot(x, ys, labels, title, xlabel, ylabel, path):
        captured["sparsities"] = list(x)
        captured["energies"] = list(ys[0])
        return real_line(x, ys, labels, title, xlabel, ylabel, path)

    monkeypatch.setattr(results, "line_plot", retain_real_plot)
    results.generate_connectivity_analysis_figures(tmp_path, EnergyCoefficients(baseline_w=0.05))
    expected = []
    for rho in captured["sparsities"]:
        active = int(50000 * rho)
        memory = active * 100 * 4 + 50000 * 4
        expected.append(active ** 2 * 1e-12 + memory * 0.1e-12 + active * 0.1 * 1e-18 + 0.05 * 0.01)
    assert captured["energies"] == pytest.approx(expected, rel=1e-12, abs=0)
    assert (tmp_path / "sparsity_energy_scaling.png").read_bytes().startswith(b"\x89PNG")
    # The complete enhanced Brain additionally has AL/CX/plasticity and 15% DRAM.
    active = int(50000 * captured["sparsities"][0])
    sparse_memory = active * 100 * 4 + 50000 * 4
    full_flops = 128 * 15 + active ** 2 + 64 * 12 + 64 ** 2 * 0.5 + active * 0.1 * 5
    full_memory = 128 * 8 + sparse_memory + 64 * 4 + active * 4
    full_energy = full_flops * 1e-12 + full_memory * (0.85 * 0.1 + 0.15 * 20) * 1e-12 + active * 0.1e-18 + 0.05 * 0.01
    assert abs(captured["energies"][0] - full_energy) > 1e-8


def test_manuscript_module_components_sum_three_isolated_idle_charges(tmp_path, monkeypatch):
    real_bar = manuscript.bar_plot
    values = {}

    def retain_real_plot(labels, heights, title, path, **kwargs):
        values[Path(path).stem] = list(heights)
        return real_bar(labels, heights, title, path, **kwargs)

    monkeypatch.setattr(manuscript, "bar_plot", retain_real_plot)
    manuscript.generate_energy_breakdown_figures(tmp_path, image_format="svg")
    components = values["energy_component_breakdown"]
    assert components[-1] == pytest.approx(3 * 0.5 * 0.01 * 1000)
    assert sum(components) == pytest.approx(sum(values["module_energy_comparison"]))
    assert (tmp_path / "energy_component_breakdown.svg").read_bytes().startswith(b"<?xml")


@pytest.mark.parametrize("wrapper", ["generate_manuscript_figures.py", "generate_results_figures.py", "generate_publication_figures.py", "generate_multipanel_figures.py", "generate_tables_and_numbers.py"])
def test_actual_wrappers_require_output_and_refuse_occupied_family(tmp_path, wrapper):
    script = Path(__file__).parents[2] / "scripts/complexity_energetics" / wrapper
    historical = tmp_path / "complexity_energetics" / "assets"
    historical.mkdir(parents=True)
    (historical / "prior.png").write_bytes(b"historical default must survive")
    default_previous = _bytes(tmp_path)
    missing = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True, check=False)
    assert missing.returncode != 0
    assert _bytes(tmp_path) == default_previous
    root = tmp_path / "occupied"
    root.mkdir()
    (root / "historical.png").write_bytes(b"original historical bytes")
    previous = _bytes(root)
    refused = subprocess.run([sys.executable, str(script), "--output", str(root)], cwd=tmp_path,
                             capture_output=True, text=True, check=False)
    assert refused.returncode != 0
    assert _bytes(root) == previous


@pytest.mark.parametrize("existing_empty", [False, True])
def test_enhanced_supplement_failure_never_publishes_partial_family(tmp_path, monkeypatch, existing_empty):
    from antstack_core.analysis import ce_tables_and_numbers as tables
    from antstack_core.publishing import provenance
    root = tmp_path / "supplements"
    if existing_empty:
        root.mkdir(mode=0o750)
    previous_mode = root.stat().st_mode if existing_empty else None
    original = provenance.atomic_text
    calls = 0
    def fail_after_table(path, content):
        nonlocal calls
        calls += 1
        if calls == 3:
            assert len(list(Path(path).parent.glob("*.md"))) == 2
            raise OSError("injected supplement write failure")
        return original(path, content)
    monkeypatch.setattr(provenance, "atomic_text", fail_after_table)
    with pytest.raises(OSError, match="supplement write failure"):
        tables.main(["--output", str(root)])
    if existing_empty:
        assert list(root.iterdir()) == []
        assert root.stat().st_mode == previous_mode
    else:
        assert not root.exists()
    assert not list(tmp_path.glob(".*.stage-*"))


def test_enhanced_supplements_refuse_occupied_family_and_nonfinite_payload(tmp_path, monkeypatch):
    from antstack_core.analysis import ce_tables_and_numbers as tables
    root = tmp_path / "operator-results"
    root.mkdir()
    (root / "key_numbers.json").write_bytes(b"retained operator data\n")
    previous = _bytes(root)
    with pytest.raises(ValueError):
        tables.main(["--output", str(root)])
    assert _bytes(root) == previous
    fresh = tmp_path / "nonfinite"
    monkeypatch.setattr(tables, "generate_key_numbers", lambda: {"energy_j": float("nan")})
    with pytest.raises(ValueError):
        tables.main(["--output", str(fresh)])
    assert not fresh.exists()
    assert not list(tmp_path.glob(".*.stage-*"))


def test_tex_artifact_keeps_commands_literal_and_counters_attached(tmp_path):
    from antstack_core.analysis.ce_multipanel_figures import generate_latex_figure_code
    generate_latex_figure_code({"comprehensive_scaling": "scaling.png", "theoretical_limits": "references.png"}, tmp_path)
    document = (tmp_path / "multipanel_figures.tex").read_text()
    assert "\t" not in document and "\b" not in document
    assert document.count(r"\caption{\textbf{") == 2
    for key, image in (("comprehensive_scaling", "scaling.png"), ("theoretical_limits", "references.png")):
        assert r"\includegraphics[width=\linewidth]{" + image + "}" in document
        assert r"\label{fig:" + key + "_multipanel}" in document
