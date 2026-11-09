"""Behavioral regressions for the standalone comparative analysis publishers."""

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from antstack_core.analysis import ce_active_inference as active
from antstack_core.analysis import ce_contact_dynamics as contact
from antstack_core.analysis import ce_neural_networks as neural
from antstack_core.analysis.energy import EnergyCoefficients


ROOT = Path(__file__).resolve().parents[2]
CASES = (
    (contact, "contact_dynamics_analysis.json", "max_contacts", 50, 1, 4),
    (neural, "neural_network_analysis.json", "max_size", 100000, 1, 6),
    (active, "active_inference_analysis.json", "max_horizon", 15, 5, 7),
)


def _args(module, output, maximum):
    option = next(case[2] for case in CASES if case[0] is module)
    values = {"output": output, option: maximum}
    if module is active:
        values["branching"] = 4
    return SimpleNamespace(**values)


def _wrapper(module, *arguments):
    script = ROOT / "scripts" / "complexity_energetics" / (
        "analyze_" + module.__name__.rsplit(".", 1)[-1].removeprefix("ce_") + ".py"
    )
    environment = {**os.environ, "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run(
        [sys.executable, "-B", str(script), *map(str, arguments)],
        cwd=ROOT, env=environment, capture_output=True, text=True, check=False,
    )


def _assert_family(output, result_name, plot_count):
    manifest = json.loads((output / "analysis_manifest.json").read_text())
    assert manifest["complete"] is True
    assert manifest["artifact_path_base"] == "family_root"
    expected_products = {result_name, *(path.relative_to(output).as_posix()
                                      for path in (output / "plots").glob("*.png"))}
    assert len(expected_products) == plot_count + 1
    assert set(manifest["required_paths"]) == expected_products
    inventory_path = output / "provenance" / "output_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    assert inventory["complete"] is True
    artifacts = inventory["artifacts"]
    assert {item["path"] for item in artifacts} == {
        path.relative_to(output).as_posix() for path in output.rglob("*")
        if path.is_file() and path != inventory_path
    }
    for item in artifacts:
        relative = Path(item["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        data = (output / relative).read_bytes()
        assert len(data) == item["bytes"] > 0
        assert hashlib.sha256(data).hexdigest() == item["checksum"]
        if relative.suffix == ".png":
            assert data.startswith(b"\x89PNG\r\n\x1a\n")

    result = json.loads((output / result_name).read_text())

    def check_links(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "plots":
                    for link in item.values():
                        relative = Path(link)
                        assert not relative.is_absolute() and ".." not in relative.parts
                        assert (output / relative).is_file()
                else:
                    check_links(item)
        elif isinstance(value, list):
            for item in value:
                check_links(item)

    check_links(result)
    return result


def _assert_bounds(module, result, maximum):
    config = result["configuration"]
    if module is contact:
        grid = config["contact_counts"]
        for solver in ("pgs", "si", "ldlt"):
            assert result["solver_analysis"][solver]["contact_counts"] == grid
        for name in ("Smooth", "Rough", "Steep", "Soft"):
            assert 0 < result["terrain_analysis"][name]["params"]["C"] <= maximum
    elif module is neural:
        grid = config["network_sizes"]
        assert result["brain_scaling_analysis"]["summary"]["network_size"] <= maximum
        assert 0 < config["reference_network_size"] <= maximum
        for pattern in ("random", "structured"):
            assert result["connectivity_analysis"][pattern]["network_sizes"] == grid
    else:
        grid = config["horizons"]
        for name in ("B_2", "B_3", "B_4", "B_5", "B_6"):
            assert result["branching_analysis"][name]["horizon"] <= maximum
        for name in ("No Bounds", "Policy Sampling", "Hierarchical", "Conservative"):
            assert result["bounded_rationality_analysis"][name]["horizon"] <= maximum
    assert grid == sorted(set(grid))
    assert all(0 < value <= maximum for value in grid)
    assert grid[-1] == maximum


@pytest.mark.parametrize("module,result_name,option,default,small,plot_count", CASES)
def test_default_wrapper_publishes_current_byte_portable_family(
    tmp_path, module, result_name, option, default, small, plot_count
):
    output = tmp_path / "default-family"
    completed = _wrapper(module, "--output", output)
    assert completed.returncode == 0, completed.stderr
    result = _assert_family(output, result_name, plot_count)
    _assert_bounds(module, result, default)
    relocated = tmp_path / "relocated-family"
    output.rename(relocated)
    _assert_family(relocated, result_name, plot_count)


@pytest.mark.parametrize("module,result_name,option,default,small,plot_count", CASES)
def test_small_cap_has_no_invented_fit_and_preserves_empty_root_mode(
    tmp_path, module, result_name, option, default, small, plot_count
):
    output = tmp_path / "small-family"
    output.mkdir(mode=0o750)
    output.chmod(0o750)
    arguments = _args(module, output, small)
    original_arguments = vars(arguments).copy()
    assert module.run(arguments) == output / "analysis_manifest.json"
    assert vars(arguments) == original_arguments
    assert stat.S_IMODE(output.stat().st_mode) == 0o750
    result = _assert_family(output, result_name, plot_count)
    _assert_bounds(module, result, small)
    if module is contact:
        fits = [result["solver_analysis"][solver]["flops_scaling"]
                for solver in ("pgs", "si", "ldlt")]
    elif module is neural:
        fits = [result["connectivity_analysis"][pattern]["flops_scaling"]
                for pattern in ("random", "structured")]
    else:
        fits = [result["horizon_analysis"]["summary"]["energy_scaling"]]
    for fit in fits:
        assert fit["valid"] is False
        assert fit["error"]
        assert "scaling_exponent" not in fit


@pytest.mark.parametrize("module,result_name,option,default,small,plot_count", CASES)
@pytest.mark.parametrize("existing_empty_root", (False, True))
def test_real_late_producer_failure_does_not_publish_or_change_empty_root(
    tmp_path, monkeypatch, module, result_name, option, default, small, plot_count,
    existing_empty_root,
):
    output = tmp_path / "failed-family"
    if existing_empty_root:
        output.mkdir(mode=0o750)
        output.chmod(0o750)
    producer = module._produce

    def fail_after_real_products(stage, *arguments):
        producer(stage, *arguments)
        assert (stage / result_name).is_file()
        assert len(list((stage / "plots").glob("*.png"))) == plot_count
        raise RuntimeError("deterministic failure after real analysis products")

    monkeypatch.setattr(module, "_produce", fail_after_real_products)
    with pytest.raises(RuntimeError, match="deterministic failure"):
        module.run(_args(module, output, small))
    if existing_empty_root:
        assert list(output.iterdir()) == []
        assert stat.S_IMODE(output.stat().st_mode) == 0o750
    else:
        assert not output.exists()
    assert not list(tmp_path.glob(".failed-family.stage-*"))


@pytest.mark.parametrize("module,result_name,option,default,small,plot_count", CASES)
def test_missing_occupied_link_and_runtime_package_roots_refuse_without_writes(
    tmp_path, module, result_name, option, default, small, plot_count
):
    with pytest.raises(ValueError):
        module.run(_args(module, None, small))
    occupied = tmp_path / "occupied"
    occupied.mkdir(mode=0o750)
    sentinel = occupied / result_name
    sentinel.write_bytes(b"retain previous product bytes\x00")
    before = (sentinel.read_bytes(), occupied.stat().st_mode)
    with pytest.raises(ValueError):
        module.run(_args(module, occupied, small))
    assert (sentinel.read_bytes(), occupied.stat().st_mode) == before
    assert list(occupied.iterdir()) == [sentinel]

    target = tmp_path / "link-target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    marker = runtime / "pyvenv.cfg"
    marker.write_bytes(b"retain runtime marker\n")
    package = Path(module.__file__).resolve().parents[1] / tmp_path.name
    for forbidden in (linked, runtime / "output", package):
        with pytest.raises(ValueError):
            module.run(_args(module, forbidden, small))
    assert list(target.iterdir()) == []
    assert marker.read_bytes() == b"retain runtime marker\n"
    assert not (runtime / "output").exists()
    assert not package.exists()


@pytest.mark.parametrize("module,result_name,option,default,small,plot_count", CASES)
def test_actual_wrapper_requires_output_and_refuses_occupied_root(
    tmp_path, module, result_name, option, default, small, plot_count
):
    missing = _wrapper(module)
    assert missing.returncode == 2
    assert "--output" in missing.stderr
    output = tmp_path / "occupied-wrapper"
    output.mkdir()
    sentinel = output / result_name
    sentinel.write_bytes(b"existing wrapper product")
    completed = _wrapper(module, "--output", output)
    assert completed.returncode == 2
    assert sentinel.read_bytes() == b"existing wrapper product"
    assert list(output.iterdir()) == [sentinel]


@pytest.mark.parametrize("module,flag,value", (
    (contact, "--solver", "pgs"),
    (neural, "--pattern", "random"),
))
def test_removed_label_only_options_are_not_accepted(tmp_path, module, flag, value):
    output = tmp_path / "not-published"
    completed = _wrapper(module, "--output", output, flag, value)
    assert completed.returncode == 2
    assert not output.exists()


@pytest.mark.parametrize("module,maximum,branching", (
    (contact, 0, None), (contact, -1, None), (contact, True, None),
    (contact, 1.5, None), (neural, 0, None), (neural, -1, None),
    (neural, True, None), (neural, 1.5, None),
    (active, 4, 4), (active, True, 4), (active, 5.5, 4),
    (active, 5, 0), (active, 5, -1), (active, 5, True), (active, 5, 1.5),
))
def test_invalid_domains_refuse_before_publication(tmp_path, module, maximum, branching):
    output = tmp_path / "not-published"
    arguments = _args(module, output, maximum)
    if module is active:
        arguments.branching = branching
    with pytest.raises(ValueError):
        module.run(arguments)
    assert not output.exists()
    assert not list(tmp_path.glob(".not-published.stage-*"))


def test_empty_sampled_below_cut_set_has_no_fabricated_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(active, "EnergyCoefficients", lambda: EnergyCoefficients(baseline_w=200.0))
    output = tmp_path / "high-baseline"
    active.run(_args(active, output, 5))
    result = _assert_family(output, "active_inference_analysis.json", 7)
    assert result["horizon_analysis"]["H_p_5"]["energy_total"] >= 2.0
    summary = result["sampled_model_energy_summary"]
    assert summary["cutoff_j"] == 1.0
    assert summary["horizons_strictly_below_cutoff"] == []
    assert summary["largest_sampled_horizon_below_cutoff"] is None


def test_even_planning_cap_retains_requested_endpoint(tmp_path):
    output = tmp_path / "even-cap"
    active.run(_args(active, output, 6))
    result = _assert_family(output, "active_inference_analysis.json", 7)
    _assert_bounds(active, result, 6)
    assert result["configuration"]["horizons"] == [5, 6]
    fit = result["horizon_analysis"]["summary"]["energy_scaling"]
    assert fit["valid"] is False and "scaling_exponent" not in fit
