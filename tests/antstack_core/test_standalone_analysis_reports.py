"""Real analytical families: completeness, portable bytes and failure isolation."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest
import yaml

from antstack_core.analysis import (
    ExperimentManifest,
    enhanced_body_workload_closed_form,
    enhanced_brain_workload_closed_form,
    enhanced_mind_workload_closed_form,
    estimate_detailed_energy,
)


ADAPTERS = [
    pytest.param(('ce_comprehensive_analysis', 'comprehensive_analysis_results.json',
                  'module_analyses', 'generate_veridical_report'), id='comprehensive'),
    pytest.param(('ce_module_scaling_analysis', 'analysis_results.json',
                  'scaling_analyses', 'generate_comprehensive_report'), id='module-scaling'),
]
SOURCE_MANIFEST = Path(__file__).resolve().parents[2] / 'papers/complexity_energetics/manifest.example.yaml'


@pytest.fixture(params=ADAPTERS)
def adapter(request):
    name, results, analyses, report = request.param
    return SimpleNamespace(module=importlib.import_module('antstack_core.analysis.' + name),
                           results=results, analyses=analyses, report=report)


@pytest.fixture
def manifest_path(tmp_path):
    # A real nondefault coefficient catches accidental fallback to default prices.
    data = yaml.safe_load(SOURCE_MANIFEST.read_text())
    data['coefficients']['flops_pj'] = 7.25
    data['workloads']['body']['params']['J'] = 9
    data['workloads']['brain']['params']['K'] = 96
    path = tmp_path / 'supplied.yaml'
    path.write_text(yaml.safe_dump(data), encoding='utf-8')
    path.chmod(0o600)
    return path


def _strict_json(path):
    def invalid(token):
        raise AssertionError(f'Nonfinite JSON token: {token}')
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=invalid)


def _args(output, manifest, modules=('body', 'brain', 'mind')):
    return SimpleNamespace(output=str(output), manifest=str(manifest), modules=list(modules))


def test_real_default_selection_publishes_complete_portable_current_family(adapter, manifest_path, tmp_path):
    output = tmp_path / 'family'
    args = _args(output, manifest_path)
    original_args = {**vars(args), 'modules': args.modules.copy()}
    source_bytes = manifest_path.read_bytes()
    receipt = adapter.module.run(args)
    assert receipt == output / 'analysis_manifest.json'
    assert vars(args) == original_args
    assert manifest_path.read_bytes() == source_bytes
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600

    family = _strict_json(receipt)
    assert family['complete'] is True
    assert family['parameters']['modules'] == ['body', 'brain', 'mind']
    expected_plots = (9 if adapter.analyses == 'module_analyses' else 10)
    required_plots = [name for name in family['required_paths'] if name.endswith('.png')]
    assert len(required_plots) == expected_plots
    for name in family['required_paths']:
        product = output / name
        assert product.is_file() and product.stat().st_size > 0
        if product.suffix == '.png':
            assert product.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    bindings = list(family['input_snapshots'].values())
    assert len(bindings) == 1
    assert (output / bindings[0]['snapshot']).read_bytes() == source_bytes
    assert bindings[0]['sha256'] == hashlib.sha256(source_bytes).hexdigest()

    inventory = _strict_json(output / 'provenance/output_inventory.json')
    inventoried = {item['path'] for item in inventory['artifacts']}
    assert set(family['required_paths']) <= inventoried
    for item in inventory['artifacts']:
        data = (output / item['path']).read_bytes()
        assert item['checksum'] == hashlib.sha256(data).hexdigest()
        assert item['bytes'] == len(data)
    for path in output.rglob('*.json'):
        _strict_json(path)
    for path in (*output.rglob('*.json'), *output.rglob('*.md')):
        text = path.read_text(encoding='utf-8')
        assert '.stage-' not in text

    results = _strict_json(output / adapter.results)
    configs = results['configuration']['analysis_configs']
    assert configs['body']['values'] == [6, 12, 18, 24, 36]
    assert configs['body']['base_params']['J'] == 18
    coeffs = ExperimentManifest.load(manifest_path).coefficients.to_energy_coefficients()
    workloads = {'body': enhanced_body_workload_closed_form,
                 'brain': enhanced_brain_workload_closed_form,
                 'mind': enhanced_mind_workload_closed_form}
    for result in results[adapter.analyses]:
        config = configs[result['module']]
        params = {**config['base_params'], config['param']: config['values'][0]}
        load = workloads[result['module']](0.01, params)
        expected = estimate_detailed_energy(load, coeffs, 0.01).total
        if adapter.analyses == 'module_analyses':
            assert result['computational_data'][0]['energy'] == pytest.approx(expected)
            diagram = result['complexity_entropy_analysis']
            assert len(diagram['complexity']) == len(diagram['entropy']) == 1
            assert result['complexity_analysis']['complexity_entropy_analysis'] == diagram
            network = result['network_complexity']
            assert network['path_length'] is None
            assert network['path_length_status']
            assert result['complexity_analysis']['network_analysis'] == network
        else:
            assert result['energy_values'][0] == pytest.approx(expected)
        paths = result.get('visualizations', result.get('plots'))
        for relative in paths.values():
            assert not Path(relative).is_absolute()
            assert (output / relative).is_file()
    report = next(path for path in output.rglob('*.md'))
    assert report.stat().st_size > 0


@pytest.mark.parametrize('root_state', ['absent', 'empty', 'occupied'])
def test_real_manifest_failures_preserve_destination_and_source(adapter, tmp_path, root_state):
    output = tmp_path / 'family'
    if root_state != 'absent':
        output.mkdir(mode=0o700)
    sentinel = output / 'prior.json'
    if root_state == 'occupied':
        sentinel.write_bytes(b'{"prior":"must remain"}\n')
        sentinel.chmod(0o600)
    for supplied in ('missing', 'malformed', 'invalid'):
        manifest = tmp_path / f'{supplied}.yaml'
        if supplied == 'malformed':
            manifest.write_text('workloads: [unterminated')
        elif supplied == 'invalid':
            data = yaml.safe_load(SOURCE_MANIFEST.read_text())
            data['coefficients']['flops_pj'] = -1
            manifest.write_text(yaml.safe_dump(data))
        before = manifest.read_bytes() if manifest.exists() else None
        with pytest.raises((ValueError, FileNotFoundError, yaml.YAMLError)):
            adapter.module.run(_args(output, manifest, ('body',)))
        if root_state != 'absent':
            assert stat.S_IMODE(output.stat().st_mode) == 0o700
        assert (manifest.read_bytes() if manifest.exists() else None) == before
        if root_state == 'occupied':
            assert sentinel.read_bytes() == b'{"prior":"must remain"}\n'
            assert stat.S_IMODE(sentinel.stat().st_mode) == 0o600
            assert list(output.iterdir()) == [sentinel]
        elif root_state == 'empty':
            assert list(output.iterdir()) == []
        else:
            assert not output.exists()


@pytest.mark.parametrize('modules', [[], ['body', 'unknown'], ['unknown']])
def test_empty_or_unknown_selection_cannot_publish_partial_family(adapter, manifest_path, tmp_path, modules):
    output = tmp_path / 'family'
    with pytest.raises(ValueError):
        adapter.module.run(_args(output, manifest_path, modules))
    assert not output.exists()


@pytest.mark.parametrize('existing_empty', [False, True])
def test_late_report_failure_rolls_back_real_prior_generation(adapter, manifest_path, tmp_path, monkeypatch, existing_empty):
    output = tmp_path / 'family'
    if existing_empty:
        output.mkdir(mode=0o700)
    original = getattr(adapter.module, adapter.report)
    def fail_after_real_report(results, stage):
        original(results, stage)
        assert list(stage.rglob('*.png'))
        assert (stage / adapter.results).is_file()
        raise RuntimeError('injected late report failure')
    monkeypatch.setattr(adapter.module, adapter.report, fail_after_real_report)
    source_bytes = manifest_path.read_bytes()
    with pytest.raises(RuntimeError):
        adapter.module.run(_args(output, manifest_path, ('body',)))
    assert manifest_path.read_bytes() == source_bytes
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600
    if existing_empty:
        assert output.is_dir() and not list(output.iterdir())
        assert stat.S_IMODE(output.stat().st_mode) == 0o700
    else:
        assert not output.exists()
    assert not list(tmp_path.glob('.family.stage-*'))


def test_missing_critical_plot_cannot_be_hidden_by_inventory(adapter, manifest_path, tmp_path, monkeypatch):
    output = tmp_path / 'family'
    plot_name = 'multi_scale_visualization' if adapter.analyses == 'module_analyses' else 'scatter_plot'
    original = getattr(adapter.module, plot_name)
    def remove_real_plot(*args, **kwargs):
        result = original(*args, **kwargs)
        Path(args[-1]).unlink()
        return result
    monkeypatch.setattr(adapter.module, plot_name, remove_real_plot)
    with pytest.raises(ValueError):
        adapter.module.run(_args(output, manifest_path, ('body',)))
    assert not output.exists()
    assert not list(tmp_path.glob('.family.stage-*'))


def test_late_plot_failure_does_not_publish_earlier_plots(adapter, manifest_path, tmp_path, monkeypatch):
    output = tmp_path / 'family'
    plot_name = 'network_visualization' if adapter.analyses == 'module_analyses' else 'scatter_plot'
    def fail_plot(*args, **kwargs):
        stage_plots = Path(args[-1]).parent
        assert list(stage_plots.glob('*.png'))
        raise RuntimeError('injected late plotting failure')
    monkeypatch.setattr(adapter.module, plot_name, fail_plot)
    with pytest.raises(RuntimeError):
        adapter.module.run(_args(output, manifest_path, ('body',)))
    assert not output.exists()


def test_changed_input_cannot_publish_results_bound_to_stale_bytes(adapter, manifest_path, tmp_path, monkeypatch):
    output = tmp_path / 'family'
    original = getattr(adapter.module, adapter.report)
    def change_input_after_report(results, stage):
        original(results, stage)
        manifest_path.write_text(manifest_path.read_text() + '\n# changed during generation\n')
    monkeypatch.setattr(adapter.module, adapter.report, change_input_after_report)
    with pytest.raises(ValueError):
        adapter.module.run(_args(output, manifest_path, ('body',)))
    assert not output.exists()


def test_nonfinite_generated_json_cannot_publish(adapter, manifest_path, tmp_path, monkeypatch):
    output = tmp_path / 'family'
    original = getattr(adapter.module, adapter.report)
    def corrupt_after_report(results, stage):
        original(results, stage)
        path = stage / adapter.results
        data = json.loads(path.read_text())
        data['nonfinite_regression'] = float('nan')
        path.write_text(json.dumps(data))
    monkeypatch.setattr(adapter.module, adapter.report, corrupt_after_report)
    with pytest.raises(ValueError):
        adapter.module.run(_args(output, manifest_path, ('body',)))
    assert not output.exists()


@pytest.mark.parametrize('wrapper', ['comprehensive_analysis', 'generate_comprehensive_analysis'])
def test_wrappers_require_explicit_output(wrapper, monkeypatch):
    module = importlib.import_module('scripts.complexity_energetics.' + wrapper)
    monkeypatch.setattr(sys, 'argv', [wrapper])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2


def test_occupied_valid_family_preserves_product_bytes_and_modes(adapter, manifest_path, tmp_path):
    output = tmp_path / 'family'
    output.mkdir(mode=0o700)
    original = output / adapter.results
    original.write_bytes(b'previous scientific result\\n')
    original.chmod(0o600)
    source_bytes = manifest_path.read_bytes()
    with pytest.raises(ValueError):
        adapter.module.run(_args(output, manifest_path))
    assert original.read_bytes() == b'previous scientific result\\n'
    assert stat.S_IMODE(original.stat().st_mode) == 0o600
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert list(output.iterdir()) == [original]
    assert manifest_path.read_bytes() == source_bytes
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600
