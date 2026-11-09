"""Generate Markdown tables and numbers for the separate enhanced analytical model.

The output is not canonical base-run evidence, empirical validation or device
calibration. All energy coefficients are assumptions, and mechanics are omitted.
"""

import argparse
import os
import stat
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from antstack_core.analysis import (
    EnergyCoefficients, analyze_scaling_relationship, estimate_detailed_energy,
    enhanced_body_workload_closed_form, enhanced_brain_workload_closed_form,
    enhanced_mind_workload_closed_form, calculate_contact_complexity,
)
from antstack_core.analysis.ce_publication_figures import _standard_module_data

PROVENANCE = ('Enhanced analytical workload model, not canonical base-run evidence. '
              'Compute-plus-idle energy with assumed default EnergyCoefficients; no mechanical work, '
              'sensing power, device calibration or empirical validation. Mind max_policies=1000. '
              'Fits describe the displayed deterministic sweep, not independent samples.')


def generate_complexity_table() -> str:
    """Report real workload counters rather than unsupported asymptotic claims."""
    data = _standard_module_data()
    lines = ['### Table: Enhanced Workload Counters per 10 ms {#tab:module_complexities}', '',
             '| Module | FLOPs/Decision | Memory Access (bytes) | Spikes |',
             '|--------|----------------|-----------------------|--------|']
    for name, item in data.items():
        load = item['load']
        lines.append(f'| {name} | {load.flops:.6g} | {load.sram_bytes + load.dram_bytes:.6g} | {load.spikes:.6g} |')
    return '\n'.join(lines) + '\n\n' + PROVENANCE


def generate_energy_coefficients_table() -> str:
    coeffs = EnergyCoefficients()
    lines = ['### Table: Assumed Energy Coefficients (Not Device-Calibrated) {#tab:energy_coefficients}', '',
             '| Component | Value | Units | Basis |', '|-----------|-------|-------|-------|']
    for name, value, unit in [('FLOPs', coeffs.flops_pj, 'pJ/FLOP'),
                              ('SRAM', coeffs.sram_pj_per_byte, 'pJ/byte'),
                              ('DRAM', coeffs.dram_pj_per_byte, 'pJ/byte'),
                              ('Spikes', coeffs.spike_aj, 'aJ/spike'),
                              ('Idle baseline per isolated module', coeffs.baseline_w, 'W')]:
        lines.append(f'| {name} | {value:.6g} | {unit} | Default model assumption, not measured calibration |')
    return '\n'.join(lines)


def _scaling_sweeps():
    return [
        ('AntBody', 'J', [6, 12, 18, 24, 36], enhanced_body_workload_closed_form,
         {'C': 12, 'S': 256, 'hz': 100, 'contact_solver': 'pgs'}),
        ('AntBrain', 'K', [64, 128, 256, 512, 1024], enhanced_brain_workload_closed_form,
         {'N_KC': 50000, 'rho': 0.02, 'H': 64, 'hz': 100}),
        ('AntMind', 'H_p', [5, 7, 9, 11, 13], enhanced_mind_workload_closed_form,
         {'B': 3, 'hz': 100, 'state_dim': 12, 'action_dim': 4, 'max_policies': 1000}),
    ]


def _format_fit(fit):
    if not fit.get('valid', False):
        return 'unavailable | unavailable'
    r_squared = 'undefined' if fit['r_squared'] is None else f'{fit["r_squared"]:.6g}'
    return f'{fit["scaling_exponent"]:.6g} | {r_squared}'


def generate_scaling_results_table() -> str:
    coeffs = EnergyCoefficients()
    lines = ['### Table: Descriptive Enhanced Model Fits {#tab:scaling_laws}', '',
             '| Module | Parameter Values | Energy Exponent | Log-log R² | FLOPs Exponent | Log-log R² |',
             '|--------|------------------|-----------------|------------|----------------|------------|']
    for name, parameter, values, workload, params in _scaling_sweeps():
        loads = [workload(0.01, dict(params, **{parameter: value})) for value in values]
        energies = [estimate_detailed_energy(load, coeffs, 0.01).total for load in loads]
        energy_fit = analyze_scaling_relationship(values, energies)
        flops_fit = analyze_scaling_relationship(values, [load.flops for load in loads])
        lines.append(f'| {name} | {parameter}: {values} | {_format_fit(energy_fit)} | {_format_fit(flops_fit)} |')
    return '\n'.join(lines) + '\n\n' + PROVENANCE


def generate_efficiency_analysis_table() -> str:
    lines = ['### Table: Conditional Erasure Reference Comparison {#tab:efficiency_analysis}', '',
             '| Module | Modeled Energy (J) | Conditional Erasure Reference (J) | Model / Reference |',
             '|--------|--------------------|----------------------------------|-------------------|']
    for name, item in _standard_module_data().items():
        energy, reference = item['breakdown'].total, item['reference_j']
        lines.append(f'| {name} | {energy:.6e} | {reference:.6e} | {energy/reference:.6g} |')
    return ('\n'.join(lines) + '\n\nAssumes one erased bit per FLOP at 300 K; FLOPs do not inherently determine bit erasures. '
            'No mechanical work is supplied. Ratios are not physical efficiencies or achievable improvement factors.\n\n' + PROVENANCE)


def generate_contact_solver_table() -> str:
    """Use supported solvers and publish counters actually computed at each size."""
    lines = ['### Table: Supported Contact Solver Model Counters {#tab:contact_solvers}', '',
             '| Solver | Joints | Contacts | FLOPs/Solve | Memory (bytes) | Iterations |',
             '|--------|--------|----------|-------------|----------------|------------|']
    for solver in ('pgs', 'si', 'ldlt'):
        for contacts in (5, 10, 15, 20, 25):
            flops, memory, iterations = calculate_contact_complexity(18, contacts, solver)
            lines.append(f'| {solver.upper()} | 18 | {contacts} | {flops:.6g} | {memory:.6g} | {iterations} |')
    return '\n'.join(lines) + '\n\nAnalytical workload estimates, not measured execution times or solver accuracy benchmarks.'


def generate_parameter_ranges_table() -> str:
    lines = ['### Table: Enhanced Model Sweep Configuration {#tab:param_ranges}', '',
             '| Module | Varied Parameter | Evaluated Values | Fixed Parameters |',
             '|--------|------------------|------------------|------------------|']
    for name, parameter, values, _workload, params in _scaling_sweeps():
        lines.append(f'| {name} | {parameter} | {values} | {params} |')
    return '\n'.join(lines) + '\n\nThese are selected model inputs, not measured biological ranges.'


def generate_all_tables() -> Dict[str, str]:
    return {'complexity_table': generate_complexity_table(),
            'energy_coefficients': generate_energy_coefficients_table(),
            'scaling_results': generate_scaling_results_table(),
            'efficiency_analysis': generate_efficiency_analysis_table(),
            'contact_solvers': generate_contact_solver_table(),
            'parameter_ranges': generate_parameter_ranges_table()}


def generate_key_numbers() -> Dict[str, Any]:
    data = _standard_module_data()
    coeffs = EnergyCoefficients()
    energies = {name.removeprefix('Ant').lower() + '_mj': item['breakdown'].total * 1000 for name, item in data.items()}
    energies['total_mj'] = sum(energies.values())
    body = _scaling_sweeps()[0]
    _name, parameter, values, workload, params = body
    sweep_energies = [estimate_detailed_energy(workload(0.01, dict(params, **{parameter: value})), coeffs, 0.01).total for value in values]
    fit = analyze_scaling_relationship(values, sweep_energies)
    load_data = {}
    for name, item in data.items():
        prefix, load = name.removeprefix('Ant').lower(), item['load']
        load_data[prefix + '_flops'] = load.flops
        load_data[prefix + '_memory_kb'] = (load.sram_bytes + load.dram_bytes) / 1024
    return {'schema_version': 1, 'model_family': 'enhanced_analytical',
            'provenance': PROVENANCE, 'assumed_energy_coefficients': asdict(coeffs),
            'per_decision_energy': energies, 'computational_load': load_data,
            'scaling_exponents': {'body_energy': fit['scaling_exponent'] if fit.get('valid', False) else None,
                                  'body_r_squared': fit['r_squared'] if fit.get('valid', False) else None,
                                  'body_regime': fit.get('scaling_regime', 'unavailable')},
            'system_parameters': {'control_frequency_hz': 100, 'decision_period_ms': 10,
                                  'baseline_power_mw': coeffs.baseline_w * 1000,
                                  'total_power_w': energies['total_mj'] / 1000 / 0.01,
                                  'total_power_boundary': 'sum of isolated module compute-plus-idle estimates, not whole-device power'}}


def main(argv=None):
    from antstack_core.publishing.provenance import (
        _fresh_family_path, atomic_json, atomic_text, build_run_provenance, capture_source_state,
        file_identity, write_provenance,
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='Fresh absent/empty enhanced-model supplement family root')
    args = parser.parse_args(argv)
    destination = _fresh_family_path(args.output)
    source_state = capture_source_state()
    tables, numbers = generate_all_tables(), generate_key_numbers()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{destination.name}.stage-', dir=destination.parent) as directory:
        stage = Path(directory)
        for name, content in tables.items():
            atomic_text(stage / f'{name}.md', content + '\n')
        atomic_json(stage / 'key_numbers.json', numbers)
        combined = '# Enhanced Analytical Model Tables\n\n' + PROVENANCE + '\n\n' + '\n\n'.join(tables.values()) + '\n'
        atomic_text(stage / 'generated_tables.md', combined)
        products = sorted(path.name for path in stage.iterdir()) + ['provenance/provenance.json']
        inventory_path = 'provenance/output_inventory.json'
        provenance = build_run_provenance(
            project='ant-stack-enhanced-tables', output_paths=products + [inventory_path],
            parameters={'schema_version': 1, 'model_family': 'enhanced_analytical',
                        'assumed_energy_coefficients': asdict(EnergyCoefficients()),
                        'model_definition': 'Fixed enhanced analytical inputs and descriptive sweeps in source generator',
                        'evidence_class': 'model estimates, not empirical validation or canonical base-run evidence'},
            source_state=source_state,
        )
        write_provenance(stage / 'provenance/provenance.json', provenance)
        artifacts = []
        for name in products:
            identity = file_identity(stage / name)
            artifacts.append({'path': name, 'artifact_type': 'markdown' if name.endswith('.md') else 'provenance' if name.startswith('provenance/') else 'json',
                              'producing_task': 'enhanced_tables.generate', 'source_config': 'key_numbers.json',
                              'checksum': identity['sha256'], 'bytes': identity['bytes']})
        atomic_json(stage / inventory_path, {'schema_version': 1, 'complete': True,
                                            'artifact_path_base': 'family_root', 'artifacts': artifacts})
        for artifact in artifacts:
            if file_identity(stage / artifact['path']) != {'sha256': artifact['checksum'], 'bytes': artifact['bytes']}:
                raise RuntimeError(f"Supplement changed before publication: {artifact['path']}")
        _fresh_family_path(destination)
        if destination.exists():
            os.chmod(stage, stat.S_IMODE(destination.stat().st_mode))
        os.replace(stage, destination)
    print(f'Published enhanced analytical supplements: {destination}')


if __name__ == '__main__':
    main()
