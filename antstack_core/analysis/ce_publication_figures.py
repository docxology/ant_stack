"""Generate descriptive publication figures from supported analytical workloads.

All energies use the ENHANCED analytical model and assumed default EnergyCoefficients.
They are compute-plus-idle estimates for one 10 ms decision, not
measurements. Fits describe the displayed sweep, without empirical uncertainty.
"""

import json
from pathlib import Path
from typing import Any, Dict

from antstack_core.analysis import (
    EnergyCoefficients, analyze_scaling_relationship, estimate_detailed_energy,
    estimate_theoretical_limits, enhanced_body_workload_closed_form,
    enhanced_brain_workload_closed_form, enhanced_mind_workload_closed_form,
    calculate_contact_complexity, calculate_sparse_neural_complexity,
)
from antstack_core.figures import FigureManager, interval_plot, publication_bar_plot, publication_line_plot


def _image_path(output_dir, filename, image_format):
    if image_format not in {'png', 'svg', 'pdf'}:
        raise ValueError(f'Unsupported figure format: {image_format}')
    return str((output_dir / filename).with_suffix('.' + image_format))


def _standard_module_data(coeffs=None):
    """Return consistent compute-only estimates and conditional erasure references.

    No mechanical/sensing power is supplied. The Landauer reference assumes one
    erased bit per FLOP at 300 K; this is an assumption, not a measured floor for
    the complete device or evidence of achievable optimization.
    """
    coeffs = EnergyCoefficients() if coeffs is None else coeffs
    specifications = [
        ('AntBody', enhanced_body_workload_closed_form,
         {'J': 18, 'C': 12, 'S': 256, 'hz': 100, 'contact_solver': 'pgs'}),
        ('AntBrain', enhanced_brain_workload_closed_form,
         {'K': 128, 'N_KC': 50000, 'rho': 0.02, 'H': 64, 'hz': 100,
          'connectivity_pattern': 'random'}),
        ('AntMind', enhanced_mind_workload_closed_form,
         {'B': 4, 'H_p': 12, 'hz': 100, 'state_dim': 16, 'action_dim': 6,
          'max_policies': 1000}),
    ]
    results = {}
    for name, workload, params in specifications:
        load = workload(0.01, params)
        breakdown = estimate_detailed_energy(load, coeffs, 0.01)
        reference = estimate_theoretical_limits({'bits_erased': load.flops}, 300.0)['landauer_limit_j']
        results[name] = {'load': load, 'breakdown': breakdown, 'reference_j': reference}
    return results


def _component_totals_mj(breakdowns):
    """Aggregate every energy component in joules, then convert once to mJ."""
    components = {'Actuation': 0.0, 'Sensing': 0.0, 'Compute': 0.0, 'Baseline': 0.0}
    for energy in breakdowns:
        components['Actuation'] += energy.actuation
        components['Sensing'] += energy.sensing
        components['Compute'] += energy.compute_flops + energy.compute_memory + energy.compute_spikes
        components['Baseline'] += energy.baseline
    return {name: joules * 1000 for name, joules in components.items()}


def _descriptive_scaling_stats(x, y):
    fit = analyze_scaling_relationship(x, y)
    result = {'Displayed range': f'{min(y):.6g} - {max(y):.6g}'}
    if fit.get('valid', False):
        result['Descriptive power-law exponent'] = f'{fit["scaling_exponent"]:.6g}'
        result['Log-log fit R²'] = 'undefined (constant response)' if fit['r_squared'] is None else f'{fit["r_squared"]:.6g}'
    else:
        result['Power-law fit'] = 'Not identifiable for this sweep'
    return result


def generate_energy_breakdown_figures(output_dir: Path, figure_manager: FigureManager, image_format: str = "png") -> Dict[str, str]:
    data = _standard_module_data()
    energies = [item['breakdown'] for item in data.values()]
    totals_mj = [energy.total * 1000 for energy in energies]
    caption = ('Enhanced analytical model with assumed default EnergyCoefficients (not device calibration). Estimates per 10 ms decision, including each module compute and baseline. '
               'Actuation and sensing power are not supplied and are therefore not modeled. '
               'These deterministic estimates are not independent empirical samples; no significance test is claimed.')
    module_id = publication_bar_plot(
        list(data), totals_mj, 'Enhanced Compute-plus-Idle Module Energy', _image_path(output_dir, 'module_energy_comparison.png', image_format),
        ylabel='Energy per Decision (mJ)', figure_manager=figure_manager,
        detailed_caption=caption, stats={'Total Energy (mJ)': f'{sum(totals_mj):.6g}'})
    components = _component_totals_mj(energies)
    component_id = publication_bar_plot(
        list(components), list(components.values()), 'Enhanced Compute-plus-Idle Energy Components',
        _image_path(output_dir, 'energy_component_breakdown.png', image_format), ylabel='Energy per Decision (mJ)',
        figure_manager=figure_manager, detailed_caption=caption,
        stats={f'{name} Energy (mJ)': f'{value:.6g}' for name, value in components.items()})
    return {'module_comparison': module_id, 'component_breakdown': component_id}


def generate_scaling_analysis_figures(output_dir: Path, figure_manager: FigureManager, image_format: str = "png") -> Dict[str, str]:
    coeffs = EnergyCoefficients()
    sweeps = [
        ('body_energy_scaling', 'body_joint_energy_scaling.png', 'AntBody', 'J', [6, 12, 18, 24, 30],
         enhanced_body_workload_closed_form, {'C': 12, 'S': 256, 'hz': 100, 'contact_solver': 'pgs'}),
        ('brain_energy_scaling', 'brain_channel_energy_scaling.png', 'AntBrain', 'K', [32, 64, 128, 256, 512],
         enhanced_brain_workload_closed_form, {'N_KC': 50000, 'rho': 0.02, 'H': 64, 'hz': 100}),
        ('mind_energy_scaling', 'mind_horizon_energy_scaling.png', 'AntMind', 'H_p', [4, 8, 12, 16, 20],
         enhanced_mind_workload_closed_form, {'B': 4, 'hz': 100, 'state_dim': 16, 'action_dim': 6, 'max_policies': 1000}),
    ]
    figures = {}
    for key, filename, module, parameter, values, workload, params in sweeps:
        energies = [estimate_detailed_energy(workload(0.01, dict(params, **{parameter: value})), coeffs, 0.01).total * 1000
                    for value in values]
        figures[key] = publication_line_plot(
            values, [energies], [module], f'{module} Enhanced Model Energy vs {parameter}', _image_path(output_dir, filename, image_format),
            xlabel=parameter, ylabel='Energy per Decision (mJ)', figure_manager=figure_manager,
            detailed_caption=('Enhanced analytical model with assumed default EnergyCoefficients; compute-plus-idle boundary, no mechanics. '
                              'Any fitted exponent describes only these displayed values, not an asymptotic law, '
                              'measured transition, confidence interval or hypothesis test. Mind uses at most 1000 policies.'),
            stats=_descriptive_scaling_stats(values, energies))
    return figures


def generate_connectivity_analysis_figures(output_dir: Path, figure_manager: FigureManager, image_format: str = "png") -> Dict[str, str]:
    sparsities = [0.01, 0.02, 0.05, 0.1, 0.2]
    counts = [calculate_sparse_neural_complexity(50000, rho, 'random') for rho in sparsities]
    coeffs = EnergyCoefficients()
    energies = [estimate_detailed_energy(enhanced_brain_workload_closed_form(0.01, {
        'K': 128, 'N_KC': 50000, 'rho': rho, 'H': 64, 'hz': 100, 'connectivity_pattern': 'random'}), coeffs, 0.01).total * 1000
        for rho in sparsities]
    figures = {}
    for key, filename, title, values, unit in [
        ('connectivity_flops', 'connectivity_patterns_flops.png', 'Sparse Neural FLOPs', [row[0] for row in counts], 'FLOPs per Decision'),
        ('connectivity_spikes', 'connectivity_patterns_spikes.png', 'Sparse Neural Spike Count', [row[2] for row in counts], 'Spikes per Decision'),
        ('sparsity_energy', 'sparsity_energy_scaling.png', 'Brain Energy vs Sparsity', energies, 'Energy per Decision (mJ)'),
    ]:
        figures[key] = publication_line_plot(
            sparsities, [values], [title], title, _image_path(output_dir, filename, image_format), xlabel='Sparsity (ρ)', ylabel=unit,
            figure_manager=figure_manager, detailed_caption=(
                'Enhanced analytical model, assumed default EnergyCoefficients and compute-plus-idle boundary, no mechanics. Random connectivity. Counts are model outputs, not measurements of neural activity. '
                'Fits are descriptive; no significance or sampling uncertainty is inferred.'),
            stats=_descriptive_scaling_stats(sparsities, values))
    return figures


def generate_theoretical_limits_figures(output_dir: Path, figure_manager: FigureManager, image_format: str = "png") -> Dict[str, str]:
    data = _standard_module_data()
    references = [item['reference_j'] for item in data.values()]
    energies = [item['breakdown'].total for item in data.values()]
    ratios = [energy / reference for energy, reference in zip(energies, references)]
    caption = ('Enhanced analytical model with assumed default EnergyCoefficients and 1000-policy Mind cap. Compute-plus-idle estimates compared with a conditional Landauer erasure reference '
               '(one erased bit per FLOP at 300 K). Mechanical work is not modeled. '
               'Ratios are overhead relative to that assumption, not physical efficiencies or achievable improvement factors.')
    reference_path = _image_path(output_dir, 'theoretical_limits_comparison.png', image_format)
    reference_title = 'Model vs Conditional Erasure Reference (No Mechanics)'
    interval_plot([f'{name} estimate' for name in data] + [f'{name} reference' for name in data],
                  energies + references, reference_title, reference_path, 'Energy per Decision (J)')
    reference_id = figure_manager.get_next_figure_id('interval_plot')
    figure_manager.add_figure(reference_id, reference_title, caption, reference_path)
    ratio_id = publication_bar_plot(
        list(data), ratios, 'Modeled Energy / Erasure Reference', _image_path(output_dir, 'efficiency_ratios.png', image_format),
        ylabel='Energy / Conditional Reference', figure_manager=figure_manager, detailed_caption=caption)
    return {'theoretical_limits': reference_id, 'efficiency_ratios': ratio_id}


def generate_contact_solver_figures(output_dir: Path, figure_manager: FigureManager, image_format: str = "png") -> Dict[str, str]:
    contacts = [5, 10, 15, 20, 25, 30]
    solvers = ['pgs', 'si', 'ldlt']
    counts = {solver: [calculate_contact_complexity(J=18, C=count, solver=solver) for count in contacts] for solver in solvers}
    figures = {}
    for key, filename, index, ylabel in [
        ('solver_flops', 'contact_solver_flops_comparison.png', 0, 'FLOPs per Solve'),
        ('solver_memory', 'contact_solver_memory_comparison.png', 1, 'Memory (bytes)'),
    ]:
        figures[key] = publication_line_plot(
            contacts, [[row[index] for row in counts[solver]] for solver in solvers], [solver.upper() for solver in solvers],
            'Supported Contact Solver Workload Estimates', _image_path(output_dir, filename, image_format), xlabel='Number of Contacts', ylabel=ylabel,
            figure_manager=figure_manager, detailed_caption=(
                'Analytical workload counters for PGS, SI and LDLT at 18 joints. '
                'This comparison does not measure solver accuracy, execution time or statistical significance.'))
    return figures


def run(args):
    from antstack_core.analysis.ce_manuscript_figures import _publish_family
    coeffs = EnergyCoefficients()
    def produce(stage):
        manager = FigureManager(stage)
        for producer in (generate_energy_breakdown_figures, generate_scaling_analysis_figures,
                         generate_connectivity_analysis_figures, generate_theoretical_limits_figures,
                         generate_contact_solver_figures):
            producer(stage, manager, image_format=args.format)
        figures, models = {}, {}
        for identifier, item in manager.figures.items():
            path = Path(item['file_path'])
            figures[identifier] = str(path)
            models[path.stem] = {'model_definition': 'ce_publication_figures analytical producer',
                                 'caption': item['caption'], 'statistics': item['stats'],
                                 'assumed_coefficients': coeffs.to_dict(),
                                 'evidence_class': 'descriptive analytical estimates, not empirical validation'}
            item['file_path'] = path.name
        manager.save_captions()
        return {'figures': figures, 'model_definitions': models}
    return _publish_family(args.output, 'publication', 'enhanced_figure_manifest.json',
                           args.format, coeffs, produce, supplementary=('figure_captions.md',))
