"""Comprehensive complexity energetics analysis with advanced methodologies. Moved verbatim from scripts/complexity_energetics/comprehensive_analysis.py; thin wrapper retained for CLI compatibility."""

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Any

from antstack_core.analysis import (
    ExperimentManifest, EnergyCoefficients, estimate_detailed_energy,
    analyze_scaling_relationship, calculate_energy_efficiency_metrics,
    estimate_theoretical_limits, enhanced_body_workload_closed_form,
    enhanced_brain_workload_closed_form, enhanced_mind_workload_closed_form
)
from antstack_core.analysis.complexity_analysis import (
    AgentBasedModel, NetworkComplexityAnalyzer, ThermodynamicComplexityAnalyzer,
    ComplexityEntropyAnalyzer, EnhancedBootstrapAnalyzer, comprehensive_complexity_analysis
)
from antstack_core.figures.advanced_plots import (
    complexity_entropy_diagram, network_visualization, thermodynamic_phase_diagram,
    multi_scale_visualization
)
from antstack_core.figures.publication_plots import FigureManager
from antstack_core.publishing.provenance import _publish_analysis_family


def analyze_module_with_advanced_methods(module_name: str, 
                                       param_name: str, 
                                       param_values: List[float],
                                       base_params: Dict[str, Any], 
                                       coeffs: EnergyCoefficients,
                                       output_dir: Path) -> Dict[str, Any]:
    """Analyze module with advanced complexity analysis methods.
    
    Performs comprehensive analysis including:
    - Traditional scaling analysis
    - Agent-based modeling for emergent behavior
    - Network complexity analysis
    - Thermodynamic efficiency analysis
    - Complexity-entropy analysis
    - Resampling sensitivity of deterministic model points (not empirical validation)
    
    Args:
        module_name: Name of module ('body', 'brain', 'mind')
        param_name: Parameter to vary
        param_values: List of parameter values to test
        base_params: Base parameter configuration
        coeffs: Energy coefficients
        output_dir: Directory for output files
    
    Returns:
        Dictionary containing comprehensive analysis results
    """
    print(f"Performing advanced analysis for {module_name} module...")
    
    # Select appropriate workload function
    workload_funcs = {
        'body': enhanced_body_workload_closed_form,
        'brain': enhanced_brain_workload_closed_form,
        'mind': enhanced_mind_workload_closed_form
    }
    
    if module_name not in workload_funcs:
        raise ValueError(f"Unknown module: {module_name}")
    
    workload_func = workload_funcs[module_name]
    
    # Calculate energy and computational data for each parameter value
    energy_data = []
    computational_data = []
    network_data = []
    time_series_data = []
    
    for i, param_val in enumerate(param_values):
        params = dict(base_params)
        params[param_name] = param_val
        
        # Calculate workload
        load = workload_func(0.01, params)  # 10ms decision
        breakdown = estimate_detailed_energy(load, coeffs, 0.01)
        
        energy_data.append(breakdown)
        computational_data.append({
            'parameter_value': param_val,
            'energy': breakdown.total,
            'flops': load.flops,
            'memory': load.sram_bytes + load.dram_bytes,
            'spikes': load.spikes,
            'computational_work': load.flops * coeffs.flops_pj * 1e-12,
            'information_processed': load.flops * 64  # Assume 64-bit operations
        })
        
        # Generate network data (simplified adjacency matrix)
        n_nodes = min(10, int(param_val / 10) + 5)  # Scale with parameter
        adjacency_matrix = [[0.0] * n_nodes for _ in range(n_nodes)]
        for j in range(n_nodes):
            for k in range(j + 1, n_nodes):
                if (j + k) % 3 == 0:  # Sparse connections
                    adjacency_matrix[j][k] = adjacency_matrix[k][j] = 0.5
        
        network_data.append(adjacency_matrix)
        
        # Ordered parameter sweep, not repeated time observations or independent samples.
        time_series_data.append(breakdown.total)
    
    # Perform comprehensive complexity analysis
    complexity_results = comprehensive_complexity_analysis(
        energy_data, computational_data, network_data[0], time_series_data
    )
    
    # Agent-based modeling analysis
    agent_model = AgentBasedModel(
        agents=[{'initial_state': 0.0, 'initial_energy': 1.0} for _ in range(5)],
        interactions=[(i, (i + 1) % 5, 0.1) for i in range(5)],
        environment_params={'temperature': 300.0, 'pressure': 101325.0},
        time_steps=100
    )
    
    emergent_metrics = agent_model.simulate()
    
    # Network complexity analysis
    network_analyzer = NetworkComplexityAnalyzer()
    network_complexity = asdict(network_analyzer.analyze_network_complexity(network_data[0]))
    if math.isinf(network_complexity['path_length']):
        # JSON has no infinity token; retain its mathematical meaning explicitly.
        network_complexity['path_length'] = None
        network_complexity['path_length_status'] = 'infinite: constructed graph is disconnected'
    complexity_results['network_analysis'] = network_complexity
    
    # Thermodynamic analysis
    thermo_analyzer = ThermodynamicComplexityAnalyzer()
    thermo_results = []
    for energy, comp in zip(energy_data, computational_data):
        thermo_result = thermo_analyzer.analyze_thermodynamic_efficiency(
            energy, comp['computational_work'], comp['information_processed']
        )
        thermo_results.append(thermo_result)
    
    # Complexity-entropy analysis
    ce_analyzer = ComplexityEntropyAnalyzer()
    # One whole-grid diagnostic, not a time-series window or empirical sample.
    ce_results = ce_analyzer.create_complexity_entropy_diagram(
        time_series_data, window_size=len(time_series_data)
    )
    complexity_results['complexity_entropy_analysis'] = ce_results
    
    # Bootstrap analysis
    bootstrap_analyzer = EnhancedBootstrapAnalyzer()
    x_values = [comp['parameter_value'] for comp in computational_data]
    y_values = [comp['energy'] for comp in computational_data]
    bootstrap_results = bootstrap_analyzer.analyze_scaling_with_bootstrap(x_values, y_values)
    
    # Generate advanced visualizations
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Complexity-entropy diagram
    ce_plot_path = plot_dir / f"{module_name}_{param_name}_complexity_entropy.png"
    complexity_entropy_diagram(
        ce_results['complexity'], ce_results['entropy'],
        f"{module_name.title()} Ordered Model-Sweep Complexity-Entropy Diagnostic",
        str(ce_plot_path)
    )
    
    # The constructed graph is illustrative, not biological topology evidence.
    network_plot_path = plot_dir / f"{module_name}_{param_name}_network.png"
    network_visualization(
        network_data[0],
        [f"Node {i}" for i in range(len(network_data[0]))],
        f"{module_name.title()} Illustrative Constructed Network",
        str(network_plot_path)
    )
    
    
    # Multi-scale visualization
    multi_scale_data = {
        'micro_scale': {
            'x': [comp['flops'] for comp in computational_data],
            'y': [comp['energy'] for comp in computational_data]
        },
        'meso_scale': {
            'x': [comp['memory'] for comp in computational_data],
            'y': [comp['energy'] for comp in computational_data]
        },
        'macro_scale': {
            'x': [comp['parameter_value'] for comp in computational_data],
            'y': [comp['energy'] for comp in computational_data]
        }
    }
    
    multi_scale_plot_path = plot_dir / f"{module_name}_{param_name}_multi_scale.png"
    multi_scale_visualization(
        multi_scale_data,
        f"{module_name.title()} Multi-Scale Analysis",
        str(multi_scale_plot_path)
    )
    
    return {
        'provenance': {'energy': 'deterministic analytical estimates',
                       'network': 'illustrative constructed graph, not module topology',
                       'agent_simulation': 'illustrative model, not independent validation',
                       'sequence': 'ordered parameter sweep, not measured time series',
                       'bootstrap': 'resampling sensitivity of model sweep, not empirical uncertainty',
                       'complexity_entropy': 'one whole fixed-grid window; descriptive sequence diagnostic, not temporal evidence'},
        'module': module_name,
        'parameter': param_name,
        'parameter_values': param_values,
        'energy_data': energy_data,
        'computational_data': computational_data,
        'complexity_analysis': complexity_results,
        'agent_based_modeling': emergent_metrics,
        'network_complexity': network_complexity,
        'thermodynamic_analysis': thermo_results,
        'complexity_entropy_analysis': ce_results,
        'bootstrap_analysis': bootstrap_results,
        'visualizations': {
            'complexity_entropy': str(ce_plot_path),
            'network': str(network_plot_path),
            'multi_scale': str(multi_scale_plot_path)
        }
    }


def generate_veridical_report(all_results: Dict[str, Any], output_dir: Path) -> None:
    """Report analytical sweeps without manufacturing empirical evidence.

    Model points at different parameters and illustrative simulations are not
    independent physical measurements, replication or calibration evidence.
    """
    description = ('Enhanced analytical compute-plus-idle workload model with assumed manifest coefficients. '
                   'Fixed parameter grids are independent of canonical manifest workload selection. '
                   'Mind uses at most 1000 policies. No mechanical power, independent measurements, '
                   'device calibration or external validation is supplied. This separate model family '
                   'does not validate canonical base-run manuscript numbers.')
    summary = {'provenance': description, 'configuration': all_results.get('configuration', {}), 'modules': []}
    lines = ['# Enhanced Analytical Model Report', '', description, '']
    for result in all_results.get('module_analyses', []):
        points = result['computational_data']
        x = [point['parameter_value'] for point in points]
        energies = [point['energy'] for point in points]
        fit = analyze_scaling_relationship(x, energies)
        summary['modules'].append({'module': result['module'], 'parameter': result['parameter'],
                                   'points': points, 'descriptive_power_fit': fit})
        lines.extend([f"## {result['module'].title()}: {result['parameter']}", '',
                      '| Parameter | Modeled Energy (J/Decision) | FLOPs/Decision |',
                      '|-----------|-----------------------------|----------------|'])
        lines.extend(f"| {point['parameter_value']} | {point['energy']:.9g} | {point['flops']:.9g} |" for point in points)
        if fit.get('valid', False):
            r_squared = 'undefined (constant response)' if fit['r_squared'] is None else format(fit['r_squared'], '.6g')
            lines.extend(['', f"Descriptive log-log fit: exponent {fit['scaling_exponent']:.6g}, R² {r_squared}."])
        else:
            lines.extend(['', 'Power-law fit unavailable for these model points.'])
        lines.extend(['', 'No sampling confidence, significance, structural phase or replication claim is inferred.', ''])
    report_dir = output_dir / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / 'comprehensive_analysis_report.md').write_text('\n'.join(lines), encoding='utf-8')
    (output_dir / 'data_driven_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _produce_analysis(output_dir, *, manifest_path, modules, analysis_configs):
    """Generate the declared family only inside the publisher's private stage."""
    print("=" * 80)
    print("COMPREHENSIVE COMPLEXITY ENERGETICS ANALYSIS")
    print("=" * 80)
    print(f"Modules to analyze: {modules}")
    print()
    
    # Manifest failure is fatal: do not silently substitute unrelated parameters.
    manifest = ExperimentManifest.load(manifest_path)
    valid, errors = manifest.validate()
    if not valid:
        raise ValueError("Invalid manifest: " + "; ".join(errors))
    coeffs = manifest.coefficients.to_energy_coefficients()
    
    
    # Perform comprehensive analyses
    module_analyses = []
    start_time = time.time()
    
    for module in modules:
        if module in analysis_configs:
            print(f"\nAnalyzing {module} module...")
            config = analysis_configs[module]
            
            module_start = time.time()
            result = analyze_module_with_advanced_methods(
                module, config['param'], config['values'],
                config['base_params'], coeffs, output_dir
            )
            module_analyses.append(result)
            
            module_time = time.time() - module_start
            print(f"  Completed in {module_time:.2f} seconds")
            print(f"  Generated {len(result['visualizations'])} visualizations")
        else:
            raise ValueError(f"Unknown module: {module}")
    
    total_analysis_time = time.time() - start_time
    print(f"\nTotal analysis time: {total_analysis_time:.2f} seconds")
    
    # Compile all results
    all_results = {
        'provenance': 'enhanced analytical compute-plus-idle model; fixed grids independent of canonical manifest workload selection; assumed manifest coefficients; Mind max_policies=1000; no mechanics, device calibration or independent empirical validation; not canonical base-run evidence',
        'module_analyses': module_analyses,
        'configuration': {
            'manifest_path': str(manifest_path),
            'coefficients': manifest.coefficients.to_dict(),
            'modules_analyzed': modules,
            'analysis_configs': analysis_configs,
            'decision_seconds': 0.01,
            'analysis_time': total_analysis_time
        },
        'timestamp': time.time()
    }
    
    # Save results as JSON
    results_path = output_dir / "comprehensive_analysis_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=asdict, allow_nan=False)
    
    print(f"Analysis results saved: {results_path}")
    
    # Generate veridical report
    generate_veridical_report(all_results, output_dir)
    


def run(args):
    """Publish a complete, portable analytical family into an absent/empty root."""
    modules = list(args.modules)
    if not modules:
        raise ValueError("At least one analysis module is required")
    unknown = [module for module in modules if module not in ('body', 'brain', 'mind')]
    if unknown:
        raise ValueError(f"Unknown modules: {unknown}")
    analysis_configs = {
        'body': {
            'param': 'J',
            'values': [6, 12, 18, 24, 36],
            'base_params': {'J': 18, 'C': 12, 'S': 256, 'hz': 100, 'contact_solver': 'pgs'}
        },
        'brain': {
            'param': 'K',
            'values': [64, 128, 256, 512, 1024],
            'base_params': {'K': 128, 'N_KC': 50000, 'rho': 0.02, 'H': 64, 'hz': 100, 'connectivity_pattern': 'random'}
        },
        'mind': {
            'param': 'H_p',
            'values': [5, 10, 15, 20, 25],
            'base_params': {'B': 4, 'H_p': 20, 'hz': 100, 'state_dim': 16, 'action_dim': 6, 'hierarchical': False}
        }
    }
    selected_configs = {module: analysis_configs[module] for module in modules}
    required = ['comprehensive_analysis_results.json', 'data_driven_summary.json',
                'reports/comprehensive_analysis_report.md']
    required.extend(
        f"plots/{module}_{config['param']}_{kind}.png"
        for module, config in selected_configs.items()
        for kind in ('complexity_entropy', 'network', 'multi_scale')
    )
    manifest_path = Path(args.manifest)
    result = _publish_analysis_family(
        args.output, project='ce-comprehensive-analysis',
        produce=lambda stage: _produce_analysis(
            stage, manifest_path=manifest_path, modules=modules,
            analysis_configs=selected_configs),
        required_paths=required, input_paths=(manifest_path,),
        parameters={
            'modules': modules, 'analysis_configs': selected_configs,
            'model_scope': 'enhanced analytical compute-plus-idle; fixed grids, not canonical manifest workloads',
            'coefficient_source': 'original manifest snapshot',
            'decision_seconds': 0.01, 'mind_max_policies': 1000,
            'complexity_entropy_window': 'entire ordered parameter grid',
            'bootstrap_seed': 0, 'bootstrap_attempted': 1000,
        },
    )
    print(f"Complete analytical family published: {result.parent}")
    print(f"- Family manifest: {result}")
    return result
