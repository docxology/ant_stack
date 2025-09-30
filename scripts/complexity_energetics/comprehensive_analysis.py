#!/usr/bin/env python3
"""Comprehensive complexity energetics analysis with advanced methodologies.

This script orchestrates a complete analysis pipeline incorporating:
- Advanced complexity analysis with agent-based modeling
- Network complexity metrics and structural analysis
- Thermodynamic efficiency frameworks
- Complexity-entropy diagrams for intrinsic computation
- Multi-scale visualization and statistical validation
- Veridical reporting with complete empirical evidence

Usage:
    python scripts/complexity_energetics/comprehensive_analysis.py [--manifest path/to/manifest.yaml] [--output output_dir]

References:
- Agent-based modeling: https://eprints.whiterose.ac.uk/81723/
- Complexity-entropy analysis: https://arxiv.org/abs/0806.4789
- Thermodynamic computing: https://pubmed.ncbi.nlm.nih.gov/28505845/
- Reproducible research: https://doi.org/10.1038/s41586-020-2196-x
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

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
from antstack_core.analysis.veridical_reporting import VeridicalReporter, EmpiricalEvidence, CaseStudy
from antstack_core.figures.advanced_plots import (
    complexity_entropy_diagram, network_visualization, thermodynamic_phase_diagram,
    multi_scale_visualization
)
from antstack_core.figures.publication_plots import FigureManager


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
    - Bootstrap validation with bias correction
    
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
        
        # Generate time series data (energy over time)
        time_series_data.extend([breakdown.total] * 10)  # 10 time points per parameter
    
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
    
    emergent_metrics = agent_model.simulate(seed=42)
    
    # Network complexity analysis
    network_analyzer = NetworkComplexityAnalyzer()
    network_complexity = network_analyzer.analyze_network_complexity(network_data[0])
    
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
    ce_results = ce_analyzer.create_complexity_entropy_diagram(time_series_data)
    
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
        f"{module_name.title()} Complexity-Entropy Analysis",
        str(ce_plot_path)
    )
    
    # Network visualization
    network_plot_path = plot_dir / f"{module_name}_{param_name}_network.png"
    network_visualization(
        network_data[0],
        [f"Node {i}" for i in range(len(network_data[0]))],
        f"{module_name.title()} Network Structure",
        str(network_plot_path)
    )
    
    # Thermodynamic phase diagram
    temp_values = [300.0] * len(param_values)  # Constant temperature
    pressure_values = [101325.0 + i * 1000 for i in range(len(param_values))]  # Varying pressure
    efficiency_values = [thermo.thermodynamic_efficiency for thermo in thermo_results]
    
    phase_plot_path = plot_dir / f"{module_name}_{param_name}_phase_diagram.png"
    thermodynamic_phase_diagram(
        temp_values, pressure_values, efficiency_values,
        f"{module_name.title()} Thermodynamic Phase Diagram",
        str(phase_plot_path)
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
            'phase_diagram': str(phase_plot_path),
            'multi_scale': str(multi_scale_plot_path)
        }
    }


def generate_veridical_report(all_results: Dict[str, Any], output_dir: Path) -> None:
    """Generate comprehensive veridical report with complete empirical evidence.
    
    Args:
        all_results: Complete analysis results
        output_dir: Output directory for report
    """
    print("Generating comprehensive veridical report...")
    
    # Initialize veridical reporter
    reporter = VeridicalReporter(output_dir)
    
    # Extract experimental data
    experimental_data = {}
    for result in all_results.get('module_analyses', []):
        module = result['module']
        param_name = result['parameter']
        energy_values = [comp_data['energy'] for comp_data in result['computational_data']]
        experimental_data[f"{module}_{param_name}"] = energy_values
    
    # Generate empirical evidence
    evidence = reporter.generate_empirical_evidence(experimental_data, all_results)
    
    # Create case studies
    case_studies = []
    
    # Case Study 1: Body Module Analysis
    body_result = next((r for r in all_results.get('module_analyses', []) if r['module'] == 'body'), None)
    if body_result:
        case_study_1 = reporter.create_case_study(
            "body_analysis",
            "AntBody Mechanical Efficiency Analysis",
            {f"joint_count_{v}": [v] for v in body_result['parameter_values']},
            body_result,
            "Comprehensive analysis of mechanical efficiency in AntBody module using contact dynamics simulation and energy modeling."
        )
        case_studies.append(case_study_1)
    
    # Case Study 2: Brain Module Analysis
    brain_result = next((r for r in all_results.get('module_analyses', []) if r['module'] == 'brain'), None)
    if brain_result:
        case_study_2 = reporter.create_case_study(
            "brain_analysis",
            "AntBrain Neural Processing Efficiency Analysis",
            {f"channels_{v}": [v] for v in brain_result['parameter_values']},
            brain_result,
            "Advanced analysis of neural processing efficiency using sparse neural networks and biological connectivity patterns."
        )
        case_studies.append(case_study_2)
    
    # Case Study 3: Mind Module Analysis
    mind_result = next((r for r in all_results.get('module_analyses', []) if r['module'] == 'mind'), None)
    if mind_result:
        case_study_3 = reporter.create_case_study(
            "mind_analysis",
            "AntMind Cognitive Inference Analysis",
            {f"horizon_{v}": [v] for v in mind_result['parameter_values']},
            mind_result,
            "Comprehensive analysis of cognitive inference efficiency using active inference and bounded rationality frameworks."
        )
        case_studies.append(case_study_3)
    
    # Generate comprehensive report
    report_path = reporter.generate_comprehensive_report(evidence, case_studies, all_results)
    
    # Generate data-driven summary
    summary = reporter.generate_data_driven_summary(evidence, case_studies)
    
    # Save summary
    summary_path = output_dir / "data_driven_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Comprehensive veridical report generated: {report_path}")
    print(f"Data-driven summary saved: {summary_path}")


def main():
    """Main analysis orchestration function."""
    parser = argparse.ArgumentParser(description="Generate comprehensive complexity energetics analysis")
    parser.add_argument("--manifest", default="complexity_energetics/manifest.example.yaml",
                       help="Path to experiment manifest")
    parser.add_argument("--output", default="comprehensive_analysis_output",
                       help="Output directory for results")
    parser.add_argument("--modules", nargs="+", default=["body", "brain", "mind"],
                       help="Modules to analyze")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("COMPREHENSIVE COMPLEXITY ENERGETICS ANALYSIS")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Modules to analyze: {args.modules}")
    print()
    
    # Load configuration
    try:
        manifest = ExperimentManifest.load(args.manifest)
        print(f"Loaded manifest: {args.manifest}")
    except Exception as e:
        print(f"Error loading manifest: {e}")
        print("Using default configuration...")
        manifest = ExperimentManifest(
            seed=42,
            coefficients={
                'e_flop_pj': 1.0,
                'e_sram_pj_per_byte': 0.1,
                'e_dram_pj_per_byte': 20.0,
                'e_spike_aj': 1.0,
                'baseline_w': 0.05
            }
        )
    
    # Create energy coefficients
    coeffs = EnergyCoefficients(
        flops_pj=float(manifest.coefficients.get("e_flop_pj", 1.0)),
        sram_pj_per_byte=float(manifest.coefficients.get("e_sram_pj_per_byte", 0.1)),
        dram_pj_per_byte=float(manifest.coefficients.get("e_dram_pj_per_byte", 20.0)),
        spike_aj=float(manifest.coefficients.get("e_spike_aj", 1.0)),
        baseline_w=float(manifest.coefficients.get("baseline_w", 0.05))
    )
    
    # Define analysis configurations
    analysis_configs = {
        'body': {
            'param': 'J',
            'values': [6, 12, 18, 24, 36],
            'base_params': {'J': 18, 'C': 12, 'S': 256, 'hz': 100, 'contact_solver': 'pgs'}
        },
        'brain': {
            'param': 'K',
            'values': [64, 128, 256, 512, 1024],
            'base_params': {'K': 128, 'N_KC': 50000, 'rho': 0.02, 'H': 64, 'hz': 100, 'connectivity_pattern': 'biological'}
        },
        'mind': {
            'param': 'H_p',
            'values': [5, 10, 15, 20, 25],
            'base_params': {'B': 4, 'H_p': 20, 'hz': 100, 'state_dim': 16, 'action_dim': 6, 'hierarchical': False}
        }
    }
    
    # Perform comprehensive analyses
    module_analyses = []
    start_time = time.time()
    
    for module in args.modules:
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
            print(f"Warning: Unknown module '{module}', skipping...")
    
    total_analysis_time = time.time() - start_time
    print(f"\nTotal analysis time: {total_analysis_time:.2f} seconds")
    
    # Compile all results
    all_results = {
        'module_analyses': module_analyses,
        'configuration': {
            'manifest_path': args.manifest,
            'coefficients': manifest.coefficients,
            'modules_analyzed': args.modules,
            'analysis_time': total_analysis_time
        },
        'timestamp': time.time()
    }
    
    # Save results as JSON
    results_path = output_dir / "comprehensive_analysis_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"Analysis results saved: {results_path}")
    
    # Generate veridical report
    generate_veridical_report(all_results, output_dir)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved in: {output_dir}")
    print(f"- Analysis results: {results_path}")
    print(f"- Comprehensive report: {output_dir / 'reports' / 'comprehensive_analysis_report.md'}")
    print(f"- Data summary: {output_dir / 'data_driven_summary.json'}")
    print(f"- Plots directory: {output_dir / 'plots'}")
    print(f"- Supporting data: {output_dir / 'data'}")
    print(f"- Analysis details: {output_dir / 'analysis'}")


if __name__ == "__main__":
    main()
