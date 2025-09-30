#!/usr/bin/env python3
"""Comprehensive analysis script for complexity and energetics research.

This script orchestrates the complete analysis pipeline including:
- Enhanced workload calculations with realistic complexity models
- Detailed energy breakdown and efficiency analysis
- Scaling relationship identification and power law analysis
- Theoretical limit comparisons and efficiency benchmarking
- Publication-quality figure generation with comprehensive captions

Usage:
    python scripts/generate_comprehensive_analysis.py [--manifest path/to/manifest.yaml] [--output output_dir]

References:
- Scientific computing workflows: https://doi.org/10.1371/journal.pcbi.1004668
- Reproducible research practices: https://doi.org/10.1038/s41586-020-2196-x
"""

import argparse
import os
import sys
import json
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
from antstack_core.figures import bar_plot, line_plot, scatter_plot


def analyze_module_scaling(module_name: str, param_name: str, param_values: List[float],
                          base_params: Dict[str, Any], coeffs: EnergyCoefficients,
                          output_dir: Path) -> Dict[str, Any]:
    """Analyze scaling behavior of a specific module with comprehensive metrics.
    
    Performs detailed scaling analysis including:
    - Energy scaling with parameter variation
    - Theoretical scaling law identification
    - Efficiency metrics calculation
    - Statistical validation of scaling relationships
    
    Args:
        module_name: Name of module ('body', 'brain', 'mind')
        param_name: Parameter to vary (e.g., 'K', 'J', 'C')
        param_values: List of parameter values to test
        base_params: Base parameter configuration
        coeffs: Energy coefficients for calculations
        output_dir: Directory for output files
    
    Returns:
        Dictionary containing scaling analysis results
    """
    print(f"Analyzing {module_name} scaling vs {param_name}...")
    
    # Select appropriate workload function
    workload_funcs = {
        'body': enhanced_body_workload_closed_form,
        'brain': enhanced_brain_workload_closed_form,
        'mind': enhanced_mind_workload_closed_form
    }
    
    if module_name not in workload_funcs:
        raise ValueError(f"Unknown module: {module_name}")
    
    workload_func = workload_funcs[module_name]
    
    # Calculate energy for each parameter value
    energy_values = []
    flops_values = []
    memory_values = []
    
    for param_val in param_values:
        params = dict(base_params)
        params[param_name] = param_val
        
        # Calculate workload
        load = workload_func(0.01, params)  # 10ms decision
        breakdown = estimate_detailed_energy(load, coeffs, 0.01)
        
        energy_values.append(breakdown.total)
        flops_values.append(load.flops)
        memory_values.append(load.sram_bytes + load.dram_bytes)
    
    # Analyze scaling relationships
    energy_scaling = analyze_scaling_relationship(param_values, energy_values)
    flops_scaling = analyze_scaling_relationship(param_values, flops_values)
    memory_scaling = analyze_scaling_relationship(param_values, memory_values)
    
    # Calculate efficiency metrics
    performance_proxy = [1.0 / p for p in param_values]  # Inverse relationship
    efficiency_metrics = calculate_energy_efficiency_metrics(
        energy_values, performance_proxy
    )
    
    # Generate plots
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Energy scaling plot
    energy_plot_path = plot_dir / f"{module_name}_{param_name}_energy_scaling.png"
    line_plot(
        param_values, [energy_values], [f"{module_name.title()} Energy"],
        f"{module_name.title()} Energy Scaling vs {param_name}",
        f"{param_name} (parameter value)",
        "Energy per Decision (J)",
        str(energy_plot_path)
    )
    
    # FLOPs scaling plot
    flops_plot_path = plot_dir / f"{module_name}_{param_name}_flops_scaling.png"
    line_plot(
        param_values, [flops_values], [f"{module_name.title()} FLOPs"],
        f"{module_name.title()} Computational Complexity vs {param_name}",
        f"{param_name} (parameter value)",
        "FLOPs per Decision",
        str(flops_plot_path)
    )
    
    # Combined analysis plot
    combined_plot_path = plot_dir / f"{module_name}_{param_name}_combined.png"
    scatter_plot(
        flops_values, energy_values,
        f"{module_name.title()}: Energy vs Computational Complexity",
        "FLOPs per Decision",
        "Energy per Decision (J)",
        str(combined_plot_path)
    )
    
    return {
        'module': module_name,
        'parameter': param_name,
        'parameter_values': param_values,
        'energy_values': energy_values,
        'flops_values': flops_values,
        'memory_values': memory_values,
        'energy_scaling': energy_scaling,
        'flops_scaling': flops_scaling,
        'memory_scaling': memory_scaling,
        'efficiency_metrics': efficiency_metrics,
        'plots': {
            'energy_scaling': str(energy_plot_path),
            'flops_scaling': str(flops_plot_path),
            'combined': str(combined_plot_path)
        }
    }


def compare_theoretical_limits(analysis_results: List[Dict[str, Any]], 
                             coeffs: EnergyCoefficients,
                             output_dir: Path) -> Dict[str, Any]:
    """Compare actual energy consumption with theoretical limits.
    
    Calculates theoretical minimum energy requirements based on:
    - Landauer's principle for computation
    - Thermodynamic limits for mechanical work
    - Information-theoretic bounds
    
    Args:
        analysis_results: Results from module scaling analyses
        coeffs: Energy coefficients
        output_dir: Output directory for plots
    
    Returns:
        Dictionary containing theoretical limit analysis
    """
    print("Comparing with theoretical limits...")
    
    theoretical_data = []
    actual_data = []
    module_labels = []
    
    for result in analysis_results:
        module = result['module']
        avg_flops = sum(result['flops_values']) / len(result['flops_values'])
        avg_energy = sum(result['energy_values']) / len(result['energy_values'])
        
        # Calculate theoretical limits
        limits = estimate_theoretical_limits({
            'flops': avg_flops,
            'bits_processed': avg_flops * 64,  # Assume 64-bit operations
            'mechanical_work_j': 0.001 if module == 'body' else 0.0
        })
        
        theoretical_data.append(limits['total_theoretical_j'])
        actual_data.append(avg_energy)
        module_labels.append(module.title())
    
    # Calculate efficiency ratios
    efficiency_ratios = [actual / theoretical for actual, theoretical in zip(actual_data, theoretical_data)]
    
    # Generate comparison plot
    plot_dir = output_dir / "plots"
    comparison_plot_path = plot_dir / "theoretical_limits_comparison.png"
    
    bar_plot(
        module_labels + [f"{label} (Theoretical)" for label in module_labels],
        actual_data + theoretical_data,
        "Actual vs Theoretical Energy Limits",
        str(comparison_plot_path),
        ylabel="Energy per Decision (J)"
    )
    
    return {
        'modules': module_labels,
        'actual_energy': actual_data,
        'theoretical_energy': theoretical_data,
        'efficiency_ratios': efficiency_ratios,
        'comparison_plot': str(comparison_plot_path)
    }


def generate_comprehensive_report(all_results: Dict[str, Any], output_dir: Path) -> None:
    """Generate comprehensive analysis report with detailed findings.
    
    Creates a detailed markdown report including:
    - Executive summary of key findings
    - Detailed scaling analysis results
    - Theoretical limit comparisons
    - Efficiency recommendations
    - Figure references and captions
    
    Args:
        all_results: Complete analysis results
        output_dir: Output directory for report
    """
    report_path = output_dir / "comprehensive_analysis_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Comprehensive Complexity and Energetics Analysis Report\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write("This report presents a comprehensive analysis of computational complexity ")
        f.write("and energy consumption across the Ant Stack architecture, including ")
        f.write("AntBody (physical dynamics), AntBrain (neural processing), and AntMind ")
        f.write("(cognitive inference). The analysis incorporates realistic complexity ")
        f.write("models, theoretical limit comparisons, and scaling law identification.\n\n")
        
        # Key Findings
        f.write("### Key Findings\n\n")
        
        scaling_results = all_results.get('scaling_analyses', [])
        for result in scaling_results:
            module = result['module']
            param = result['parameter']
            scaling = result.get('energy_scaling', {})
            
            if 'scaling_exponent' in scaling:
                exponent = scaling['scaling_exponent']
                regime = scaling.get('regime', 'unknown')
                r_squared = scaling.get('r_squared', 0)
                
                f.write(f"- **{module.title()} Module**: Energy scales as {param}^{exponent:.2f} ")
                f.write(f"({regime} regime, R² = {r_squared:.3f})\n")
        
        f.write("\n")
        
        # Theoretical Limits Analysis
        if 'theoretical_limits' in all_results:
            limits = all_results['theoretical_limits']
            f.write("### Theoretical Efficiency Analysis\n\n")
            
            for i, (module, ratio) in enumerate(zip(limits['modules'], limits['efficiency_ratios'])):
                f.write(f"- **{module}**: Operating at {ratio:.1f}× theoretical minimum ")
                f.write(f"(efficiency = {100/ratio:.1f}%)\n")
            
            f.write("\n")
        
        # Detailed Results
        f.write("## Detailed Analysis Results\n\n")
        
        for result in scaling_results:
            module = result['module']
            param = result['parameter']
            
            f.write(f"### {module.title()} Module Scaling Analysis\n\n")
            
            # Scaling relationships
            energy_scaling = result.get('energy_scaling', {})
            flops_scaling = result.get('flops_scaling', {})
            
            f.write("#### Scaling Relationships\n\n")
            if 'scaling_exponent' in energy_scaling:
                f.write(f"- **Energy Scaling**: E ∝ {param}^{energy_scaling['scaling_exponent']:.3f} ")
                f.write(f"(R² = {energy_scaling.get('r_squared', 0):.3f})\n")
            
            if 'scaling_exponent' in flops_scaling:
                f.write(f"- **Computational Scaling**: FLOPs ∝ {param}^{flops_scaling['scaling_exponent']:.3f} ")
                f.write(f"(R² = {flops_scaling.get('r_squared', 0):.3f})\n")
            
            # Efficiency metrics
            efficiency = result.get('efficiency_metrics', {})
            if efficiency:
                f.write("\n#### Efficiency Metrics\n\n")
                f.write(f"- **Cost of Transport**: {efficiency.get('cost_of_transport', 0):.4f}\n")
                f.write(f"- **Performance per Joule**: {efficiency.get('performance_per_joule', 0):.2e}\n")
                f.write(f"- **Energy Variance**: {efficiency.get('efficiency_variance', 0):.2e}\n")
            
            # Figures
            plots = result.get('plots', {})
            if plots:
                f.write("\n#### Generated Figures\n\n")
                for plot_type, plot_path in plots.items():
                    plot_name = plot_type.replace('_', ' ').title()
                    f.write(f"- **{plot_name}**: `{plot_path}`\n")
            
            f.write("\n")
        
        # Recommendations
        f.write("## Design Recommendations\n\n")
        f.write("Based on the scaling analysis and efficiency metrics:\n\n")
        
        f.write("1. **Computational Efficiency**: Focus optimization efforts on modules ")
        f.write("showing super-linear scaling (exponent > 1.5) as these will dominate ")
        f.write("energy consumption at scale.\n\n")
        
        f.write("2. **Parameter Tuning**: For modules operating significantly above ")
        f.write("theoretical limits (>10× minimum), investigate algorithmic improvements ")
        f.write("or hardware acceleration opportunities.\n\n")
        
        f.write("3. **Scaling Considerations**: Design systems to operate in linear or ")
        f.write("sub-linear scaling regimes where possible, using techniques like ")
        f.write("sparsity, approximation, and bounded rationality.\n\n")
        
        # References
        f.write("## References\n\n")
        f.write("- Scaling laws in complex systems: https://doi.org/10.1126/science.1062081\n")
        f.write("- Energy-efficient computing: https://ieeexplore.ieee.org/document/8845760\n")
        f.write("- Landauer's principle: https://doi.org/10.1038/nature10872\n")
        f.write("- Neuromorphic energy analysis: https://arxiv.org/abs/2505.03764\n")
        f.write("- Active inference complexity: https://doi.org/10.1098/rsif.2017.0792\n")
    
    print(f"Comprehensive report generated: {report_path}")


def main():
    """Main analysis orchestration function."""
    parser = argparse.ArgumentParser(description="Generate comprehensive complexity and energetics analysis")
    parser.add_argument("--manifest", default="complexity_energetics/manifest.example.yaml",
                       help="Path to experiment manifest")
    parser.add_argument("--output", default="analysis_output",
                       help="Output directory for results")
    parser.add_argument("--modules", nargs="+", default=["body", "brain", "mind"],
                       help="Modules to analyze")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Load configuration
    try:
        manifest = ExperimentManifest.load(args.manifest)
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
    
    # Perform scaling analyses
    scaling_results = []
    for module in args.modules:
        if module in analysis_configs:
            config = analysis_configs[module]
            result = analyze_module_scaling(
                module, config['param'], config['values'],
                config['base_params'], coeffs, output_dir
            )
            scaling_results.append(result)
    
    # Compare with theoretical limits
    theoretical_analysis = compare_theoretical_limits(scaling_results, coeffs, output_dir)
    
    # Compile all results
    all_results = {
        'scaling_analyses': scaling_results,
        'theoretical_limits': theoretical_analysis,
        'configuration': {
            'manifest_path': args.manifest,
            'coefficients': manifest.coefficients,
            'modules_analyzed': args.modules
        }
    }
    
    # Save results as JSON
    results_path = output_dir / "analysis_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"Analysis results saved: {results_path}")
    
    # Generate comprehensive report
    generate_comprehensive_report(all_results, output_dir)
    
    print(f"\nComprehensive analysis complete!")
    print(f"Results saved in: {output_dir}")
    print(f"- Analysis results: {results_path}")
    print(f"- Comprehensive report: {output_dir / 'comprehensive_analysis_report.md'}")
    print(f"- Plots directory: {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
