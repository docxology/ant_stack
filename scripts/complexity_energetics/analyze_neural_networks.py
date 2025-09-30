#!/usr/bin/env python3
"""Neural network analysis orchestrator.

Thin orchestrator for sparse neural network analysis using ce.workloads methods.
Analyzes different connectivity patterns and sparsity levels.

Usage:
    python scripts/analyze_neural_networks.py [--output output_dir] [--pattern biological]
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from antstack_core.analysis import (
    calculate_sparse_neural_complexity, enhanced_brain_workload_closed_form
)
from antstack_core.analysis import (
    estimate_detailed_energy, analyze_scaling_relationship, 
    calculate_energy_efficiency_metrics
)
from antstack_core.analysis import EnergyCoefficients
from antstack_core.figures import line_plot, scatter_plot, bar_plot


def analyze_connectivity_patterns(network_sizes: List[int], sparsity: float, 
                                output_dir: Path) -> Dict[str, Any]:
    """Compare different neural connectivity patterns across network sizes."""
    print("Analyzing neural connectivity patterns...")
    
    patterns = ["random", "small_world", "scale_free", "biological"]
    results = {}
    
    for pattern in patterns:
        flops_data = []
        memory_data = []
        spikes_data = []
        
        for N in network_sizes:
            flops, memory, spikes = calculate_sparse_neural_complexity(N, sparsity, pattern)
            flops_data.append(flops)
            memory_data.append(memory)
            spikes_data.append(spikes)
        
        # Analyze scaling relationships
        flops_scaling = analyze_scaling_relationship(network_sizes, flops_data)
        memory_scaling = analyze_scaling_relationship(network_sizes, memory_data)
        spikes_scaling = analyze_scaling_relationship(network_sizes, spikes_data)
        
        results[pattern] = {
            'network_sizes': network_sizes,
            'flops': flops_data,
            'memory': memory_data,
            'spikes': spikes_data,
            'flops_scaling': flops_scaling,
            'memory_scaling': memory_scaling,
            'spikes_scaling': spikes_scaling
        }
    
    # Generate comparison plots
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # FLOPs comparison
    flops_series = [results[pattern]['flops'] for pattern in patterns]
    flops_plot = plot_dir / "connectivity_flops_comparison.png"
    line_plot(
        network_sizes, flops_series, patterns,
        "Neural Connectivity FLOPs Comparison",
        "Network Size (neurons)", "FLOPs per Decision",
        str(flops_plot)
    )
    
    # Spikes comparison
    spikes_series = [results[pattern]['spikes'] for pattern in patterns]
    spikes_plot = plot_dir / "connectivity_spikes_comparison.png"
    line_plot(
        network_sizes, spikes_series, patterns,
        "Neural Connectivity Spikes Comparison",
        "Network Size (neurons)", "Spikes per Decision", 
        str(spikes_plot)
    )
    
    results['plots'] = {
        'flops_comparison': str(flops_plot),
        'spikes_comparison': str(spikes_plot)
    }
    
    return results


def analyze_sparsity_effects(network_size: int, sparsity_levels: List[float],
                           output_dir: Path) -> Dict[str, Any]:
    """Analyze effects of different sparsity levels on neural network performance."""
    print("Analyzing sparsity effects...")
    
    coeffs = EnergyCoefficients()
    results = {}
    
    energy_values = []
    flops_values = []
    spikes_values = []
    
    for rho in sparsity_levels:
        # Use biological connectivity as baseline
        flops, memory, spikes = calculate_sparse_neural_complexity(
            network_size, rho, "biological"
        )
        
        # Calculate energy using brain workload model
        params = {
            'K': 128, 'N_KC': network_size, 'rho': rho, 'H': 64, 'hz': 100,
            'connectivity_pattern': 'biological'
        }
        load = enhanced_brain_workload_closed_form(0.01, params)
        breakdown = estimate_detailed_energy(load, coeffs, 0.01)
        
        energy_values.append(breakdown.total)
        flops_values.append(load.flops)
        spikes_values.append(load.spikes)
        
        results[f"rho_{rho:.3f}"] = {
            'sparsity': rho,
            'energy_total': breakdown.total,
            'energy_breakdown': breakdown.to_dict(),
            'flops': load.flops,
            'spikes': load.spikes,
            'memory': load.sram_bytes
        }
    
    # Analyze scaling relationships
    energy_scaling = analyze_scaling_relationship(sparsity_levels, energy_values)
    flops_scaling = analyze_scaling_relationship(sparsity_levels, flops_values)
    spikes_scaling = analyze_scaling_relationship(sparsity_levels, spikes_values)
    
    # Generate sparsity analysis plots
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    energy_plot = plot_dir / "sparsity_energy_scaling.png"
    line_plot(
        sparsity_levels, [energy_values], ["Energy"],
        "Energy vs Neural Sparsity",
        "Sparsity Level (ρ)", "Energy per Decision (J)",
        str(energy_plot)
    )
    
    spikes_plot = plot_dir / "sparsity_spikes_scaling.png"
    line_plot(
        sparsity_levels, [spikes_values], ["Spikes"],
        "Spike Count vs Neural Sparsity", 
        "Sparsity Level (ρ)", "Spikes per Decision",
        str(spikes_plot)
    )
    
    results['summary'] = {
        'sparsity_levels': sparsity_levels,
        'energy_values': energy_values,
        'flops_values': flops_values,
        'spikes_values': spikes_values,
        'energy_scaling': energy_scaling,
        'flops_scaling': flops_scaling,
        'spikes_scaling': spikes_scaling,
        'plots': {
            'energy_scaling': str(energy_plot),
            'spikes_scaling': str(spikes_plot)
        }
    }
    
    return results


def analyze_brain_scaling(K_values: List[int], output_dir: Path) -> Dict[str, Any]:
    """Analyze brain energy scaling with different AL input channel counts."""
    print("Analyzing brain scaling with AL channels...")
    
    coeffs = EnergyCoefficients()
    results = {}
    
    energy_values = []
    flops_values = []
    memory_values = []
    
    for K in K_values:
        params = {
            'K': K, 'N_KC': 50000, 'rho': 0.02, 'H': 64, 'hz': 100,
            'connectivity_pattern': 'biological'
        }
        
        load = enhanced_brain_workload_closed_form(0.01, params)
        breakdown = estimate_detailed_energy(load, coeffs, 0.01)
        
        energy_values.append(breakdown.total)
        flops_values.append(load.flops)
        memory_values.append(load.sram_bytes + load.dram_bytes)
        
        results[f"K_{K}"] = {
            'K': K,
            'energy_total': breakdown.total,
            'energy_breakdown': breakdown.to_dict(),
            'flops': load.flops,
            'memory': load.sram_bytes + load.dram_bytes,
            'spikes': load.spikes
        }
    
    # Analyze scaling relationships
    energy_scaling = analyze_scaling_relationship(K_values, energy_values)
    flops_scaling = analyze_scaling_relationship(K_values, flops_values)
    memory_scaling = analyze_scaling_relationship(K_values, memory_values)
    
    # Calculate efficiency metrics
    performance_proxy = [1.0 / K for K in K_values]  # Inverse for efficiency analysis
    efficiency_metrics = calculate_energy_efficiency_metrics(
        energy_values, performance_proxy
    )
    
    # Generate brain scaling plots
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    energy_plot = plot_dir / "brain_K_energy_scaling.png"
    line_plot(
        K_values, [energy_values], ["Brain Energy"],
        "Brain Energy Scaling vs AL Channels",
        "AL Input Channels (K)", "Energy per Decision (J)",
        str(energy_plot)
    )
    
    combined_plot = plot_dir / "brain_K_energy_flops.png"
    scatter_plot(
        flops_values, energy_values,
        "Brain: Energy vs Computational Complexity",
        "FLOPs per Decision", "Energy per Decision (J)",
        str(combined_plot)
    )
    
    results['summary'] = {
        'K_values': K_values,
        'energy_values': energy_values,
        'flops_values': flops_values,
        'memory_values': memory_values,
        'energy_scaling': energy_scaling,
        'flops_scaling': flops_scaling,
        'memory_scaling': memory_scaling,
        'efficiency_metrics': efficiency_metrics,
        'plots': {
            'energy_scaling': str(energy_plot),
            'combined': str(combined_plot)
        }
    }
    
    return results


def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(description="Analyze neural network patterns and scaling")
    parser.add_argument("--output", default="neural_analysis_output",
                       help="Output directory for results")
    parser.add_argument("--pattern", choices=["random", "small_world", "scale_free", "biological"],
                       default="biological", help="Primary connectivity pattern")
    parser.add_argument("--max-size", type=int, default=100000,
                       help="Maximum network size to analyze")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Neural Network Analysis Orchestrator")
    print("=" * 60)
    
    # Define analysis parameters
    network_sizes = [1000, 5000, 10000, 25000, 50000, args.max_size]
    sparsity_levels = [0.005, 0.01, 0.02, 0.05, 0.1]
    K_values = [32, 64, 128, 256, 512, 1024]
    
    # Analyze connectivity patterns
    connectivity_results = analyze_connectivity_patterns(
        network_sizes, 0.02, output_dir
    )
    
    # Analyze sparsity effects
    sparsity_results = analyze_sparsity_effects(50000, sparsity_levels, output_dir)
    
    # Analyze brain scaling
    brain_results = analyze_brain_scaling(K_values, output_dir)
    
    # Compile comprehensive results
    all_results = {
        'connectivity_analysis': connectivity_results,
        'sparsity_analysis': sparsity_results,
        'brain_scaling_analysis': brain_results,
        'configuration': {
            'network_sizes': network_sizes,
            'sparsity_levels': sparsity_levels,
            'K_values': K_values,
            'primary_pattern': args.pattern,
            'max_size': args.max_size
        }
    }
    
    # Save results
    results_path = output_dir / "neural_network_analysis.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nNeural network analysis complete!")
    print(f"Results saved in: {output_dir}")
    print(f"- Analysis results: {results_path}")
    print(f"- Plots directory: {output_dir / 'plots'}")
    
    # Print key findings
    print(f"\nKey Findings:")
    
    # Best connectivity pattern
    patterns = ["random", "small_world", "scale_free", "biological"]
    best_pattern = min(patterns, key=lambda p: connectivity_results[p]['flops'][-1])
    print(f"- Most efficient connectivity: {best_pattern}")
    
    # Sparsity scaling
    sparsity_regime = sparsity_results['summary']['energy_scaling'].get('regime', 'unknown')
    print(f"- Sparsity energy scaling: {sparsity_regime}")
    
    # Brain scaling
    brain_regime = brain_results['summary']['energy_scaling'].get('regime', 'unknown')
    brain_exponent = brain_results['summary']['energy_scaling'].get('scaling_exponent', 0)
    print(f"- Brain K scaling: {brain_regime} (exponent: {brain_exponent:.3f})")


if __name__ == "__main__":
    main()
