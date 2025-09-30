#!/usr/bin/env python3
"""Contact dynamics analysis orchestrator.

Thin orchestrator for contact complexity analysis using ce.workloads methods.
Analyzes different solver algorithms and contact configurations.

Usage:
    python scripts/analyze_contact_dynamics.py [--output output_dir] [--solver pgs|lcp|mlcp]
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from antstack_core.analysis import calculate_contact_complexity, enhanced_body_workload_closed_form
from antstack_core.analysis import estimate_detailed_energy, analyze_scaling_relationship
from antstack_core.analysis import EnergyCoefficients
from antstack_core.figures import line_plot, scatter_plot, bar_plot


def analyze_solver_comparison(contact_counts: List[int], output_dir: Path) -> Dict[str, Any]:
    """Compare different contact solver algorithms across contact counts."""
    print("Analyzing contact solver performance...")
    
    solvers = ["pgs", "lcp", "mlcp"]
    results = {}
    
    for solver in solvers:
        flops_data = []
        memory_data = []
        
        for C in contact_counts:
            flops, memory, _ = calculate_contact_complexity(18, C, solver)
            flops_data.append(flops)
            memory_data.append(memory)
        
        # Analyze scaling relationships
        flops_scaling = analyze_scaling_relationship(contact_counts, flops_data)
        memory_scaling = analyze_scaling_relationship(contact_counts, memory_data)
        
        results[solver] = {
            'contact_counts': contact_counts,
            'flops': flops_data,
            'memory': memory_data,
            'flops_scaling': flops_scaling,
            'memory_scaling': memory_scaling
        }
    
    # Generate comparison plots
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # FLOPs comparison
    flops_series = [results[solver]['flops'] for solver in solvers]
    flops_plot = plot_dir / "contact_solver_flops_comparison.png"
    line_plot(
        contact_counts, flops_series, solvers,
        "Contact Solver FLOPs Comparison",
        "Number of Contacts", "FLOPs per Solve",
        str(flops_plot)
    )
    
    # Memory comparison
    memory_series = [results[solver]['memory'] for solver in solvers]
    memory_plot = plot_dir / "contact_solver_memory_comparison.png"
    line_plot(
        contact_counts, memory_series, solvers,
        "Contact Solver Memory Comparison", 
        "Number of Contacts", "Memory (bytes)",
        str(memory_plot)
    )
    
    results['plots'] = {
        'flops_comparison': str(flops_plot),
        'memory_comparison': str(memory_plot)
    }
    
    return results


def analyze_terrain_effects(terrain_configs: List[Dict[str, Any]], output_dir: Path) -> Dict[str, Any]:
    """Analyze energy effects of different terrain configurations."""
    print("Analyzing terrain effects on energy consumption...")
    
    coeffs = EnergyCoefficients()
    results = {}
    
    terrain_names = []
    energy_values = []
    flops_values = []
    
    for config in terrain_configs:
        name = config['name']
        params = config['params']
        
        # Calculate workload with terrain-specific parameters
        load = enhanced_body_workload_closed_form(0.01, params)  # 10ms decision
        breakdown = estimate_detailed_energy(load, coeffs, 0.01)
        
        terrain_names.append(name)
        energy_values.append(breakdown.total)
        flops_values.append(load.flops)
        
        results[name] = {
            'energy_total': breakdown.total,
            'energy_breakdown': breakdown.to_dict(),
            'flops': load.flops,
            'memory': load.sram_bytes + load.dram_bytes,
            'params': params
        }
    
    # Generate terrain comparison plots
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    energy_plot = plot_dir / "terrain_energy_comparison.png"
    bar_plot(
        terrain_names, energy_values,
        "Energy Consumption by Terrain Type",
        str(energy_plot), ylabel="Energy per Decision (J)"
    )
    
    flops_plot = plot_dir / "terrain_flops_comparison.png"
    bar_plot(
        terrain_names, flops_values,
        "Computational Load by Terrain Type",
        str(flops_plot), ylabel="FLOPs per Decision"
    )
    
    results['summary'] = {
        'terrain_names': terrain_names,
        'energy_values': energy_values,
        'flops_values': flops_values,
        'plots': {
            'energy_comparison': str(energy_plot),
            'flops_comparison': str(flops_plot)
        }
    }
    
    return results


def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(description="Analyze contact dynamics and terrain effects")
    parser.add_argument("--output", default="contact_analysis_output",
                       help="Output directory for results")
    parser.add_argument("--solver", choices=["pgs", "lcp", "mlcp"], default="pgs",
                       help="Primary solver to analyze")
    parser.add_argument("--max-contacts", type=int, default=50,
                       help="Maximum number of contacts to analyze")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Contact Dynamics Analysis Orchestrator")
    print("=" * 60)
    
    # Define analysis parameters
    contact_counts = [5, 10, 15, 20, 25, 30, 40, args.max_contacts]
    
    # Analyze solver comparison
    solver_results = analyze_solver_comparison(contact_counts, output_dir)
    
    # Define terrain configurations for analysis
    terrain_configs = [
        {
            'name': 'Smooth',
            'params': {
                'J': 18, 'C': 8, 'S': 256, 'hz': 100,
                'contact_solver': 'pgs'
            }
        },
        {
            'name': 'Rough',
            'params': {
                'J': 18, 'C': 16, 'S': 256, 'hz': 100,
                'contact_solver': 'pgs'
            }
        },
        {
            'name': 'Steep',
            'params': {
                'J': 18, 'C': 20, 'S': 256, 'hz': 100,
                'contact_solver': 'lcp'  # More contacts need robust solver
            }
        },
        {
            'name': 'Soft',
            'params': {
                'J': 18, 'C': 12, 'S': 256, 'hz': 100,
                'contact_solver': 'mlcp'
            }
        }
    ]
    
    # Analyze terrain effects
    terrain_results = analyze_terrain_effects(terrain_configs, output_dir)
    
    # Compile comprehensive results
    all_results = {
        'solver_analysis': solver_results,
        'terrain_analysis': terrain_results,
        'configuration': {
            'contact_counts': contact_counts,
            'primary_solver': args.solver,
            'max_contacts': args.max_contacts
        }
    }
    
    # Save results
    results_path = output_dir / "contact_dynamics_analysis.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nContact dynamics analysis complete!")
    print(f"Results saved in: {output_dir}")
    print(f"- Analysis results: {results_path}")
    print(f"- Plots directory: {output_dir / 'plots'}")
    
    # Print key findings
    print(f"\nKey Findings:")
    print(f"- PGS scaling: {solver_results['pgs']['flops_scaling'].get('regime', 'unknown')}")
    print(f"- LCP scaling: {solver_results['lcp']['flops_scaling'].get('regime', 'unknown')}")
    print(f"- MLCP scaling: {solver_results['mlcp']['flops_scaling'].get('regime', 'unknown')}")
    
    max_energy_terrain = max(terrain_results['summary']['terrain_names'], 
                           key=lambda x: terrain_results[x]['energy_total'])
    min_energy_terrain = min(terrain_results['summary']['terrain_names'],
                           key=lambda x: terrain_results[x]['energy_total'])
    
    print(f"- Highest energy terrain: {max_energy_terrain}")
    print(f"- Lowest energy terrain: {min_energy_terrain}")


if __name__ == "__main__":
    main()
