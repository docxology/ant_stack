#!/usr/bin/env python3
"""Active inference analysis orchestrator.

Thin orchestrator for active inference complexity analysis using ce.workloads methods.
Analyzes policy horizons, branching factors, and bounded rationality effects.

Usage:
    python scripts/analyze_active_inference.py [--output output_dir] [--max-horizon 20]
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from antstack_core.analysis import (
    calculate_active_inference_complexity, enhanced_mind_workload_closed_form
)
from antstack_core.analysis import (
    estimate_detailed_energy, analyze_scaling_relationship,
    estimate_theoretical_limits, calculate_energy_efficiency_metrics
)
from antstack_core.analysis import EnergyCoefficients
from antstack_core.figures import line_plot, scatter_plot, bar_plot


def analyze_horizon_scaling(horizons: List[int], branching: int, state_dim: int, 
                          action_dim: int, output_dir: Path) -> Dict[str, Any]:
    """Analyze computational explosion with policy horizon scaling."""
    print("Analyzing policy horizon scaling...")
    
    coeffs = EnergyCoefficients()
    results = {}
    
    energy_values = []
    flops_values = []
    memory_values = []
    theoretical_limits = []
    
    for H_p in horizons:
        # Calculate active inference complexity
        ai_flops, ai_memory = calculate_active_inference_complexity(
            H_p, branching, state_dim, action_dim
        )
        
        # Calculate energy using mind workload model
        params = {
            'B': branching, 'H_p': H_p, 'hz': 100,
            'state_dim': state_dim, 'action_dim': action_dim,
            'hierarchical': False
        }
        
        load = enhanced_mind_workload_closed_form(0.01, params)
        breakdown = estimate_detailed_energy(load, coeffs, 0.01)
        
        # Calculate theoretical limits
        limits = estimate_theoretical_limits({
            'flops': load.flops,
            'bits_processed': load.flops * 64,
            'mechanical_work_j': 0.0
        })
        
        energy_values.append(breakdown.total)
        flops_values.append(load.flops)
        memory_values.append(load.sram_bytes)
        theoretical_limits.append(limits['total_theoretical_j'])
        
        results[f"H_p_{H_p}"] = {
            'horizon': H_p,
            'energy_total': breakdown.total,
            'energy_breakdown': breakdown.to_dict(),
            'flops': load.flops,
            'memory': load.sram_bytes,
            'theoretical_limit': limits['total_theoretical_j'],
            'efficiency_ratio': breakdown.total / limits['total_theoretical_j']
        }
    
    # Analyze scaling relationships
    energy_scaling = analyze_scaling_relationship(horizons, energy_values)
    flops_scaling = analyze_scaling_relationship(horizons, flops_values)
    memory_scaling = analyze_scaling_relationship(horizons, memory_values)
    
    # Generate horizon scaling plots
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    energy_plot = plot_dir / "horizon_energy_scaling.png"
    line_plot(
        horizons, [energy_values], ["Mind Energy"],
        "Mind Energy Scaling vs Policy Horizon",
        "Policy Horizon (steps)", "Energy per Decision (J)",
        str(energy_plot)
    )
    
    flops_plot = plot_dir / "horizon_flops_scaling.png"
    line_plot(
        horizons, [flops_values], ["Mind FLOPs"],
        "Mind FLOPs Scaling vs Policy Horizon",
        "Policy Horizon (steps)", "FLOPs per Decision",
        str(flops_plot)
    )
    
    # Theoretical limits comparison
    limits_plot = plot_dir / "horizon_theoretical_limits.png"
    line_plot(
        horizons, [energy_values, theoretical_limits], 
        ["Actual Energy", "Theoretical Limit"],
        "Actual vs Theoretical Energy Limits",
        "Policy Horizon (steps)", "Energy per Decision (J)",
        str(limits_plot)
    )
    
    results['summary'] = {
        'horizons': horizons,
        'energy_values': energy_values,
        'flops_values': flops_values,
        'memory_values': memory_values,
        'theoretical_limits': theoretical_limits,
        'energy_scaling': energy_scaling,
        'flops_scaling': flops_scaling,
        'memory_scaling': memory_scaling,
        'plots': {
            'energy_scaling': str(energy_plot),
            'flops_scaling': str(flops_plot),
            'theoretical_limits': str(limits_plot)
        }
    }
    
    return results


def analyze_branching_effects(horizon: int, branching_factors: List[int], 
                            state_dim: int, action_dim: int, output_dir: Path) -> Dict[str, Any]:
    """Analyze effects of different branching factors on complexity."""
    print("Analyzing branching factor effects...")
    
    coeffs = EnergyCoefficients()
    results = {}
    
    energy_values = []
    flops_values = []
    policy_counts = []
    
    for B in branching_factors:
        # Calculate raw complexity
        ai_flops, ai_memory = calculate_active_inference_complexity(
            horizon, B, state_dim, action_dim
        )
        
        # Calculate total policy count (before sampling)
        total_policies = B ** horizon
        effective_policies = min(1000, total_policies)  # Bounded rationality
        
        params = {
            'B': B, 'H_p': horizon, 'hz': 100,
            'state_dim': state_dim, 'action_dim': action_dim,
            'hierarchical': False
        }
        
        load = enhanced_mind_workload_closed_form(0.01, params)
        breakdown = estimate_detailed_energy(load, coeffs, 0.01)
        
        energy_values.append(breakdown.total)
        flops_values.append(load.flops)
        policy_counts.append(effective_policies)
        
        results[f"B_{B}"] = {
            'branching': B,
            'total_policies': total_policies,
            'effective_policies': effective_policies,
            'energy_total': breakdown.total,
            'flops': load.flops,
            'memory': load.sram_bytes,
            'bounded_rationality_active': total_policies > 1000
        }
    
    # Analyze scaling relationships
    energy_scaling = analyze_scaling_relationship(branching_factors, energy_values)
    flops_scaling = analyze_scaling_relationship(branching_factors, flops_values)
    
    # Generate branching analysis plots
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    energy_plot = plot_dir / "branching_energy_scaling.png"
    line_plot(
        branching_factors, [energy_values], ["Energy"],
        "Energy vs Branching Factor",
        "Branching Factor (B)", "Energy per Decision (J)",
        str(energy_plot)
    )
    
    policies_plot = plot_dir / "branching_policy_count.png"
    line_plot(
        branching_factors, [policy_counts], ["Effective Policies"],
        "Effective Policy Count vs Branching Factor",
        "Branching Factor (B)", "Effective Policies (bounded)",
        str(policies_plot)
    )
    
    results['summary'] = {
        'branching_factors': branching_factors,
        'energy_values': energy_values,
        'flops_values': flops_values,
        'policy_counts': policy_counts,
        'energy_scaling': energy_scaling,
        'flops_scaling': flops_scaling,
        'plots': {
            'energy_scaling': str(energy_plot),
            'policy_count': str(policies_plot)
        }
    }
    
    return results


def analyze_bounded_rationality(horizon: int, branching: int, state_dim: int,
                              action_dim: int, output_dir: Path) -> Dict[str, Any]:
    """Analyze effectiveness of bounded rationality strategies."""
    print("Analyzing bounded rationality effectiveness...")
    
    coeffs = EnergyCoefficients()
    
    # Compare different bounded rationality strategies
    strategies = [
        {'name': 'No Bounds', 'max_policies': None, 'hierarchical': False},
        {'name': 'Policy Sampling', 'max_policies': 1000, 'hierarchical': False},
        {'name': 'Hierarchical', 'max_policies': 1000, 'hierarchical': True},
        {'name': 'Conservative', 'max_policies': 100, 'hierarchical': False}
    ]
    
    results = {}
    strategy_names = []
    energy_values = []
    flops_values = []
    
    for strategy in strategies:
        name = strategy['name']
        
        # Calculate complexity with strategy
        if strategy['max_policies'] is None:
            # Unbounded (theoretical)
            total_policies = branching ** horizon
            effective_policies = total_policies
        else:
            total_policies = branching ** horizon
            effective_policies = min(strategy['max_policies'], total_policies)
        
        params = {
            'B': branching, 'H_p': horizon, 'hz': 100,
            'state_dim': state_dim, 'action_dim': action_dim,
            'hierarchical': strategy['hierarchical']
        }
        
        # For unbounded case, estimate theoretical energy
        if strategy['max_policies'] is None and total_policies > 10000:
            # Extrapolate from bounded case
            bounded_load = enhanced_mind_workload_closed_form(0.01, dict(params, hierarchical=False))
            scaling_factor = total_policies / 1000
            theoretical_flops = bounded_load.flops * scaling_factor
            theoretical_energy = theoretical_flops * 1e-12  # Rough estimate
            
            energy_values.append(theoretical_energy)
            flops_values.append(theoretical_flops)
        else:
            load = enhanced_mind_workload_closed_form(0.01, params)
            breakdown = estimate_detailed_energy(load, coeffs, 0.01)
            
            energy_values.append(breakdown.total)
            flops_values.append(load.flops)
        
        strategy_names.append(name)
        
        results[name] = {
            'strategy': strategy,
            'total_policies': total_policies,
            'effective_policies': effective_policies,
            'energy_total': energy_values[-1],
            'flops': flops_values[-1],
            'tractable': effective_policies <= 1000
        }
    
    # Generate bounded rationality comparison plots
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    energy_plot = plot_dir / "bounded_rationality_energy.png"
    bar_plot(
        strategy_names, energy_values,
        "Energy by Bounded Rationality Strategy",
        str(energy_plot), ylabel="Energy per Decision (J)"
    )
    
    flops_plot = plot_dir / "bounded_rationality_flops.png"
    bar_plot(
        strategy_names, flops_values,
        "FLOPs by Bounded Rationality Strategy",
        str(flops_plot), ylabel="FLOPs per Decision"
    )
    
    results['summary'] = {
        'strategy_names': strategy_names,
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
    parser = argparse.ArgumentParser(description="Analyze active inference complexity and scaling")
    parser.add_argument("--output", default="active_inference_output",
                       help="Output directory for results")
    parser.add_argument("--max-horizon", type=int, default=15,
                       help="Maximum policy horizon to analyze")
    parser.add_argument("--branching", type=int, default=4,
                       help="Default branching factor")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Active Inference Analysis Orchestrator")
    print("=" * 60)
    
    # Define analysis parameters
    horizons = list(range(5, args.max_horizon + 1, 2))
    branching_factors = [2, 3, 4, 5, 6]
    state_dim = 16
    action_dim = 6
    
    # Analyze horizon scaling
    horizon_results = analyze_horizon_scaling(
        horizons, args.branching, state_dim, action_dim, output_dir
    )
    
    # Analyze branching effects
    branching_results = analyze_branching_effects(
        10, branching_factors, state_dim, action_dim, output_dir
    )
    
    # Analyze bounded rationality
    bounded_results = analyze_bounded_rationality(
        12, args.branching, state_dim, action_dim, output_dir
    )
    
    # Compile comprehensive results
    all_results = {
        'horizon_analysis': horizon_results,
        'branching_analysis': branching_results,
        'bounded_rationality_analysis': bounded_results,
        'configuration': {
            'horizons': horizons,
            'branching_factors': branching_factors,
            'state_dim': state_dim,
            'action_dim': action_dim,
            'max_horizon': args.max_horizon,
            'default_branching': args.branching
        }
    }
    
    # Save results
    results_path = output_dir / "active_inference_analysis.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nActive inference analysis complete!")
    print(f"Results saved in: {output_dir}")
    print(f"- Analysis results: {results_path}")
    print(f"- Plots directory: {output_dir / 'plots'}")
    
    # Print key findings
    print(f"\nKey Findings:")
    
    # Horizon scaling
    horizon_regime = horizon_results['summary']['energy_scaling'].get('regime', 'unknown')
    horizon_exponent = horizon_results['summary']['energy_scaling'].get('scaling_exponent', 0)
    print(f"- Horizon energy scaling: {horizon_regime} (exponent: {horizon_exponent:.1f})")
    
    # Branching effects
    branching_regime = branching_results['summary']['energy_scaling'].get('regime', 'unknown')
    print(f"- Branching energy scaling: {branching_regime}")
    
    # Bounded rationality effectiveness
    strategies = bounded_results['summary']['strategy_names']
    energies = bounded_results['summary']['energy_values']
    best_strategy = strategies[energies.index(min(energies))]
    print(f"- Most efficient strategy: {best_strategy}")
    
    # Tractability threshold
    max_tractable_horizon = max(h for h in horizons 
                               if horizon_results[f"H_p_{h}"]['energy_total'] < 1.0)
    print(f"- Tractability threshold: H_p ≤ {max_tractable_horizon}")


if __name__ == "__main__":
    main()
