#!/usr/bin/env python3
"""Manuscript figure generation orchestrator.

Generates all figures for the complexity_energetics manuscript using enhanced methods.
Creates publication-quality figures with proper captions and statistical analysis.

Usage:
    python scripts/generate_manuscript_figures.py [--output assets_dir] [--format png]
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
    estimate_detailed_energy, analyze_scaling_relationship,
    calculate_energy_efficiency_metrics, estimate_theoretical_limits,
    bootstrap_mean_ci
)
from antstack_core.analysis import (
    enhanced_body_workload_closed_form, enhanced_brain_workload_closed_form,
    enhanced_mind_workload_closed_form, calculate_contact_complexity,
    calculate_sparse_neural_complexity, calculate_active_inference_complexity
)
from antstack_core.analysis import EnergyCoefficients
from antstack_core.figures import bar_plot, line_plot, scatter_plot


def generate_complexity_overview_data() -> Dict[str, Any]:
    """Generate data for complexity overview figure."""
    print("Generating complexity overview data...")
    
    # Standard parameters
    body_params = {'J': 18, 'C': 12, 'S': 256, 'hz': 100, 'contact_solver': 'pgs'}
    brain_params = {'K': 128, 'N_KC': 50000, 'rho': 0.02, 'H': 64, 'hz': 100}
    mind_params = {'B': 4, 'H_p': 15, 'hz': 100, 'state_dim': 16, 'action_dim': 6}
    
    # Calculate workloads
    body_load = enhanced_body_workload_closed_form(0.01, body_params)
    brain_load = enhanced_brain_workload_closed_form(0.01, brain_params)
    mind_load = enhanced_mind_workload_closed_form(0.01, mind_params)
    
    return {
        'body': {
            'flops': body_load.flops,
            'memory': body_load.sram_bytes + body_load.dram_bytes,
            'spikes': body_load.spikes,
            'params': body_params
        },
        'brain': {
            'flops': brain_load.flops,
            'memory': brain_load.sram_bytes + brain_load.dram_bytes,
            'spikes': brain_load.spikes,
            'params': brain_params
        },
        'mind': {
            'flops': mind_load.flops,
            'memory': mind_load.sram_bytes + mind_load.dram_bytes,
            'spikes': mind_load.spikes,
            'params': mind_params
        }
    }


def generate_energy_breakdown_figures(output_dir: Path) -> Dict[str, str]:
    """Generate detailed energy breakdown figures."""
    print("Generating energy breakdown figures...")
    
    coeffs = EnergyCoefficients()
    
    # Per-decision energy breakdown
    body_load = enhanced_body_workload_closed_form(0.01, {
        'J': 18, 'C': 12, 'S': 256, 'hz': 100, 'contact_solver': 'pgs'
    })
    brain_load = enhanced_brain_workload_closed_form(0.01, {
        'K': 128, 'N_KC': 50000, 'rho': 0.02, 'H': 64, 'hz': 100
    })
    mind_load = enhanced_mind_workload_closed_form(0.01, {
        'B': 4, 'H_p': 12, 'hz': 100, 'state_dim': 16, 'action_dim': 6
    })
    
    # Calculate detailed breakdowns
    body_breakdown = estimate_detailed_energy(body_load, coeffs, 0.01)
    brain_breakdown = estimate_detailed_energy(brain_load, coeffs, 0.01)
    mind_breakdown = estimate_detailed_energy(mind_load, coeffs, 0.01)
    
    # Module comparison
    modules = ['AntBody', 'AntBrain', 'AntMind']
    total_energies = [body_breakdown.total, brain_breakdown.total, mind_breakdown.total]
    
    module_comparison_plot = output_dir / "module_energy_comparison.png"
    bar_plot(
        modules, [e * 1000 for e in total_energies],  # Convert to mJ
        "Energy Consumption by Module",
        str(module_comparison_plot), ylabel="Energy per Decision (mJ)"
    )
    
    # Detailed component breakdown
    components = ['Compute (FLOPs)', 'Compute (Memory)', 'Compute (Spikes)', 'Baseline']
    
    # Aggregate compute components across modules
    flops_energy = body_breakdown.compute_flops + brain_breakdown.compute_flops + mind_breakdown.compute_flops
    memory_energy = body_breakdown.compute_memory + brain_breakdown.compute_memory + mind_breakdown.compute_memory
    spikes_energy = body_breakdown.compute_spikes + brain_breakdown.compute_spikes + mind_breakdown.compute_spikes
    baseline_energy = body_breakdown.baseline + brain_breakdown.baseline + mind_breakdown.baseline
    
    component_energies = [flops_energy, memory_energy, spikes_energy, baseline_energy]
    
    component_breakdown_plot = output_dir / "energy_component_breakdown.png"
    bar_plot(
        components, [e * 1e6 for e in component_energies],  # Convert to μJ
        "Energy Breakdown by Component Type",
        str(component_breakdown_plot), ylabel="Energy per Decision (μJ)"
    )
    
    return {
        'module_comparison': str(module_comparison_plot),
        'component_breakdown': str(component_breakdown_plot)
    }


def generate_scaling_analysis_figures(output_dir: Path) -> Dict[str, str]:
    """Generate comprehensive scaling analysis figures."""
    print("Generating scaling analysis figures...")
    
    coeffs = EnergyCoefficients()
    figures = {}
    
    # Body scaling (Joint count)
    J_values = [6, 12, 18, 24, 36]
    body_energies = []
    body_flops = []
    
    for J in J_values:
        params = {'J': J, 'C': 12, 'S': 256, 'hz': 100, 'contact_solver': 'pgs'}
        load = enhanced_body_workload_closed_form(0.01, params)
        breakdown = estimate_detailed_energy(load, coeffs, 0.01)
        body_energies.append(breakdown.total)
        body_flops.append(load.flops)
    
    # Body energy scaling
    body_energy_plot = output_dir / "body_joint_energy_scaling.png"
    line_plot(
        J_values, [body_energies], ["Body Energy"],
        "AntBody: Energy Scaling vs Joint Count",
        "Number of Joints (J)", "Energy per Decision (J)",
        str(body_energy_plot)
    )
    figures['body_energy_scaling'] = str(body_energy_plot)
    
    # Body energy vs complexity
    body_complexity_plot = output_dir / "body_energy_complexity.png"
    scatter_plot(
        body_flops, body_energies,
        "AntBody: Energy vs Computational Complexity",
        "FLOPs per Decision", "Energy per Decision (J)",
        str(body_complexity_plot)
    )
    figures['body_complexity'] = str(body_complexity_plot)
    
    # Brain scaling (AL channels)
    K_values = [64, 128, 256, 512, 1024]
    brain_energies = []
    brain_spikes = []
    
    for K in K_values:
        params = {'K': K, 'N_KC': 50000, 'rho': 0.02, 'H': 64, 'hz': 100}
        load = enhanced_brain_workload_closed_form(0.01, params)
        breakdown = estimate_detailed_energy(load, coeffs, 0.01)
        brain_energies.append(breakdown.total)
        brain_spikes.append(load.spikes)
    
    # Brain energy scaling
    brain_energy_plot = output_dir / "brain_channel_energy_scaling.png"
    line_plot(
        K_values, [brain_energies], ["Brain Energy"],
        "AntBrain: Energy Scaling vs AL Channels",
        "AL Input Channels (K)", "Energy per Decision (J)",
        str(brain_energy_plot)
    )
    figures['brain_energy_scaling'] = str(brain_energy_plot)
    
    # Brain spikes scaling
    brain_spikes_plot = output_dir / "brain_channel_spikes_scaling.png"
    line_plot(
        K_values, [brain_spikes], ["Spike Count"],
        "AntBrain: Spike Generation vs AL Channels",
        "AL Input Channels (K)", "Spikes per Decision",
        str(brain_spikes_plot)
    )
    figures['brain_spikes_scaling'] = str(brain_spikes_plot)
    
    # Mind scaling (Policy horizon) - limited range to avoid explosion
    H_p_values = [5, 7, 9, 11, 13]
    mind_energies = []
    mind_flops = []
    
    for H_p in H_p_values:
        params = {'B': 3, 'H_p': H_p, 'hz': 100, 'state_dim': 12, 'action_dim': 4}
        load = enhanced_mind_workload_closed_form(0.01, params)
        breakdown = estimate_detailed_energy(load, coeffs, 0.01)
        mind_energies.append(breakdown.total)
        mind_flops.append(load.flops)
    
    # Mind energy scaling
    mind_energy_plot = output_dir / "mind_horizon_energy_scaling.png"
    line_plot(
        H_p_values, [mind_energies], ["Mind Energy"],
        "AntMind: Energy Scaling vs Policy Horizon",
        "Policy Horizon (steps)", "Energy per Decision (J)",
        str(mind_energy_plot)
    )
    figures['mind_energy_scaling'] = str(mind_energy_plot)
    
    # Mind computational explosion
    mind_flops_plot = output_dir / "mind_horizon_flops_scaling.png"
    line_plot(
        H_p_values, [mind_flops], ["Mind FLOPs"],
        "AntMind: Computational Explosion vs Policy Horizon",
        "Policy Horizon (steps)", "FLOPs per Decision",
        str(mind_flops_plot)
    )
    figures['mind_flops_scaling'] = str(mind_flops_plot)
    
    return figures


def generate_connectivity_analysis_figures(output_dir: Path) -> Dict[str, str]:
    """Generate neural connectivity pattern analysis figures."""
    print("Generating connectivity analysis figures...")
    
    network_size = 50000
    sparsity = 0.02
    patterns = ["random", "small_world", "scale_free", "biological"]
    
    # Analyze different connectivity patterns
    pattern_flops = []
    pattern_spikes = []
    
    for pattern in patterns:
        flops, memory, spikes = calculate_sparse_neural_complexity(
            network_size, sparsity, pattern
        )
        pattern_flops.append(flops)
        pattern_spikes.append(spikes)
    
    # Connectivity pattern comparison
    connectivity_flops_plot = output_dir / "connectivity_patterns_flops.png"
    bar_plot(
        [p.replace('_', ' ').title() for p in patterns], pattern_flops,
        "Neural Connectivity Patterns: Computational Cost",
        str(connectivity_flops_plot), ylabel="FLOPs per Decision"
    )
    
    connectivity_spikes_plot = output_dir / "connectivity_patterns_spikes.png"
    bar_plot(
        [p.replace('_', ' ').title() for p in patterns], pattern_spikes,
        "Neural Connectivity Patterns: Spike Generation",
        str(connectivity_spikes_plot), ylabel="Spikes per Decision"
    )
    
    # Sparsity effects analysis
    sparsity_levels = [0.005, 0.01, 0.02, 0.05, 0.1]
    sparsity_energies = []
    sparsity_spikes = []
    
    coeffs = EnergyCoefficients()
    
    for rho in sparsity_levels:
        params = {'K': 128, 'N_KC': 50000, 'rho': rho, 'H': 64, 'hz': 100}
        load = enhanced_brain_workload_closed_form(0.01, params)
        breakdown = estimate_detailed_energy(load, coeffs, 0.01)
        sparsity_energies.append(breakdown.total)
        sparsity_spikes.append(load.spikes)
    
    sparsity_energy_plot = output_dir / "sparsity_energy_scaling.png"
    line_plot(
        sparsity_levels, [sparsity_energies], ["Energy"],
        "Neural Sparsity Effects on Energy Consumption",
        "Sparsity Level (ρ)", "Energy per Decision (J)",
        str(sparsity_energy_plot)
    )
    
    return {
        'connectivity_flops': str(connectivity_flops_plot),
        'connectivity_spikes': str(connectivity_spikes_plot),
        'sparsity_energy': str(sparsity_energy_plot)
    }


def generate_theoretical_limits_figures(output_dir: Path) -> Dict[str, str]:
    """Generate theoretical limits comparison figures."""
    print("Generating theoretical limits figures...")
    
    coeffs = EnergyCoefficients()
    
    # Calculate actual vs theoretical for each module
    modules = ['AntBody', 'AntBrain', 'AntMind']
    actual_energies = []
    theoretical_energies = []
    efficiency_ratios = []
    
    # Body
    body_load = enhanced_body_workload_closed_form(0.01, {
        'J': 18, 'C': 12, 'S': 256, 'hz': 100
    })
    body_energy = estimate_detailed_energy(body_load, coeffs, 0.01).total
    body_limits = estimate_theoretical_limits({
        'flops': body_load.flops,
        'mechanical_work_j': 0.001
    })
    
    actual_energies.append(body_energy)
    theoretical_energies.append(body_limits['total_theoretical_j'])
    efficiency_ratios.append(body_energy / body_limits['total_theoretical_j'])
    
    # Brain
    brain_load = enhanced_brain_workload_closed_form(0.01, {
        'K': 128, 'N_KC': 50000, 'rho': 0.02, 'H': 64, 'hz': 100
    })
    brain_energy = estimate_detailed_energy(brain_load, coeffs, 0.01).total
    brain_limits = estimate_theoretical_limits({
        'flops': brain_load.flops,
        'mechanical_work_j': 0.0
    })
    
    actual_energies.append(brain_energy)
    theoretical_energies.append(brain_limits['total_theoretical_j'])
    efficiency_ratios.append(brain_energy / brain_limits['total_theoretical_j'])
    
    # Mind (limited horizon to avoid explosion)
    mind_load = enhanced_mind_workload_closed_form(0.01, {
        'B': 4, 'H_p': 8, 'hz': 100, 'state_dim': 16, 'action_dim': 6
    })
    mind_energy = estimate_detailed_energy(mind_load, coeffs, 0.01).total
    mind_limits = estimate_theoretical_limits({
        'flops': mind_load.flops,
        'mechanical_work_j': 0.0
    })
    
    actual_energies.append(mind_energy)
    theoretical_energies.append(mind_limits['total_theoretical_j'])
    efficiency_ratios.append(mind_energy / mind_limits['total_theoretical_j'])
    
    # Theoretical limits comparison
    all_energies = actual_energies + theoretical_energies
    all_labels = [f"{m} (Actual)" for m in modules] + [f"{m} (Theoretical)" for m in modules]
    
    limits_plot = output_dir / "theoretical_limits_comparison.png"
    bar_plot(
        all_labels, all_energies,
        "Actual vs Theoretical Energy Limits",
        str(limits_plot), ylabel="Energy per Decision (J)"
    )
    
    # Efficiency ratios
    efficiency_plot = output_dir / "efficiency_ratios.png"
    bar_plot(
        modules, efficiency_ratios,
        "Energy Efficiency Ratios (Actual/Theoretical)",
        str(efficiency_plot), ylabel="Efficiency Ratio"
    )
    
    return {
        'theoretical_limits': str(limits_plot),
        'efficiency_ratios': str(efficiency_plot)
    }


def generate_contact_solver_figures(output_dir: Path) -> Dict[str, str]:
    """Generate contact solver comparison figures."""
    print("Generating contact solver comparison figures...")
    
    contact_counts = [5, 10, 15, 20, 25, 30]
    solvers = ["pgs", "lcp", "mlcp"]
    
    solver_data = {}
    for solver in solvers:
        flops_data = []
        memory_data = []
        
        for C in contact_counts:
            flops, memory, _ = calculate_contact_complexity(J=18, C=C, solver=solver)
            flops_data.append(flops)
            memory_data.append(memory)
        
        solver_data[solver] = {'flops': flops_data, 'memory': memory_data}
    
    # FLOPs comparison
    flops_series = [solver_data[solver]['flops'] for solver in solvers]
    solver_labels = [solver.upper() for solver in solvers]
    
    solver_flops_plot = output_dir / "contact_solver_flops_comparison.png"
    line_plot(
        contact_counts, flops_series, solver_labels,
        "Contact Solver Computational Complexity",
        "Number of Contacts", "FLOPs per Solve",
        str(solver_flops_plot)
    )
    
    # Memory comparison
    memory_series = [solver_data[solver]['memory'] for solver in solvers]
    
    solver_memory_plot = output_dir / "contact_solver_memory_comparison.png"
    line_plot(
        contact_counts, memory_series, solver_labels,
        "Contact Solver Memory Requirements",
        "Number of Contacts", "Memory (bytes)",
        str(solver_memory_plot)
    )
    
    return {
        'solver_flops': str(solver_flops_plot),
        'solver_memory': str(solver_memory_plot)
    }


def main():
    """Main figure generation orchestration."""
    parser = argparse.ArgumentParser(description="Generate all manuscript figures")
    parser.add_argument("--output", default="complexity_energetics/assets",
                       help="Output directory for figures")
    parser.add_argument("--format", choices=["png", "svg", "pdf"], default="png",
                       help="Output format for figures")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Manuscript Figure Generation Orchestrator")
    print("=" * 60)
    
    # Generate all figure sets
    all_figures = {}
    
    # Energy breakdown figures
    energy_figures = generate_energy_breakdown_figures(output_dir)
    all_figures.update(energy_figures)
    
    # Scaling analysis figures
    scaling_figures = generate_scaling_analysis_figures(output_dir)
    all_figures.update(scaling_figures)
    
    # Connectivity analysis figures
    connectivity_figures = generate_connectivity_analysis_figures(output_dir)
    all_figures.update(connectivity_figures)
    
    # Theoretical limits figures
    limits_figures = generate_theoretical_limits_figures(output_dir)
    all_figures.update(limits_figures)
    
    # Contact solver figures
    solver_figures = generate_contact_solver_figures(output_dir)
    all_figures.update(solver_figures)
    
    # Generate complexity overview data for Mermaid diagrams
    complexity_data = generate_complexity_overview_data()
    
    # Save figure manifest
    figure_manifest = {
        'figures': all_figures,
        'complexity_data': complexity_data,
        'generation_info': {
            'output_directory': str(output_dir),
            'format': args.format,
            'total_figures': len(all_figures)
        }
    }
    
    manifest_path = output_dir / "figure_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(figure_manifest, f, indent=2, default=str)
    
    print(f"\nFigure generation complete!")
    print(f"Generated {len(all_figures)} figures in: {output_dir}")
    print(f"Figure manifest: {manifest_path}")
    
    # Print figure summary
    print(f"\nGenerated Figures:")
    for category, figures in [
        ("Energy Analysis", energy_figures),
        ("Scaling Analysis", scaling_figures), 
        ("Connectivity Analysis", connectivity_figures),
        ("Theoretical Limits", limits_figures),
        ("Contact Solvers", solver_figures)
    ]:
        print(f"  {category}: {len(figures)} figures")
        for name, path in figures.items():
            print(f"    - {name}: {Path(path).name}")


if __name__ == "__main__":
    main()
