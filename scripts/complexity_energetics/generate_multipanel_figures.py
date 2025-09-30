#!/usr/bin/env python3
"""Generate multipanel figures for the manuscript.

Creates sophisticated multipanel figures with proper subfigure layouts,
statistical overlays, and publication-quality formatting.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from antstack_core.analysis import (
    estimate_detailed_energy, analyze_scaling_relationship,
    calculate_energy_efficiency_metrics, estimate_theoretical_limits
)
from antstack_core.analysis import (
    enhanced_body_workload_closed_form, enhanced_brain_workload_closed_form,
    enhanced_mind_workload_closed_form
)
from antstack_core.analysis import EnergyCoefficients

# Import plotting with fallback
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    except ImportError:
        print("Warning: Seaborn not available, using matplotlib defaults")
    HAS_PLOTTING = True
except ImportError as e:
    HAS_PLOTTING = False
    print(f"Warning: Matplotlib not available, skipping plot generation: {e}")


def create_multipanel_scaling_analysis(output_dir: Path) -> Dict[str, str]:
    """Create comprehensive multipanel scaling analysis figures."""
    
    if not HAS_PLOTTING:
        return {}
    
    print("Generating multipanel scaling analysis...")
    
    coeffs = EnergyCoefficients()
    
    # Create 2x3 multipanel figure for complete scaling analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Scaling Analysis: Energy and Computational Complexity', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Body scaling analysis
    J_values = [6, 12, 18, 24, 36]
    body_energies = []
    body_flops = []
    
    for J in J_values:
        params = {'J': J, 'C': 12, 'S': 256, 'hz': 100, 'contact_solver': 'pgs'}
        load = enhanced_body_workload_closed_form(0.01, params)
        breakdown = estimate_detailed_energy(load, coeffs, 0.01)
        body_energies.append(breakdown.total * 1000)  # Convert to mJ
        body_flops.append(load.flops)
    
    # Body energy plot
    ax = axes[0, 0]
    ax.plot(J_values, body_energies, 'o-', linewidth=3, markersize=8, 
            color='#1f77b4', markerfacecolor='white', markeredgewidth=2)
    
    # Add scaling annotation
    scaling = analyze_scaling_relationship(J_values, body_energies)
    if scaling.get('valid', False):
        ax.text(0.05, 0.95, f'E ∝ J^{scaling["scaling_exponent"]:.2e}\nR² = {scaling["r_squared"]:.3f}',
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10, verticalalignment='top', fontweight='bold')
    
    ax.set_xlabel('Joint Count (J)', fontweight='bold')
    ax.set_ylabel('Energy per Decision (mJ)', fontweight='bold')
    ax.set_title('(A) AntBody Energy Scaling', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Body FLOPs plot
    ax = axes[1, 0]
    ax.plot(J_values, body_flops, 's-', linewidth=3, markersize=8,
            color='#ff7f0e', markerfacecolor='white', markeredgewidth=2)
    
    flops_scaling = analyze_scaling_relationship(J_values, body_flops)
    if flops_scaling.get('valid', False):
        ax.text(0.05, 0.95, f'FLOPs ∝ J^{flops_scaling["scaling_exponent"]:.3f}\nR² = {flops_scaling["r_squared"]:.3f}',
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10, verticalalignment='top', fontweight='bold')
    
    ax.set_xlabel('Joint Count (J)', fontweight='bold')
    ax.set_ylabel('FLOPs per Decision', fontweight='bold')
    ax.set_title('(D) AntBody Computational Scaling', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Brain scaling analysis
    K_values = [64, 128, 256, 512, 1024]
    brain_energies = []
    brain_spikes = []
    
    for K in K_values:
        params = {'K': K, 'N_KC': 50000, 'rho': 0.02, 'H': 64, 'hz': 100}
        load = enhanced_brain_workload_closed_form(0.01, params)
        breakdown = estimate_detailed_energy(load, coeffs, 0.01)
        brain_energies.append(breakdown.total * 1000)  # Convert to mJ
        brain_spikes.append(load.spikes)
    
    # Brain energy plot
    ax = axes[0, 1]
    ax.plot(K_values, brain_energies, '^-', linewidth=3, markersize=8,
            color='#2ca02c', markerfacecolor='white', markeredgewidth=2)
    
    brain_scaling = analyze_scaling_relationship(K_values, brain_energies)
    if brain_scaling.get('valid', False):
        ax.text(0.05, 0.95, f'E ∝ K^{brain_scaling["scaling_exponent"]:.2e}\nR² = {brain_scaling["r_squared"]:.3f}',
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10, verticalalignment='top', fontweight='bold')
    
    ax.set_xlabel('AL Input Channels (K)', fontweight='bold')
    ax.set_ylabel('Energy per Decision (mJ)', fontweight='bold')
    ax.set_title('(B) AntBrain Energy Scaling', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Brain spikes plot
    ax = axes[1, 1]
    ax.plot(K_values, brain_spikes, 'D-', linewidth=3, markersize=8,
            color='#d62728', markerfacecolor='white', markeredgewidth=2)
    
    spikes_scaling = analyze_scaling_relationship(K_values, brain_spikes)
    if spikes_scaling.get('valid', False):
        ax.text(0.05, 0.95, f'Spikes ∝ K^{spikes_scaling["scaling_exponent"]:.3f}\nR² = {spikes_scaling["r_squared"]:.3f}',
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10, verticalalignment='top', fontweight='bold')
    
    ax.set_xlabel('AL Input Channels (K)', fontweight='bold')
    ax.set_ylabel('Spikes per Decision', fontweight='bold')
    ax.set_title('(E) AntBrain Spike Generation', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Mind scaling analysis (limited range)
    H_p_values = [5, 7, 9, 11, 13]
    mind_energies = []
    mind_flops = []
    
    for H_p in H_p_values:
        params = {'B': 3, 'H_p': H_p, 'hz': 100, 'state_dim': 12, 'action_dim': 4}
        load = enhanced_mind_workload_closed_form(0.01, params)
        breakdown = estimate_detailed_energy(load, coeffs, 0.01)
        mind_energies.append(breakdown.total * 1000)  # Convert to mJ
        mind_flops.append(load.flops / 1e6)  # Convert to MFLOPs
    
    # Mind energy plot
    ax = axes[0, 2]
    ax.semilogy(H_p_values, mind_energies, 'v-', linewidth=3, markersize=8,
                color='#9467bd', markerfacecolor='white', markeredgewidth=2)
    
    mind_scaling = analyze_scaling_relationship(H_p_values, mind_energies)
    if mind_scaling.get('valid', False):
        ax.text(0.05, 0.95, f'E ∝ H_p^{mind_scaling["scaling_exponent"]:.1f}\nR² = {mind_scaling["r_squared"]:.3f}',
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10, verticalalignment='top', fontweight='bold')
    
    ax.set_xlabel('Policy Horizon (H_p)', fontweight='bold')
    ax.set_ylabel('Energy per Decision (mJ)', fontweight='bold')
    ax.set_title('(C) AntMind Energy Scaling', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Mind FLOPs plot
    ax = axes[1, 2]
    ax.semilogy(H_p_values, mind_flops, 'p-', linewidth=3, markersize=8,
                color='#8c564b', markerfacecolor='white', markeredgewidth=2)
    
    mind_flops_scaling = analyze_scaling_relationship(H_p_values, mind_flops)
    if mind_flops_scaling.get('valid', False):
        ax.text(0.05, 0.95, f'FLOPs ∝ H_p^{mind_flops_scaling["scaling_exponent"]:.1f}\nR² = {mind_flops_scaling["r_squared"]:.3f}',
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10, verticalalignment='top', fontweight='bold')
    
    ax.set_xlabel('Policy Horizon (H_p)', fontweight='bold')
    ax.set_ylabel('Computational Load (MFLOPs)', fontweight='bold')
    ax.set_title('(F) AntMind Computational Explosion', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save multipanel figure
    multipanel_path = output_dir / "comprehensive_scaling_multipanel.png"
    fig.savefig(multipanel_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return {"comprehensive_scaling": str(multipanel_path)}


def create_theoretical_limits_multipanel(output_dir: Path) -> Dict[str, str]:
    """Create multipanel theoretical limits analysis."""
    
    if not HAS_PLOTTING:
        return {}
    
    print("Generating theoretical limits multipanel...")
    
    coeffs = EnergyCoefficients()
    
    # Create 2x2 multipanel figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Theoretical Limits and Efficiency Analysis', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Calculate data for each module
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
    
    actual_energies.append(body_energy * 1000)  # mJ
    theoretical_energies.append(body_limits['total_theoretical_j'] * 1000)  # mJ
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
    
    actual_energies.append(brain_energy * 1000)  # mJ
    theoretical_energies.append(brain_limits['total_theoretical_j'] * 1000)  # mJ
    efficiency_ratios.append(brain_energy / brain_limits['total_theoretical_j'])
    
    # Mind (limited horizon)
    mind_load = enhanced_mind_workload_closed_form(0.01, {
        'B': 4, 'H_p': 8, 'hz': 100, 'state_dim': 16, 'action_dim': 6
    })
    mind_energy = estimate_detailed_energy(mind_load, coeffs, 0.01).total
    mind_limits = estimate_theoretical_limits({
        'flops': mind_load.flops,
        'mechanical_work_j': 0.0
    })
    
    actual_energies.append(mind_energy * 1000)  # mJ
    theoretical_energies.append(mind_limits['total_theoretical_j'] * 1000)  # mJ
    efficiency_ratios.append(mind_energy / mind_limits['total_theoretical_j'])
    
    # Plot 1: Actual vs Theoretical Energy
    ax = axes[0, 0]
    x_pos = np.arange(len(modules))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, actual_energies, width, label='Actual Energy',
                   color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x_pos + width/2, theoretical_energies, width, label='Theoretical Minimum',
                   color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, value in zip(bars1, actual_energies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(actual_energies) * 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    for bar, value in zip(bars2, theoretical_energies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(actual_energies) * 0.01,
                f'{value:.2e}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax.set_xlabel('Module', fontweight='bold')
    ax.set_ylabel('Energy per Decision (mJ)', fontweight='bold')
    ax.set_title('(A) Actual vs Theoretical Energy', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(modules)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Plot 2: Efficiency Ratios (log scale)
    ax = axes[0, 1]
    colors = ['#2ca02c', '#d62728', '#9467bd']
    bars = ax.bar(modules, efficiency_ratios, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, value in zip(bars, efficiency_ratios):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{value:.1e}×', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_yscale('log')
    ax.set_xlabel('Module', fontweight='bold')
    ax.set_ylabel('Efficiency Ratio (Actual/Theoretical)', fontweight='bold')
    ax.set_title('(B) Energy Efficiency Ratios', fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Plot 3: Energy Component Breakdown
    ax = axes[1, 0]
    
    # Get detailed breakdowns
    body_breakdown = estimate_detailed_energy(body_load, coeffs, 0.01)
    brain_breakdown = estimate_detailed_energy(brain_load, coeffs, 0.01)
    mind_breakdown = estimate_detailed_energy(mind_load, coeffs, 0.01)
    
    components = ['FLOPs', 'Memory', 'Spikes', 'Baseline']
    body_components = [body_breakdown.compute_flops * 1000, 
                      body_breakdown.compute_memory * 1000,
                      body_breakdown.compute_spikes * 1000,
                      body_breakdown.baseline * 1000]
    brain_components = [brain_breakdown.compute_flops * 1000,
                       brain_breakdown.compute_memory * 1000,
                       brain_breakdown.compute_spikes * 1000,
                       brain_breakdown.baseline * 1000]
    mind_components = [mind_breakdown.compute_flops * 1000,
                      mind_breakdown.compute_memory * 1000,
                      mind_breakdown.compute_spikes * 1000,
                      mind_breakdown.baseline * 1000]
    
    x = np.arange(len(components))
    width = 0.25
    
    ax.bar(x - width, body_components, width, label='AntBody', alpha=0.8)
    ax.bar(x, brain_components, width, label='AntBrain', alpha=0.8)
    ax.bar(x + width, mind_components, width, label='AntMind', alpha=0.8)
    
    ax.set_xlabel('Energy Component', fontweight='bold')
    ax.set_ylabel('Energy per Decision (mJ)', fontweight='bold')
    ax.set_title('(C) Energy Component Breakdown', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(components)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Plot 4: Scaling Regime Classification
    ax = axes[1, 1]
    
    # Scaling exponents from previous analysis
    scaling_exponents = [8.1e-7, 1.04e-5, 11.1]  # Body, Brain, Mind
    r_squared_values = [0.926, 0.871, 0.761]
    
    # Create scatter plot with size based on R²
    scatter = ax.scatter(range(len(modules)), scaling_exponents, 
                        s=[r*500 for r in r_squared_values], 
                        c=colors, alpha=0.7, edgecolors='black', linewidth=2)
    
    # Add regime lines
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Linear (slope=1)')
    ax.axhline(y=2.0, color='orange', linestyle='--', alpha=0.7, label='Quadratic (slope=2)')
    
    # Add annotations
    regimes = ['Sub-linear', 'Sub-linear', 'Super-linear']
    for i, (module, exp, regime) in enumerate(zip(modules, scaling_exponents, regimes)):
        ax.annotate(f'{regime}\n(R²={r_squared_values[i]:.3f})', 
                   xy=(i, exp), xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                   fontsize=9, fontweight='bold')
    
    ax.set_yscale('log')
    ax.set_xlabel('Module', fontweight='bold')
    ax.set_ylabel('Scaling Exponent', fontweight='bold')
    ax.set_title('(D) Scaling Regime Classification', fontweight='bold')
    ax.set_xticks(range(len(modules)))
    ax.set_xticklabels(modules)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save multipanel figure
    limits_path = output_dir / "theoretical_limits_multipanel.png"
    fig.savefig(limits_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return {"theoretical_limits": str(limits_path)}


def generate_latex_figure_code(figures: Dict[str, str], output_dir: Path) -> None:
    """Generate LaTeX code for multipanel figures."""
    
    latex_code = []
    
    # Comprehensive scaling figure
    if "comprehensive_scaling" in figures:
        latex_code.append("""
\\begin{figure}[ht]
  \\centering
  \\includegraphics[width=\\linewidth]{""" + figures["comprehensive_scaling"].replace(str(output_dir.parent) + "/", "") + """}
  \\caption{\\textbf{Comprehensive Scaling Analysis.} Multipanel analysis showing energy and computational scaling across all three modules. (A-C) Energy scaling vs primary parameters: AntBody shows baseline-dominated constant scaling, AntBrain demonstrates sparsity-enabled sub-linear scaling, and AntMind exhibits exponential growth despite bounded rationality. (D-F) Computational complexity scaling: FLOPs growth patterns reveal underlying algorithmic complexity, with Mind showing steepest growth due to policy tree explosion. Statistical annotations show power law exponents and goodness-of-fit (R²) values.}
  \\label{fig:comprehensive_scaling_multipanel}
\\end{figure}
""")
    
    # Theoretical limits figure
    if "theoretical_limits" in figures:
        latex_code.append("""
\\begin{figure}[ht]
  \\centering
  \\includegraphics[width=\\linewidth]{""" + figures["theoretical_limits"].replace(str(output_dir.parent) + "/", "") + """}
  \\caption{\\textbf{Theoretical Limits and Efficiency Analysis.} Comprehensive analysis of energy efficiency across modules. (A) Actual vs theoretical minimum energy consumption based on Landauer's principle and thermodynamic bounds. (B) Efficiency ratios showing AntBody near-optimal performance due to mechanical work dominance, while compute-intensive modules show substantial optimization opportunities. (C) Energy component breakdown revealing dominant energy sources in each module. (D) Scaling regime classification with statistical confidence (bubble size indicates R² values).}
  \\label{fig:theoretical_limits_multipanel}
\\end{figure}
""")
    
    # Write LaTeX code to file
    latex_file = output_dir / "multipanel_figures.tex"
    with open(latex_file, 'w') as f:
        f.write("% Auto-generated multipanel figure LaTeX code\n")
        f.write("% Include in main document with: \\input{multipanel_figures.tex}\n\n")
        f.write("\n".join(latex_code))
    
    print(f"Generated LaTeX code: {latex_file}")


def main():
    """Main function to generate all multipanel figures."""
    
    # Find the complexity_energetics directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    output_dir = project_root / "papers" / "complexity_energetics" / "assets"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Multipanel Figure Generation")
    print("=" * 60)
    
    all_figures = {}
    
    # Generate multipanel figures
    scaling_figures = create_multipanel_scaling_analysis(output_dir)
    all_figures.update(scaling_figures)
    
    limits_figures = create_theoretical_limits_multipanel(output_dir)
    all_figures.update(limits_figures)
    
    # Generate LaTeX code
    generate_latex_figure_code(all_figures, output_dir)
    
    # Save figure manifest
    manifest = {
        "multipanel_figures": all_figures,
        "generation_info": {
            "output_directory": str(output_dir),
            "total_multipanel_figures": len(all_figures)
        }
    }
    
    manifest_path = output_dir / "multipanel_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    
    print(f"\nMultipanel figure generation complete!")
    print(f"Generated {len(all_figures)} multipanel figures in: {output_dir}")
    print(f"Manifest: {manifest_path}")
    
    for name, path in all_figures.items():
        print(f"  - {name}: {Path(path).name}")


if __name__ == "__main__":
    main()
