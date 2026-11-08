#!/usr/bin/env python3
"""Publish source-bound enhanced analytical multipanels into explicit fresh families."""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json

from antstack_core.analysis import (
    estimate_detailed_energy, analyze_scaling_relationship,
    calculate_energy_efficiency_metrics, estimate_theoretical_limits
)
from antstack_core.analysis import (
    enhanced_body_workload_closed_form, enhanced_brain_workload_closed_form,
    enhanced_mind_workload_closed_form
)
from antstack_core.analysis import EnergyCoefficients


def _r_squared_label(value):
    return 'undefined' if value is None else f'{value:.3f}'

# Import plotting with fallback
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    try:
        import seaborn as sns
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
    fig.suptitle('Enhanced Compute-plus-Idle Scaling: Assumed Coefficients, No Mechanics',
                 fontsize=16, fontweight='bold', y=0.99)
    
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
        ax.text(0.05, 0.95, f'Fitted energy exponent = {scaling["scaling_exponent"]:.2e}\nlog-log R² = {_r_squared_label(scaling["r_squared"])}',
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10, verticalalignment='top', fontweight='bold')
    
    ax.set_xlabel('Joint Count (J)', fontweight='bold')
    ax.set_ylabel('Energy per Decision (mJ)', fontweight='bold')
    ax.set_title('(A) AntBody Energy Scaling', fontweight='bold')
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax.grid(True, alpha=0.3)
    
    # Body FLOPs plot
    ax = axes[1, 0]
    ax.plot(J_values, body_flops, 's-', linewidth=3, markersize=8,
            color='#ff7f0e', markerfacecolor='white', markeredgewidth=2)
    
    flops_scaling = analyze_scaling_relationship(J_values, body_flops)
    if flops_scaling.get('valid', False):
        ax.text(0.05, 0.95, f'Fitted FLOPs exponent = {flops_scaling["scaling_exponent"]:.3g}\nlog-log R² = {_r_squared_label(flops_scaling["r_squared"])}',
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
        ax.text(0.05, 0.95, f'Fitted energy exponent = {brain_scaling["scaling_exponent"]:.2e}\nlog-log R² = {_r_squared_label(brain_scaling["r_squared"])}',
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10, verticalalignment='top', fontweight='bold')
    
    ax.set_xlabel('AL Input Channels (K)', fontweight='bold')
    ax.set_ylabel('Energy per Decision (mJ)', fontweight='bold')
    ax.set_title('(B) AntBrain Energy Scaling', fontweight='bold')
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax.grid(True, alpha=0.3)
    
    # Brain spikes plot
    ax = axes[1, 1]
    ax.plot(K_values, brain_spikes, 'D-', linewidth=3, markersize=8,
            color='#d62728', markerfacecolor='white', markeredgewidth=2)
    
    spikes_scaling = analyze_scaling_relationship(K_values, brain_spikes)
    if spikes_scaling.get('valid', False):
        ax.text(0.05, 0.95, f'Fitted spikes exponent = {spikes_scaling["scaling_exponent"]:.3g}\nlog-log R² = {_r_squared_label(spikes_scaling["r_squared"])}',
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
    ax.plot(H_p_values, mind_energies, 'v-', linewidth=3, markersize=8,
                color='#9467bd', markerfacecolor='white', markeredgewidth=2)
    
    mind_scaling = analyze_scaling_relationship(H_p_values, mind_energies)
    if mind_scaling.get('valid', False):
        ax.text(0.05, 0.95, f'Fitted energy exponent = {mind_scaling["scaling_exponent"]:.2e}\nlog-log R² = {_r_squared_label(mind_scaling["r_squared"])}',
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10, verticalalignment='top', fontweight='bold')
    
    ax.set_xlabel('Policy Horizon (H_p)', fontweight='bold')
    ax.set_ylabel('Energy per Decision (mJ)', fontweight='bold')
    ax.set_title('(C) AntMind Energy Scaling', fontweight='bold')
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax.grid(True, alpha=0.3)
    
    # Mind FLOPs plot
    ax = axes[1, 2]
    ax.semilogy(H_p_values, mind_flops, 'p-', linewidth=3, markersize=8,
                color='#8c564b', markerfacecolor='white', markeredgewidth=2)
    
    mind_flops_scaling = analyze_scaling_relationship(H_p_values, mind_flops)
    if mind_flops_scaling.get('valid', False):
        ax.text(0.05, 0.95, f'Fitted FLOPs exponent = {mind_flops_scaling["scaling_exponent"]:.3g}\nlog-log R² = {_r_squared_label(mind_flops_scaling["r_squared"])}',
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10, verticalalignment='top', fontweight='bold')
    
    ax.set_xlabel('Policy Horizon (H_p)', fontweight='bold')
    ax.set_ylabel('Computational Load (MFLOPs)', fontweight='bold')
    ax.set_title('(F) Enhanced AntMind FLOPs (1000-policy cap)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    
    # Save multipanel figure
    multipanel_path = output_dir / "comprehensive_scaling_multipanel.png"
    from antstack_core.figures.plots import _save_figure
    _save_figure(fig, multipanel_path)
    
    return {"comprehensive_scaling": str(multipanel_path)}


def create_theoretical_limits_multipanel(output_dir: Path) -> Dict[str, str]:
    """Create multipanel theoretical limits analysis."""
    
    if not HAS_PLOTTING:
        return {}
    
    print("Generating theoretical limits multipanel...")
    
    coeffs = EnergyCoefficients()
    
    # Create 2x2 multipanel figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Enhanced Compute-plus-Idle Model: Conditional Erasure References',
                 fontsize=16, fontweight='bold', y=0.99)
    
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
        'bits_erased': body_load.flops
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
        'bits_erased': brain_load.flops
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
        'bits_erased': mind_load.flops
    })
    
    actual_energies.append(mind_energy * 1000)  # mJ
    theoretical_energies.append(mind_limits['total_theoretical_j'] * 1000)  # mJ
    efficiency_ratios.append(mind_energy / mind_limits['total_theoretical_j'])
    
    # Plot 1: Actual vs Theoretical Energy
    ax = axes[0, 0]
    x_pos = np.arange(len(modules))
    width = 0.35
    
    ax.scatter(x_pos - width/2, actual_energies, label='Enhanced Model Energy',
               color='#1f77b4', s=75, zorder=3)
    ax.scatter(x_pos + width/2, theoretical_energies, label='Conditional Erasure Reference',
               color='#ff7f0e', marker='D', s=65, zorder=3)
    for positions, values in ((x_pos - width/2, actual_energies), (x_pos + width/2, theoretical_energies)):
        for position, value in zip(positions, values):
            ax.annotate(f'{value:.3g}', (position, value), xytext=(0, 7),
                        textcoords='offset points', ha='center', fontsize=9)
    ax.set_yscale('log')
    ax.margins(y=.2)
    
    ax.set_xlabel('Module', fontweight='bold')
    ax.set_ylabel('Energy per Decision (mJ; log scale)', fontweight='bold')
    ax.set_title('(A) Model vs 1 Erased Bit/FLOP at 300 K', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(modules)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Plot 2: Efficiency Ratios (log scale)
    ax = axes[0, 1]
    colors = ['#2ca02c', '#d62728', '#9467bd']
    ax.scatter(x_pos, efficiency_ratios, color=colors, s=85, zorder=3)
    for position, value in zip(x_pos, efficiency_ratios):
        ax.annotate(f'{value:.2e}×', (position, value), xytext=(0, 8),
                    textcoords='offset points', ha='center', fontweight='bold', fontsize=10)
    ax.set_xticks(x_pos, modules)
    ax.set_xlim(-.5, len(modules) - .5)
    ax.set_ylim(min(efficiency_ratios) / 2, max(efficiency_ratios) * 3)
    
    ax.set_yscale('log')
    ax.set_xlabel('Module', fontweight='bold')
    ax.set_ylabel('Model / Erasure Reference (Not Efficiency)', fontweight='bold')
    ax.set_title('(B) Conditional Reference Ratios', fontweight='bold')
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
    positive_components = [value for series in (body_components, brain_components, mind_components)
                           for value in series if value > 0]
    ax.set_yscale('symlog', linthresh=min(positive_components) / 10)
    
    ax.set_xlabel('Energy Component', fontweight='bold')
    ax.set_ylabel('Energy per Decision (mJ; symlog)', fontweight='bold')
    ax.set_title('(C) Components Including Exact Zeros', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(components)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Describe fits from current enhanced model sweeps, not historical constants.
    ax = axes[1, 1]
    sweeps = [
        (enhanced_body_workload_closed_form, 'J', [6, 12, 18, 24, 36],
         {'C': 12, 'S': 256, 'hz': 100}),
        (enhanced_brain_workload_closed_form, 'K', [64, 128, 256, 512, 1024],
         {'N_KC': 50000, 'rho': 0.02, 'H': 64, 'hz': 100}),
        (enhanced_mind_workload_closed_form, 'H_p', [5, 7, 9, 11, 13],
         {'B': 3, 'hz': 100, 'state_dim': 12, 'action_dim': 4, 'max_policies': 1000}),
    ]
    for index, (workload, parameter, values, params) in enumerate(sweeps):
        energies = [estimate_detailed_energy(workload(0.01, dict(params, **{parameter: value})), coeffs, 0.01).total
                    for value in values]
        fit = analyze_scaling_relationship(values, energies)
        if fit.get('valid', False):
            exponent = fit['scaling_exponent']
            ax.scatter(index, exponent, s=100, c=colors[index])
            ax.annotate(f'log-log R²={_r_squared_label(fit["r_squared"])}', (index, exponent),
                        xytext=(-5 if index == 2 else 5, 5), textcoords='offset points',
                        ha='right' if index == 2 else 'left', fontsize=9)
        else:
            ax.text(index, 0, 'Fit unavailable', ha='center')
    ax.set_xlabel('Module', fontweight='bold')
    ax.set_ylabel('Descriptive Power-law Exponent', fontweight='bold')
    ax.set_title('(D) Current Enhanced Model Sweep Fits', fontweight='bold')
    ax.set_xticks(range(len(modules)))
    ax.set_xticklabels(modules)
    ax.set_xlim(-.4, len(modules) - .6)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    
    # Save multipanel figure
    limits_path = output_dir / "theoretical_limits_multipanel.png"
    from antstack_core.figures.plots import _save_figure
    _save_figure(fig, limits_path)
    
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
  \caption{\textbf{Enhanced Analytical Scaling Sweeps.} Energy and operation counts for the enhanced workload model using assumed default energy coefficients, not device calibration. Energy includes compute and idle baseline; mechanics are not supplied. Mind uses at most 1000 policies. Power-law fits describe displayed arrays only, not empirical uncertainty, asymptotic complexity or phase transitions.}
  \\label{fig:comprehensive_scaling_multipanel}
\\end{figure}
""")
    
    # Theoretical limits figure
    if "theoretical_limits" in figures:
        latex_code.append("""
\\begin{figure}[ht]
  \\centering
  \\includegraphics[width=\\linewidth]{""" + figures["theoretical_limits"].replace(str(output_dir.parent) + "/", "") + """}
  \caption{\textbf{Enhanced Model and Conditional Erasure References.} (A) Compute-plus-idle model estimates and conditional Landauer references assuming one erased bit per FLOP at 300 K. FLOPs do not inherently establish bit erasures. (B) Ratios to that conditional reference, not physical efficiency or achievable improvement. (C) All modeled compute and idle components. (D) Descriptive fits computed from current enhanced sweeps, not confidence estimates. Default energy coefficients are assumptions, not device calibration; mechanical work is not modeled. Mind uses at most 1000 policies.}
  \\label{fig:theoretical_limits_multipanel}
\\end{figure}
""")
    
    # Write LaTeX code to file
    latex_file = output_dir / "multipanel_figures.tex"
    from antstack_core.publishing.provenance import atomic_text
    atomic_text(latex_file, "% Generated enhanced analytical multipanels; compile from this family root.\n\n" + "\n".join(latex_code))
    
    print(f"Generated LaTeX code: {latex_file}")


def main(argv=None):
    """Publish a complete fresh multipanel family; never overwrite paper assets."""
    import argparse
    from antstack_core.analysis.ce_manuscript_figures import _publish_family
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Fresh absent/empty multipanel family root")
    args = parser.parse_args(argv)
    coeffs = EnergyCoefficients()
    def produce(stage):
        figures = {**create_multipanel_scaling_analysis(stage), **create_theoretical_limits_multipanel(stage)}
        generate_latex_figure_code({key: Path(path).name for key, path in figures.items()}, stage)
        models = {Path(path).stem: {'model_definition': key,
                                   'assumed_coefficients': coeffs.to_dict(),
                                   'duration_s': 0.01,
                                   'source': 'ce_multipanel_figures.py fixed enhanced analytical sweeps',
                                   'energy_boundary': 'compute-plus-isolated-idle; no mechanics or calibration',
                                   'erasure_reference': 'one erased bit per FLOP at300K, not physical efficiency'}
                  for key, path in figures.items()}
        return {'figures': figures, 'model_definitions': models}
    return _publish_family(args.output, 'multipanel', 'multipanel_manifest.json', 'png', coeffs,
                           produce, supplementary=('multipanel_figures.tex',))


if __name__ == "__main__":
    main()
