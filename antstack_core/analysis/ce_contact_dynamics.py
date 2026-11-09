"""Sample analytical contact-solver and terrain workloads in an isolated family."""

import json
from pathlib import Path
from typing import Dict, List, Any

from antstack_core.analysis import calculate_contact_complexity, enhanced_body_workload_closed_form
from antstack_core.analysis import estimate_detailed_energy, analyze_scaling_relationship
from antstack_core.analysis import EnergyCoefficients
from antstack_core.figures import line_plot, bar_plot
from antstack_core.publishing.provenance import _publish_analysis_family


def analyze_solver_comparison(contact_counts: List[int], output_dir: Path) -> Dict[str, Any]:
    """Compare different contact solver algorithms across contact counts."""
    print("Analyzing contact solver performance...")

    solvers = ["pgs", "si", "ldlt"]
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


def run(args):
    maximum = args.max_contacts
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError("max_contacts must be a positive integer")
    contact_counts = sorted({count for count in (5, 10, 15, 20, 25, 30, 40, maximum)
                             if 0 < count <= maximum})
    return _publish_analysis_family(
        getattr(args, "output", None), project="contact_dynamics",
        produce=lambda stage: _produce(stage, contact_counts, maximum),
        required_paths=(
            "contact_dynamics_analysis.json",
            "plots/contact_solver_flops_comparison.png",
            "plots/contact_solver_memory_comparison.png",
            "plots/terrain_energy_comparison.png",
            "plots/terrain_flops_comparison.png",
        ),
        parameters={
            "max_contacts": maximum, "contact_counts": contact_counts,
            "solvers": ["pgs", "si", "ldlt"], "solver_joints": 18,
            "terrain_contacts": {name: min(count, maximum) for name, count in
                                 (("Smooth", 8), ("Rough", 16), ("Steep", 20), ("Soft", 12))},
            "decision_duration_s": 0.01, "hz": 100,
            "evidence_class": "analytical/synthetic workload and model-energy estimates",
        },
    )


def _produce(output_dir: Path, contact_counts: List[int], maximum: int):

    print("=" * 60)
    print("Contact Dynamics Analysis Orchestrator")
    print("=" * 60)

    # The sorted grid includes the requested endpoint and never exceeds it.
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
                'contact_solver': 'ldlt'
            }
        },
        {
            'name': 'Soft',
            'params': {
                'J': 18, 'C': 12, 'S': 256, 'hz': 100,
                'contact_solver': 'si'
            }
        }
    ]
    for config in terrain_configs:
        config['params']['C'] = min(config['params']['C'], maximum)

    # Analyze terrain effects
    terrain_results = analyze_terrain_effects(terrain_configs, output_dir)

    # Compile comprehensive results
    all_results = {
        'solver_analysis': solver_results,
        'terrain_analysis': terrain_results,
        'configuration': {
            'contact_counts': contact_counts,
            'solvers': ['pgs', 'si', 'ldlt'],
            'max_contacts': maximum,
            'decision_duration_s': 0.01,
            'evidence_class': 'analytical/synthetic estimates; not empirical measurements'
        }
    }

    # Save results
    results_path = output_dir / "contact_dynamics_analysis.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nContact dynamics analysis complete!")

    # Print key findings
    print(f"\nKey Findings:")
    for solver in ("pgs", "si", "ldlt"):
        fit = solver_results[solver]['flops_scaling']
        description = fit['scaling_regime'] if fit['valid'] else f"unavailable ({fit['error']})"
        print(f"- {solver.upper()} sampled FLOP scaling: {description}")

    max_energy_terrain = max(terrain_results['summary']['terrain_names'],
                           key=lambda x: terrain_results[x]['energy_total'])
    min_energy_terrain = min(terrain_results['summary']['terrain_names'],
                           key=lambda x: terrain_results[x]['energy_total'])

    print(f"- Highest sampled model-energy terrain: {max_energy_terrain}")
    print(f"- Lowest sampled model-energy terrain: {min_energy_terrain}")
