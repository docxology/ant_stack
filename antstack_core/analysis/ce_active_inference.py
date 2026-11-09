"""Sample analytical aggregate-policy workloads in an isolated family."""

import json
from pathlib import Path
from typing import Dict, List, Any

from antstack_core.analysis import (
    enhanced_mind_workload_closed_form
)
from antstack_core.analysis import (
    estimate_detailed_energy, analyze_scaling_relationship,
    estimate_theoretical_limits
)
from antstack_core.analysis import EnergyCoefficients
from antstack_core.figures import line_plot, bar_plot
from antstack_core.publishing.provenance import _publish_analysis_family

from .workloads import effective_policy_count

def analyze_horizon_scaling(horizons: List[int], branching: int, state_dim: int,
                          action_dim: int, output_dir: Path) -> Dict[str, Any]:
    """Analyze the aggregate policy model with a fixed 1000-policy budget."""
    print("Analyzing policy horizon scaling...")

    coeffs = EnergyCoefficients()
    results = {}

    energy_values = []
    flops_values = []
    memory_values = []
    theoretical_limits = []

    for H_p in horizons:
        # Calculate energy using mind workload model
        params = {
            'B': branching, 'H_p': H_p, 'hz': 100,
            'state_dim': state_dim, 'action_dim': action_dim,
            'hierarchical': False, 'max_policies': 1000
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
            'model': 'bounded_aggregate_policy_evaluation',
            'max_policies': 1000,
            'effective_policies': effective_policy_count(branching, H_p, 1000),
            'energy_total': breakdown.total,
            'energy_breakdown': breakdown.to_dict(),
            'flops': load.flops,
            'memory': load.sram_bytes,
            'theoretical_limit': limits['total_theoretical_j'],
            'model_to_theoretical_energy_ratio': breakdown.total / limits['total_theoretical_j']
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
        ["Model Energy", "Theoretical Bound"],
        "Model Energy vs Theoretical Bound",
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
        # Calculate total policy count (before sampling)
        total_policies = B ** horizon
        effective_policies = effective_policy_count(B, horizon, 1000)

        params = {
            'B': B, 'H_p': horizon, 'hz': 100,
            'state_dim': state_dim, 'action_dim': action_dim,
            'hierarchical': False, 'max_policies': 1000
        }

        load = enhanced_mind_workload_closed_form(0.01, params)
        breakdown = estimate_detailed_energy(load, coeffs, 0.01)

        energy_values.append(breakdown.total)
        flops_values.append(load.flops)
        policy_counts.append(effective_policies)

        results[f"B_{B}"] = {
            'branching': B,
            'horizon': horizon,
            'model': 'bounded_aggregate_policy_evaluation',
            'max_policies': 1000,
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

        total_policies = branching ** horizon
        effective_policies = effective_policy_count(branching, horizon, strategy['max_policies'])

        params = {
            'B': branching, 'H_p': horizon, 'hz': 100,
            'state_dim': state_dim, 'action_dim': action_dim,
            'hierarchical': strategy['hierarchical'],
            'max_policies': strategy['max_policies'],
        }

        # Exact and bounded cases use the same component accounting and prices.
        load = enhanced_mind_workload_closed_form(0.01, params)
        breakdown = estimate_detailed_energy(load, coeffs, 0.01)
        energy_values.append(breakdown.total)
        flops_values.append(load.flops)

        strategy_names.append(name)

        results[name] = {
            'strategy': strategy,
            'horizon': horizon,
            'model': 'exact_aggregate_policy_evaluation' if strategy['max_policies'] is None else 'bounded_aggregate_policy_evaluation',
            'total_policies': total_policies,
            'effective_policies': effective_policies,
            'energy_total': energy_values[-1],
            'flops': flops_values[-1]
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


def run(args):
    maximum = args.max_horizon
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 5:
        raise ValueError("max_horizon must be an integer of at least 5")
    branching = args.branching
    if isinstance(branching, bool) or not isinstance(branching, int) or branching < 1:
        raise ValueError("branching must be a positive integer")
    horizons = sorted(set(range(5, maximum + 1, 2)) | {maximum})
    return _publish_analysis_family(
        getattr(args, "output", None), project="active_inference",
        produce=lambda stage: _produce(stage, horizons, maximum, branching),
        required_paths=(
            "active_inference_analysis.json",
            "plots/horizon_energy_scaling.png",
            "plots/horizon_flops_scaling.png",
            "plots/horizon_theoretical_limits.png",
            "plots/branching_energy_scaling.png",
            "plots/branching_policy_count.png",
            "plots/bounded_rationality_energy.png",
            "plots/bounded_rationality_flops.png",
        ),
        parameters={
            "max_horizon": maximum, "horizons": horizons, "branching": branching,
            "branching_factors": [2, 3, 4, 5, 6],
            "branching_study_horizon": min(10, maximum),
            "strategy_study_horizon": min(12, maximum), "state_dim": 16, "action_dim": 6,
            "max_policies": 1000, "strategy_budgets": [None, 1000, 1000, 100],
            "decision_duration_s": 0.01, "hz": 100,
            "evidence_class": "analytical/synthetic aggregate-policy model; not runtime feasibility",
        },
    )


def _produce(output_dir: Path, horizons: List[int], maximum: int, branching: int):

    print("=" * 60)
    print("Active Inference Analysis Orchestrator")
    print("=" * 60)

    # Define analysis parameters
    # Include the exact upper endpoint even when the stride does not reach it.
    branching_factors = [2, 3, 4, 5, 6]
    state_dim = 16
    action_dim = 6

    # Analyze horizon scaling
    horizon_results = analyze_horizon_scaling(
        horizons, branching, state_dim, action_dim, output_dir
    )

    # Analyze branching effects
    branching_results = analyze_branching_effects(
        min(10, maximum), branching_factors, state_dim, action_dim, output_dir
    )

    # Analyze bounded rationality
    bounded_results = analyze_bounded_rationality(
        min(12, maximum), branching, state_dim, action_dim, output_dir
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
            'max_horizon': maximum,
            'default_branching': branching,
            'branching_study_horizon': min(10, maximum),
            'strategy_study_horizon': min(12, maximum),
            'decision_duration_s': 0.01,
            'evidence_class': 'analytical/synthetic estimates; not runtime feasibility'
        }
    }
    # This is a descriptive cutoff over sampled model energies, not tractability.
    below_cut = [h for h in horizons if horizon_results[f"H_p_{h}"]['energy_total'] < 1.0]
    all_results['sampled_model_energy_summary'] = {
        'cutoff_j': 1.0,
        'horizons_strictly_below_cutoff': below_cut,
        'largest_sampled_horizon_below_cutoff': max(below_cut) if below_cut else None,
    }

    # Save results
    results_path = output_dir / "active_inference_analysis.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nActive inference analysis complete!")

    # Print key findings
    print(f"\nKey Findings:")

    # Horizon scaling
    fit = horizon_results['summary']['energy_scaling']
    if fit['valid']:
        print(f"- Sampled horizon model-energy scaling: {fit['scaling_regime']} "
              f"(exponent: {fit['scaling_exponent']:.3f})")
    else:
        print(f"- Sampled horizon model-energy scaling unavailable: {fit['error']}")

    # Branching effects
    fit = branching_results['summary']['energy_scaling']
    description = fit['scaling_regime'] if fit['valid'] else f"unavailable ({fit['error']})"
    print(f"- Sampled branching model-energy scaling: {description}")

    # Bounded rationality effectiveness
    strategies = bounded_results['summary']['strategy_names']
    energies = bounded_results['summary']['energy_values']
    best_strategy = strategies[energies.index(min(energies))]
    print(f"- Lowest sampled model-energy strategy: {best_strategy}")
    if below_cut:
        print(f"- Sampled horizons with model energy below 1 J: {below_cut}")
    else:
        print("- No sampled horizon has model energy below 1 J")
    print("- These aggregate-policy estimates do not establish runtime feasibility or physical efficiency")
