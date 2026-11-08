"""Fresh-root results PNG families with explicit per-figure analytical boundaries."""

from pathlib import Path

from antstack_core.analysis import (
    ComputeLoad, estimate_detailed_energy,
    enhanced_body_workload_closed_form, enhanced_brain_workload_closed_form,
    enhanced_mind_workload_closed_form, calculate_contact_complexity,
    calculate_sparse_neural_complexity,
)
from antstack_core.analysis import EnergyCoefficients
from antstack_core.figures import bar_plot, interval_plot, line_plot, scatter_plot
from antstack_core.analysis.ce_publication_figures import _standard_module_data
from antstack_core.analysis.ce_manuscript_figures import _family_models, _publish_family


def generate_body_scaling_figures(output_dir: Path, coeffs: EnergyCoefficients):
    """Generate AntBody scaling analysis figures using enhanced workload models."""
    print("Generating AntBody scaling figures...")
    
    # Joint count scaling (body_joint_energy_scaling.png)
    joint_counts = [6, 12, 18, 24, 30, 36]
    body_params = {"C": 12, "S": 256, "hz": 100}
    
    energies = []
    complexities = []
    
    for J in joint_counts:
        params = dict(body_params, J=J)
        load = enhanced_body_workload_closed_form(0.01, params)  # 10ms decision
        energy = estimate_detailed_energy(load, coeffs, 0.01).total
        energies.append(energy)
        complexities.append(load.flops)
    
    # Energy scaling plot
    line_plot(joint_counts, [energies], ["Energy (Body)"], 
             "AntBody Energy Scaling vs Joint Count", "Joint Count (J)", 
             "Energy per Decision (J)", str(output_dir / "body_joint_energy_scaling.png"))
    
    # Complexity relationship
    scatter_plot(complexities, energies, "AntBody Energy vs Computational Complexity",
                "FLOPs per Decision", "Energy per Decision (J)", 
                str(output_dir / "body_energy_complexity.png"))


def generate_brain_scaling_figures(output_dir: Path, coeffs: EnergyCoefficients):
    """Generate AntBrain scaling analysis figures using enhanced neural models."""
    print("Generating AntBrain scaling figures...")
    
    # AL channel scaling (brain_channel_energy_scaling.png)
    K_values = [64, 128, 256, 512, 1024]
    brain_params = {"N_KC": 50000, "rho": 0.02, "H": 64, "hz": 100}
    
    energies = []
    spikes = []
    
    for K in K_values:
        params = dict(brain_params, K=K)
        load = enhanced_brain_workload_closed_form(0.01, params)  # 10ms decision
        energy = estimate_detailed_energy(load, coeffs, 0.01).total
        energies.append(energy)
        spikes.append(load.spikes)
    
    # Energy scaling plot
    line_plot(K_values, [energies], ["Energy (Brain)"], 
             "AntBrain Energy Scaling vs AL Channels", "AL Input Channels (K)", 
             "Energy per Decision (J)", str(output_dir / "brain_channel_energy_scaling.png"))
    
    # Spike scaling plot
    line_plot(K_values, [spikes], ["Spikes"], 
             "AntBrain Spike Generation vs AL Channels", "AL Input Channels (K)", 
             "Spikes per Decision", str(output_dir / "brain_channel_spikes_scaling.png"))


def generate_mind_scaling_figures(output_dir: Path, coeffs: EnergyCoefficients):
    """Generate AntMind scaling analysis figures using enhanced active inference models."""
    print("Generating AntMind scaling figures...")
    
    # Policy horizon scaling (mind_horizon_energy_scaling.png)
    horizons = [5, 8, 10, 12, 15, 18, 20]
    mind_params = {"B": 4, "state_dim": 16, "action_dim": 6, "hz": 100}
    
    energies = []
    flops = []
    
    for H_p in horizons:
        params = dict(mind_params, H_p=H_p)
        load = enhanced_mind_workload_closed_form(0.01, params)  # 10ms decision
        energy = estimate_detailed_energy(load, coeffs, 0.01).total
        energies.append(energy)
        flops.append(load.flops)
    
    # Energy scaling plot (super-linear growth)
    line_plot(horizons, [energies], ["Energy (Mind)"], 
             "AntMind Energy Scaling vs Policy Horizon", "Policy Horizon (H_p)", 
             "Energy per Decision (J)", str(output_dir / "mind_horizon_energy_scaling.png"))
    
    # FLOPs explosion plot
    line_plot(horizons, [flops], ["FLOPs"], 
             "AntMind Modeled FLOPs vs Policy Horizon (1000-policy cap)", "Policy Horizon (H_p)",
             "FLOPs per Decision", str(output_dir / "mind_horizon_flops_scaling.png"))


def generate_connectivity_analysis_figures(output_dir: Path, coeffs=None):
    """Generate neural connectivity pattern analysis figures."""
    print("Generating connectivity analysis figures...")
    
    patterns = ["random", "structured"]
    N_total = 50000
    sparsity = 0.02
    
    flops_results = []
    spikes_results = []
    
    for pattern in patterns:
        flops, memory, spikes = calculate_sparse_neural_complexity(N_total, sparsity, pattern)
        flops_results.append(flops)
        spikes_results.append(spikes)
    
    # Connectivity patterns FLOPs comparison
    bar_plot(patterns, flops_results, "Neural Connectivity Computational Cost", 
             str(output_dir / "connectivity_patterns_flops.png"), "FLOPs per Decision")
    
    # Connectivity patterns spikes comparison
    bar_plot(patterns, spikes_results, "Neural Connectivity Spike Generation", 
             str(output_dir / "connectivity_patterns_spikes.png"), "Spikes per Decision")
    
    # Sparsity energy scaling
    sparsity_levels = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    sparsity_energies = []
    coeffs = EnergyCoefficients() if coeffs is None else coeffs
    
    for rho in sparsity_levels:
        flops, memory, spikes = calculate_sparse_neural_complexity(N_total, rho, "random")
        load = ComputeLoad(flops=flops, sram_bytes=memory, dram_bytes=0, spikes=spikes)
        energy = estimate_detailed_energy(load, coeffs, 0.01).total
        sparsity_energies.append(energy)
    
    line_plot(sparsity_levels, [sparsity_energies], ["Isolated sparse network"],
             "Sparse-Only Energy (All SRAM; No AL/CX/Plasticity)", "Sparsity Level (ρ)",
             "Energy per Decision (J)", str(output_dir / "sparsity_energy_scaling.png"))


def generate_theoretical_limits_figures(output_dir: Path, coeffs: EnergyCoefficients):
    """Compare consistent modeled energy with an explicit conditional reference."""
    data = _standard_module_data(coeffs)
    modules = list(data)
    energies = [item['breakdown'].total for item in data.values()]
    references = [item['reference_j'] for item in data.values()]
    ratios = [energy / reference for energy, reference in zip(energies, references)]
    interval_plot([f'{name} model' for name in modules] + [f'{name} reference' for name in modules],
                  energies + references, 'Model vs Conditional Erasure Reference (No Mechanics)',
                  str(output_dir / 'theoretical_limits_comparison.png'), 'Energy per Decision (J)')
    bar_plot(modules, ratios, 'Modeled Energy / Conditional Erasure Reference',
             str(output_dir / 'efficiency_ratios.png'), 'Energy / Reference (Not Physical Efficiency)')


def generate_contact_solver_figures(output_dir: Path):
    """Generate contact solver comparison figures."""
    print("Generating contact solver figures...")
    
    contact_counts = [5, 10, 15, 20, 25, 30]
    solvers = ["pgs", "si", "ldlt"]
    
    solver_flops = {solver: [] for solver in solvers}
    solver_memory = {solver: [] for solver in solvers}
    
    for C in contact_counts:
        for solver in solvers:
            flops, memory, _ = calculate_contact_complexity(18, C, solver)
            solver_flops[solver].append(flops)
            solver_memory[solver].append(memory)
    
    # FLOPs comparison
    flops_series = [solver_flops[solver] for solver in solvers]
    line_plot(contact_counts, flops_series, [s.upper() for s in solvers], 
             "Contact Solver Computational Complexity", "Contact Count (C)", 
             "FLOPs per Decision", str(output_dir / "contact_solver_flops_comparison.png"))
    
    # Memory comparison
    memory_series = [solver_memory[solver] for solver in solvers]
    line_plot(contact_counts, memory_series, [s.upper() for s in solvers], 
             "Contact Solver Memory Requirements", "Contact Count (C)", 
             "Memory (bytes)", str(output_dir / "contact_solver_memory_comparison.png"))


def run(args):
    """Publish the existing results model as one complete, fresh PNG family."""
    coeffs = EnergyCoefficients(
        flops_pj=1.0, sram_pj_per_byte=0.1, dram_pj_per_byte=20.0,
        spike_aj=1.0, baseline_w=0.05,
    )

    def produce(stage):
        generate_body_scaling_figures(stage, coeffs)
        generate_brain_scaling_figures(stage, coeffs)
        generate_mind_scaling_figures(stage, coeffs)
        generate_connectivity_analysis_figures(stage, coeffs)
        generate_theoretical_limits_figures(stage, coeffs)
        generate_contact_solver_figures(stage)
        return {"figures": {stem: str(stage / f"{stem}.png")
                            for stem in _family_models("results", coeffs)}}

    return _publish_family(args.output, "results", "results_figure_model.json", "png", coeffs, produce)

