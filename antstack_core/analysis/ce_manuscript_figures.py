"""Fresh-root manuscript figure families; direct producers default to PNG.

``run`` owns the complete family transaction under trusted exclusive output
ownership. Analytical estimates are not canonical base-run or calibrated data.
"""

import os
import stat
import tempfile
from pathlib import Path
from typing import Dict, Any

from antstack_core.analysis import (
    estimate_detailed_energy, estimate_theoretical_limits,
    enhanced_body_workload_closed_form, enhanced_brain_workload_closed_form,
    enhanced_mind_workload_closed_form, calculate_contact_complexity,
    calculate_sparse_neural_complexity,
)
from antstack_core.analysis import EnergyCoefficients
from antstack_core.figures import bar_plot, interval_plot, line_plot, scatter_plot
from antstack_core.publishing.provenance import (
    atomic_json, build_run_provenance, capture_source_state, file_identity,
    ordinary_path, publication_path, write_provenance,
)


def _image_path(output_dir, stem, image_format):
    if image_format not in {"png", "svg", "pdf"}:
        raise ValueError(f"Unsupported figure format: {image_format}")
    return Path(output_dir) / f"{stem}.{image_format}"


def _family_models(suite, coeffs):
    """Record the distinct existing sweeps and energy boundaries, not reconcile them."""
    manuscript = suite == "manuscript"
    body = {"J": 18, "C": 12, "S": 256, "hz": 100, "P": 4096, "contact_solver": "pgs"}
    brain = {"K": 128, "N_KC": 50000, "rho": 0.02, "H": 64, "hz": 100,
             "connectivity_pattern": "random"}
    mind = {"B": 4, "H_p": 12, "hz": 100, "state_dim": 16, "action_dim": 6,
            "max_policies": 1000, "hierarchical": False}
    energy = {"duration_s": 0.01, "energy_boundary": "compute-plus-isolated-idle; no actuation or sensing power supplied",
              "assumed_coefficients": coeffs.to_dict(), "calibration": "not performed"}
    body_model = dict(energy, model="enhanced_body_workload_closed_form", parameters=body,
                      memory_fraction={"sram": 0.75, "dram": 0.25})
    brain_model = dict(energy, model="enhanced_brain_workload_closed_form", parameters=brain,
                       memory_fraction={"sram": 0.85, "dram": 0.15},
                       components=["AL", "sparse MB", "CX", "plasticity"])
    mind_model = dict(energy, model="enhanced_mind_workload_closed_form", parameters=mind,
                      memory_fraction={"sram": 0.8, "dram": 0.2})
    body_sweep = dict(body_model, sweep={"parameter": "J", "values": [6, 12, 18, 24, 36] if manuscript else [6, 12, 18, 24, 30, 36]})
    brain_sweep = dict(brain_model, sweep={"parameter": "K", "values": [64, 128, 256, 512, 1024]})
    mind_sweep = dict(mind_model,
                      parameters=dict(mind, B=3, state_dim=12, action_dim=4) if manuscript else mind,
                      sweep={"parameter": "H_p", "values": [5, 7, 9, 11, 13] if manuscript else [5, 8, 10, 12, 15, 18, 20]})
    sparse = {"model": "calculate_sparse_neural_complexity", "parameters": {"N_total": 50000, "sparsity": 0.02},
              "patterns": ["random", "structured"], "energy_boundary": "counters only; isolated sparse network, not full Brain"}
    sparsity_sweep = {"parameter": "rho", "values": [0.005, 0.01, 0.02, 0.05, 0.1] if manuscript else [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]}
    if manuscript:
        sparse_energy = dict(brain_model, sweep=sparsity_sweep)
    else:
        sparse_energy = dict(energy, model="calculate_sparse_neural_complexity",
                             parameters={"N_total": 50000, "connectivity_pattern": "random"},
                             sweep=sparsity_sweep, memory_fraction={"sram": 1.0, "dram": 0.0},
                             components=["isolated sparse network only; no AL, CX or plasticity"])
    reference = {"modules": {"AntBody": body_model, "AntBrain": brain_model,
                             "AntMind": dict(mind_model, parameters=dict(mind, H_p=8)) if manuscript else mind_model},
                 "reference": {"bits_erased_per_flop": 1, "temperature_k": 300,
                               "interpretation": "conditional erasure reference, not physical efficiency"},
                 "idle_accounting": "each module modeled in isolation with one baseline charge"}
    contacts = {"model": "calculate_contact_complexity", "parameters": {"J": 18},
                "sweep": {"parameter": "C", "values": [5, 10, 15, 20, 25, 30]},
                "solvers": ["pgs", "si", "ldlt"], "energy_boundary": "solver counters only; no energy model"}
    models = {
        "body_joint_energy_scaling": body_sweep, "body_energy_complexity": body_sweep,
        "brain_channel_energy_scaling": brain_sweep, "brain_channel_spikes_scaling": brain_sweep,
        "mind_horizon_energy_scaling": mind_sweep, "mind_horizon_flops_scaling": mind_sweep,
        "connectivity_patterns_flops": sparse, "connectivity_patterns_spikes": sparse,
        "sparsity_energy_scaling": sparse_energy,
        "theoretical_limits_comparison": reference, "efficiency_ratios": reference,
        "contact_solver_flops_comparison": contacts, "contact_solver_memory_comparison": contacts,
    }
    if manuscript:
        modules = {"AntBody": body_model, "AntBrain": brain_model, "AntMind": mind_model}
        models["module_energy_comparison"] = {"modules": modules, "idle_accounting": "one independent baseline per module"}
        models["energy_component_breakdown"] = {"modules": modules,
                                               "idle_accounting": "sum of three independent module baselines, not one shared system baseline"}
    return models


def _fresh_family_path(output):
    if output is None:
        raise ValueError("An explicit fresh family output directory is required")
    destination = publication_path(output, tree=True)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(f"Figure family output must be absent or empty: {destination}")
    return destination


def _publish_family(output, suite, manifest_name, image_format, coeffs, produce, *, supplementary=()):
    """Stage a complete family and atomically rename it into a fresh namespace.

    The shared provenance record and relative output inventory follow the existing
    publishing contract. Stable links and collisions are rejected, not hostile
    same-UID replacement races: callers exclusively own this output namespace.
    """
    destination = _fresh_family_path(output)
    source_state = capture_source_state()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.stage-", dir=destination.parent) as directory:
        stage = Path(directory)
        manifest = produce(stage)
        models = manifest.pop("model_definitions", None)
        if models is None:
            models = _family_models(suite, coeffs)
        if not models:
            raise ValueError("A figure family must declare actual produced models")
        figures = {key: ordinary_path(path).relative_to(stage).as_posix()
                   for key, path in manifest["figures"].items()}
        expected = {f"{stem}.{image_format}" for stem in models}
        if set(figures.values()) != expected or len(figures) != len(expected):
            raise ValueError("Figure family is missing products or contains duplicate/unexpected products")
        extras = set(supplementary)
        if any(Path(name).name != name for name in extras) or extras & expected:
            raise ValueError("Invalid supplementary figure product")
        if {path.name for path in stage.iterdir()} != expected | extras:
            raise ValueError("Figure family staging contains unregistered products")
        for name in expected | extras:
            path = stage / name
            identity = file_identity(path)
            if not identity["bytes"] or path.stat().st_nlink != 1:
                raise ValueError(f"Figure product must be a nonempty exclusive file: {name}")
        manifest.update({"schema_version": 1, "complete": True, "family": suite,
                         "artifact_path_base": "family_root", "figures": figures,
                         "figure_models": {f"{stem}.{image_format}": model for stem, model in models.items()},
                         "model_provenance": "deterministic analytical estimates; not canonical base-run evidence; no empirical validation or calibration",
                         "generation_info": {"output_directory": str(destination), "format": image_format,
                                             "total_figures": len(figures)}})
        atomic_json(stage / manifest_name, manifest)
        provenance_name = "provenance/provenance.json"
        inventory_name = "provenance/output_inventory.json"
        products = sorted(expected | extras) + [manifest_name, provenance_name]
        provenance = build_run_provenance(
            project=f"ant-stack-{suite}-figures", output_paths=products + [inventory_name],
            parameters={"family": suite, "output_path_base": "family_root", "format": image_format,
                        "figure_models": manifest["figure_models"]}, source_state=source_state)
        write_provenance(stage / provenance_name, provenance)
        artifacts = []
        for name in products:
            identity = file_identity(stage / name)
            kind = "image" if name in expected else "markdown" if name.endswith(".md") else "latex" if name.endswith(".tex") else "provenance"
            artifacts.append({"path": name, "artifact_type": kind,
                              "producing_task": f"{suite}.figures", "source_config": manifest_name,
                              "checksum": identity["sha256"], "bytes": identity["bytes"]})
        atomic_json(stage / inventory_name, {"artifact_path_base": "family_root", "artifacts": artifacts})
        for artifact in artifacts:
            if file_identity(stage / artifact["path"]) != {"sha256": artifact["checksum"], "bytes": artifact["bytes"]}:
                raise RuntimeError(f"Figure family changed before publication: {artifact['path']}")
        _fresh_family_path(destination)
        if destination.exists():
            os.chmod(stage, stat.S_IMODE(destination.stat().st_mode))
        os.replace(stage, destination)
    print(f"Published {len(expected)} {image_format.upper()} figures: {destination}")
    return destination / manifest_name


def generate_complexity_overview_data() -> Dict[str, Any]:
    """Generate data for complexity overview figure."""
    print("Generating complexity overview data...")
    
    # Standard parameters
    body_params = {'J': 18, 'C': 12, 'S': 256, 'hz': 100, 'P': 4096, 'contact_solver': 'pgs'}
    brain_params = {'K': 128, 'N_KC': 50000, 'rho': 0.02, 'H': 64, 'hz': 100,
                    'connectivity_pattern': 'random'}
    mind_params = {'B': 4, 'H_p': 15, 'hz': 100, 'state_dim': 16, 'action_dim': 6,
                   'max_policies': 1000, 'hierarchical': False}
    
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


def generate_energy_breakdown_figures(output_dir: Path, image_format: str = "png") -> Dict[str, str]:
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
    
    module_comparison_plot = _image_path(output_dir, "module_energy_comparison", image_format)
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
    
    component_breakdown_plot = _image_path(output_dir, "energy_component_breakdown", image_format)
    bar_plot(
        components, [e * 1000 for e in component_energies],  # Convert J to mJ once
        "Enhanced Compute-plus-Idle Components (Assumed Coefficients, No Mechanics)",
        str(component_breakdown_plot), ylabel="Energy per Decision (mJ)"
    )
    
    return {
        'module_comparison': str(module_comparison_plot),
        'component_breakdown': str(component_breakdown_plot)
    }


def generate_scaling_analysis_figures(output_dir: Path, image_format: str = "png") -> Dict[str, str]:
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
    body_energy_plot = _image_path(output_dir, "body_joint_energy_scaling", image_format)
    line_plot(
        J_values, [body_energies], ["Body Energy"],
        "AntBody: Energy Scaling vs Joint Count",
        "Number of Joints (J)", "Energy per Decision (J)",
        str(body_energy_plot)
    )
    figures['body_energy_scaling'] = str(body_energy_plot)
    
    # Body energy vs complexity
    body_complexity_plot = _image_path(output_dir, "body_energy_complexity", image_format)
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
    brain_energy_plot = _image_path(output_dir, "brain_channel_energy_scaling", image_format)
    line_plot(
        K_values, [brain_energies], ["Brain Energy"],
        "AntBrain: Energy Scaling vs AL Channels",
        "AL Input Channels (K)", "Energy per Decision (J)",
        str(brain_energy_plot)
    )
    figures['brain_energy_scaling'] = str(brain_energy_plot)
    
    # Brain spikes scaling
    brain_spikes_plot = _image_path(output_dir, "brain_channel_spikes_scaling", image_format)
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
    mind_energy_plot = _image_path(output_dir, "mind_horizon_energy_scaling", image_format)
    line_plot(
        H_p_values, [mind_energies], ["Mind Energy"],
        "AntMind: Energy Scaling vs Policy Horizon",
        "Policy Horizon (steps)", "Energy per Decision (J)",
        str(mind_energy_plot)
    )
    figures['mind_energy_scaling'] = str(mind_energy_plot)
    
    # Mind computational explosion
    mind_flops_plot = _image_path(output_dir, "mind_horizon_flops_scaling", image_format)
    line_plot(
        H_p_values, [mind_flops], ["Mind FLOPs"],
        "AntMind: Modeled FLOPs vs Policy Horizon (1000-policy cap)",
        "Policy Horizon (steps)", "FLOPs per Decision",
        str(mind_flops_plot)
    )
    figures['mind_flops_scaling'] = str(mind_flops_plot)
    
    return figures


def generate_connectivity_analysis_figures(output_dir: Path, image_format: str = "png") -> Dict[str, str]:
    """Generate neural connectivity pattern analysis figures."""
    print("Generating connectivity analysis figures...")
    
    network_size = 50000
    sparsity = 0.02
    patterns = ["random", "structured"]
    
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
    connectivity_flops_plot = _image_path(output_dir, "connectivity_patterns_flops", image_format)
    bar_plot(
        [p.replace('_', ' ').title() for p in patterns], pattern_flops,
        "Neural Connectivity Patterns: Computational Cost",
        str(connectivity_flops_plot), ylabel="FLOPs per Decision"
    )
    
    connectivity_spikes_plot = _image_path(output_dir, "connectivity_patterns_spikes", image_format)
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
    
    sparsity_energy_plot = _image_path(output_dir, "sparsity_energy_scaling", image_format)
    line_plot(
        sparsity_levels, [sparsity_energies], ["Energy"],
        "Full Enhanced Brain: Sparsity Effects on Energy",
        "Sparsity Level (ρ)", "Energy per Decision (J)",
        str(sparsity_energy_plot)
    )
    
    return {
        'connectivity_flops': str(connectivity_flops_plot),
        'connectivity_spikes': str(connectivity_spikes_plot),
        'sparsity_energy': str(sparsity_energy_plot)
    }


def generate_theoretical_limits_figures(output_dir: Path, image_format: str = "png") -> Dict[str, str]:
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
        'bits_erased': body_load.flops
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
        'bits_erased': brain_load.flops
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
        'bits_erased': mind_load.flops
    })
    
    actual_energies.append(mind_energy)
    theoretical_energies.append(mind_limits['total_theoretical_j'])
    efficiency_ratios.append(mind_energy / mind_limits['total_theoretical_j'])
    
    # Theoretical limits comparison
    all_energies = actual_energies + theoretical_energies
    all_labels = [f"{m} (Model)" for m in modules] + [f"{m} (1 erased bit/FLOP, 300 K)" for m in modules]
    
    limits_plot = _image_path(output_dir, "theoretical_limits_comparison", image_format)
    interval_plot(
        all_labels, all_energies,
        "Modeled Energy vs Conditional Erasure Reference (No Mechanics)",
        str(limits_plot), xlabel="Energy per Decision (J)"
    )
    
    # Efficiency ratios
    efficiency_plot = _image_path(output_dir, "efficiency_ratios", image_format)
    bar_plot(
        modules, efficiency_ratios,
        "Modeled Energy / Conditional Erasure Reference",
        str(efficiency_plot), ylabel="Energy / Reference (Not Physical Efficiency)"
    )
    
    return {
        'theoretical_limits': str(limits_plot),
        'efficiency_ratios': str(efficiency_plot)
    }


def generate_contact_solver_figures(output_dir: Path, image_format: str = "png") -> Dict[str, str]:
    """Generate contact solver comparison figures."""
    print("Generating contact solver comparison figures...")
    
    contact_counts = [5, 10, 15, 20, 25, 30]
    solvers = ["pgs", "si", "ldlt"]
    
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
    
    solver_flops_plot = _image_path(output_dir, "contact_solver_flops_comparison", image_format)
    line_plot(
        contact_counts, flops_series, solver_labels,
        "Contact Solver Computational Complexity",
        "Number of Contacts", "FLOPs per Solve",
        str(solver_flops_plot)
    )
    
    # Memory comparison
    memory_series = [solver_data[solver]['memory'] for solver in solvers]
    
    solver_memory_plot = _image_path(output_dir, "contact_solver_memory_comparison", image_format)
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


def run(args):
    """Publish all fifteen figures and strict metadata together, never in place."""
    image_format = args.format
    _image_path(Path("."), "format-validation", image_format)

    def produce(stage):
        figures = {}
        for producer in (generate_energy_breakdown_figures, generate_scaling_analysis_figures,
                         generate_connectivity_analysis_figures, generate_theoretical_limits_figures,
                         generate_contact_solver_figures):
            figures.update(producer(stage, image_format=image_format))
        return {"figures": figures, "complexity_data": generate_complexity_overview_data()}

    return _publish_family(args.output, "manuscript", "figure_manifest.json", image_format,
                           EnergyCoefficients(), produce)

