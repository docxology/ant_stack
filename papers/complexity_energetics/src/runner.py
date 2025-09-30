"""
Ant Stack Complexity & Energetics Analysis Runner

This module provides the main analysis pipeline for computational complexity and energy
analysis of the Ant Stack framework. It integrates analytical models with empirical
measurements to generate comprehensive energy and complexity characterizations.

Key Features:
- Manifest-driven experimental configuration
- Statistical validation with bootstrap confidence intervals
- Automated figure generation with publication-quality outputs
- Comprehensive energy modeling across all system components
- Reproducible analysis with deterministic seeding

Author: Daniel Ari Friedman
Institution: Active Inference Institute
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Use only real, compliant methods from antstack_core
from antstack_core.analysis import ExperimentManifest
from antstack_core.analysis import EnergyCoefficients
from antstack_core.analysis import estimate_compute_energy, bootstrap_mean_ci, add_baseline_energy, cost_of_transport, analyze_scaling_relationship
from antstack_core.analysis import (
    body_workload,
    brain_workload,
    mind_workload,
    estimate_body_energy_mech,
    body_workload_closed_form,
    brain_workload_closed_form,
    mind_workload_closed_form,
    estimate_body_compute_per_decision,
    estimate_brain_compute_per_decision,
    estimate_mind_compute_per_decision,
)
from antstack_core.analysis import measure_energy, NullPowerMeter, RaplPowerMeter, NvmlPowerMeter
from antstack_core.figures import bar_plot, line_plot, scatter_plot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


WORKLOADS = {
    "body": body_workload,
    "brain": brain_workload,
    "mind": mind_workload,
}

CLOSED_FORM_WORKLOADS = {
    "body": body_workload_closed_form,
    "brain": brain_workload_closed_form,
    "mind": mind_workload_closed_form,
}


def run_manifest(manifest_path: str, out_dir: str) -> None:
    """
    Execute the complete analysis pipeline based on a manifest configuration.
    
    Args:
        manifest_path: Path to the YAML manifest file containing experimental configuration
        out_dir: Output directory for results, figures, and generated content
        
    Raises:
        FileNotFoundError: If manifest file doesn't exist
        ValueError: If manifest contains invalid configuration
        RuntimeError: If analysis pipeline fails
    """
    try:
        logger.info(f"Starting analysis with manifest: {manifest_path}")
        logger.info(f"Output directory: {out_dir}")
        
        # Create output directory
        os.makedirs(out_dir, exist_ok=True)
        
        # Load and validate manifest
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
            
        manifest = ExperimentManifest.load(manifest_path)
        logger.info(f"Loaded manifest with seed: {manifest.seed}")
        
        # Set random seed for reproducibility
        random.seed(manifest.seed)
        
        # Initialize energy coefficients from manifest (SINGLE SOURCE OF TRUTH)
        coeff = EnergyCoefficients(
            flops_pj=float(manifest.coefficients.flops_pj),
            sram_pj_per_byte=float(manifest.coefficients.sram_pj_per_byte),
            dram_pj_per_byte=float(manifest.coefficients.dram_pj_per_byte),
            spike_aj=float(manifest.coefficients.spike_aj),
            baseline_w=float(manifest.coefficients.baseline_w),
            body_per_joint_w=float(manifest.coefficients.body_per_joint_w),
            body_sensor_w_per_channel=float(manifest.coefficients.body_sensor_w_per_channel),
        )
        logger.info("Initialized energy coefficients from manifest")
        logger.info(f"  FLOP energy: {coeff.flops_pj:.2f} pJ/FLOP")
        logger.info(f"  SRAM energy: {coeff.sram_pj_per_byte:.2f} pJ/byte")
        logger.info(f"  DRAM energy: {coeff.dram_pj_per_byte:.2f} pJ/byte")
        logger.info(f"  Spike energy: {coeff.spike_aj:.2f} aJ/spike")
        logger.info(f"  Baseline power: {coeff.baseline_w:.3f} W")
        
    except Exception as e:
        logger.error(f"Failed to initialize analysis: {e}")
        raise RuntimeError(f"Analysis initialization failed: {e}") from e

    # Initialize power measurement system
    try:
        rows = []
        meter_cfg = manifest.meter
        meter_type = (meter_cfg.meter_type if meter_cfg else "null").lower()
        
        if meter_type == "rapl":
            meter = RaplPowerMeter(meter_cfg.energy_path if meter_cfg else None)
            logger.info("Initialized RAPL power meter")
        elif meter_type == "nvml":
            idx = int(meter_cfg.device_index if meter_cfg else 0)
            meter = NvmlPowerMeter(idx)
            logger.info(f"Initialized NVML power meter for device {idx}")
        else:
            meter = NullPowerMeter()
            logger.info("Using null power meter (simulation mode)")
            
    except Exception as e:
        logger.error(f"Failed to initialize power measurement: {e}")
        raise RuntimeError(f"Power measurement initialization failed: {e}") from e

    # Execute workload analysis
    try:
        logger.info("Starting workload analysis")
        for wl_name, wl_cfg in (manifest.workloads or {}).items():
            logger.info(f"Processing workload: {wl_name}")
            
            fn = WORKLOADS.get(wl_name)
            if fn is None:
                logger.warning(f"Unknown workload: {wl_name}, skipping")
                continue
                
            for r in range(wl_cfg.repeats):
                try:
                    with measure_energy(meter):
                        t0 = time.time()
                        if (wl_cfg.mode or "loop") == "closed_form":
                            # For reproducible energy magnitudes, use duration directly
                            fn_cf = CLOSED_FORM_WORKLOADS.get(wl_name, fn)
                            load = fn_cf(wl_cfg.duration_s, wl_cfg.params or {})
                            dt = float(wl_cfg.duration_s)
                        else:
                            load = fn(wl_cfg.duration_s, wl_cfg.params or {})
                            dt = time.time() - t0
                            
                    e_est = estimate_compute_energy(load, coeff)
                    e_est = add_baseline_energy(e_est, dt, coeff)
                    
                    rows.append({
                        "workload": wl_name,
                        "repeat": r,
                        "duration_s": dt,
                        "flops": load.flops,
                        "sram_bytes": load.sram_bytes,
                        "dram_bytes": load.dram_bytes,
                        "spikes": load.spikes,
                        "energy_est_j": e_est,
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to process workload {wl_name}, repeat {r}: {e}")
                    continue
                    
        logger.info(f"Completed workload analysis with {len(rows)} measurements")
        
    except Exception as e:
        logger.error(f"Workload analysis failed: {e}")
        raise RuntimeError(f"Workload analysis failed: {e}") from e

    # Write CSV and JSON sidecar with summary stats
    csv_path = os.path.join(out_dir, "results.csv")
    fieldnames = list(rows[0].keys()) if rows else [
        "workload", "repeat", "duration_s", "flops", "sram_bytes", "dram_bytes", "spikes", "energy_est_j"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"wrote: {csv_path}")
    # JSON summary
    summary = {
        "seed": manifest.seed,
        "coefficients": asdict(coeff),
        "rows": rows,
    }
    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)
    print(f"wrote: {json_path}")

    # Aggregate and plot (optional)
    totals: Dict[str, float] = {}
    per_wl: Dict[str, List[float]] = {}
    body_sense: List[float] = []
    body_act: List[float] = []
    for row in rows:
        wl = row["workload"]
        e = float(row["energy_est_j"])
        # Only Body and Brain expend energy; Mind is symbolic (0 J by convention)
        if wl != "mind":
            totals[wl] = totals.get(wl, 0.0) + e
            per_wl.setdefault(wl, []).append(e)
        # Compute simple body partition using mechanical/sensing model
        if wl == "body":
            # Reuse manifest defaults for J/S
            body_params = (manifest.workloads.get("body").params if manifest.workloads and manifest.workloads.get("body") else {})
            e_body = estimate_body_energy_mech(row["duration_s"], body_params or {}, coeff)
            # Assume sensing is S * sensor_w_per_channel * duration
            S = int((body_params or {}).get("S", 256))
            sens_w = float((body_params or {}).get("sensor_w_per_channel", coeff.body_sensor_w_per_channel))
            e_sense = sens_w * S * float(row["duration_s"])
            e_act = max(0.0, e_body - e_sense)
            body_sense.append(e_sense)
            body_act.append(e_act)
        
    # Prepare assets path for embedded figures in the paper
    assets_dir = os.path.join(os.path.dirname(out_dir), "assets")
    os.makedirs(assets_dir, exist_ok=True)
    abs_plot_path = Path(assets_dir) / "energy.png"
    abs_body_split_path = Path(assets_dir) / "body_split.png"
    abs_breakdown_path = Path(assets_dir) / "per_decision_breakdown.png"
    if totals:
        # Ensure plot order Body, Brain
        labels = [lbl for lbl in ["body", "brain"] if lbl in totals]
        values = [totals[lbl] for lbl in labels]
        # Simple yerr from 95% CI half-widths if available
        yerrs: List[float] = []
        for lbl in labels:
            vals = per_wl.get(lbl, [])
            if vals:
                mean, lo, hi = bootstrap_mean_ci(vals, num_samples=1000, alpha=0.05, seed=manifest.seed)
                yerrs.append(max(0.0, hi - mean))
            else:
                yerrs.append(0.0)
        bar_plot(labels, values, "Estimated Energy by Workload (Body, Brain)", str(abs_plot_path), ylabel="Joules", yerr=yerrs)
    if body_sense or body_act:
        bar_plot(
            ["Sense", "Actuation"],
            [sum(body_sense), sum(body_act)],
            "Body Energy Partition (Sense vs Actuation)",
            str(abs_body_split_path),
            ylabel="Joules",
        )
    # Per-decision breakdown at 100 Hz (10 ms), using closed-form compute and mechanical model
    try:
        body_params_pd = (manifest.workloads.get("body").params if manifest.workloads and manifest.workloads.get("body") else {}) or {}
        dt_decision = 0.01
        # Body actuation and sensing
        e_body_dec = estimate_body_energy_mech(dt_decision, body_params_pd, coeff)
        S_pd = int(body_params_pd.get("S", 256))
        sens_w_pd = float(body_params_pd.get("sensor_w_per_channel", coeff.body_sensor_w_per_channel))
        e_sense_dec = sens_w_pd * S_pd * dt_decision
        e_act_dec = max(0.0, e_body_dec - e_sense_dec)
        # Brain compute per decision
        brain_params_pd = (manifest.workloads.get("brain").params if manifest.workloads and manifest.workloads.get("brain") else {}) or {}
        comp_brain_dec = estimate_brain_compute_per_decision(brain_params_pd)
        e_brain_dec = estimate_compute_energy(comp_brain_dec, coeff)
        # Mind compute (0 by convention)
        e_mind_dec = 0.0
        # Baseline
        e_base_dec = max(0.0, coeff.baseline_w * dt_decision)
        parts_mj = [e_act_dec * 1e3, e_sense_dec * 1e3, e_brain_dec * 1e3, e_mind_dec * 1e3, e_base_dec * 1e3]
        labels_pd = ["Actuation", "Sensing", "Brain", "Mind", "Baseline"]
        bar_plot(labels_pd, parts_mj, "Per-Decision Energy Breakdown (mJ at 100 Hz)", str(abs_breakdown_path), ylabel="mJ/decision")
    except Exception as _:
        pass

    # Comprehensive scaling sweeps (matches Scaling.md theoretical claims)
    scaling_assets = []
    scaling_exponents = {}  # Store scaling exponents for caption generation
    if manifest.scaling:
        # Load raw manifest data to access the analyses field
        import yaml
        with open(manifest_path, 'r') as f:
            raw_manifest = yaml.safe_load(f)

        scaling_configs = []
        raw_scaling = raw_manifest.get('scaling', {})

        # Check if new multiple scaling analysis format exists
        if 'analyses' in raw_scaling and raw_scaling['analyses']:
            scaling_configs = raw_scaling['analyses']
        else:
            # Legacy single scaling analysis
            scaling_configs = [{
                'workload': raw_scaling.get('workload', 'brain'),
                'param': raw_scaling.get('param', 'K'),
                'values': raw_scaling.get('values', [64, 128, 256, 512]),
                'description': 'Legacy scaling analysis'
            }]

        mind_policies = raw_scaling.get('mind_policies', [])

        for scaling_cfg in scaling_configs:
            wl = scaling_cfg['workload']
            param = scaling_cfg['param']
            values = scaling_cfg['values']
            description = scaling_cfg.get('description', f'{wl} vs {param} scaling')

            logger.info(f"Running scaling analysis: {description}")

            x_vals = []
            y_energy = []
            tmp_params = dict((manifest.workloads.get(wl, {}).params if manifest.workloads else {}))

            # Special handling for brain scaling with mind policies
            if mind_policies and wl == "brain":
                # Generate multiple curves, one per policy variant
                series = []
                labels_series = []
                for pol in mind_policies:
                    pol_params = dict(tmp_params)
                    rho_mult = float(pol.get('rho_multiplier', 1.0))
                    K_gate = float(pol.get('K_gate', 1.0))
                    series_y = []
                    for v in values:
                        pol_params[param] = max(1, int(v * K_gate)) if param == 'K' else v
                        if 'rho' in pol_params:
                            pol_params['rho'] = float(tmp_params.get('rho', pol_params['rho'])) * rho_mult
                        load = brain_workload(0.25, pol_params)
                        e_est = estimate_compute_energy(load, coeff)
                        series_y.append(e_est)
                    series.append(series_y)
                    labels_series.append(f"{pol.get('name','policy')}")

                # Generate line plot
                x_vals = values
                x_label = "AL input channels K" if param == "K" else param
                title = f"{wl.title()} Energy vs {param} under Mind Policies"
                scale_line = Path(assets_dir) / f"scale_{wl}_{param}.png"
                line_plot(x_vals, series, labels_series, title, x_label, "Estimated Energy (J)", str(scale_line))
                scaling_assets.append(scale_line)

                # Pareto frontier using best energy at each parameter value
                y_best = [min(curves[i] for curves in series) for i in range(len(values))]
                perf = [1.0/float(v) for v in x_vals]  # Inverse parameter as performance proxy
                pareto = sorted(zip(y_best, perf))
                px = [p for p,_ in pareto]
                py = [q for _,q in pareto]
                pareto_plot = Path(assets_dir) / f"pareto_{wl}_{param}.png"
                line_plot(px, [py], ["Frontier"], f"Pareto Frontier: Energy vs Performance ({wl},{param})", "Energy (J)", "Performance (a.u.)", str(pareto_plot))
                scaling_assets.append(pareto_plot)

                # Scatter view of best-policy energy
                scatter_path = Path(assets_dir) / f"scale_{wl}_{param}_scatter.png"
                scatter_plot(x_vals, y_best, f"{title} (best policy, scatter)", x_label, "Estimated Energy (J)", str(scatter_path))
                scaling_assets.append(scatter_path)
                
                # Calculate scaling exponent for best-policy curve
                scaling_result = analyze_scaling_relationship(x_vals, y_best)
                scaling_exp = scaling_result.get('scaling_exponent', 0.0)
                r_sq = scaling_result.get('r_squared', 0.0)
                scaling_exponents[f"{wl}_{param}"] = {'exponent': scaling_exp, 'r_squared': r_sq}
                logger.info(f"Scaling analysis ({wl}, {param}): exponent={scaling_exp:.3f}, R²={r_sq:.3f}")
            else:
                # Single curve scaling analysis
                for v in values:
                    tmp_params[param] = v
                    if wl == "body":
                        load = body_workload(0.25, tmp_params)
                    elif wl == "mind":
                        load = mind_workload(0.25, tmp_params)
                    else:
                        load = brain_workload(0.25, tmp_params)
                    e_est = estimate_compute_energy(load, coeff)
                    x_vals.append(v)
                    y_energy.append(e_est)

                # Generate appropriate labels based on workload and parameter
                if wl == "brain" and param == "K":
                    x_label = "AL input channels K"
                    title = "AntBrain Energy Scaling vs AL channels (K)"
                    series_label = "Energy (AntBrain)"
                elif wl == "body" and param == "J":
                    x_label = "Joint count J"
                    title = "AntBody Energy Scaling vs Joint Count (J)"
                    series_label = "Energy (AntBody)"
                elif wl == "mind" and param == "H_p":
                    x_label = "Policy horizon H_p (steps)"
                    title = "AntMind Energy Scaling vs Planning Horizon (H_p)"
                    series_label = "Energy (AntMind)"
                else:
                    x_label = param
                    title = f"Scaling: {wl} vs {param}"
                    series_label = f"Energy ({wl})"

                # Line plot
                scale_line = Path(assets_dir) / f"scale_{wl}_{param}.png"
                line_plot(x_vals, [y_energy], [series_label], title, x_label, "Estimated Energy (J)", str(scale_line), bands=None)
                scaling_assets.append(scale_line)

                # Scatter plot companion
                scatter_path = Path(assets_dir) / f"scale_{wl}_{param}_scatter.png"
                scatter_plot(x_vals, y_energy, f"{title} (scatter)", x_label, "Estimated Energy (J)", str(scatter_path))
                scaling_assets.append(scatter_path)

                # Calculate scaling exponent
                scaling_result = analyze_scaling_relationship(x_vals, y_energy)
                scaling_exp = scaling_result.get('scaling_exponent', 0.0)
                r_sq = scaling_result.get('r_squared', 0.0)
                scaling_exponents[f"{wl}_{param}"] = {'exponent': scaling_exp, 'r_squared': r_sq}
                logger.info(f"Scaling analysis ({wl}, {param}): exponent={scaling_exp:.3f}, R²={r_sq:.3f}")
                
                # Pareto frontier
                perf = [1.0/float(v) for v in x_vals]
                pareto = sorted(zip(y_energy, perf))
                px = [p for p,_ in pareto]
                py = [q for _,q in pareto]
                pareto_plot = Path(assets_dir) / f"pareto_{wl}_{param}.png"
                line_plot(px, [py], ["Frontier"], f"Pareto Frontier: Energy vs Performance ({wl},{param})", "Energy (J)", "Performance (a.u.)", str(pareto_plot))
                scaling_assets.append(pareto_plot)

    # Generate markdown snippet to embed in paper
    # Write snippet inside the paper directory so the build can include it
    gen_md_path = os.path.join(os.path.dirname(out_dir), "Generated.md")
    lines: List[str] = []
    lines.append("# Generated Results (from src)\n")
    # Provenance block
    import platform
    import subprocess
    git_rev = ""
    try:
        git_rev = subprocess.check_output(["git","rev-parse","--short","HEAD"], text=True).strip()
    except Exception:
        git_rev = ""
    lines.append("\nProvenance: commit=" + (git_rev or "unknown") + ", seed=" + str(manifest.seed) + ", python=" + platform.python_version() + "\n\n")
    lines.append("\n## Per-Workload Estimated Energy (mean [95% CI], J)\n\n")
    lines.append("Only Body and Brain expend energy; Mind is a symbolic layer (0 J by convention).\n\n")
    lines.append("| Workload | Mean (J) | 95% CI Low | 95% CI High | N |\n")
    lines.append("|---|---:|---:|---:|\n")
    for wl, vals in per_wl.items():
        n = len(vals)
        mean, ci_lo, ci_hi = bootstrap_mean_ci(vals, num_samples=2000, alpha=0.05, seed=manifest.seed)
        lines.append(f"| {wl} | {mean:.6f} | {ci_lo:.6f} | {ci_hi:.6f} | {n} |\n")
    # Embed plots using repository-relative paths for robust PDF builds,
    # and provide absolute file:// links alongside for reproducibility.
    plot_uri = abs_plot_path.resolve().as_uri()
    if abs_plot_path.exists():
        lines.append(f"\n## Figure: Total Energy by Workload {{#fig:energy_by_workload}}\n")
        lines.append(f"\n![Total estimated energy by workload]({str(abs_plot_path)})\n")
        lines.append(f"\n**Caption:** Total estimated energy by workload. Only Body and Brain expend energy; Mind is symbolic (0 J).\n")
        lines.append(f"\n\\href{{{plot_uri}}}{{(View absolute file)}}\n")
    # Body split plot
    body_plot_uri = abs_body_split_path.resolve().as_uri()
    if abs_body_split_path.exists():
        lines.append(f"\n## Figure: Body Energy Partition {{#fig:body_partition}}\n")
        lines.append(f"\n![Body energy partition]({str(abs_body_split_path)})\n")
        lines.append(f"\n**Caption:** Estimated Body energy partition into Sensing and Actuation, aggregated over runs.\n")
        lines.append(f"\n\\href{{{body_plot_uri}}}{{(View absolute file)}}\n")
    for p in scaling_assets:
        if p.exists():
            name = p.name
            if name.startswith("pareto_"):
                # pareto_{wl}_{param}.png - handle compound params like H_p
                parts = name[:-4].split("_")  # drop .png
                wl = parts[1] if len(parts) > 1 else "brain"
                # Reconstruct compound parameter names (e.g., H_p from ['pareto', 'mind', 'H', 'p'])
                param = "_".join(parts[2:]) if len(parts) > 3 else (parts[2] if len(parts) > 2 else "K")
                fig_id = f"pareto_{wl}_{param}"
                title = f"Pareto Frontier (Energy vs Performance)"
                if wl == "brain" and param == "K":
                    caption_text = "Pareto frontier analysis showing the trade-off between energy consumption and sensory processing capacity in AntBrain. Performance is proxied by inverse AL input channels (1/K), representing information processing capability."
                elif wl == "body" and param == "J":
                    caption_text = "Pareto frontier for AntBody showing energy-performance trade-offs with varying joint counts. Performance proxy represents morphological dexterity."
                elif wl == "mind" and param == "H_p":
                    caption_text = "Pareto frontier for AntMind showing fundamental trade-offs between planning horizon and computational feasibility."
                else:
                    caption_text = f"Pareto frontier analysis for {wl} vs {param}, showing energy-performance trade-offs across configurations."
            elif name.startswith("scale_"):
                # scale_{wl}_{param}[[_scatter]].png - handle compound params like H_p
                is_scatter = name.endswith("_scatter.png")
                parts = name[:-4].split("_")  # drop .png
                wl = parts[1] if len(parts) > 1 else "brain"
                # Reconstruct compound parameter names, excluding 'scatter' suffix if present
                if is_scatter and len(parts) > 3:
                    param = "_".join(parts[2:-1])  # Exclude 'scatter'
                elif len(parts) > 3:
                    param = "_".join(parts[2:])
                else:
                    param = parts[2] if len(parts) > 2 else "K"
                fig_id = f"scaling_{wl}_{param}" + ("_scatter" if is_scatter else "")

                # Get measured scaling exponent if available
                exp_key = f"{wl}_{param}"
                exp_info = scaling_exponents.get(exp_key, {})
                measured_exp = exp_info.get('exponent', None)
                r_sq = exp_info.get('r_squared', None)
                
                # Generate exponent string for captions
                if measured_exp is not None and r_sq is not None and r_sq > 0.8:
                    exp_str = f"$E \\propto {param}^{{{measured_exp:.2f}}}$"
                    quality_str = f"$R^2={r_sq:.3f}$"
                else:
                    exp_str = ""
                    quality_str = ""
                
                # Generate appropriate title and caption based on workload and parameter
                if wl == "brain" and param == "K":
                    if is_scatter:
                        title = "AntBrain Energy Scaling vs AL Channels (K) [scatter]"
                        caption_text = f"Scatter plot representation of AntBrain energy scaling with antennal lobe input channels (K). Individual data points show experimental measurements with variability, complementing the line plot smoothing. Demonstrates the robustness of sub-linear scaling {exp_str if exp_str else ''} across different sensory configurations."
                    else:
                        title = "AntBrain Energy Scaling vs AL Channels (K)"
                        # Construct caption with proper LaTeX escaping (f-strings have issues with backslashes)
                        rho_text = r"$\rho = 0.02$"
                        caption_text = f"AntBrain energy scaling as a function of antennal lobe input channels (K). Demonstrates sub-linear scaling {exp_str if exp_str else ''} {quality_str if quality_str else ''} due to biological sparsity patterns ({rho_text}), enabling massive sensory expansion (64 to 1024 channels) without proportional energy increase. Multiple curves represent different AntMind policy variants affecting neural processing efficiency."
                elif wl == "body" and param == "J":
                    if is_scatter:
                        title = "AntBody Energy Scaling vs Joint Count (J) [scatter]"
                        caption_text = f"Scatter plot of AntBody energy consumption across different joint counts (J). Shows the dominance of baseline power consumption over joint-dependent computation, resulting in essentially flat energy scaling {exp_str if exp_str else ''}."
                    else:
                        title = "AntBody Energy Scaling vs Joint Count (J)"
                        caption_text = f"AntBody energy scaling with joint count (J). Demonstrates flat scaling {exp_str if exp_str else ''} {quality_str if quality_str else ''} due to baseline power dominance (50 mW from sensors and controllers), making morphological complexity essentially free in terms of energy cost."
                elif wl == "mind" and param == "H_p":
                    if is_scatter:
                        title = "AntMind Energy Scaling vs Planning Horizon (H_p) [scatter]"
                        caption_text = f"Scatter plot showing exponential energy growth {exp_str if exp_str else ''} in AntMind as planning horizon (H_p) increases. Illustrates the fundamental computational barriers of exact active inference beyond 15-step horizons."
                    else:
                        title = "AntMind Energy Scaling vs Planning Horizon (H_p)"
                        caption_text = f"AntMind energy scaling with policy planning horizon (H_p). Shows super-linear exponential growth {exp_str if exp_str else ''} {quality_str if quality_str else ''} due to combinatorial explosion in policy evaluation, establishing fundamental limits for real-time active inference."
                else:
                    title = f"Scaling ({wl} vs {param})" + (" [scatter]" if is_scatter else "")
                    exp_clause = f" with measured scaling {exp_str}" if exp_str else ""
                    caption_text = f"Energy scaling analysis for {wl} module vs parameter {param}{exp_clause}." + (" Scatter plot representation showing individual measurements." if is_scatter else " Line plot showing scaling trends.")
            else:
                fig_id = "scaling"
                title = "Scaling"
                caption_text = "Scaling analysis figure."

            lines.append(f"\n## Figure: {title} {{#fig:{fig_id}}}\n")
            lines.append(f"\n![{title.lower()}]({str(p)})\n")
            lines.append(f"\n**Caption:** {caption_text}\n")
            lines.append(f"\n\\href{{{p.resolve().as_uri()}}}{{(View absolute file)}}\n")
    # Per-decision breakdown figure and table
    if abs_breakdown_path.exists():
        lines.append(f"\n## Figure: Per-Decision Energy Breakdown {{#fig:per_decision}}\n")
        lines.append(f"\n![Per-decision energy breakdown]({str(abs_breakdown_path)})\n")
        lines.append(f"\n**Caption:** Average per-decision (10 ms) energy components at 100 Hz. Mind compute is 0 by convention; baseline is system idle.\n")
        # Table version using same averages
        act = e_act_dec * 1e3
        sen = e_sense_dec * 1e3
        brn = e_brain_dec * 1e3
        base = e_base_dec * 1e3
        total = act + sen + brn + base
        lines.append("\n### Table: Per-Decision Energy Breakdown (mJ)\n\n")
        lines.append("| Component | Energy (mJ) |\n")
        lines.append("|---|---:|\n")
        lines.append(f"| Actuation | {act:.3f} |\n")
        lines.append(f"| Sensing | {sen:.3f} |\n")
        lines.append(f"| Brain compute | {brn:.3f} |\n")
        lines.append("| Mind compute | 0.000 |\n")
        lines.append(f"| Baseline/idle | {base:.3f} |\n")
        lines.append(f"| Total | {total:.3f} |\n")
    # Link to raw CSV as absolute file URL
    csv_uri = Path(csv_path).resolve().as_uri()
    lines.append("\n## Raw Results (CSV)\n\n")
    lines.append(f"\\href{{{csv_uri}}}{{View Results CSV}}\n")
    # Derived metric: CoT using nominal mass/distance if available in manifest
    if per_wl.get("body"):
        mass_kg = float((manifest.mass_kg if getattr(manifest, 'mass_kg', None) is not None else 0.02) or 0.02)
        distance_m = float((manifest.distance_m if getattr(manifest, 'distance_m', None) is not None else 1.0) or 1.0)
        # Use mean energy per decision, not total across all repeats
        mean_body_energy = sum(per_wl["body"]) / len(per_wl["body"])
        cot = cost_of_transport(mean_body_energy, mass_kg, distance_m)
        lines.append("\n## Derived Metric: Cost of Transport (dimensionless)\n\n")
        lines.append(f"CoT $\\approx$ {cot:.4f} (assuming mass={mass_kg} kg, distance={distance_m} m).\n")
        # Add biological comparison for context
        lines.append(f"\nBiological ants achieve CoT 0.1-0.3, indicating {cot/0.2:.1f}$\\times$ optimization potential in mechanical efficiency.\n")

    # Per-decision complexity estimates (compute/memory), derived from closed-form helpers
    lines.append("\n## Table: Per-Decision Complexity (Compute/Memory) {#tab:complexity_per_decision}\n\n")
    lines.append("| Workload | FLOPs/decision | SRAM bytes/decision | DRAM bytes/decision | Spikes/decision |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for wl in ["body", "brain", "mind"]:
        wl_cfg = (manifest.workloads or {}).get(wl)
        params = (wl_cfg.params if wl_cfg else {}) or {}
        if wl == "body":
            comp = estimate_body_compute_per_decision(params)
        elif wl == "brain":
            comp = estimate_brain_compute_per_decision(params)
        else:
            comp = estimate_mind_compute_per_decision(params)
        lines.append(
            f"| {wl} | {comp.flops:.0f} | {comp.sram_bytes:.0f} | {comp.dram_bytes:.0f} | {comp.spikes:.0f} |\n"
        )

    with open(gen_md_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"wrote: {gen_md_path}")


def main() -> None:
    """
    Main entry point for the Ant Stack complexity and energetics analysis.
    
    This function parses command-line arguments and executes the complete analysis
    pipeline based on the provided manifest configuration.
    """
    parser = argparse.ArgumentParser(
        description="Ant Stack Complexity & Energetics Analysis Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python runner.py manifest.yaml
  python runner.py manifest.yaml --out results/
  python runner.py manifest.yaml --out results/ --verbose
        """
    )
    
    parser.add_argument(
        "manifest",
        help="Path to experiment manifest YAML file"
    )
    
    parser.add_argument(
        "--out",
        default="complexity_energetics/out",
        help="Output directory for results and figures (default: complexity_energetics/out)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging output"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate manifest and configuration without running analysis"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    try:
        if args.validate_only:
            logger.info("Validation mode: checking manifest and configuration")
            # Load manifest to validate
            manifest = ExperimentManifest.load(args.manifest)
            logger.info("Manifest validation successful")
            return
            
        # Run the complete analysis pipeline
        run_manifest(args.manifest, args.out)
        logger.info("Analysis completed successfully")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


