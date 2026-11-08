"""Manifest-driven analytical compute accounting with explicit observation boundaries.

``--out`` is the artifact root: CSV, Markdown, provenance and assets all live
inside it. A nonempty root is immutable; run-all supplies fresh staging roots.
Output ownership must remain trusted and exclusive throughout execution.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from antstack_core.analysis import (
    EnergyCoefficients, ExperimentManifest, NullPowerMeter, NvmlPowerMeter,
    RaplPowerMeter, analyze_scaling_relationship, body_workload,
    body_workload_closed_form, bootstrap_mean_ci, brain_workload,
    brain_workload_closed_form, estimate_compute_energy, estimate_body_energy_mech, measure_energy,
    mind_workload, mind_workload_closed_form,
)
from antstack_core.figures import bar_plot, interval_plot, line_plot, scatter_plot
from antstack_core.publishing.provenance import (
    atomic_json, build_run_provenance, capture_source_state, ordinary_path,
    publication_path, preserve_tree_modes, write_provenance,
)

logger = logging.getLogger(__name__)
WORKLOADS = {"body": body_workload, "brain": brain_workload, "mind": mind_workload}
CLOSED_FORM_WORKLOADS = {"body": body_workload_closed_form,
                         "brain": brain_workload_closed_form, "mind": mind_workload_closed_form}


class RunManifestResult:
    """Current-invocation output inventory (never inferred from directory contents)."""
    def __init__(self, *, manifest_path, output_dir, csv_path, summary_path,
                 generated_markdown_path, provenance_path, output_paths, rows, scaling_exponents):
        self.manifest_path = manifest_path
        self.output_dir = output_dir
        self.csv_path = csv_path
        self.summary_path = summary_path
        self.generated_markdown_path = generated_markdown_path
        self.provenance_path = provenance_path
        self.output_paths = output_paths
        self.rows = rows
        self.scaling_exponents = scaling_exponents


def markdown_image_path(path: Path, base: Path | None = None) -> str:
    """Portable link relative to the containing generated Markdown document."""
    return Path(os.path.relpath(path, base or path.parent.parent)).as_posix()


def _workload_config(manifest, workload):
    return (manifest.workloads or {}).get(workload)


def _workload_params(manifest, workload):
    cfg = _workload_config(manifest, workload)
    return dict((cfg.params if cfg else {}) or {})


def _load_for_manifest_mode(manifest, workload, duration_s, params):
    cfg = _workload_config(manifest, workload)
    if cfg is None:
        raise ValueError(f"Scaling requests unconfigured workload: {workload}")
    functions = CLOSED_FORM_WORKLOADS if cfg.mode == "closed_form" else WORKLOADS
    return functions[workload](duration_s, params)


def _finite_tree(value, name="manifest"):
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, f"{name}.{key}")
    elif isinstance(value, list):
        for child in value:
            _finite_tree(child, name)
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
        raise ValueError(f"{name} must contain finite numbers")


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def load_manifest_snapshot(path: str | Path):
    """Read once, validate raw semantics, and retain all raw scaling extensions."""
    source = ordinary_path(path)
    data = source.read_bytes()
    raw = yaml.safe_load(data)
    if not isinstance(raw, dict):
        raise ValueError("Manifest must be a mapping")
    _finite_tree(raw)
    for section in ("coefficients", "meter", "statistics", "scaling"):
        if section in raw and raw[section] is not None and not isinstance(raw[section], dict):
            raise ValueError(f"{section} must be a mapping")
    _integer(raw.get("seed", 0), "seed", 0)
    workloads = raw.get("workloads")
    if not isinstance(workloads, dict) or not workloads:
        raise ValueError("At least one workload is required")
    for name, cfg in workloads.items():
        if name not in WORKLOADS or not isinstance(cfg, dict):
            raise ValueError(f"Unsupported workload: {name}")
        if cfg.get("mode", "loop") not in {"loop", "closed_form"}:
            raise ValueError(f"Unsupported workload mode: {cfg.get('mode')}")
        _integer(cfg.get("repeats", 1), f"{name}.repeats")
        duration = cfg.get("duration_s", 1.0)
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
            raise ValueError(f"{name}.duration_s must be positive and finite")
        params = cfg.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"{name}.params must be a mapping")
        for key in ("hierarchical",):
            if key in params and type(params[key]) is not bool:
                raise ValueError(f"{name}.{key} must be a boolean")
        if "contact_solver" in params and params["contact_solver"] not in {"pgs", "si", "ldlt"}:
            raise ValueError("Unsupported contact_solver")
        if "connectivity_pattern" in params and params["connectivity_pattern"] not in {"random", "structured"}:
            raise ValueError("Unsupported connectivity_pattern")
    manifest = ExperimentManifest.from_dict(raw)
    meter = manifest.meter.meter_type
    # YAML's unquoted null is the native offline configuration spelling.
    meter = "null" if meter is None else meter
    if meter not in {"null", "rapl", "nvml"}:
        raise ValueError(f"Unsupported meter_type: {meter}")
    notes = raw.get("provenance")
    if notes is not None:
        if not isinstance(notes, dict):
            raise ValueError("provenance must be a metadata mapping")
        if {"include_git", "include_versions"} & set(notes):
            raise ValueError("provenance is metadata, not privacy controls; source/version identities are mandatory")
    if meter != "null" and any(cfg.get("mode", "loop") == "closed_form" for cfg in workloads.values()):
        raise ValueError("closed_form is analytical; hardware measurement requires loop workloads")
    valid, errors = manifest.validate()
    if not valid:
        raise ValueError("; ".join(errors))
    coeff = EnergyCoefficients(**asdict(manifest.coefficients))
    raw_coefficients = raw.get("coefficients") or {}
    if not isinstance(raw_coefficients, dict) or set(raw_coefficients) - set(asdict(manifest.coefficients)):
        raise ValueError("Use canonical coefficient names: " + ", ".join(asdict(manifest.coefficients)))
    if any(type(value) not in (int, float) for value in raw_coefficients.values()):
        raise ValueError("Energy coefficients must be finite numeric values, not booleans or strings")
    for name, cfg in manifest.workloads.items():
        # Domain validation only: never execute timed loop or hardware in validation mode.
        CLOSED_FORM_WORKLOADS[name](cfg.duration_s, dict(cfg.params or {}))
    settings = raw.get("statistics") or raw.get("scaling") or {}
    confidence = settings.get("confidence_level", 0.95)
    samples = settings.get("bootstrap_samples", 1000)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 < confidence < 1:
        raise ValueError("confidence_level must lie strictly between zero and one")
    _integer(samples, "bootstrap_samples")
    scaling_curves(manifest, raw, coeff, validate_only=True)
    return data, raw, manifest, coeff, float(confidence), samples, meter


def scaling_curves(manifest, raw, coeff, *, validate_only=False):
    scaling = raw.get("scaling") or {}
    analyses = scaling.get("analyses")
    if analyses is None:
        analyses = [scaling] if "values" in scaling else []
    if not isinstance(analyses, list):
        raise ValueError("scaling.analyses must be a list")
    curves = []
    identities = set()
    for analysis in analyses:
        workload, parameter = analysis.get("workload", "brain"), analysis.get("param", "K")
        if workload not in manifest.workloads or not isinstance(parameter, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", parameter):
            raise ValueError("Scaling needs a configured workload and safe parameter name")
        supported = {"body": {"J", "C", "S", "hz"},
                     "brain": {"K", "N_KC", "rho", "H", "hz"},
                     "mind": {"B", "H_p", "state_dim", "action_dim", "precision_steps", "hz"}}
        if parameter not in supported[workload]:
            raise ValueError(f"Unsupported scaling parameter for {workload}: {parameter}")
        identity = (workload, parameter)
        if identity in identities:
            raise ValueError(f"Duplicate scaling identity: {identity}")
        identities.add(identity)
        values = analysis.get("values")
        if not isinstance(values, list) or len(values) < 2 or any(type(v) not in (int, float) or v <= 0 for v in values) or len(set(values)) != len(values):
            raise ValueError("Scaling values must contain at least two distinct positive finite numbers")
        policies = scaling.get("mind_policies") if workload == "brain" else None
        policies = policies or [{"name": "configured model"}]
        for policy in policies:
            params = _workload_params(manifest, workload)
            rho_multiplier, gate = policy.get("rho_multiplier", 1), policy.get("K_gate", 1)
            if type(rho_multiplier) not in (int, float) or rho_multiplier <= 0 or type(gate) not in (int, float) or gate <= 0:
                raise ValueError("Policy multipliers must be positive finite numbers")
            energies = []
            for value in values:
                changed = dict(params)
                changed[parameter] = max(1, int(value * gate)) if parameter == "K" else value
                if "rho" in changed:
                    changed["rho"] *= rho_multiplier
                load = CLOSED_FORM_WORKLOADS[workload](0.25, changed) if validate_only else _load_for_manifest_mode(manifest, workload, 0.25, changed)
                energies.append(estimate_compute_energy(load, coeff))
            curves.append({"name": str(policy.get("name", "policy")), "workload": workload,
                           "parameter": parameter, "x": values, "y": energies, "duration_s": 0.25})
    return curves


def run_manifest(manifest_path: str, out_dir: str, *, source_state=None, statistics_seed=None) -> RunManifestResult:
    data, raw, manifest, coeff, confidence, samples, meter_type = load_manifest_snapshot(manifest_path)
    interval_seed = manifest.seed if statistics_seed is None else statistics_seed
    _integer(interval_seed, "statistics_seed", 0)
    source = ordinary_path(manifest_path)
    root = publication_path(out_dir, tree=True)
    if root == source or root in source.parents:
        raise ValueError("Output must not contain the input manifest")
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f"CE artifact root must be new or empty: {root}")
    state = dict(source_state) if source_state is not None else capture_source_state(Path.cwd())
    meter_cfg = manifest.meter
    meter = (RaplPowerMeter(meter_cfg.energy_path) if meter_type == "rapl" else
             NvmlPowerMeter(meter_cfg.device_index) if meter_type == "nvml" else NullPowerMeter())
    rows = []
    for workload, cfg in manifest.workloads.items():
        for repeat in range(cfg.repeats):
            with measure_energy(meter) as observation:
                start = time.monotonic()
                load = _load_for_manifest_mode(manifest, workload, cfg.duration_s, dict(cfg.params or {}))
                duration = cfg.duration_s if cfg.mode == "closed_form" else time.monotonic() - start
            compute = estimate_compute_energy(load, coeff)
            idle = coeff.baseline_w * duration
            measured = meter_type != "null" and observation.get("measured", False)
            if meter_type != "null" and not measured:
                raise RuntimeError("Requested hardware observation is unavailable")
            rows.append({"workload": workload, "repeat": repeat, "duration_s": duration,
                         "flops": load.flops, "sram_bytes": load.sram_bytes,
                         "dram_bytes": load.dram_bytes, "spikes": load.spikes,
                         "compute_energy_j": compute, "isolated_idle_energy_j": idle,
                         "energy_est_j": compute + idle,
                         "record_kind": "analytical" if cfg.mode == "closed_form" else "synthetic_workload",
                         "measured_energy_j": observation["energy_joules"] if measured else None,
                         "measurement_status": "measured" if measured else "not_measured"})
    expected = {(name, repeat) for name, cfg in manifest.workloads.items() for repeat in range(cfg.repeats)}
    if {(row["workload"], row["repeat"]) for row in rows} != expected or len(rows) != len(expected):
        raise RuntimeError("Incomplete workload/repeat coverage")
    curves = scaling_curves(manifest, raw, coeff)
    _finite_tree(rows, "computed workload records")
    _finite_tree(curves, "computed scaling curves")
    root.mkdir(parents=True, exist_ok=True)
    assets = root / "assets"
    assets.mkdir()
    snapshot = root / "manifest.input.yaml"
    snapshot.write_bytes(data)
    csv_path = root / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    outputs = [snapshot, csv_path]
    by_workload = {name: [row["energy_est_j"] for row in rows if row["workload"] == name]
                   for name in manifest.workloads}
    intervals = {name: bootstrap_mean_ci(values, num_samples=samples, alpha=1-confidence, seed=interval_seed)
                 for name, values in by_workload.items()}
    energy_plot = assets / "energy.png"
    labels = list(intervals)
    interval_plot([name.title() for name in labels], [intervals[name][0] for name in labels],
                  "Module-isolated energy: computed means and supplied intervals", str(energy_plot),
                  xlabel="J / workload window",
                  confidence_intervals=[(intervals[name][1], intervals[name][2]) for name in labels])
    outputs.append(energy_plot)
    percent = confidence * 100
    lines = ["# Analytical Complexity and Energy Results\n\n", "These are analytical/synthetic workloads, not independent biological or hardware observations.\n\n",
             "## Per-Workload Estimated Energy\n\n",
             "Each row includes that module's compute and isolated idle over its workload duration. These isolated rows must not be summed as a system total; locomotion is excluded.\n\n",
             f"| Workload | Mean (J/window) | {percent:g}% CI Low | {percent:g}% CI High | N |\n",
             "|---|---:|---:|---:|---:|\n"]
    for name, (mean, low, high) in intervals.items():
        lines.append(f"| {name} | {mean:.8g} | {low:.8g} | {high:.8g} | {len(by_workload[name])} |\n")
    lines.extend(["\n## Figure: Module-Isolated Energy {#fig:energy_by_workload}\n\n",
                  f"![Mean module-isolated compute plus idle, with {percent:g}% bootstrap intervals for the mean. Repeating deterministic values does not establish empirical precision; positive wide-range values use a labelled logarithmic axis.](assets/energy.png)\n\n"])
    scaling_exponents = {}
    grouped = {}
    for curve in curves:
        grouped.setdefault((curve["workload"], curve["parameter"]), []).append(curve)
    for (workload, parameter), group in grouped.items():
        identity = f"{workload}_{parameter}"
        plot = assets / f"scale_{identity}.png"
        scatter = assets / f"scale_{identity}_scatter.png"
        proxy = assets / f"inverse_parameter_{identity}.png"
        x = group[0]["x"]
        series = [curve["y"] for curve in group]
        parameter_label = {"J": "Joint count J", "K": "Input-channel count K", "H_p": "Decision horizon H_p (steps)"}.get(parameter, parameter)
        line_plot(x, series, [curve["name"] for curve in group], f"{workload.title()} compute vs {parameter_label}",
                  parameter_label, "Compute energy (J / 0.25 s)", str(plot))
        best = [min(values) for values in zip(*series)]
        scatter_plot(x, best, "Lowest modeled compute among configured parameterizations", parameter_label,
                     "Compute energy (J / 0.25 s)", str(scatter))
        line_plot(best, [[1/value for value in x]], ["Illustrative proxy"], "Inverse parameter: an illustration, not a Pareto frontier",
                  "Compute energy (J / 0.25 s)", f"Inverse {parameter_label} (proxy only)", str(proxy))
        relation = analyze_scaling_relationship(x, best)
        scaling_exponents[identity] = {"exponent": relation.get("scaling_exponent"),
                                       "r_squared": relation.get("r_squared"), "valid": relation.get("valid", False)}
        for image, suffix in ((plot, ""), (scatter, "_scatter"), (proxy, "_proxy")):
            outputs.append(image)
            figure_id = f"inverse_parameter_{identity}" if suffix == "_proxy" else f"scaling_{identity}{suffix}"
            view = "inverse-parameter illustration" if suffix == "_proxy" else "lowest configured compute points" if suffix else "configured compute curves"
            caption = ("Calculated values over a 0.25 s compute-only window; lines connect model outputs, not empirical replicates. "
                       + ("The inverse-parameter curve is a proxy, not measured performance or a nondominated frontier." if suffix == "_proxy" else "No fixed biological scaling law is inferred."))
            lines.extend([f"\n## Figure: {workload.title()} {view} {{#fig:{figure_id}}}\n\n",
                          f"![{caption}](assets/{image.name})\n\n"])
    # Same selected workload model and same common interval, including nonzero Mind compute.
    frequencies = [float(_workload_params(manifest, name).get("hz", 10 if name == "mind" else 100)) for name in manifest.workloads]
    dt = 1.0 / max(frequencies)
    window = f"{dt*1000:g} ms"
    parts, complexities = {}, {}
    for name in manifest.workloads:
        load = _load_for_manifest_mode(manifest, name, dt, _workload_params(manifest, name))
        complexities[name] = asdict(load)
        parts[name] = estimate_compute_energy(load, coeff)
    if "body" in manifest.workloads:
        body_params = _workload_params(manifest, "body")
        mechanical = estimate_body_energy_mech(dt, body_params, coeff)
        sensing = float(body_params.get("sensor_w_per_channel", coeff.body_sensor_w_per_channel)) * int(body_params.get("S", 256)) * dt
        parts["Body actuation"] = mechanical - sensing
        parts["Body sensing"] = sensing
        body_plot = assets / "body_split.png"
        bar_plot(["Sensing", "Actuation"], [sensing*1000, (mechanical-sensing)*1000],
                 f"Modeled body components ({window})", str(body_plot), ylabel=f"mJ / {window}")
        outputs.append(body_plot)
        lines.extend(["\n## Figure: Body Energy Partition {#fig:body_partition}\n\n",
                      f"![Separate sensing and actuation assumptions integrated over the common {window} interval, not the compute-only workload row.](assets/body_split.png)\n\n"])
    parts["Shared idle"] = coeff.baseline_w * dt
    decision_plot = assets / "per_decision_breakdown.png"
    component_labels = [f"{name.title()} compute" if name in WORKLOADS else name for name in parts]
    interval_plot(component_labels, [value*1000 for value in parts.values()],
                  f"Modeled system energy components: common {window} window", str(decision_plot),
                  xlabel=f"mJ / {window} window")
    outputs.append(decision_plot)
    lines.extend(["\n## Figure: Common-Window Compute Accounting {#fig:per_decision}\n\n",
                  f"![Selected workload counters over a common {window} window, inverse maximum configured update rate, with shared idle counted once. Integer update flooring can omit slower modules; this is not their time-averaged load or necessarily one decision each. Sensing and actuation are separate assumptions, not observed locomotion.](assets/per_decision_breakdown.png)\n\n",
                  f"| Component | Energy (mJ / {window}) |\n|---|---:|\n"])
    for name, energy in parts.items():
        lines.append(f"| {name} | {energy*1000:.8g} |\n")
    lines.append(f"| Modeled system total | {sum(parts.values())*1000:.8g} |\n")
    lines.extend(["\n## Table: Common-Window Complexity {#tab:complexity_per_decision}\n\n",
                  f"| Workload | FLOPs / {window} | SRAM bytes / {window} | DRAM bytes / {window} | Spikes / {window} |\n",
                  "|---|---:|---:|---:|---:|\n"])
    for name, load in complexities.items():
        lines.append(f"| {name} | {load['flops']:.8g} | {load['sram_bytes']:.8g} | {load['dram_bytes']:.8g} | {load['spikes']:.8g} |\n")
    lines.extend(["\n## Cost of Transport\n\nUnavailable: compute-window energy is not locomotion energy, and no matching modeled locomotion interval/distance is supplied by this producer.\n",
                  "\n## Raw Results\n\n[Results CSV](results.csv)\n"])
    generated = root / "Generated.md"
    generated.write_text("".join(lines), encoding="utf-8")
    outputs.append(generated)
    summary_path, provenance_path = root / "summary.json", root / "provenance.json"
    outputs.extend([summary_path, provenance_path])
    provenance = build_run_provenance(project="ant-stack-complexity-energetics", input_paths=[snapshot],
                                      output_paths=[str(path.relative_to(root)) for path in outputs],
                                      parameters={"seed": manifest.seed, "statistics_seed": interval_seed, "confidence_level": confidence,
                                                  "bootstrap_samples": samples, "meter_type": meter_type,
                                                  "input_sha256": hashlib.sha256(data).hexdigest(),
                                                  "output_path_base": "artifact_root", "complete": True,
                                                  "requested_repeats": {name: cfg.repeats for name, cfg in manifest.workloads.items()}},
                                      source_state=state)
    from dataclasses import replace
    from antstack_core.publishing.provenance import file_identity
    provenance = replace(provenance, input_paths=["manifest.input.yaml"],
                         input_identities={"manifest.input.yaml": file_identity(snapshot)})
    write_provenance(provenance_path, provenance)
    atomic_json(summary_path, {"schema_version": 2, "seed": manifest.seed, "statistics_seed": interval_seed, "coefficients": asdict(coeff),
                               "rows": rows, "complete": True, "confidence_level": confidence,
                               "bootstrap_samples": samples, "scaling_curves": curves,
                               "provenance": provenance.to_dict(), "output_paths": provenance.output_paths})
    return RunManifestResult(manifest_path=str(source), output_dir=str(root), csv_path=str(csv_path),
                             summary_path=str(summary_path), generated_markdown_path=str(generated),
                             provenance_path=str(provenance_path), output_paths=[str(path) for path in outputs],
                             rows=rows, scaling_exponents=scaling_exponents)



def project_result(result: RunManifestResult, paper_root: str | Path, *, publish: bool = True) -> dict[str, Any]:
    """Publish only current produced files, retaining the prior paper tree intact.

    Existing guides and unrelated assets are copied unchanged; previous generated
    destinations may be replaced because their original tree is retained in a
    sibling archive. Stable symlinks anywhere in the paper tree are rejected.
    The caller must hold exclusive trusted ownership, not rely on race isolation.
    """
    import shutil
    import uuid
    from antstack_core.publishing.provenance import file_identity

    if type(publish) is not bool:
        raise ValueError("publish must be a boolean")
    source_root = ordinary_path(result.output_dir, tree=True)
    destination = publication_path(paper_root, tree=True)
    if source_root == destination or source_root in destination.parents or destination in source_root.parents:
        raise ValueError("CE source and projection namespaces must be disjoint")
    produced = [ordinary_path(path) for path in result.output_paths]
    if len(set(produced)) != len(produced):
        raise ValueError("Duplicate producer output identity")
    required = {ordinary_path(path) for path in (result.csv_path, result.summary_path,
                result.generated_markdown_path, result.provenance_path)}
    if not required.issubset(set(produced)):
        raise ValueError("Projection requires every mandatory producer output")
    if json.loads(Path(result.summary_path).read_text()).get("complete") is not True:
        raise ValueError("Projection requires a completed CE result")
    input_source = ordinary_path(result.manifest_path)
    selected = {}
    for source in produced:
        relative = source.relative_to(source_root)
        file_identity(source)
        if source.name in {"README.md", "AGENTS.md"}:
            raise ValueError("Guides cannot be declared as generated CE artifacts")
        if relative == Path("Generated.md"):
            target = relative
        elif relative.parts[0] == "assets":
            target = relative
        else:
            target = Path("out") / relative
        if target in selected:
            raise ValueError(f"Duplicate projection target: {target}")
        selected[target] = source
        projected = destination / target
        if projected == input_source or projected in input_source.parents:
            raise ValueError("Projection output overlaps its source input")
    if Path("Generated.md") not in selected:
        raise ValueError("Producer did not supply generated Markdown")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name(f".{destination.name}.stage-{uuid.uuid4().hex}")
    archive = destination.with_name(f".{destination.name}.archive-{uuid.uuid4().hex}")
    if destination.exists():
        shutil.copytree(destination, stage)
    else:
        stage.mkdir()
    inventory_name = ".antstack-ce-projection.json"
    prior = destination / inventory_name
    if prior.is_file():
        old = json.loads(prior.read_text(encoding="utf-8"))
        for name in old.get("generated_paths", []):
            relative = Path(name)
            if (not relative.parts or relative.is_absolute() or ".." in relative.parts
                    or relative.parts[0] not in {"assets", "out", "Generated.md"}
                    or relative.name in {"README.md", "AGENTS.md"}):
                raise ValueError("Invalid previous projection ownership inventory")
            target = stage / relative
            if relative not in selected and target.is_file():
                target.unlink()
    identities = {}
    for relative, source in selected.items():
        target = ordinary_path(stage / relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_dir():
            raise ValueError(f"Projection file collides with directory: {relative}")
        if relative == Path("Generated.md"):
            # Assets already have document-relative links; only raw CSV moves to out/.
            target.write_text(source.read_text(encoding="utf-8").replace(
                "[Results CSV](results.csv)", "[Results CSV](out/results.csv)"), encoding="utf-8")
        else:
            shutil.copy2(source, target)
        identities[relative.as_posix()] = file_identity(target)
    atomic_json(stage / inventory_name, {"schema_version": 1, "canonical_source": str(source_root),
                                       "generated_paths": sorted(identities), "identities": identities})
    if not publish:
        return {"projection_root": str(destination), "staged_root": str(stage),
                "generated_paths": sorted(identities), "identities": identities}
    moved = False
    preserve_tree_modes(stage, destination)
    try:
        if destination.exists():
            os.replace(destination, archive)
            moved = True
        os.replace(stage, destination)
    except BaseException:
        if moved and not destination.exists():
            os.replace(archive, destination)
        raise
    return {"projection_root": str(destination), "archive": str(archive) if moved else None,
            "generated_paths": sorted(identities), "identities": identities}


def main():
    parser = argparse.ArgumentParser(description="Ant Stack Complexity & Energetics Analysis Runner")
    parser.add_argument("manifest")
    parser.add_argument("--out", help="Required for generation: fresh exclusive artifact root")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only and args.out is None:
        parser.error("--out is required for generation; choose a fresh exclusive artifact root")
    try:
        if args.validate_only:
            load_manifest_snapshot(args.manifest)
        else:
            run_manifest(args.manifest, args.out)
    except Exception as error:
        logger.error("Analysis failed: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
