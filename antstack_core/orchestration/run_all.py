"""Transactional artifact orchestration under trusted exclusive output ownership.

Runs are built in fresh siblings, validated from current bytes, then published by
rename. clean_run_dir permits replacement only of owned completed runs; the old
tree is retained in an archive. Stable symlinks are rejected, but these checks do
not claim isolation from hostile concurrent same-UID filesystem mutation.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import math
import os
import re
import shutil
import sys
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml
from antstack_core.analysis import analyze_scaling_relationship, bootstrap_mean_ci
from antstack_core.publishing.provenance import (
    DEFAULT_DEPENDENCIES, atomic_json, build_run_provenance, capture_source_state,
    collect_dependency_versions, file_identity, ordinary_path, publication_path,
    preserve_tree_modes, write_provenance,
)
from .complexity_energetics import load_manifest_snapshot, project_result, run_manifest, scaling_curves

TASK_NAMES = ("data", "statistics", "visualizations", "animations", "reports", "papers", "validation")
RUN_SUBDIRECTORIES = ("data/raw", "data/derived", "statistics", "visualizations/static",
                      "visualizations/animations", "reports", "papers", "logs", "provenance")
DEFAULT_REQUIRED_ARTIFACT_TYPES = ("csv", "json", "markdown", "image", "animation", "log", "provenance")


def _bool(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _formats(value):
    return (value,) if isinstance(value, str) else tuple(value)


def _resolve_repo_path(path):
    candidate = Path(path).expanduser()
    return ordinary_path(candidate if candidate.is_absolute() else Path.cwd() / candidate)


def _mapping(raw, key):
    value = raw.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_type: str
    path: str
    absolute_path: str
    producing_task: str
    source_config: str
    checksum: str
    bytes: int

    def to_dict(self):
        # v2 has exactly one portable address space; absolute_path is runtime-only.
        value = asdict(self)
        value.pop("absolute_path")
        value["content_identity"] = f"sha256:{self.checksum}"
        return value


@dataclass(frozen=True)
class TaskSelection:
    data: bool = True
    statistics: bool = True
    visualizations: bool = True
    animations: bool = True
    reports: bool = True
    papers: bool = True
    validation: bool = True

    @classmethod
    def from_mapping(cls, data=None):
        data = data or {}
        unknown = set(data) - set(TASK_NAMES)
        if unknown:
            raise ValueError(f"Unknown task name(s): {sorted(unknown)}")
        return cls(**{name: _bool(data.get(name, True), f"tasks.{name}") for name in TASK_NAMES})

    @classmethod
    def only(cls, names):
        selected = parse_task_names(names)
        return cls(**{name: name in selected for name in TASK_NAMES})

    def enabled(self):
        return tuple(name for name in TASK_NAMES if getattr(self, name))


@dataclass(frozen=True)
class RunSection:
    run_id: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    seed: int = 123
    output_root: str = "outputs"
    clean_run_dir: bool = False
    log_level: str = "INFO"

    @classmethod
    def from_mapping(cls, data=None):
        values = dict(data or {})
        if "run_id" not in values or values["run_id"] is None:
            values["run_id"] = cls().run_id
        return cls(**values)


@dataclass(frozen=True)
class ComplexityEnergeticsSection:
    manifest: str = "papers/complexity_energetics/manifest.example.yaml"
    paper_projection: bool = True
    paper_root: str = "papers/complexity_energetics"
    build_pdf: bool = False
    pdf_sources: tuple[str, ...] = ("2_complexity_energetics.pdf",)

    @classmethod
    def from_mapping(cls, data=None):
        values = dict(data or {})
        if "pdf_sources" in values:
            values["pdf_sources"] = _formats(values["pdf_sources"])
        return cls(**values)


@dataclass(frozen=True)
class VisualizationSection:
    output_formats: tuple[str, ...] = ("png",)
    dpi: int = 160
    figure_size: tuple[float, float] = (8.0, 5.0)
    statistical_annotations: bool = True
    animation_fps: int = 8
    animation_duration_s: float = 3.0
    animation_formats: tuple[str, ...] = ("gif", "html")

    @classmethod
    def from_mapping(cls, data=None):
        values = dict(data or {})
        for name in ("output_formats", "animation_formats"):
            if name in values:
                values[name] = _formats(values[name])
        if "figure_size" in values:
            values["figure_size"] = tuple(values["figure_size"])
        return cls(**values)


@dataclass(frozen=True)
class ValidationSection:
    required_artifact_types: tuple[str, ...] = DEFAULT_REQUIRED_ARTIFACT_TYPES
    fail_on_missing: bool = True
    fail_on_empty: bool = True

    @classmethod
    def from_mapping(cls, data=None):
        values = dict(data or {})
        if "required_artifact_types" in values:
            required = values["required_artifact_types"]
            values["required_artifact_types"] = tuple(required.split(",")) if isinstance(required, str) else tuple(required)
        return cls(**values)


@dataclass(frozen=True)
class RunAllConfig:
    config_path: Path
    run: RunSection = field(default_factory=RunSection)
    tasks: TaskSelection = field(default_factory=TaskSelection)
    complexity_energetics: ComplexityEnergeticsSection = field(default_factory=ComplexityEnergeticsSection)
    visualization: VisualizationSection = field(default_factory=VisualizationSection)
    validation: ValidationSection = field(default_factory=ValidationSection)
    source_bytes: bytes | None = field(default=None, repr=False)

    @classmethod
    def from_file(cls, path):
        path = _resolve_repo_path(path)
        source_bytes = path.read_bytes()
        raw = yaml.safe_load(source_bytes)
        if not isinstance(raw, Mapping):
            raise ValueError("Run-all config must be a YAML mapping")
        return cls(path, RunSection.from_mapping(_mapping(raw, "run")),
                   TaskSelection.from_mapping(_mapping(raw, "tasks")),
                   ComplexityEnergeticsSection.from_mapping(_mapping(raw, "complexity_energetics")),
                   VisualizationSection.from_mapping(_mapping(raw, "visualization")),
                   ValidationSection.from_mapping(_mapping(raw, "validation")), source_bytes).validate()

    def with_overrides(self, *, output_root=None, run_id=None, tasks=None, log_level=None):
        values = {key: value for key, value in {"output_root": output_root, "run_id": run_id, "log_level": log_level}.items() if value is not None}
        return replace(self, run=replace(self.run, **values),
                       tasks=self.tasks if tasks is None else TaskSelection.only(tasks)).validate()

    def validate(self):
        if not isinstance(self.run.run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", self.run.run_id):
            raise ValueError("run.run_id must be a safe dot-free single path segment")
        _integer(self.run.seed, "run.seed", 0)
        for name, value in asdict(self.tasks).items():
            _bool(value, f"tasks.{name}")
        for name, value in (("clean_run_dir", self.run.clean_run_dir),
                            ("paper_projection", self.complexity_energetics.paper_projection),
                            ("build_pdf", self.complexity_energetics.build_pdf),
                            ("statistical_annotations", self.visualization.statistical_annotations)):
            _bool(value, name)
        for name in ("fail_on_missing", "fail_on_empty"):
            _bool(getattr(self.validation, name), name)
        _integer(self.visualization.dpi, "dpi")
        _integer(self.visualization.animation_fps, "animation_fps")
        _positive(self.visualization.animation_duration_s, "animation_duration_s")
        if len(self.visualization.figure_size) != 2:
            raise ValueError("figure_size requires width and height")
        for size in self.visualization.figure_size:
            _positive(size, "figure_size")
        if not self.visualization.output_formats or set(self.visualization.output_formats) - {"png", "svg", "pdf"}:
            raise ValueError("unsupported visualization output format")
        if not self.visualization.animation_formats or set(self.visualization.animation_formats) - {"gif", "html", "mp4"}:
            raise ValueError("unsupported animation format")
        if self.run.log_level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("Unsupported log_level")
        source = _resolve_repo_path(self.complexity_energetics.manifest)
        if not source.is_file():
            raise ValueError(f"complexity_energetics.manifest not found: {source}")
        load_manifest_snapshot(source)
        _preflight_paths(self)
        if self.tasks.animations and "mp4" in self.visualization.animation_formats and shutil.which("ffmpeg") is None:
            raise ValueError("Requested MP4 requires ffmpeg")
        if self.tasks.papers:
            for source in self.complexity_energetics.pdf_sources:
                if not _resolve_repo_path(source).is_file():
                    raise ValueError(f"Requested historical PDF source is absent: {source}")
        if self.tasks.papers and self.complexity_energetics.build_pdf:
            for executable in ("pandoc", "xelatex"):
                if shutil.which(executable) is None:
                    raise ValueError(f"Requested PDF requires {executable}")
        return self

    def to_dict(self):
        payload = asdict(self)
        payload.pop("source_bytes")
        payload["config_path"] = str(self.config_path)
        return payload


@dataclass
class RunContext:
    config: RunAllConfig
    run_dir: Path
    layout: dict[str, Path]
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    ce_result: Any = None
    ce_summary: Any = None
    ce_manifest: Any = None
    compatibility_projected: bool = False
    log_path: Path | None = None
    source_state: dict = field(default_factory=dict)
    manifest_snapshot: Path | None = None
    snapshot_data: bytes | None = None
    generation_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass(frozen=True)
class RunAllResult:
    run_id: str
    run_dir: Path
    manifest_path: Path | None
    provenance_path: Path | None
    artifacts: tuple[ArtifactRecord, ...]
    validate_only: bool = False

    def to_dict(self):
        return {"run_id": self.run_id, "run_dir": str(self.run_dir),
                "manifest_path": str(self.manifest_path) if self.manifest_path else None,
                "provenance_path": str(self.provenance_path) if self.provenance_path else None,
                "artifacts": [item.to_dict() for item in self.artifacts], "validate_only": self.validate_only}


def parse_task_names(value):
    names = tuple(part.strip() for part in value.split(",") if part.strip()) if isinstance(value, str) else tuple(value)
    if set(names) - set(TASK_NAMES):
        raise ValueError(f"Unknown task name(s): {set(names) - set(TASK_NAMES)}")
    return tuple(name for name in TASK_NAMES if name in names)


def _overlap(a, b):
    return a == b or a in b.parents or b in a.parents


def _preflight_paths(config):
    root = _resolve_repo_path(config.run.output_root)
    run = publication_path(root / config.run.run_id, tree=True)
    sources = [_resolve_repo_path(config.config_path), _resolve_repo_path(config.complexity_energetics.manifest)]
    sources += [_resolve_repo_path(path) for path in config.complexity_energetics.pdf_sources]
    for source in sources:
        if _overlap(run, source):
            raise ValueError("Run output and source namespaces overlap")
    if run.exists():
        if not config.run.clean_run_dir:
            raise FileExistsError(f"Run already exists; owned replacement requires clean_run_dir=true: {run}")
        marker = run / ".antstack-owned.json"
        completed = run / "manifest.json"
        if not marker.is_file() or not completed.is_file():
            raise ValueError(f"Refusing replacement of unowned/incomplete run: {run}")
        ownership = json.loads(marker.read_text())
        manifest = json.loads(completed.read_text())
        if ownership != {"owner": "ant-stack-run-all", "run_id": config.run.run_id} or manifest.get("run_id") != config.run.run_id or manifest.get("complete") is not True:
            raise ValueError("Existing run ownership/completion is invalid")
    if config.complexity_energetics.paper_projection:
        projection = publication_path(_resolve_repo_path(config.complexity_energetics.paper_root), tree=True)
        if _overlap(root, projection):
            raise ValueError("Canonical output and projection namespaces overlap")
        for source in sources:
            for generated in (projection / "out", projection / "assets", projection / "Generated.md"):
                if _overlap(source, generated):
                    raise ValueError("Projection and source namespaces overlap")
    return run


def create_run_layout(config):
    """Create a fresh run only; orchestration stages owned replacements separately."""
    config.validate()
    run = publication_path(_resolve_repo_path(config.run.output_root) / config.run.run_id, tree=True)
    run.parent.mkdir(parents=True, exist_ok=True)
    run.mkdir(exist_ok=False)
    return run, _layout(run)


def _layout(root):
    layout = {name: root / name for name in RUN_SUBDIRECTORIES}
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def checksum_file(path):
    return file_identity(path)["sha256"]




def _artifact_type(path):
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "provenance" if "provenance" in path.name or path.parent.name == "provenance" else "json"
    return {".csv": "csv", ".md": "markdown", ".png": "image", ".svg": "image", ".pdf": "paper",
            ".gif": "animation", ".mp4": "animation", ".html": "animation", ".log": "log", ".yaml": "yaml"}.get(suffix, "artifact")


def _record_artifact(ctx, path, artifact_type, producing_task, source_config):
    if path is None:
        return None
    path = ordinary_path(path)
    relative = path.relative_to(ctx.run_dir).as_posix()
    identity = file_identity(path)
    record = ArtifactRecord(artifact_type or _artifact_type(path), relative, str(path), producing_task,
                            source_config, identity["sha256"], identity["bytes"])
    if any(item.path == relative for item in ctx.artifacts):
        raise ValueError(f"Duplicate artifact identity: {relative}")
    ctx.artifacts.append(record)
    return record


def _write_json(path, payload):
    return atomic_json(path, payload)


def _copy_file(src, dst):
    src, dst = ordinary_path(src), ordinary_path(dst)
    file_identity(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src != dst:
        shutil.copy2(src, dst)
    return dst


def _configure_logging(path, level_name):
    # An unregistered per-run logger cannot capture or reconfigure the host app.
    logger = logging.Logger(__name__, level_name.upper())
    logger.propagate = False
    handler = logging.FileHandler(publication_path(path), mode="x", encoding="utf-8")
    handler.stream.write(f"Ant Stack workflow log; configured level={level_name.upper()}\n")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _close_logging(logger):
    for handler in tuple(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.flush()
        finally:
            handler.close()


def _write_initial_inputs(ctx):
    copied = ctx.layout["provenance"] / "run_config.yaml"
    copied.write_bytes(ctx.config.source_bytes if ctx.config.source_bytes is not None else ctx.config.config_path.read_bytes())
    _record_artifact(ctx, copied, "provenance", "configuration", "run_all.config")
    source = _resolve_repo_path(ctx.config.complexity_energetics.manifest)
    data = ctx.snapshot_data if ctx.snapshot_data is not None else load_manifest_snapshot(source)[0]
    ctx.manifest_snapshot = ctx.layout["data/raw"] / "complexity_energetics_manifest.yaml"
    ctx.manifest_snapshot.write_bytes(data)
    load_manifest_snapshot(ctx.manifest_snapshot)
    _record_artifact(ctx, ctx.manifest_snapshot, "yaml", "data", "complexity_energetics.manifest")


def _ensure_ce_result(ctx):
    if ctx.ce_result is not None:
        return
    if ctx.manifest_snapshot is None:
        raise ValueError("A validated input snapshot is required")
    ctx.ce_result = run_manifest(str(ctx.manifest_snapshot), str(ctx.run_dir / "papers/complexity_energetics/out"),
                                 source_state=ctx.source_state or None, statistics_seed=ctx.config.run.seed)
    ctx.ce_manifest = load_manifest_snapshot(ctx.manifest_snapshot)[2]
    ctx.ce_summary = json.loads(Path(ctx.ce_result.summary_path).read_text())
    if ctx.ce_summary.get("complete") is not True:
        raise RuntimeError("CE producer did not complete")
    for path in ctx.ce_result.output_paths:
        _record_artifact(ctx, path, None, "complexity_energetics", "complexity_energetics.manifest")


def _ce_rows(ctx):
    return list(ctx.ce_result.rows) if ctx.ce_result is not None else []


def _task_data(ctx):
    _ensure_ce_result(ctx)
    for source, name in ((ctx.ce_result.csv_path, "results.csv"), (ctx.ce_result.summary_path, "summary.json")):
        destination = _copy_file(source, ctx.layout["data/derived"] / f"complexity_energetics_{name}")
        _record_artifact(ctx, destination, None, "data", "complexity_energetics.manifest")


def _summarize_workloads(rows, *, seed, confidence=0.95, samples=1000):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["workload"], []).append(row["energy_est_j"])
    result = {"workloads": {}, "row_count": len(rows), "seed": seed, "confidence_level": confidence, "bootstrap_samples": samples}
    for name, values in grouped.items():
        mean, lo, hi = bootstrap_mean_ci(values, num_samples=samples, alpha=1-confidence, seed=seed)
        variance = sum((value-mean)**2 for value in values) / max(1, len(values)-1)
        result["workloads"][name] = {"n": len(values), "mean_energy_est_j": mean,
                                      "min_energy_est_j": min(values), "max_energy_est_j": max(values),
                                      "std_energy_est_j": math.sqrt(variance),
                                      "ci_low_energy_est_j": lo, "ci_high_energy_est_j": hi}
    return result


def _statistics(ctx):
    loaded = load_manifest_snapshot(ctx.manifest_snapshot)
    return _summarize_workloads(_ce_rows(ctx), seed=ctx.config.run.seed, confidence=loaded[4], samples=loaded[5])


def _scaling_curves(ctx):
    if ctx.ce_summary is not None:
        return list(ctx.ce_summary["scaling_curves"])
    _, raw, manifest, coeff, *_ = load_manifest_snapshot(ctx.manifest_snapshot)
    return scaling_curves(manifest, raw, coeff)


def _scaling_summary(ctx):
    return {"curves": [{**curve, **analyze_scaling_relationship(curve["x"], curve["y"])} for curve in _scaling_curves(ctx)]}


def _task_statistics(ctx):
    _ensure_ce_result(ctx)
    stats = _statistics(ctx)
    for name, payload in (("workload_summary.json", stats), ("scaling_summary.json", _scaling_summary(ctx))):
        _record_artifact(ctx, _write_json(ctx.layout["statistics"] / name, payload), "json", "statistics", "manifest.statistics")
    path = ctx.layout["statistics"] / "bootstrap_intervals.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["workload", "metric", "n", "mean", "ci_low", "ci_high"])
        for name, value in stats["workloads"].items():
            writer.writerow([name, "energy_est_j", value["n"], value["mean_energy_est_j"], value["ci_low_energy_est_j"], value["ci_high_energy_est_j"]])
    _record_artifact(ctx, path, "csv", "statistics", "manifest.statistics")


def _task_visualizations(ctx):
    from antstack_core.figures import PlotConfig, interval_plot
    _ensure_ce_result(ctx)
    for source in ctx.ce_result.output_paths:
        source = Path(source)
        if source.suffix == ".png" and source.parent.name == "assets":
            destination = _copy_file(source, ctx.layout["visualizations/static"] / f"complexity_energetics_{source.name}")
            _record_artifact(ctx, destination, "image", "visualizations", "complexity_energetics.assets")
    stats = _statistics(ctx)["workloads"]
    for fmt in ctx.config.visualization.output_formats:
        path = ctx.layout["visualizations/static"] / f"workload_energy_summary.{fmt}"
        labels = list(stats)
        means = [stats[name]["mean_energy_est_j"] for name in labels]
        bounds = ([(stats[name]["ci_low_energy_est_j"], stats[name]["ci_high_energy_est_j"]) for name in labels]
                  if ctx.config.visualization.statistical_annotations else None)
        interval_plot([name.title() for name in labels], means,
                      "Module-isolated modeled energy: current workload means", str(path),
                      xlabel="J / workload window", confidence_intervals=bounds,
                      config=PlotConfig(figure_size=ctx.config.visualization.figure_size,
                                        dpi=ctx.config.visualization.dpi))
        _record_artifact(ctx, path, "image", "visualizations", "manifest.statistics")


def _task_animations(ctx):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    _ensure_ce_result(ctx)
    visualization = ctx.config.visualization
    directory = ctx.layout["visualizations/animations"]
    formats = set(visualization.animation_formats)
    media = []
    if "html" in formats:
        formats.add("gif")
    stats = _statistics(ctx)["workloads"]
    curves = _scaling_curves(ctx)
    definitions = [("energy_components", list(stats), [stats[name]["mean_energy_est_j"] for name in stats])]
    if curves:
        definitions.append(("scaling_sweep", curves[0]["x"], curves[0]["y"]))
    frames = max(2, int(visualization.animation_fps * visualization.animation_duration_s))
    for name, labels, values in definitions:
        for fmt in sorted(formats - {"html"}):
            figure, ax = plt.subplots(figsize=visualization.figure_size)
            bars = ax.bar([str(label) for label in labels], [0.0]*len(values))
            ax.set_ylim(0, max(values)*1.15 if max(values) > 0 else 1)
            ax.set_ylabel("Modeled energy (J)")
            positive = [value for value in values if value > 0]
            if len(positive) == len(values) and max(positive) / min(positive) > 100:
                ax.set_yscale("symlog", linthresh=min(positive)/10)
                ax.set_ylabel("Modeled energy (J; symmetric-log axis)")
            ax.set_title("Illustrative reveal of fixed modeled values; not elapsed experimental time")
            def update(frame, bars=bars, values=values):
                for bar, value in zip(bars, values):
                    bar.set_height(value*(frame+1)/frames)
                return bars
            animation = FuncAnimation(figure, update, frames=frames, interval=1000/visualization.animation_fps)
            path = directory / f"{name}.{fmt}"
            writer = FFMpegWriter(fps=visualization.animation_fps) if fmt == "mp4" else PillowWriter(fps=visualization.animation_fps)
            try:
                animation.save(path, writer=writer)
            finally:
                plt.close(figure)
            _record_artifact(ctx, path, "animation", "animations", "visualization.animation_formats")
            if fmt == "gif":
                media.append(path)
    if "html" in visualization.animation_formats:
        path = directory / "preview.html"
        images = [f'<img alt="{image.stem}" src="data:image/gif;base64,{base64.b64encode(image.read_bytes()).decode()}" />' for image in media]
        path.write_text('<!doctype html><html><head><meta charset="utf-8"><title>Modeled energy animations</title></head><body>' + "".join(images) + '</body></html>', encoding="utf-8")
        _record_artifact(ctx, path, "animation", "animations", "visualization.animation_formats")


def _task_reports(ctx):
    _ensure_ce_result(ctx)
    # Preserve linked assets next to the copied report, not broken source-bound links.
    source = Path(ctx.ce_result.generated_markdown_path)
    target = ctx.layout["reports"] / "complexity_energetics_generated.md"
    text = source.read_text()
    text = text.replace("](assets/", "](../papers/complexity_energetics/out/assets/").replace("](results.csv)", "](../papers/complexity_energetics/out/results.csv)")
    target.write_text(text, encoding="utf-8")
    _record_artifact(ctx, target, "markdown", "reports", "complexity_energetics.generated")
    path = ctx.layout["reports"] / "run_summary.md"
    lines = ["# Ant Stack Run Summary", "", f"Run: {ctx.config.run.run_id}", "", "Analytical/synthetic workload outputs; not empirical validation.", "", "| Workload | N | Mean J/window | CI low | CI high |", "|---|---:|---:|---:|---:|"]
    for name, value in _statistics(ctx)["workloads"].items():
        lines.append(f"| {name} | {value['n']} | {value['mean_energy_est_j']:.8g} | {value['ci_low_energy_est_j']:.8g} | {value['ci_high_energy_est_j']:.8g} |")
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    _record_artifact(ctx, path, "markdown", "reports", "run_all.config")


def _task_papers(ctx):
    _ensure_ce_result(ctx)
    sources = []
    for source in ctx.config.complexity_energetics.pdf_sources:
        path = _resolve_repo_path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Requested historical PDF source is absent: {path}")
        destination = ctx.layout["papers"] / path.name
        _copy_file(path, destination)
        _record_artifact(ctx, destination, "historical_paper", "papers", "complexity_energetics.pdf_sources")
        sources.append(path.name)
    if ctx.config.complexity_energetics.build_pdf:
        from antstack_core.publishing.pdf_generator import PDFGenerator
        destination = ctx.layout["papers"] / "complexity_energetics.pdf"
        result = PDFGenerator().generate_pdf([Path(ctx.ce_result.generated_markdown_path)], destination,
                    {"title": "Ant Stack Generated Analytical Results", "subtitle": "Model outputs; not an empirical study or the full manuscript"})
        if not result.success:
            raise RuntimeError(f"Requested PDF build failed: {result.error_message}")
        _record_artifact(ctx, destination, "paper", "papers", "complexity_energetics.build_pdf")
    path = ctx.layout["papers"] / "paper_artifacts.md"
    path.write_text("# Paper Artifacts\n\nGenerated analytical report: [Markdown](complexity_energetics/out/Generated.md).\n\n" +
                    f"Historical copied PDFs (not a new build): {', '.join(sources) or 'none'}.\n", encoding="utf-8")
    _record_artifact(ctx, path, "markdown", "papers", "complexity_energetics.paper_projection")


def _validate_current(ctx):
    identities = set()
    for record in ctx.artifacts:
        if record.path in identities:
            raise ValueError(f"Duplicate artifact identity: {record.path}")
        identities.add(record.path)
        path = ordinary_path(ctx.run_dir / record.path)
        if path != Path(record.absolute_path):
            raise ValueError("Artifact escapes its canonical run namespace")
        current = file_identity(path)
        if current["bytes"] != record.bytes or current["sha256"] != record.checksum:
            raise RuntimeError(f"Artifact changed after registration: {record.path}")
        if ctx.config.validation.fail_on_empty and current["bytes"] == 0:
            raise RuntimeError(f"Empty artifact: {record.path}")


def _task_validation(ctx):
    _validate_current(ctx)
    present = {item.artifact_type for item in ctx.artifacts}
    missing = set(ctx.config.validation.required_artifact_types)-present
    valid = not (missing and ctx.config.validation.fail_on_missing)
    path = _write_json(ctx.layout["statistics"] / "validation_summary.json",
                       {"valid": valid, "missing_artifact_types": sorted(missing),
                        "checksums": {item.path: item.checksum for item in ctx.artifacts},
                        "validation_scope": "artifact completeness and current byte integrity, not scientific validation"})
    _record_artifact(ctx, path, "json", "validation", "validation.config")
    if not valid:
        raise RuntimeError(f"Run validation failed; missing artifact types: {sorted(missing)}")


def write_manifest(path, config, artifacts, *, generation_id=None):
    destination = ordinary_path(path)
    if destination in {_resolve_repo_path(config.config_path), _resolve_repo_path(config.complexity_energetics.manifest)}:
        raise ValueError("Manifest output must not replace an input source")
    paths = [record.path for record in artifacts]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate manifest identity")
    return _write_json(Path(path), {"schema_version": 2, "complete": True,
                                   "artifact_path_base": ".." if Path(path).parent.name == "provenance" else ".", "run_id": config.run.run_id,
                                   "generation_id": generation_id or uuid.uuid4().hex,
                                   "created_at_utc": datetime.now(timezone.utc).isoformat(),
                                   "tasks": config.tasks.enabled(), "config": config.to_dict(),
                                   "artifact_count": len(artifacts), "artifacts": [item.to_dict() for item in artifacts]})


def _write_final_metadata(ctx):
    for name, payload in (("dependency_versions.json", collect_dependency_versions(DEFAULT_DEPENDENCIES)),
                          ("git_state.json", ctx.source_state)):
        _record_artifact(ctx, _write_json(ctx.layout["provenance"] / name, payload), "provenance", "provenance", name)
    path = ctx.layout["provenance"] / "provenance.json"
    inputs = [ctx.layout["provenance"] / "run_config.yaml", ctx.manifest_snapshot]
    provenance = build_run_provenance(project="ant-stack-run-all", input_paths=inputs,
                                      output_paths=[item.path for item in ctx.artifacts] + ["manifest.json", "provenance/provenance.json", "provenance/output_inventory.json"],
                                      parameters={"run_id": ctx.config.run.run_id, "output_path_base": "run_root",
                                                  "seed_scope": "run.seed controls all bootstrap intervals; manifest seed identifies workload input"},
                                      source_state=ctx.source_state)
    # Portable snapshot identities rather than ephemeral staging directory names.
    provenance = replace(provenance, input_paths=[str(p.relative_to(ctx.run_dir)) for p in inputs],
                         input_identities={str(p.relative_to(ctx.run_dir)): file_identity(p) for p in inputs})
    write_provenance(path, provenance)
    _record_artifact(ctx, path, "provenance", "provenance", "run_all.provenance")
    inventory = _write_json(ctx.layout["provenance"] / "output_inventory.json",
                            {"artifact_path_base": "run_root", "artifacts": [item.to_dict() for item in ctx.artifacts]})
    _record_artifact(ctx, inventory, "provenance", "provenance", "run_all.inventory")
    _validate_current(ctx)
    # Required metadata first; the root completion manifest is atomic and last.
    write_manifest(ctx.layout["provenance"] / "run_manifest.json", ctx.config, ctx.artifacts, generation_id=ctx.generation_id)
    _record_artifact(ctx, ctx.layout["provenance"] / "run_manifest.json", "provenance", "provenance", "run_all.manifest_snapshot")
    marker = ctx.run_dir / ".antstack-owned.json"
    _write_json(marker, {"owner": "ant-stack-run-all", "run_id": ctx.config.run.run_id})
    _record_artifact(ctx, marker, "provenance", "provenance", "run_all.ownership")
    _validate_current(ctx)
    write_manifest(ctx.run_dir / "manifest.json", ctx.config, ctx.artifacts, generation_id=ctx.generation_id)
    return ctx.run_dir / "manifest.json", path


def run_all(config_path, *, output_root=None, run_id=None, validate_only=False, tasks=None, log_level=None):
    _bool(validate_only, "validate_only")
    config = RunAllConfig.from_file(config_path).with_overrides(output_root=output_root, run_id=run_id, tasks=tasks, log_level=log_level)
    destination = _preflight_paths(config)
    snapshot = load_manifest_snapshot(_resolve_repo_path(config.complexity_energetics.manifest))[0]
    if validate_only:
        return RunAllResult(config.run.run_id, destination, None, None, (), True)
    source_state = capture_source_state(Path.cwd())
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name(f".{destination.name}.stage-{uuid.uuid4().hex}")
    stage.mkdir()
    ctx = RunContext(config, stage, _layout(stage), source_state=source_state, snapshot_data=snapshot)
    ctx.log_path = ctx.layout["logs"] / "run_all_antstack.log"
    logging_state = _configure_logging(ctx.log_path, config.run.log_level)
    try:
        logging_state.info("Starting workflow: %s", config.run.run_id)
        _write_initial_inputs(ctx)
        for task in TASK_NAMES:
            if task != "validation" and getattr(config.tasks, task):
                globals()[f"_task_{task}"](ctx)
        logging_state.info("Finished producing requested tasks")
    finally:
        _close_logging(logging_state)
    _record_artifact(ctx, ctx.log_path, "log", "logging", "run.log_level")
    if config.tasks.validation:
        _task_validation(ctx)
    # Prepare compatibility publication without touching the current projection.
    projection_stage = projection = None
    if config.complexity_energetics.paper_projection and ctx.ce_result is not None:
        projection = _resolve_repo_path(config.complexity_energetics.paper_root)
        prepared = project_result(ctx.ce_result, projection, publish=False)
        projection_stage = Path(prepared["staged_root"])
        metadata_path = projection_stage / ".antstack-ce-projection.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["canonical_source"] = str(destination / "papers/complexity_energetics/out")
        metadata["canonical_generation_id"] = ctx.generation_id
        _write_json(metadata_path, metadata)
    _write_final_metadata(ctx)
    archive = destination.with_name(f".{destination.name}.archive-{uuid.uuid4().hex}")
    projection_archive = projection.with_name(f".{projection.name}.archive-{uuid.uuid4().hex}") if projection else None
    moved_run = moved_projection = published_projection = False
    preserve_tree_modes(stage, destination)
    if projection_stage is not None:
        preserve_tree_modes(projection_stage, projection)
    try:
        if projection_stage is not None:
            if projection.exists():
                os.replace(projection, projection_archive)
                moved_projection = True
            os.replace(projection_stage, projection)
            published_projection = True
        if destination.exists():
            os.replace(destination, archive)
            moved_run = True
        # Canonical completed view is published last: no required copy/move follows.
        os.replace(stage, destination)
    except BaseException:
        if published_projection:
            os.replace(projection, projection_stage)
        if moved_projection:
            os.replace(projection_archive, projection)
        if moved_run:
            os.replace(archive, destination)
        raise
    artifacts = tuple(replace(item, absolute_path=str(destination / item.path)) for item in ctx.artifacts)
    return RunAllResult(config.run.run_id, destination, destination / "manifest.json", destination / "provenance/provenance.json", artifacts)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate all Ant Stack outputs from one YAML config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out")
    parser.add_argument("--run-id")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--tasks")
    parser.add_argument("--log-level")
    args = parser.parse_args(argv)
    try:
        result = run_all(args.config, output_root=args.out, run_id=args.run_id, validate_only=args.validate_only, tasks=args.tasks, log_level=args.log_level)
    except Exception as error:
        print(f"run-all-antstack failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), indent=2))
    return 0


__all__ = ["ArtifactRecord", "ComplexityEnergeticsSection", "DEFAULT_REQUIRED_ARTIFACT_TYPES", "RUN_SUBDIRECTORIES",
           "RunAllConfig", "RunAllResult", "RunContext", "RunSection", "TASK_NAMES", "TaskSelection", "ValidationSection",
           "VisualizationSection", "checksum_file", "create_run_layout", "main", "parse_task_names", "run_all", "write_manifest"]

if __name__ == "__main__":
    raise SystemExit(main())
