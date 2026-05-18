"""Canonical run-all orchestration for Ant Stack generated artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import shutil
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from antstack_core.analysis import (
    EnergyCoefficients,
    ExperimentManifest,
    add_baseline_energy,
    analyze_scaling_relationship,
    body_workload_closed_form,
    bootstrap_mean_ci,
    brain_workload_closed_form,
    estimate_compute_energy,
    mind_workload_closed_form,
)
from antstack_core.publishing import (
    DEFAULT_DEPENDENCIES,
    build_run_provenance,
    collect_dependency_versions,
    detect_git_state,
    write_provenance,
)


ROOT = Path(__file__).resolve().parents[2]
TASK_NAMES = (
    "data",
    "statistics",
    "visualizations",
    "animations",
    "reports",
    "papers",
    "validation",
)
RUN_SUBDIRECTORIES = (
    "data/raw",
    "data/derived",
    "statistics",
    "visualizations/static",
    "visualizations/animations",
    "reports",
    "papers",
    "logs",
    "provenance",
)
DEFAULT_REQUIRED_ARTIFACT_TYPES = (
    "csv",
    "json",
    "markdown",
    "image",
    "animation",
    "log",
    "provenance",
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArtifactRecord:
    """Manifest entry for one generated or projected artifact."""

    artifact_type: str
    path: str
    absolute_path: str
    producing_task: str
    source_config: str
    checksum: str
    bytes: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable artifact manifest entry."""
        return asdict(self)


@dataclass(frozen=True)
class TaskSelection:
    """Boolean task switches for modular Ant Stack output generation."""

    data: bool = True
    statistics: bool = True
    visualizations: bool = True
    animations: bool = True
    reports: bool = True
    papers: bool = True
    validation: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "TaskSelection":
        """Create task switches from a mapping, defaulting all tasks to enabled."""
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise TypeError("tasks must be a mapping of task names to booleans")
        unknown = sorted(set(data) - set(TASK_NAMES))
        if unknown:
            raise ValueError(f"Unknown task name(s): {', '.join(unknown)}")
        values = {name: bool(data.get(name, True)) for name in TASK_NAMES}
        return cls(**values)

    @classmethod
    def only(cls, names: Iterable[str]) -> "TaskSelection":
        """Create switches with only the named tasks enabled."""
        parsed = parse_task_names(names)
        values = {name: name in parsed for name in TASK_NAMES}
        return cls(**values)

    def enabled(self) -> tuple[str, ...]:
        """Return enabled task names in canonical execution order."""
        return tuple(name for name in TASK_NAMES if bool(getattr(self, name)))


@dataclass(frozen=True)
class RunSection:
    """Run identity, output root, cleanup, and logging options."""

    run_id: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ"))
    seed: int = 123
    output_root: str = "outputs"
    clean_run_dir: bool = False
    log_level: str = "INFO"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "RunSection":
        """Create run options from a mapping."""
        data = data or {}
        return cls(
            run_id=str(data.get("run_id") or cls().run_id),
            seed=int(data.get("seed", 123)),
            output_root=str(data.get("output_root", "outputs")),
            clean_run_dir=bool(data.get("clean_run_dir", False)),
            log_level=str(data.get("log_level", "INFO")).upper(),
        )


@dataclass(frozen=True)
class ComplexityEnergeticsSection:
    """Complexity energetics orchestration options."""

    manifest: str = "papers/complexity_energetics/manifest.example.yaml"
    paper_projection: bool = True
    paper_root: str = "papers/complexity_energetics"
    build_pdf: bool = False
    pdf_sources: tuple[str, ...] = ("2_complexity_energetics.pdf",)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ComplexityEnergeticsSection":
        """Create complexity energetics options from a mapping."""
        data = data or {}
        pdf_sources = data.get("pdf_sources", ("2_complexity_energetics.pdf",))
        if isinstance(pdf_sources, str):
            pdf_sources = (pdf_sources,)
        return cls(
            manifest=str(data.get("manifest", cls.manifest)),
            paper_projection=bool(data.get("paper_projection", True)),
            paper_root=str(data.get("paper_root", cls.paper_root)),
            build_pdf=bool(data.get("build_pdf", False)),
            pdf_sources=tuple(str(path) for path in pdf_sources),
        )


@dataclass(frozen=True)
class VisualizationSection:
    """Visualization and animation rendering options."""

    output_formats: tuple[str, ...] = ("png",)
    dpi: int = 160
    figure_size: tuple[float, float] = (8.0, 5.0)
    statistical_annotations: bool = True
    animation_fps: int = 8
    animation_duration_s: float = 3.0
    animation_formats: tuple[str, ...] = ("gif", "html")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "VisualizationSection":
        """Create visualization options from a mapping."""
        data = data or {}
        output_formats = data.get("output_formats", ("png",))
        if isinstance(output_formats, str):
            output_formats = (output_formats,)
        animation_formats = data.get("animation_formats", ("gif", "html"))
        if isinstance(animation_formats, str):
            animation_formats = (animation_formats,)
        figure_size = data.get("figure_size", (8.0, 5.0))
        if not isinstance(figure_size, Sequence) or len(figure_size) != 2:
            raise ValueError("visualization.figure_size must contain width and height")
        return cls(
            output_formats=tuple(str(fmt).lower().lstrip(".") for fmt in output_formats),
            dpi=int(data.get("dpi", 160)),
            figure_size=(float(figure_size[0]), float(figure_size[1])),
            statistical_annotations=bool(data.get("statistical_annotations", True)),
            animation_fps=int(data.get("animation_fps", 8)),
            animation_duration_s=float(data.get("animation_duration_s", 3.0)),
            animation_formats=tuple(str(fmt).lower().lstrip(".") for fmt in animation_formats),
        )


@dataclass(frozen=True)
class ValidationSection:
    """Validation behavior for run-all artifacts."""

    required_artifact_types: tuple[str, ...] = DEFAULT_REQUIRED_ARTIFACT_TYPES
    fail_on_missing: bool = True
    fail_on_empty: bool = True
    deterministic_checksum_checks: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ValidationSection":
        """Create validation options from a mapping."""
        data = data or {}
        required = data.get("required_artifact_types", DEFAULT_REQUIRED_ARTIFACT_TYPES)
        if isinstance(required, str):
            required = tuple(part.strip() for part in required.split(",") if part.strip())
        return cls(
            required_artifact_types=tuple(str(item) for item in required),
            fail_on_missing=bool(data.get("fail_on_missing", True)),
            fail_on_empty=bool(data.get("fail_on_empty", True)),
            deterministic_checksum_checks=bool(data.get("deterministic_checksum_checks", True)),
        )


@dataclass(frozen=True)
class RunAllConfig:
    """Validated configuration for package-owned run-all orchestration."""

    config_path: Path
    run: RunSection = field(default_factory=RunSection)
    tasks: TaskSelection = field(default_factory=TaskSelection)
    complexity_energetics: ComplexityEnergeticsSection = field(default_factory=ComplexityEnergeticsSection)
    visualization: VisualizationSection = field(default_factory=VisualizationSection)
    validation: ValidationSection = field(default_factory=ValidationSection)

    @classmethod
    def from_file(cls, path: str | Path) -> "RunAllConfig":
        """Load and validate the run-all YAML configuration structure."""
        config_path = _resolve_repo_path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Run-all config not found: {config_path}")
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, Mapping):
            raise ValueError("Run-all config must be a YAML mapping")
        return cls(
            config_path=config_path,
            run=RunSection.from_mapping(_mapping(raw, "run")),
            tasks=TaskSelection.from_mapping(_mapping(raw, "tasks")),
            complexity_energetics=ComplexityEnergeticsSection.from_mapping(
                _mapping(raw, "complexity_energetics")
            ),
            visualization=VisualizationSection.from_mapping(_mapping(raw, "visualization")),
            validation=ValidationSection.from_mapping(_mapping(raw, "validation")),
        ).validate()

    def with_overrides(
        self,
        *,
        output_root: str | None = None,
        run_id: str | None = None,
        tasks: Iterable[str] | None = None,
        log_level: str | None = None,
    ) -> "RunAllConfig":
        """Return a copy with CLI overrides applied."""
        run = self.run
        if output_root is not None:
            run = replace(run, output_root=output_root)
        if run_id is not None:
            run = replace(run, run_id=run_id)
        if log_level is not None:
            run = replace(run, log_level=log_level.upper())
        selected_tasks = self.tasks if tasks is None else TaskSelection.only(tasks)
        return replace(self, run=run, tasks=selected_tasks).validate()

    def validate(self) -> "RunAllConfig":
        """Validate paths, task names, numeric options, and linked CE manifest."""
        errors: list[str] = []
        if not self.run.run_id or any(part in self.run.run_id for part in ("/", "\\")):
            errors.append("run.run_id must be a non-empty single path segment")
        if self.run.seed < 0:
            errors.append("run.seed must be non-negative")
        if self.visualization.dpi <= 0:
            errors.append("visualization.dpi must be positive")
        if self.visualization.figure_size[0] <= 0 or self.visualization.figure_size[1] <= 0:
            errors.append("visualization.figure_size values must be positive")
        if self.visualization.animation_fps <= 0:
            errors.append("visualization.animation_fps must be positive")
        if self.visualization.animation_duration_s <= 0:
            errors.append("visualization.animation_duration_s must be positive")
        invalid_formats = set(self.visualization.output_formats) - {"png", "svg", "pdf"}
        if invalid_formats:
            errors.append(f"unsupported visualization output format(s): {sorted(invalid_formats)}")
        invalid_animation_formats = set(self.visualization.animation_formats) - {"gif", "html", "mp4"}
        if invalid_animation_formats:
            errors.append(f"unsupported animation format(s): {sorted(invalid_animation_formats)}")
        manifest_path = _resolve_repo_path(self.complexity_energetics.manifest)
        if not manifest_path.exists():
            errors.append(f"complexity_energetics.manifest not found: {manifest_path}")
        else:
            try:
                manifest = ExperimentManifest.load(manifest_path)
                valid, manifest_errors = manifest.validate()
                non_contract_errors = {"Experiment name is required"}
                actionable = [msg for msg in manifest_errors if msg not in non_contract_errors]
                if not valid and actionable:
                    errors.extend(f"complexity manifest: {msg}" for msg in actionable)
            except Exception as exc:  # pragma: no cover - exact YAML parser errors vary
                errors.append(f"complexity_energetics.manifest is invalid: {exc}")
        if errors:
            raise ValueError("; ".join(errors))
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable config snapshot."""
        return {
            "config_path": str(self.config_path),
            "run": asdict(self.run),
            "tasks": asdict(self.tasks),
            "complexity_energetics": asdict(self.complexity_energetics),
            "visualization": asdict(self.visualization),
            "validation": asdict(self.validation),
        }


@dataclass
class RunContext:
    """Mutable runtime state shared by orchestration tasks."""

    config: RunAllConfig
    run_dir: Path
    layout: dict[str, Path]
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    ce_result: Any | None = None
    ce_summary: dict[str, Any] | None = None
    ce_manifest: ExperimentManifest | None = None
    compatibility_projected: bool = False
    log_path: Path | None = None


@dataclass(frozen=True)
class RunAllResult:
    """Result returned by programmatic run-all execution."""

    run_id: str
    run_dir: Path
    manifest_path: Path | None
    provenance_path: Path | None
    artifacts: tuple[ArtifactRecord, ...]
    validate_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result summary."""
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "provenance_path": str(self.provenance_path) if self.provenance_path else None,
            "validate_only": self.validate_only,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


def parse_task_names(value: str | Iterable[str]) -> tuple[str, ...]:
    """Parse comma-separated or iterable task names and validate them."""
    if isinstance(value, str):
        names = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        names = tuple(str(part).strip() for part in value if str(part).strip())
    unknown = sorted(set(names) - set(TASK_NAMES))
    if unknown:
        raise ValueError(f"Unknown task name(s): {', '.join(unknown)}")
    return tuple(name for name in TASK_NAMES if name in names)


def create_run_layout(config: RunAllConfig) -> tuple[Path, dict[str, Path]]:
    """Create the canonical output directory tree for a run."""
    output_root = _resolve_repo_path(config.run.output_root)
    run_dir = output_root / config.run.run_id
    if config.run.clean_run_dir and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    layout = {name: run_dir / name for name in RUN_SUBDIRECTORIES}
    for directory in layout.values():
        directory.mkdir(parents=True, exist_ok=True)
    return run_dir, layout


def write_manifest(path: str | Path, config: RunAllConfig, artifacts: Sequence[ArtifactRecord]) -> Path:
    """Write the run artifact manifest with checksums and config snapshot."""
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "run_id": config.run.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(_resolve_repo_path(config.run.output_root)),
        "source_config": str(config.config_path),
        "tasks": config.tasks.enabled(),
        "config": config.to_dict(),
        "artifact_count": len(artifacts),
        "artifacts": [artifact.to_dict() for artifact in artifacts],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def checksum_file(path: str | Path) -> str:
    """Return the SHA-256 checksum for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_all(
    config_path: str | Path,
    *,
    output_root: str | None = None,
    run_id: str | None = None,
    validate_only: bool = False,
    tasks: str | Iterable[str] | None = None,
    log_level: str | None = None,
) -> RunAllResult:
    """Run all configured Ant Stack output generation tasks."""
    task_names: tuple[str, ...] | None = None
    if tasks is not None:
        task_names = parse_task_names(tasks)
    config = RunAllConfig.from_file(config_path).with_overrides(
        output_root=output_root,
        run_id=run_id,
        tasks=task_names,
        log_level=log_level,
    )
    run_dir = _resolve_repo_path(config.run.output_root) / config.run.run_id
    if validate_only:
        return RunAllResult(
            run_id=config.run.run_id,
            run_dir=run_dir,
            manifest_path=None,
            provenance_path=None,
            artifacts=(),
            validate_only=True,
        )

    run_dir, layout = create_run_layout(config)
    ctx = RunContext(config=config, run_dir=run_dir, layout=layout)
    ctx.log_path = layout["logs"] / "run_all_antstack.log"
    _configure_logging(ctx.log_path, config.run.log_level)

    LOGGER.info("Starting run-all Ant Stack workflow: run_id=%s", config.run.run_id)
    LOGGER.info("Enabled tasks: %s", ", ".join(config.tasks.enabled()))

    try:
        _write_initial_inputs(ctx)
        if config.tasks.data:
            _task_data(ctx)
        if config.tasks.statistics:
            _task_statistics(ctx)
        if config.tasks.visualizations:
            _task_visualizations(ctx)
        if config.tasks.animations:
            _task_animations(ctx)
        if config.tasks.reports:
            _task_reports(ctx)
        if config.tasks.papers:
            _task_papers(ctx)
        _record_artifact(ctx, ctx.log_path, "log", "logging", "run.log_level")
        if config.tasks.validation:
            _task_validation(ctx)
        manifest_path, provenance_path = _write_final_metadata(ctx)
        LOGGER.info("Completed run-all Ant Stack workflow with %d artifacts", len(ctx.artifacts))
    finally:
        _flush_logging()

    return RunAllResult(
        run_id=config.run.run_id,
        run_dir=run_dir,
        manifest_path=manifest_path,
        provenance_path=provenance_path,
        artifacts=tuple(ctx.artifacts),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ``run-all-antstack``."""
    parser = argparse.ArgumentParser(description="Generate all Ant Stack outputs from one YAML config.")
    parser.add_argument("--config", required=True, help="Path to run_all_antstack YAML config.")
    parser.add_argument("--out", help="Override output root. Defaults to config run.output_root.")
    parser.add_argument("--run-id", help="Override run id. Defaults to config run.run_id.")
    parser.add_argument("--validate-only", action="store_true", help="Validate config without creating run artifacts.")
    parser.add_argument(
        "--tasks",
        help="Comma-separated task list: data,statistics,visualizations,animations,reports,papers,validation.",
    )
    parser.add_argument("--log-level", help="Override logging level, for example INFO or DEBUG.")
    args = parser.parse_args(argv)

    try:
        result = run_all(
            args.config,
            output_root=args.out,
            run_id=args.run_id,
            validate_only=args.validate_only,
            tasks=args.tasks,
            log_level=args.log_level,
        )
    except Exception as exc:
        print(f"run-all-antstack failed: {exc}", file=sys.stderr)
        return 1

    if result.validate_only:
        print(f"validated: {Path(args.config)}")
        return 0
    print(f"run_id: {result.run_id}")
    print(f"run_dir: {result.run_dir}")
    print(f"manifest: {result.manifest_path}")
    print(f"provenance: {result.provenance_path}")
    print(f"artifacts: {len(result.artifacts)}")
    return 0


def _mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def _relative_path(path: Path, base: Path = ROOT) -> str:
    resolved = path.resolve()
    for root in (base.resolve(), ROOT.resolve()):
        try:
            return str(resolved.relative_to(root))
        except ValueError:
            continue
    return str(resolved)


def _artifact_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        if "provenance" in path.name or path.parent.name == "provenance":
            return "provenance"
        return "json"
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix in {".png", ".svg"}:
        return "image"
    if suffix == ".pdf":
        return "paper"
    if suffix in {".gif", ".mp4", ".html"}:
        return "animation" if path.parent.name == "animations" else "html"
    if suffix == ".log":
        return "log"
    if suffix == ".yaml" or suffix == ".yml":
        return "yaml"
    return "artifact"


def _record_artifact(
    ctx: RunContext,
    path: str | Path | None,
    artifact_type: str | None,
    producing_task: str,
    source_config: str,
) -> ArtifactRecord | None:
    if path is None:
        return None
    artifact_path = Path(path)
    if not artifact_path.exists() or not artifact_path.is_file():
        return None
    absolute = artifact_path.resolve()
    checksum = checksum_file(absolute)
    record = ArtifactRecord(
        artifact_type=artifact_type or _artifact_type(absolute),
        path=_relative_path(absolute, ctx.run_dir),
        absolute_path=str(absolute),
        producing_task=producing_task,
        source_config=source_config,
        checksum=checksum,
        bytes=absolute.stat().st_size,
    )
    existing = {item.absolute_path for item in ctx.artifacts}
    if record.absolute_path not in existing:
        ctx.artifacts.append(record)
    return record


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _copy_file(src: str | Path, dst: str | Path) -> Path:
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if src_path.resolve() != dst_path.resolve():
        shutil.copy2(src_path, dst_path)
    return dst_path


def _configure_logging(path: Path, level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if getattr(handler, "_antstack_run_all", False):
            root_logger.removeHandler(handler)
            handler.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    file_handler._antstack_run_all = True  # type: ignore[attr-defined]
    root_logger.addHandler(file_handler)
    root_logger.setLevel(level)


def _flush_logging() -> None:
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            continue


def _write_initial_inputs(ctx: RunContext) -> None:
    config_copy = _copy_file(ctx.config.config_path, ctx.layout["provenance"] / "run_config.yaml")
    _record_artifact(ctx, config_copy, "provenance", "configuration", "run_all.config")
    manifest_source = _resolve_repo_path(ctx.config.complexity_energetics.manifest)
    manifest_copy = _copy_file(manifest_source, ctx.layout["data/raw"] / "complexity_energetics_manifest.yaml")
    _record_artifact(ctx, manifest_copy, "yaml", "data", "complexity_energetics.manifest")


def _ensure_ce_result(ctx: RunContext) -> None:
    if ctx.ce_result is not None:
        return
    from antstack_core.cli import ce as ce_cli

    manifest_path = _resolve_repo_path(ctx.config.complexity_energetics.manifest)
    out_dir = ctx.run_dir / "papers" / "complexity_energetics" / "out"
    LOGGER.info("Running complexity energetics manifest into %s", out_dir)
    ctx.ce_result = ce_cli.run_manifest(str(manifest_path), str(out_dir))
    ctx.ce_manifest = ExperimentManifest.load(manifest_path)
    summary_path = Path(ctx.ce_result.summary_path)
    if summary_path.exists():
        ctx.ce_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for output_path in getattr(ctx.ce_result, "output_paths", ()):
        _record_artifact(
            ctx,
            output_path,
            _artifact_type(Path(output_path)),
            "complexity_energetics",
            "complexity_energetics.manifest",
        )
    if ctx.config.complexity_energetics.paper_projection:
        _project_complexity_energetics(ctx)


def _project_complexity_energetics(ctx: RunContext) -> None:
    if ctx.compatibility_projected:
        return
    paper_root = _resolve_repo_path(ctx.config.complexity_energetics.paper_root)
    source_base = ctx.run_dir / "papers" / "complexity_energetics"
    source_out = source_base / "out"
    for src in sorted(source_out.glob("*")):
        if src.is_file():
            dst = _copy_file(src, paper_root / "out" / src.name)
            _record_artifact(ctx, dst, _artifact_type(dst), "paper_projection", "paper_projection")
    source_assets = source_base / "assets"
    for src in sorted(source_assets.glob("*")):
        if src.is_file():
            dst = _copy_file(src, paper_root / "assets" / src.name)
            _record_artifact(ctx, dst, _artifact_type(dst), "paper_projection", "paper_projection")
    generated = source_base / "Generated.md"
    if generated.exists():
        dst = _copy_file(generated, paper_root / "Generated.md")
        _record_artifact(ctx, dst, "markdown", "paper_projection", "paper_projection")
    ctx.compatibility_projected = True


def _task_data(ctx: RunContext) -> None:
    _ensure_ce_result(ctx)
    result = ctx.ce_result
    copies = [
        (result.csv_path, ctx.layout["data/derived"] / "complexity_energetics_results.csv", "csv"),
        (result.summary_path, ctx.layout["data/derived"] / "complexity_energetics_summary.json", "json"),
    ]
    for src, dst, artifact_type in copies:
        copied = _copy_file(src, dst)
        _record_artifact(ctx, copied, artifact_type, "data", "complexity_energetics.manifest")


def _task_statistics(ctx: RunContext) -> None:
    _ensure_ce_result(ctx)
    rows = _ce_rows(ctx)
    workload_summary = _summarize_workloads(rows, seed=ctx.config.run.seed)
    workload_path = _write_json(ctx.layout["statistics"] / "workload_summary.json", workload_summary)
    _record_artifact(ctx, workload_path, "json", "statistics", "complexity_energetics.summary")

    scaling_summary = _scaling_summary(ctx)
    scaling_path = _write_json(ctx.layout["statistics"] / "scaling_summary.json", scaling_summary)
    _record_artifact(ctx, scaling_path, "json", "statistics", "complexity_energetics.scaling")

    bootstrap_path = ctx.layout["statistics"] / "bootstrap_intervals.csv"
    bootstrap_path.parent.mkdir(parents=True, exist_ok=True)
    with bootstrap_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("workload", "metric", "n", "mean", "ci_low", "ci_high"),
        )
        writer.writeheader()
        for workload, stats in workload_summary["workloads"].items():
            writer.writerow(
                {
                    "workload": workload,
                    "metric": "energy_est_j",
                    "n": stats["n"],
                    "mean": stats["mean_energy_est_j"],
                    "ci_low": stats["ci_low_energy_est_j"],
                    "ci_high": stats["ci_high_energy_est_j"],
                }
            )
    _record_artifact(ctx, bootstrap_path, "csv", "statistics", "complexity_energetics.summary")


def _task_visualizations(ctx: RunContext) -> None:
    _ensure_ce_result(ctx)
    source_assets = ctx.run_dir / "papers" / "complexity_energetics" / "assets"
    for src in sorted(source_assets.glob("*.png")):
        dst = _copy_file(src, ctx.layout["visualizations/static"] / f"complexity_energetics_{src.name}")
        _record_artifact(ctx, dst, "image", "visualizations", "complexity_energetics.assets")

    rows = _ce_rows(ctx)
    for fmt in ctx.config.visualization.output_formats:
        plot_path = ctx.layout["visualizations/static"] / f"workload_energy_summary.{fmt}"
        _save_workload_energy_plot(rows, plot_path, ctx.config.visualization)
        _record_artifact(ctx, plot_path, "image", "visualizations", "complexity_energetics.summary")

    inventory = {
        "static_visualizations": [
            artifact.to_dict()
            for artifact in ctx.artifacts
            if artifact.producing_task == "visualizations"
        ],
        "formats": ctx.config.visualization.output_formats,
        "dpi": ctx.config.visualization.dpi,
        "figure_size": ctx.config.visualization.figure_size,
    }
    inventory_path = _write_json(ctx.layout["visualizations/static"] / "visualization_inventory.json", inventory)
    _record_artifact(ctx, inventory_path, "json", "visualizations", "visualization.config")


def _task_animations(ctx: RunContext) -> None:
    _ensure_ce_result(ctx)
    gif_paths: list[Path] = []
    if "gif" in ctx.config.visualization.animation_formats:
        energy_gif = ctx.layout["visualizations/animations"] / "energy_components.gif"
        _save_energy_animation(_ce_rows(ctx), energy_gif, ctx.config.visualization)
        gif_paths.append(energy_gif)
        _record_artifact(ctx, energy_gif, "animation", "animations", "complexity_energetics.summary")

        scaling_gif = ctx.layout["visualizations/animations"] / "scaling_sweep.gif"
        _save_scaling_animation(_scaling_curves(ctx), scaling_gif, ctx.config.visualization)
        gif_paths.append(scaling_gif)
        _record_artifact(ctx, scaling_gif, "animation", "animations", "complexity_energetics.scaling")
    if "mp4" in ctx.config.visualization.animation_formats:
        note_path = ctx.layout["visualizations/animations"] / "mp4_writer_status.json"
        _write_json(
            note_path,
            {
                "requested": True,
                "created": False,
                "reason": "MP4 output requires an ffmpeg runtime; GIF and HTML preview are always generated.",
            },
        )
        _record_artifact(ctx, note_path, "json", "animations", "visualization.animation_formats")
    if "html" in ctx.config.visualization.animation_formats:
        html_path = ctx.layout["visualizations/animations"] / "preview.html"
        _write_animation_preview(html_path, gif_paths)
        _record_artifact(ctx, html_path, "animation", "animations", "visualization.animation_formats")


def _task_reports(ctx: RunContext) -> None:
    _ensure_ce_result(ctx)
    generated = Path(ctx.ce_result.generated_markdown_path)
    if generated.exists():
        copied = _copy_file(generated, ctx.layout["reports"] / "complexity_energetics_generated.md")
        _record_artifact(ctx, copied, "markdown", "reports", "complexity_energetics.generated")
    summary_path = ctx.layout["reports"] / "run_summary.md"
    summary_path.write_text(_render_run_summary(ctx), encoding="utf-8")
    _record_artifact(ctx, summary_path, "markdown", "reports", "run_all.config")


def _task_papers(ctx: RunContext) -> None:
    _ensure_ce_result(ctx)
    paper_note = ctx.layout["papers"] / "paper_artifacts.md"
    copied_pdfs: list[str] = []
    for source in ctx.config.complexity_energetics.pdf_sources:
        src = _resolve_repo_path(source)
        if src.exists():
            dst = _copy_file(src, ctx.layout["papers"] / src.name)
            copied_pdfs.append(str(dst))
            _record_artifact(ctx, dst, "paper", "papers", "complexity_energetics.pdf_sources")
    paper_note.write_text(
        "\n".join(
            [
                "# Paper Artifacts",
                "",
                "Complexity energetics paper-ready projections are generated under this run's `papers/complexity_energetics/` folder.",
                f"Compatibility projection enabled: {ctx.config.complexity_energetics.paper_projection}",
                f"PDF build requested: {ctx.config.complexity_energetics.build_pdf}",
                f"Copied PDFs: {', '.join(copied_pdfs) if copied_pdfs else 'none available in this checkout'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _record_artifact(ctx, paper_note, "paper", "papers", "complexity_energetics.paper_projection")


def _task_validation(ctx: RunContext) -> None:
    artifact_types = {artifact.artifact_type for artifact in ctx.artifacts}
    missing = [
        artifact_type
        for artifact_type in ctx.config.validation.required_artifact_types
        if artifact_type not in artifact_types
    ]
    empty = [artifact.to_dict() for artifact in ctx.artifacts if artifact.bytes <= 0]
    checksums = {
        artifact.path: artifact.checksum
        for artifact in ctx.artifacts
        if ctx.config.validation.deterministic_checksum_checks
    }
    valid = True
    if ctx.config.validation.fail_on_missing and missing:
        valid = False
    if ctx.config.validation.fail_on_empty and empty:
        valid = False
    payload = {
        "valid": valid,
        "required_artifact_types": ctx.config.validation.required_artifact_types,
        "present_artifact_types": sorted(artifact_types),
        "missing_artifact_types": missing,
        "empty_artifacts": empty,
        "artifact_count": len(ctx.artifacts),
        "checksums": checksums,
    }
    validation_path = _write_json(ctx.layout["statistics"] / "validation_summary.json", payload)
    _record_artifact(ctx, validation_path, "json", "validation", "validation.config")
    if not valid:
        raise RuntimeError(f"Run validation failed: {payload}")


def _write_final_metadata(ctx: RunContext) -> tuple[Path, Path]:
    dependency_path = _write_json(
        ctx.layout["provenance"] / "dependency_versions.json",
        collect_dependency_versions((*DEFAULT_DEPENDENCIES, "pandas", "pillow")),
    )
    _record_artifact(ctx, dependency_path, "provenance", "provenance", "runtime.dependencies")

    commit, dirty = detect_git_state(ROOT)
    git_path = _write_json(
        ctx.layout["provenance"] / "git_state.json",
        {"git_commit": commit, "git_dirty": dirty, "root": str(ROOT)},
    )
    _record_artifact(ctx, git_path, "provenance", "provenance", "git")

    manifest_path = ctx.run_dir / "manifest.json"
    run_manifest_path = ctx.layout["provenance"] / "run_manifest.json"
    provenance_path = ctx.layout["provenance"] / "provenance.json"
    inventory_path = ctx.layout["provenance"] / "output_inventory.json"
    output_paths = [
        artifact.absolute_path for artifact in ctx.artifacts
    ] + [
        str(manifest_path.resolve()),
        str(run_manifest_path.resolve()),
        str(provenance_path.resolve()),
        str(inventory_path.resolve()),
    ]
    provenance = build_run_provenance(
        project="ant-stack-run-all",
        command=sys.argv,
        input_paths=[ctx.config.config_path, _resolve_repo_path(ctx.config.complexity_energetics.manifest)],
        output_paths=output_paths,
        parameters={
            "run_id": ctx.config.run.run_id,
            "tasks": ctx.config.tasks.enabled(),
            "paper_projection": ctx.config.complexity_energetics.paper_projection,
        },
        cwd=ROOT,
        dependency_names=(*DEFAULT_DEPENDENCIES, "pandas", "pillow"),
    )
    write_provenance(provenance_path, provenance)
    _record_artifact(ctx, provenance_path, "provenance", "provenance", "run_all.provenance")

    inventory_path = _write_json(
        inventory_path,
        {
            "run_id": ctx.config.run.run_id,
            "artifact_count": len(ctx.artifacts),
            "artifacts": [artifact.to_dict() for artifact in ctx.artifacts],
        },
    )
    _record_artifact(ctx, inventory_path, "provenance", "provenance", "run_all.inventory")

    write_manifest(manifest_path, ctx.config, ctx.artifacts)
    _copy_file(manifest_path, run_manifest_path)
    return manifest_path, provenance_path


def _ce_rows(ctx: RunContext) -> list[dict[str, Any]]:
    if ctx.ce_summary and "rows" in ctx.ce_summary:
        return list(ctx.ce_summary["rows"])
    if ctx.ce_result is not None and getattr(ctx.ce_result, "rows", None) is not None:
        return list(ctx.ce_result.rows)
    return []


def _summarize_workloads(rows: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, Any]:
    by_workload: dict[str, list[float]] = {}
    for row in rows:
        by_workload.setdefault(str(row["workload"]), []).append(float(row["energy_est_j"]))
    summary: dict[str, Any] = {"workloads": {}, "row_count": len(rows)}
    for workload, values in sorted(by_workload.items()):
        mean, ci_low, ci_high = bootstrap_mean_ci(values, num_samples=1000, alpha=0.05, seed=seed)
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
        summary["workloads"][workload] = {
            "n": len(values),
            "mean_energy_est_j": mean,
            "min_energy_est_j": min(values),
            "max_energy_est_j": max(values),
            "std_energy_est_j": math.sqrt(variance),
            "ci_low_energy_est_j": ci_low,
            "ci_high_energy_est_j": ci_high,
        }
    return summary


def _scaling_summary(ctx: RunContext) -> dict[str, Any]:
    curves = _scaling_curves(ctx)
    summary = {
        "curves": [],
        "runner_scaling_exponents": getattr(ctx.ce_result, "scaling_exponents", {}) if ctx.ce_result else {},
    }
    for curve in curves:
        relationship = analyze_scaling_relationship(curve["x"], curve["y"])
        summary["curves"].append(
            {
                "name": curve["name"],
                "workload": curve["workload"],
                "parameter": curve["parameter"],
                "x": curve["x"],
                "energy_est_j": curve["y"],
                "scaling_exponent": relationship.get("scaling_exponent"),
                "r_squared": relationship.get("r_squared"),
            }
        )
    return summary


def _scaling_curves(ctx: RunContext) -> list[dict[str, Any]]:
    manifest_path = _resolve_repo_path(ctx.config.complexity_energetics.manifest)
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    manifest = ctx.ce_manifest or ExperimentManifest.load(manifest_path)
    coeff = EnergyCoefficients(
        flops_pj=float(manifest.coefficients.flops_pj),
        sram_pj_per_byte=float(manifest.coefficients.sram_pj_per_byte),
        dram_pj_per_byte=float(manifest.coefficients.dram_pj_per_byte),
        spike_aj=float(manifest.coefficients.spike_aj),
        baseline_w=float(manifest.coefficients.baseline_w),
        body_per_joint_w=float(manifest.coefficients.body_per_joint_w),
        body_sensor_w_per_channel=float(manifest.coefficients.body_sensor_w_per_channel),
    )
    analyses = ((raw.get("scaling") or {}).get("analyses") or [])
    if not analyses:
        scaling = raw.get("scaling") or {}
        analyses = [
            {
                "workload": scaling.get("workload", "brain"),
                "param": scaling.get("param", "K"),
                "values": scaling.get("values", [64, 128, 256, 512]),
                "description": "Legacy scaling analysis",
            }
        ]
    curves: list[dict[str, Any]] = []
    for item in analyses:
        workload = str(item["workload"])
        parameter = str(item["param"])
        values = [float(value) for value in item["values"]]
        workload_cfg = (manifest.workloads or {}).get(workload)
        params = dict((workload_cfg.params if workload_cfg else {}) or {})
        y_values: list[float] = []
        for value in values:
            params[parameter] = int(value) if float(value).is_integer() else value
            load = _closed_form_load(workload, 0.25, params)
            y_values.append(add_baseline_energy(estimate_compute_energy(load, coeff), 0.25, coeff))
        curves.append(
            {
                "name": str(item.get("description", f"{workload} vs {parameter}")),
                "workload": workload,
                "parameter": parameter,
                "x": values,
                "y": y_values,
            }
        )
    return curves


def _closed_form_load(workload: str, duration_s: float, params: Mapping[str, Any]):
    if workload == "body":
        return body_workload_closed_form(duration_s, dict(params))
    if workload == "brain":
        return brain_workload_closed_form(duration_s, dict(params))
    if workload == "mind":
        return mind_workload_closed_form(duration_s, dict(params))
    raise ValueError(f"Unsupported workload in scaling curve: {workload}")


def _save_workload_energy_plot(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    visualization: VisualizationSection,
) -> None:
    import matplotlib.pyplot as plt

    summary = _summarize_workloads(rows, seed=123)["workloads"]
    labels = list(summary)
    values = [summary[label]["mean_energy_est_j"] for label in labels]
    yerr = [
        max(0.0, summary[label]["ci_high_energy_est_j"] - summary[label]["mean_energy_est_j"])
        for label in labels
    ]
    fig, ax = plt.subplots(figsize=visualization.figure_size)
    ax.bar(labels, values, yerr=yerr if visualization.statistical_annotations else None, capsize=4)
    ax.set_title("Workload Energy Summary")
    ax.set_ylabel("Mean estimated energy (J)")
    ax.set_xlabel("Workload")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=visualization.dpi)
    plt.close(fig)


def _save_energy_animation(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    visualization: VisualizationSection,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    summary = _summarize_workloads(rows, seed=123)["workloads"]
    labels = list(summary)
    values = [summary[label]["mean_energy_est_j"] for label in labels]
    frames = max(2, int(visualization.animation_fps * visualization.animation_duration_s))
    fig, ax = plt.subplots(figsize=visualization.figure_size)
    bars = ax.bar(labels, [0.0 for _ in labels])
    ymax = max(values) * 1.15 if values else 1.0
    ax.set_ylim(0.0, ymax)
    ax.set_ylabel("Mean estimated energy (J)")
    ax.set_title("Energy Components Across Workloads")

    def update(frame: int):
        fraction = (frame + 1) / frames
        for bar, value in zip(bars, values):
            bar.set_height(value * fraction)
        return bars

    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / visualization.animation_fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=visualization.animation_fps))
    plt.close(fig)


def _save_scaling_animation(
    curves: Sequence[Mapping[str, Any]],
    path: Path,
    visualization: VisualizationSection,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    if not curves:
        curves = [{"name": "empty", "x": [0.0, 1.0], "y": [0.0, 0.0]}]
    frames = max(2, int(visualization.animation_fps * visualization.animation_duration_s))
    fig, ax = plt.subplots(figsize=visualization.figure_size)
    max_y = max(max(float(value) for value in curve["y"]) for curve in curves)
    ax.set_ylim(0.0, max_y * 1.15 if max_y > 0 else 1.0)
    min_x = min(min(float(value) for value in curve["x"]) for curve in curves)
    max_x = max(max(float(value) for value in curve["x"]) for curve in curves)
    ax.set_xlim(min_x, max_x)
    ax.set_xlabel("Scaling parameter value")
    ax.set_ylabel("Estimated energy (J)")
    ax.set_title("Scaling Sweep")
    lines = []
    for curve in curves:
        (line,) = ax.plot([], [], marker="o", label=str(curve["name"])[:48])
        lines.append(line)
    ax.legend(loc="best", fontsize="small")

    def update(frame: int):
        fraction = (frame + 1) / frames
        for line, curve in zip(lines, curves):
            x_values = list(curve["x"])
            y_values = [float(value) * fraction for value in curve["y"]]
            line.set_data(x_values, y_values)
        return lines

    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / visualization.animation_fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=visualization.animation_fps))
    plt.close(fig)


def _write_animation_preview(path: Path, gif_paths: Sequence[Path]) -> None:
    links = []
    for gif_path in gif_paths:
        links.append(
            f'<figure><img src="{gif_path.name}" alt="{gif_path.stem}" /><figcaption>{gif_path.stem}</figcaption></figure>'
        )
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset=\"utf-8\"><title>Ant Stack Animations</title></head>",
                "<body>",
                "<h1>Ant Stack Animation Preview</h1>",
                *links,
                "</body></html>",
            ]
        ),
        encoding="utf-8",
    )


def _render_run_summary(ctx: RunContext) -> str:
    stats = _summarize_workloads(_ce_rows(ctx), seed=ctx.config.run.seed)
    lines = [
        "# Ant Stack Run Summary",
        "",
        f"- Run ID: `{ctx.config.run.run_id}`",
        f"- Output root: `{ctx.run_dir}`",
        f"- Tasks: `{', '.join(ctx.config.tasks.enabled())}`",
        f"- Artifact count at report time: `{len(ctx.artifacts)}`",
        "",
        "## Workload Statistics",
        "",
        "| Workload | N | Mean energy (J) | 95% CI low | 95% CI high |",
        "|---|---:|---:|---:|---:|",
    ]
    for workload, values in stats["workloads"].items():
        lines.append(
            f"| {workload} | {values['n']} | {values['mean_energy_est_j']:.6g} | {values['ci_low_energy_est_j']:.6g} | {values['ci_high_energy_est_j']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Canonical Folders",
            "",
            "- `data/raw/`: copied source manifests and run config.",
            "- `data/derived/`: normalized CSV and JSON outputs.",
            "- `statistics/`: workload summaries, scaling metrics, bootstrap intervals, and validation summary.",
            "- `visualizations/static/`: PNG/SVG/PDF figures.",
            "- `visualizations/animations/`: GIF/HTML animation previews.",
            "- `reports/`: Markdown summaries and generated manuscript fragments.",
            "- `papers/`: paper-ready projections and copied PDFs when available.",
            "- `provenance/`: run manifest, provenance, dependency versions, git state, and output inventory.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "ArtifactRecord",
    "ComplexityEnergeticsSection",
    "DEFAULT_REQUIRED_ARTIFACT_TYPES",
    "RUN_SUBDIRECTORIES",
    "RunAllConfig",
    "RunAllResult",
    "RunContext",
    "RunSection",
    "TASK_NAMES",
    "TaskSelection",
    "ValidationSection",
    "VisualizationSection",
    "checksum_file",
    "create_run_layout",
    "main",
    "parse_task_names",
    "run_all",
    "write_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
