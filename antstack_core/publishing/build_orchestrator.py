"""Dependency-aware configured publication builds with source/output-bound caching."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import tempfile
from typing import Any

from .pdf_generator import PDFGenerator, validate_pdf_environment
from .quality_validator import QualityValidator, validate_build_readiness
from .reference_manager import ReferenceManager
from .provenance import atomic_json, publication_path, publish_file
from antstack_core import __version__


def _digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


@dataclass
class BuildTarget:
    """Ordered sources, declared dependencies and observed build identities."""

    name: str
    source_files: list[Path]
    output_path: Path
    dependencies: list[str] = field(default_factory=list)
    build_config: dict[str, Any] = field(default_factory=dict)
    last_build_hash: str | None = None
    build_status: str = "pending"
    dependency_files: list[Path] = field(default_factory=list)
    last_output_hash: str | None = None
    source_root: Path | None = None
    _render_config: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.name) or self.name in (".", ".."):
            raise ValueError("Target name must be one ordinary component")
        self.source_files = [Path(path).absolute() for path in self.source_files]
        self.dependency_files = [Path(path).absolute() for path in self.dependency_files]
        self.output_path = Path(self.output_path).absolute()
        if self.output_path.suffix.lower() != ".pdf" or self.output_path in self.source_files:
            raise ValueError("Target output must be a PDF distinct from its sources")

    @property
    def needs_rebuild(self) -> bool:
        return (self.output_path.is_symlink() or self.last_build_hash is None or self.last_output_hash is None
                or _digest(self.output_path) != self.last_output_hash
                or self._calculate_source_hash() != self.last_build_hash)

    def _calculate_source_hash(self) -> str:
        result = hashlib.sha256(json.dumps({"target": self.build_config, "renderer": self._render_config}, sort_keys=True, default=str).encode())
        for path in [*self.source_files, *sorted(set(self.dependency_files))]:
            result.update(str(path).encode())
            result.update((_digest(path) or "MISSING").encode())
        return result.hexdigest()

    def update_build_hash(self):
        self.last_build_hash = self._calculate_source_hash()
        self.last_output_hash = _digest(self.output_path)


@dataclass
class BuildResult:
    target_name: str
    success: bool
    output_path: Path | None = None
    error_message: str | None = None
    build_time_seconds: float = 0.0
    artifacts: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BuildOrchestratorConfig:
    build_dir: Path = Path("build")
    temp_dir: Path = Path("temp")
    cache_dir: Path = Path("cache")
    max_workers: int = 4
    enable_parallel: bool = True
    enable_incremental: bool = True
    enable_validation: bool = True
    generate_reports: bool = True

    def __post_init__(self):
        for name in ("build_dir", "temp_dir", "cache_dir"):
            setattr(self, name, Path(getattr(self, name)).absolute())
        for name in ("enable_parallel", "enable_incremental", "enable_validation", "generate_reports"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if isinstance(self.max_workers, bool) or not isinstance(self.max_workers, int) or self.max_workers < 1:
            raise ValueError("max_workers must be a positive integer")


class BuildOrchestrator:
    """Build configured documents in dependency waves; failures block dependents."""

    def __init__(self, config: BuildOrchestratorConfig | None = None):
        self.config = config or BuildOrchestratorConfig()
        self.targets: dict[str, BuildTarget] = {}
        self.build_results: dict[str, BuildResult] = {}
        for path in (self.config.build_dir, self.config.temp_dir, self.config.cache_dir, self.config.build_dir / "reports"):
            for ancestor in (path, *path.parents):
                if ancestor.is_symlink():
                    raise ValueError(f"Symlink build directory is not allowed: {ancestor}")
            path.mkdir(parents=True, exist_ok=True)
        self.pdf_generator = PDFGenerator()
        self.quality_validator = QualityValidator()
        self.reference_manager = ReferenceManager()

    def _sync_render_config(self, target: BuildTarget):
        settings = asdict(self.pdf_generator.config)
        settings['package_version'] = __version__
        if self.pdf_generator.config.template_dir is not None:
            settings['template_sha256'] = _digest(Path(self.pdf_generator.config.template_dir) / 'default.latex')
        target._render_config = settings

    def add_build_target(self, target: BuildTarget):
        destination = target.output_path.resolve()
        for name, registered in self.targets.items():
            if name != target.name and registered.output_path.resolve() == destination:
                raise ValueError(f"Duplicate PDF output destination for targets '{name}' and '{target.name}': {destination}")
        self._sync_render_config(target)
        self.targets[target.name] = target

    def create_paper_target(self, paper_name: str, paper_dir: Path,
                            output_dir: Path | None = None) -> BuildTarget:
        from .paper_builder import ModularPaperBuilder
        directory = Path(paper_dir).absolute()
        loader = ModularPaperBuilder(directory.parent)
        loader.papers_dir = directory.parent
        config = loader.load_paper_config(directory.name)
        sources = [directory / name for name in config["content"]["files"]]
        dependencies = [directory / "paper_config.yaml"]
        bibliography = config["latex"].get("bibliography")
        if bibliography:
            dependencies.append(directory / bibliography)
        assets = directory / "assets"
        if assets.is_dir():
            dependencies.extend(path for path in assets.rglob("*") if path.is_file())
        destination = Path(output_dir) if output_dir is not None else self.config.build_dir / "papers"
        filename = config["paper"].get("output_filename", loader._default_pdf_name(directory.name))
        target = BuildTarget(paper_name, sources, destination / filename, build_config=config,
                             dependency_files=dependencies, source_root=directory)
        self.add_build_target(target)
        return target

    def _record(self, result: BuildResult, status: str | None = None) -> BuildResult:
        self.build_results[result.target_name] = result
        if result.target_name in self.targets:
            self.targets[result.target_name].build_status = status or ("success" if result.success else "failed")
        return result

    def _resolve_dependencies(self) -> list[str]:
        complete = set()
        active = set()
        ordered = []
        def visit(name):
            if name in active:
                raise ValueError(f"Cyclic build dependency at {name}")
            if name not in self.targets:
                raise ValueError(f"Unknown build dependency: {name}")
            if name in complete:
                return
            active.add(name)
            for dependency in self.targets[name].dependencies:
                visit(dependency)
            active.remove(name)
            complete.add(name)
            ordered.append(name)
        for name in self.targets:
            visit(name)
        return ordered

    def build_target(self, target_name: str, force: bool = False) -> BuildResult:
        if target_name not in self.targets:
            return self._record(BuildResult(target_name, False, error_message=f"Target '{target_name}' not found"))
        order = self._resolve_dependencies()
        required = set()
        def include(name):
            if name not in required:
                required.add(name)
                for dependency in self.targets[name].dependencies:
                    include(dependency)
        include(target_name)
        current = {}
        for name in order:
            if name in required:
                blocked = [dependency for dependency in self.targets[name].dependencies if not current[dependency].success]
                current[name] = (self._record(BuildResult(name, False, error_message=f"Failed dependencies: {blocked}"), "blocked")
                                 if blocked else self._build_one(name, force))
        return current[target_name]

    def _build_one(self, name: str, force: bool) -> BuildResult:
        import time
        started = time.monotonic()
        target = self.targets[name]
        self._sync_render_config(target)
        target.build_status = "building"
        try:
            if self.config.enable_incremental and not force and not target.needs_rebuild:
                return self._record(BuildResult(name, True, target.output_path, metadata={"reused": True}))
            validation = self._validate_target(target)
            if not validation["passed"]:
                raise ValueError("; ".join(validation["errors"]))
            source_hash = target._calculate_source_hash()
            publication_path(target.output_path)
            target.output_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=name + "-", dir=self.config.temp_dir) as temporary:
                staged = Path(temporary) / "paper.pdf"
                result = self._generate_pdf(replace(target, output_path=staged), target.source_files)
                if not result.success:
                    return self._record(result)
                if not staged.is_file():
                    raise RuntimeError("Renderer did not create the requested current output")
                if target._calculate_source_hash() != source_hash:
                    raise RuntimeError("Source/dependencies changed during the build")
                for filename, digest in result.metadata.get("resource_sha256", {}).items():
                    resource = Path(filename)
                    if _digest(resource) != digest:
                        raise RuntimeError(f"Resource changed during the build: {resource}")
                    if resource not in target.dependency_files:
                        target.dependency_files.append(resource)
                publish_file(staged, target.output_path)
            target.update_build_hash()
            result.output_path = target.output_path
            result.artifacts = [target.output_path]
            result.build_time_seconds = time.monotonic() - started
            if self.config.generate_reports:
                self._generate_build_report(target, result)
            return self._record(result)
        except (OSError, ValueError, RuntimeError) as error:
            return self._record(BuildResult(name, False, error_message=str(error), build_time_seconds=time.monotonic() - started))

    def build_all(self, force: bool = False, parallel: bool = True) -> dict[str, BuildResult]:
        order = self._resolve_dependencies()
        if not order:
            raise ValueError("No build targets configured")
        results = {}
        remaining = set(order)
        workers = self.config.max_workers if parallel and self.config.enable_parallel else 1
        with ThreadPoolExecutor(max_workers=workers) as executor:
            while remaining:
                ready = [name for name in order if name in remaining and all(dep in results for dep in self.targets[name].dependencies)]
                futures = {}
                for name in ready:
                    failed = [dep for dep in self.targets[name].dependencies if not results[dep].success]
                    if failed:
                        results[name] = self._record(BuildResult(name, False, error_message=f"Failed dependencies: {failed}"), "blocked")
                    else:
                        futures[executor.submit(self._build_one, name, force)] = name
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
                remaining.difference_update(ready)
        return results

    def _validate_target(self, target: BuildTarget) -> dict[str, Any]:
        for ancestor in (target.output_path, *target.output_path.parents):
            if ancestor.is_symlink():
                raise ValueError(f"Symlink output is not allowed: {ancestor}")
        errors = [f"Source file not found: {path}" for path in [*target.source_files, *target.dependency_files] if not path.is_file()]
        warnings = []
        if not target.source_files:
            errors.append("No source files specified")
        if not errors and self.config.enable_validation:
            report = validate_build_readiness(target.source_files)
            errors.extend(issue.message for issue in report.issues if issue.issue_type == "error")
            warnings.extend(issue.message for issue in report.issues if issue.issue_type == "warning")
        return {"passed": not errors, "errors": errors, "warnings": warnings}

    def _generate_pdf(self, target: BuildTarget, processed_files: list[Path]) -> BuildResult:
        paper = target.build_config.get("paper", target.build_config.get("metadata", {}))
        latex = target.build_config.get("latex", {})
        arguments = list(self.pdf_generator.config.pandoc_args)
        for name in ("geometry", "mainfont", "mathfont"):
            if latex.get(name):
                arguments.extend(["--variable", f"{name}:{latex[name]}"])
        if latex.get("bibliography"):
            bibliography = (target.source_root or Path.cwd()) / latex["bibliography"]
            arguments.extend(["--bibliography", str(bibliography.absolute()), "--citeproc"])
        renderer = PDFGenerator(replace(self.pdf_generator.config, pandoc_args=arguments))
        rendered = renderer.generate_pdf(processed_files, target.output_path, paper)
        return BuildResult(target.name, rendered.success, rendered.output_path, rendered.error_message,
                           rendered.build_time_seconds, warnings=rendered.warnings, metadata=rendered.metadata)

    def _generate_build_report(self, target: BuildTarget, result: BuildResult):
        atomic_json(self.config.build_dir / "reports" / (target.name + "_build_report.json"),
                     {"target_name": target.name, "success": result.success, "output_path": str(result.output_path),
                      "source_files": list(map(str, target.source_files)), "warnings": result.warnings,
                      "metadata": result.metadata, "created_at": datetime.now(timezone.utc).isoformat()})

    def get_build_status(self) -> dict[str, Any]:
        for target in self.targets.values():
            self._sync_render_config(target)
        built = sum(target.build_status == "success" and not target.needs_rebuild for target in self.targets.values())
        succeeded = sum(result.success for result in self.build_results.values())
        return {"total_targets": len(self.targets), "built_targets": built, "successful_builds": succeeded,
                "failed_builds": len(self.build_results) - succeeded,
                "build_completion_rate": built / len(self.targets) if self.targets else 0,
                "success_rate": succeeded / len(self.build_results) if self.build_results else 0}

    def clean_build_artifacts(self, target_name: str | None = None):
        targets = [self.targets[target_name]] if target_name is not None else list(self.targets.values())
        for target in targets:
            if target.output_path.is_symlink():
                raise ValueError(f"Refusing to remove a symlink output: {target.output_path}")
            if target.output_path.exists():
                if target.last_output_hash is None or _digest(target.output_path) != target.last_output_hash:
                    raise ValueError(f"Refusing to remove an unowned or changed output: {target.output_path}")
                target.output_path.unlink()
            target.last_build_hash = target.last_output_hash = None
            target.build_status = "pending"
        # Per-build TemporaryDirectory owns cleanup; never recursively erase a caller's temp root.

    def load_build_state(self, state_file: Path):
        path = Path(state_file)
        if not path.exists():
            return
        state = json.loads(path.read_text(encoding="utf-8"))
        for name, saved in state.get("targets", {}).items():
            if name in self.targets:
                self.targets[name].dependency_files.extend(Path(value) for value in saved.get("dependency_files", [])
                    if Path(value) not in self.targets[name].dependency_files)
                self.targets[name].last_build_hash = saved.get("last_build_hash")
                self.targets[name].last_output_hash = saved.get("last_output_hash")
                self.targets[name].build_status = saved.get("build_status", "pending")

    def save_build_state(self, state_file: Path):
        atomic_json(Path(state_file), {"schema_version": 2, "targets": {
            name: {"last_build_hash": target.last_build_hash, "last_output_hash": target.last_output_hash,
                   "build_status": target.build_status, "dependency_files": list(map(str, target.dependency_files))}
            for name, target in self.targets.items()}})


def create_paper_build_target(paper_name: str, paper_dir: Path, output_dir: Path | None = None) -> BuildTarget:
    return BuildOrchestrator().create_paper_target(paper_name, paper_dir, output_dir)


def build_paper(paper_name: str, paper_dir: Path, output_dir: Path | None = None, force: bool = False) -> BuildResult:
    orchestrator = BuildOrchestrator()
    target = orchestrator.create_paper_target(paper_name, paper_dir, output_dir)
    return orchestrator.build_target(target.name, force)


def validate_build_environment() -> dict[str, Any]:
    pdf = validate_pdf_environment()
    packages = ("matplotlib", "numpy", "scipy", "pandas", "yaml", "jinja2")
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    return {"pdf_environment": pdf, "python_packages": {"available": [p for p in packages if p not in missing], "missing": missing},
            "build_tools": pdf["validation"], "environment_ready": pdf["all_valid"] and not missing}
