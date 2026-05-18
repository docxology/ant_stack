"""Provenance helpers for generated research artifacts."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from antstack_core import __version__


DEFAULT_DEPENDENCIES = (
    "jinja2",
    "matplotlib",
    "numpy",
    "pandas",
    "pyyaml",
    "scipy",
)


@dataclass(frozen=True)
class ProvenanceRecord:
    """Serializable metadata describing how a generated artifact was produced."""

    project: str
    package_version: str
    created_at_utc: str
    command: list[str]
    input_paths: list[str]
    output_paths: list[str]
    parameters: dict[str, Any] = field(default_factory=dict)
    git_commit: str | None = None
    git_dirty: bool | None = None
    python_version: str = field(default_factory=platform.python_version)
    platform: str = field(default_factory=platform.platform)
    dependencies: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable provenance dictionary."""
        return asdict(self)


def _stringify_paths(paths: Iterable[str | Path]) -> list[str]:
    """Convert paths to stable strings without requiring that they exist."""
    return [str(Path(path)) for path in paths]


def collect_dependency_versions(package_names: Iterable[str] = DEFAULT_DEPENDENCIES) -> dict[str, str]:
    """Return installed versions for importable/distribution package names."""
    versions: dict[str, str] = {}
    for package_name in package_names:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def detect_git_state(cwd: str | Path = ".") -> tuple[str | None, bool | None]:
    """Return the current git commit hash and dirty-state flag for ``cwd``."""
    root = Path(cwd)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None, None

    dirty = subprocess.run(
        ["git", "diff", "--quiet"],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode != 0
    return commit, dirty


def build_run_provenance(
    *,
    project: str = "ant-stack",
    command: Sequence[str] | None = None,
    input_paths: Iterable[str | Path] = (),
    output_paths: Iterable[str | Path] = (),
    parameters: Mapping[str, Any] | None = None,
    cwd: str | Path = ".",
    dependency_names: Iterable[str] = DEFAULT_DEPENDENCIES,
    created_at_utc: str | None = None,
) -> ProvenanceRecord:
    """Create a provenance record for a command, script, or package workflow."""
    git_commit, git_dirty = detect_git_state(cwd)
    timestamp = created_at_utc or datetime.now(timezone.utc).isoformat()
    return ProvenanceRecord(
        project=project,
        package_version=__version__,
        created_at_utc=timestamp,
        command=list(command if command is not None else sys.argv),
        input_paths=_stringify_paths(input_paths),
        output_paths=_stringify_paths(output_paths),
        parameters=dict(parameters or {}),
        git_commit=git_commit,
        git_dirty=git_dirty,
        dependencies=collect_dependency_versions(dependency_names),
    )


def write_provenance(path: str | Path, record: ProvenanceRecord) -> Path:
    """Write a provenance record as indented JSON and return the output path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return output_path


__all__ = [
    "DEFAULT_DEPENDENCIES",
    "ProvenanceRecord",
    "build_run_provenance",
    "collect_dependency_versions",
    "detect_git_state",
    "write_provenance",
]
