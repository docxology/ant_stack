"""Reproducibility metadata and trusted-exclusive-output custody helpers.

These checks reject stable links and accidental overlap, not hostile same-UID
replacement races. Callers must exclusively own output namespaces throughout a run.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from antstack_core import __version__

DEFAULT_DEPENDENCIES = ("jinja2", "matplotlib", "numpy", "pandas", "pillow", "psutil", "pyyaml", "scipy", "seaborn")


def ordinary_path(path: str | Path, *, tree: bool = False) -> Path:
    """Reject symlinks/special nodes without resolving away evidence of a link."""
    candidate = Path(os.path.abspath(Path(path).expanduser()))
    for node in (*reversed(candidate.parents), candidate):
        try:
            mode = node.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise ValueError(f"Output/input path is not ordinary: {node}")
        if node != candidate and not stat.S_ISDIR(mode):
            raise ValueError(f"Path ancestor is not a directory: {node}")
    if tree and candidate.is_dir():
        for root, directories, files in os.walk(candidate, followlinks=False):
            for name in directories + files:
                ordinary_path(Path(root) / name)
    return candidate


def file_identity(path: str | Path) -> dict[str, Any]:
    source = ordinary_path(path)
    if not source.is_file():
        raise ValueError(f"Required ordinary file is absent: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": source.stat().st_size}


def publication_path(path: str | Path, *, tree: bool = False) -> Path:
    """Protect implementation, Git and runtime namespaces before any tree scan."""
    candidate = ordinary_path(path)
    package = Path(__file__).resolve().parents[1]
    package_identity = package.stat()
    for node in (candidate, *candidate.parents):
        try:
            if os.path.samestat(node.stat(), package_identity):
                raise ValueError(f"Publication cannot overwrite the generating package: {candidate}")
        except FileNotFoundError:
            pass
    for node in (candidate, *candidate.parents):
        if (node.name.casefold() == ".git" or
                (node / "pyvenv.cfg").is_file() or
                node.name.casefold() == "node_modules" or
                ((node / "HEAD").is_file() and (node / "objects").is_dir())):
            raise ValueError(f"Publication cannot use a Git/runtime namespace: {candidate}")
    if tree:
        if (candidate == Path(candidate.anchor) or candidate == Path.home() or
                (candidate.exists() and any(candidate.samefile(parent) for parent in package.parents)) or
                (candidate / ".git").exists()):
            raise ValueError(f"Publication cannot replace a project/control root: {candidate}")
        if candidate.is_dir():
            for root, directories, files in os.walk(candidate, followlinks=False):
                if any(name.casefold() in (".git", "pyvenv.cfg", "node_modules") for name in directories + files):
                    raise ValueError(f"Publication tree contains a Git/runtime namespace: {root}")
        return ordinary_path(candidate, tree=True)
    return candidate


def publish_file(source: str | Path, destination: str | Path) -> Path:
    """Atomically replace one ordinary file without broadening its prior mode."""
    source, destination = ordinary_path(source), publication_path(destination)
    if not source.is_file():
        raise ValueError(f"Missing produced file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file():
            raise ValueError(f"File publication collides with a directory: {destination}")
        os.chmod(source, stat.S_IMODE(destination.stat().st_mode))
    try:
        os.replace(source, destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        staged = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.",
                                             suffix=".tmp", delete=False) as handle:
                staged = Path(handle.name)
                with source.open("rb") as incoming:
                    shutil.copyfileobj(incoming, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(staged, stat.S_IMODE(source.stat().st_mode))
            os.replace(staged, destination)
        finally:
            if staged is not None:
                staged.unlink(missing_ok=True)
    return destination


def preserve_tree_modes(staged: str | Path, previous: str | Path) -> None:
    """Retain existing path permissions on corresponding new-generation nodes."""
    staged, previous = ordinary_path(staged, tree=True), ordinary_path(previous, tree=True)
    if not previous.exists():
        return
    # Bottom-up: applying a restrictive parent mode must not prevent its scan.
    for root, _directories, files in os.walk(staged, topdown=False, followlinks=False):
        for node in [*(Path(root) / name for name in files), Path(root)]:
            old = previous / node.relative_to(staged)
            if old.exists() and stat.S_IFMT(old.stat().st_mode) == stat.S_IFMT(node.stat().st_mode):
                os.chmod(node, stat.S_IMODE(old.stat().st_mode))


def atomic_text(path: str | Path, payload: str) -> Path:
    """Publish complete text atomically without following links or broadening file modes."""
    destination = publication_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        publish_file(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    return atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


@dataclass(frozen=True)
class ProvenanceRecord:
    """Input hashes and distinct working-project/package implementation identities."""
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
    input_identities: dict[str, dict[str, Any]] = field(default_factory=dict)
    source_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_dependency_versions(package_names: Iterable[str] = DEFAULT_DEPENDENCIES) -> dict[str, str]:
    versions = {}
    for name in package_names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def _git_snapshot(cwd: str | Path) -> dict[str, Any]:
    root = Path(cwd).absolute()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True, timeout=10).strip()
        status = subprocess.check_output(["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
                                         cwd=root, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        top = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=root,
                                          stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True, timeout=10).strip())
        changes = []
        entries = iter(status.split(b"\0"))
        for entry in entries:
            if not entry:
                continue
            code, name = entry[:2].decode(), os.fsdecode(entry[3:])
            record = {"status": code, "path": name}
            if "R" in code or "C" in code:
                record["original_path"] = os.fsdecode(next(entries))
            source = top / name
            if source.is_file() and not source.is_symlink():
                record.update(file_identity(source))
            changes.append(record)
        return {"root": str(top), "commit": commit, "dirty": bool(status), "changes": changes}
    except (OSError, ValueError, StopIteration, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {"root": str(root), "commit": None, "dirty": None, "changes": None}


def detect_git_state(cwd: str | Path = ".") -> tuple[str | None, bool | None]:
    state = _git_snapshot(cwd)
    return state["commit"], state["dirty"]


def capture_source_state(cwd: str | Path = ".") -> dict[str, Any]:
    """Capture before generation; package file hashes work for installed distributions too."""
    package = Path(__file__).resolve().parents[1]
    implementation = {str(path.relative_to(package)): file_identity(path)
                      for path in sorted(package.rglob("*.py")) if not path.is_symlink()}
    checkout = _git_snapshot(package)
    if Path(checkout["root"]) != package.parent:
        checkout = {"root": str(package), "commit": None, "dirty": None, "changes": None}
    return {"working_project": _git_snapshot(cwd),
            "generating_package": {"version": __version__, "root": str(package),
                                   "files": implementation, "source_checkout": checkout}}


def build_run_provenance(*, project: str = "ant-stack", command: Sequence[str] | None = None,
                         input_paths: Iterable[str | Path] = (), output_paths: Iterable[str | Path] = (),
                         parameters: Mapping[str, Any] | None = None, cwd: str | Path = ".",
                         dependency_names: Iterable[str] = DEFAULT_DEPENDENCIES,
                         created_at_utc: str | None = None,
                         source_state: Mapping[str, Any] | None = None) -> ProvenanceRecord:
    inputs = [str(Path(path)) for path in input_paths]
    state = dict(source_state) if source_state is not None else capture_source_state(cwd)
    working = state["working_project"]
    identities = {path: file_identity(path) for path in inputs}
    return ProvenanceRecord(project, __version__, created_at_utc or datetime.now(timezone.utc).isoformat(),
                            list(command) if command is not None else [], inputs,
                            [str(Path(path)) for path in output_paths], dict(parameters or {}),
                            working["commit"], working["dirty"],
                            dependencies=collect_dependency_versions(dependency_names),
                            input_identities=identities, source_state=state)


def write_provenance(path: str | Path, record: ProvenanceRecord) -> Path:
    return atomic_json(path, record.to_dict())


__all__ = ["DEFAULT_DEPENDENCIES", "ProvenanceRecord", "build_run_provenance",
           "collect_dependency_versions", "detect_git_state", "write_provenance"]
