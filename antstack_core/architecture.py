"""Executable architecture contracts for Ant Stack modularity.

The architecture contract is data-first so it can be validated, rendered, and
consumed by tools without scraping prose. The same contract shape applies at
multiple levels: package modules, scripts, docs, tests, papers, configs, and
generated outputs.
"""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ContractIssue:
    """One architecture validation issue."""

    contract: str
    path: str
    message: str
    severity: str = "error"

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable issue record."""
        return asdict(self)


@dataclass(frozen=True)
class ModuleContract:
    """Fractal contract for one repository module, folder, or workflow layer."""

    name: str
    path: str
    layer: str
    responsibilities: tuple[str, ...]
    module_name: str | None = None
    public_exports: tuple[str, ...] = ()
    commands: tuple[str, ...] = ()
    children: tuple["ModuleContract", ...] = field(default_factory=tuple)

    def resolved_path(self, root: str | Path = ROOT) -> Path:
        """Return the filesystem path for this contract."""
        base = Path(root)
        return base if self.path == "." else base / self.path

    def iter_contracts(self) -> Iterator["ModuleContract"]:
        """Yield this contract and every nested child contract."""
        yield self
        for child in self.children:
            yield from child.iter_contracts()

    def export_names(self) -> tuple[str, ...]:
        """Return expected public exports for this contract."""
        if self.public_exports:
            return self.public_exports
        if self.module_name is None:
            return ()
        module = importlib.import_module(self.module_name)
        return tuple(getattr(module, "__all__", ()))

    def validate(self, root: str | Path = ROOT) -> tuple[ContractIssue, ...]:
        """Validate path, signposting docs, importability, and exported names."""
        issues: list[ContractIssue] = []
        target = self.resolved_path(root)
        if not target.exists():
            return (ContractIssue(self.name, self.path, f"Expected path does not exist: {target}"),)

        if target.is_dir():
            for filename in ("README.md", "AGENTS.md"):
                if not (target / filename).is_file():
                    issues.append(ContractIssue(self.name, self.path, f"Missing local {filename}"))

        if self.module_name:
            try:
                module = importlib.import_module(self.module_name)
            except Exception as exc:  # pragma: no cover - exact import failures vary by host
                issues.append(
                    ContractIssue(
                        self.name,
                        self.path,
                        f"Module {self.module_name} is not importable: {exc}",
                    )
                )
            else:
                exports = self.export_names()
                if not exports:
                    issues.append(
                        ContractIssue(
                            self.name,
                            self.path,
                            f"Module {self.module_name} does not expose __all__",
                        )
                    )
                missing = [name for name in exports if not hasattr(module, name)]
                if missing:
                    issues.append(
                        ContractIssue(
                            self.name,
                            self.path,
                            f"Missing public export(s): {', '.join(missing)}",
                        )
                    )

        for child in self.children:
            issues.extend(child.validate(root))
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable contract tree."""
        return {
            "name": self.name,
            "path": self.path,
            "layer": self.layer,
            "responsibilities": self.responsibilities,
            "module_name": self.module_name,
            "public_exports": self.public_exports,
            "commands": self.commands,
            "children": [child.to_dict() for child in self.children],
        }


@dataclass(frozen=True)
class AntStackArchitecture:
    """Top-level executable architecture model for the repository."""

    name: str
    tagline: str
    contracts: tuple[ModuleContract, ...]

    def iter_contracts(self) -> Iterator[ModuleContract]:
        """Yield every contract in deterministic traversal order."""
        for contract in self.contracts:
            yield from contract.iter_contracts()

    def find_contract(self, name_or_path: str) -> ModuleContract | None:
        """Find a contract by display name or repository path."""
        for contract in self.iter_contracts():
            if contract.name == name_or_path or contract.path == name_or_path:
                return contract
        return None

    def layer_index(self) -> Mapping[str, tuple[str, ...]]:
        """Group contract paths by architecture layer."""
        layers: dict[str, list[str]] = {}
        for contract in self.iter_contracts():
            layers.setdefault(contract.layer, []).append(contract.path)
        return {layer: tuple(paths) for layer, paths in sorted(layers.items())}

    def validate(self, root: str | Path = ROOT) -> tuple[ContractIssue, ...]:
        """Validate every nested contract and check for duplicate names."""
        issues: list[ContractIssue] = []
        seen: dict[str, str] = {}
        for contract in self.iter_contracts():
            previous = seen.get(contract.name)
            if previous is not None:
                issues.append(
                    ContractIssue(
                        contract.name,
                        contract.path,
                        f"Duplicate contract name also used by {previous}",
                    )
                )
            seen[contract.name] = contract.path
            issues.extend(contract.validate(root))
        return tuple(issues)

    def mermaid(self) -> str:
        """Render the architecture as a Mermaid flowchart body."""
        lines = ["flowchart TD"]
        for contract in self.contracts:
            self._append_mermaid_contract(lines, contract, parent=None)
        return "\n".join(lines)

    def _append_mermaid_contract(
        self,
        lines: list[str],
        contract: ModuleContract,
        *,
        parent: ModuleContract | None,
    ) -> None:
        node_id = _node_id(contract.path)
        label = f"{contract.name}\\n{contract.layer}"
        lines.append(f'    {node_id}["{label}"]')
        if parent is not None:
            lines.append(f"    {_node_id(parent.path)} --> {node_id}")
        for child in contract.children:
            self._append_mermaid_contract(lines, child, parent=contract)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable architecture tree."""
        return {
            "name": self.name,
            "tagline": self.tagline,
            "contracts": [contract.to_dict() for contract in self.contracts],
        }


def build_default_architecture() -> AntStackArchitecture:
    """Build the canonical Ant Stack architecture contract."""
    package = ModuleContract(
        name="antstack_core",
        path="antstack_core",
        layer="package kernel",
        module_name="antstack_core",
        public_exports=(
            "analysis",
            "architecture",
            "check_runtime_dependencies",
            "cohereants",
            "figures",
            "mathematics",
            "orchestration",
            "publishing",
        ),
        responsibilities=(
            "Expose the importable Python package surface.",
            "Keep scientific and publication logic out of scripts.",
            "Provide composable modules with explicit public exports.",
        ),
        commands=("uv run pytest -q tests/antstack_core", "uv run ruff check antstack_core"),
        children=(
            ModuleContract(
                name="analysis",
                path="antstack_core/analysis",
                layer="scientific APIs",
                module_name="antstack_core.analysis",
                responsibilities=(
                    "Energy, workloads, statistics, scaling, empirical reporting, and limits.",
                    "Keep numerical methods deterministic and directly testable.",
                ),
            ),
            ModuleContract(
                name="cohereants",
                path="antstack_core/cohereants",
                layer="scientific APIs",
                module_name="antstack_core.cohereants",
                responsibilities=(
                    "Spectroscopy, physical conversions, and behavioral analysis helpers.",
                    "Bridge empirical ant data with reusable analysis surfaces.",
                ),
            ),
            ModuleContract(
                name="figures",
                path="antstack_core/figures",
                layer="visualization APIs",
                module_name="antstack_core.figures",
                responsibilities=("Publication plots, Mermaid preprocessing, references, and assets.",),
            ),
            ModuleContract(
                name="orchestration",
                path="antstack_core/orchestration",
                layer="workflow kernel",
                module_name="antstack_core.orchestration",
                responsibilities=(
                    "Validated run-all config, task selection, outputs, manifests, and provenance.",
                ),
                commands=("uv run run-all-antstack --config configs/run_all_antstack.example.yaml",),
            ),
            ModuleContract(
                name="publishing",
                path="antstack_core/publishing",
                layer="publication APIs",
                module_name="antstack_core.publishing",
                responsibilities=("Paper validation, PDF generation, templates, references, and provenance.",),
            ),
            ModuleContract(
                name="cli",
                path="antstack_core/cli",
                layer="thin entrypoints",
                module_name="antstack_core.cli",
                responsibilities=("Expose installed commands while delegating work to package APIs.",),
            ),
            ModuleContract(
                name="mathematics",
                path="antstack_core/mathematics",
                layer="publishing helpers",
                module_name="antstack_core.mathematics",
                responsibilities=("Unicode math normalization and LaTeX label extraction.",),
            ),
        ),
    )

    repo_shell = ModuleContract(
        name="repo shell",
        path=".",
        layer="workspace",
        responsibilities=(
            "Coordinate package code, papers, configs, tests, docs, scripts, and outputs.",
            "Keep every intentional directory locally documented.",
        ),
        commands=("uv run pytest -q", "uv run python tools/ensure_folder_docs.py --check"),
        children=(
            ModuleContract(
                name="configs",
                path="configs",
                layer="configuration",
                responsibilities=("Validated YAML configs for package-owned workflows.",),
            ),
            ModuleContract(
                name="docs",
                path="docs",
                layer="knowledge base",
                responsibilities=("Curated guides, API contracts, validation, and reproducibility.",),
            ),
            ModuleContract(
                name="outputs",
                path="outputs",
                layer="generated artifacts",
                responsibilities=("Canonical run artifacts under outputs/<run_id>/.",),
            ),
            ModuleContract(
                name="papers",
                path="papers",
                layer="publication sources",
                responsibilities=("Manuscript sources, assets, paper configs, and projections.",),
            ),
            ModuleContract(
                name="scripts",
                path="scripts",
                layer="thin wrappers",
                responsibilities=("Operational wrappers around package and paper APIs.",),
            ),
            ModuleContract(
                name="tests",
                path="tests",
                layer="validation",
                responsibilities=("Package, CLI, docs, rendering, and workflow tests.",),
            ),
            ModuleContract(
                name="tools",
                path="tools",
                layer="maintenance",
                responsibilities=("Folder-doc checks, render helpers, and Pandoc filters.",),
            ),
        ),
    )

    return AntStackArchitecture(
        name="Ant Stack",
        tagline="A modular scientific operating system for ant-inspired analysis and publishing.",
        contracts=(repo_shell, package),
    )


def validate_architecture(root: str | Path = ROOT) -> tuple[ContractIssue, ...]:
    """Validate the default architecture contract."""
    return build_default_architecture().validate(root)


def render_architecture_mermaid() -> str:
    """Render the default architecture as Mermaid flowchart source."""
    return build_default_architecture().mermaid()


def _node_id(path: str) -> str:
    clean = "root" if path == "." else path
    return "n_" + "".join(char if char.isalnum() else "_" for char in clean)


__all__ = [
    "AntStackArchitecture",
    "ContractIssue",
    "ModuleContract",
    "build_default_architecture",
    "render_architecture_mermaid",
    "validate_architecture",
]
