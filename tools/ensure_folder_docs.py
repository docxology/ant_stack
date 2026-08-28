"""Ensure intentional repository directories have README.md and AGENTS.md files."""

from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DIRECTORIES: dict[str, dict[str, str]] = {
    ".": {
        "role": "Repository root for the Ant Stack Python package, paper sources, canonical outputs, compatibility paper artifacts, validation scripts, and development tooling.",
        "commands": "uv sync --extra dev; bun install; uv run pytest -q; uv run ruff check .; uv run python tools/ensure_folder_docs.py --check",
    },
    "antstack_core": {
        "role": "Installable Python package surface for analysis, executable architecture contracts, figures, orchestration, publishing helpers, CLI wrappers, cohereants helpers, and math utilities.",
        "commands": "uv run pytest -q tests/antstack_core; uv run pytest -q tests/antstack_core/test_architecture_contract.py; uv run ruff check antstack_core; uv run run-all-antstack --config configs/run_all_antstack.example.yaml --validate-only",
    },
    "antstack_core/analysis": {
        "role": "Energy, workload, complexity, statistics, experiment configuration, key-number, power-meter, and theoretical-limit APIs.",
        "commands": "uv run pytest -q tests/antstack_core/test_analysis_energy.py tests/antstack_core/test_statistical_analysis.py tests/antstack_core/test_theoretical_limits.py",
    },
    "antstack_core/cli": {
        "role": "Thin console-script entrypoints that delegate to package or paper-runner APIs.",
        "commands": "uv run antstack-build --validate-only; uv run antstack-ce papers/complexity_energetics/manifest.example.yaml --out papers/complexity_energetics/out; uv run run-all-antstack --config configs/run_all_antstack.example.yaml --validate-only",
    },
    "antstack_core/cohereants": {
        "role": "Packaged Cohereants scientific helper APIs for spectral, behavioral, and core physical calculations.",
        "commands": "uv run pytest -q tests/antstack_core/test_cohereants_core.py tests/antstack_core/test_cohereants_behavioral.py tests/antstack_core/test_spectroscopy.py",
    },
    "antstack_core/figures": {
        "role": "Matplotlib, Mermaid, asset-management, and figure-reference utilities for publication rendering.",
        "commands": "uv run pytest -q tests/antstack_core/test_figures_plots.py tests/antstack_core/test_visualization.py",
    },
    "antstack_core/mathematics": {
        "role": "Small publishing-safe math helpers, including Unicode math normalization and LaTeX label extraction.",
        "commands": "uv run pytest -q tests/antstack_core/test_core_package.py",
    },
    "antstack_core/orchestration": {
        "role": "Package-owned orchestration for canonical Ant Stack data, statistics, visualizations, animations, reports, paper projections, logs, manifests, and provenance.",
        "commands": "uv run pytest -q tests/antstack_core/test_run_all_antstack.py; uv run run-all-antstack --config configs/run_all_antstack.example.yaml --validate-only",
    },
    "antstack_core/publishing": {
        "role": "Build orchestration, PDF generation, quality validation, reference management, and template helpers.",
        "commands": "uv run antstack-build --validate-only; uv run pytest -q tests/core_rendering",
    },
    "antstack_core/publishing/templates": {
        "role": "Reusable publication templates consumed by publishing helpers and paper builds.",
        "commands": "uv run antstack-build --validate-only",
    },
    "complexity_energetics": {
        "role": "Legacy top-level complexity energetics generated figures and tables retained for comparison and provenance.",
        "commands": "uv run pytest -q tests/complexity_energetics",
    },
    "complexity_energetics/assets": {
        "role": "Versioned generated image artifacts for the legacy complexity energetics layout.",
        "commands": "uv run antstack-ce papers/complexity_energetics/manifest.example.yaml --out papers/complexity_energetics/out",
    },
    "complexity_energetics/generated_content": {
        "role": "Versioned generated tables, key numbers, and TeX fragments for the legacy complexity energetics layout.",
        "commands": "uv run antstack-ce papers/complexity_energetics/manifest.example.yaml --out papers/complexity_energetics/out",
    },
    "configs": {
        "role": "Validated YAML configuration files for package-owned Ant Stack workflows.",
        "commands": "uv run run-all-antstack --config configs/run_all_antstack.example.yaml --validate-only",
    },
    "docs": {
        "role": "General project documentation, validation guidance, theory notes, and empirical ant-data integration guidance.",
        "commands": "uv run pytest -q tests/antstack_core/test_docs_contract.py; uv run python tools/ensure_folder_docs.py --check",
    },
    "examples": {
        "role": "Small examples demonstrating supported paper syntax and rendering conventions.",
        "commands": "uv run antstack-build --validate-only",
    },
    "papers": {
        "role": "Paper source roots, paper configs, manuscript sections, references, assets, and generated outputs.",
        "commands": "uv run antstack-build --validate-only",
    },
    "papers/ant_stack": {
        "role": "Ant Stack framework manuscript sources, assets, config, and retained rendered PDF.",
        "commands": "uv run antstack-build --validate-only",
    },
    "papers/ant_stack/assets": {
        "role": "Local assets and render intermediates for the Ant Stack manuscript.",
        "commands": "uv run antstack-build --validate-only",
    },
    "papers/ant_stack/assets/figures": {
        "role": "Figure assets for the Ant Stack manuscript.",
        "commands": "uv run antstack-build --validate-only",
    },
    "papers/ant_stack/assets/mermaid": {
        "role": "Rendered Mermaid diagram assets for the Ant Stack manuscript.",
        "commands": "uv run antstack-build --validate-only",
    },
    "papers/ant_stack/assets/tmp_images": {
        "role": "Temporary image staging area used during Ant Stack manuscript rendering.",
        "commands": "uv run antstack-build --validate-only",
    },
    "papers/complexity_energetics": {
        "role": "Complexity energetics manuscript, manifest, runner sources, generated content, outputs, and references.",
        "commands": "uv run antstack-ce papers/complexity_energetics/manifest.example.yaml --out papers/complexity_energetics/out",
    },
    "papers/complexity_energetics/assets": {
        "role": "Paper-local complexity energetics assets and render intermediates.",
        "commands": "uv run antstack-ce papers/complexity_energetics/manifest.example.yaml --out papers/complexity_energetics/out",
    },
    "papers/complexity_energetics/assets/figures": {
        "role": "Paper-local generated figure images for complexity energetics.",
        "commands": "uv run antstack-ce papers/complexity_energetics/manifest.example.yaml --out papers/complexity_energetics/out",
    },
    "papers/complexity_energetics/assets/mermaid": {
        "role": "Rendered Mermaid diagrams for the complexity energetics manuscript.",
        "commands": "uv run antstack-build --validate-only",
    },
    "papers/complexity_energetics/assets/tmp_images": {
        "role": "Temporary image staging area used during complexity energetics rendering.",
        "commands": "uv run antstack-build --validate-only",
    },
    "papers/complexity_energetics/generated_content": {
        "role": "Generated manuscript fragments and key-number files for complexity energetics.",
        "commands": "uv run antstack-ce papers/complexity_energetics/manifest.example.yaml --out papers/complexity_energetics/out",
    },
    "papers/complexity_energetics/out": {
        "role": "Compatibility output directory for complexity energetics paper-runner artifacts.",
        "commands": "uv run antstack-ce papers/complexity_energetics/manifest.example.yaml --out papers/complexity_energetics/out",
    },
    "papers/complexity_energetics/src": {
        "role": "Paper-local runner implementation for complexity energetics workflows.",
        "commands": "uv run pytest -q tests/complexity_energetics/test_ce.py tests/complexity_energetics/test_orchestrators.py",
    },
    "outputs": {
        "role": "Default canonical root for generated Ant Stack run artifacts and provenance.",
        "commands": "uv run run-all-antstack --config configs/run_all_antstack.example.yaml",
    },
    "papers/documentation": {
        "role": "Documentation-oriented paper configuration and PDF rendering guide.",
        "commands": "uv run antstack-build --validate-only",
    },
    "papers/documentation/assets": {
        "role": "Assets and render intermediates for the documentation paper.",
        "commands": "uv run antstack-build --validate-only",
    },
    "papers/documentation/assets/figures": {
        "role": "Figure assets for the documentation paper.",
        "commands": "uv run antstack-build --validate-only",
    },
    "papers/documentation/assets/mermaid": {
        "role": "Rendered Mermaid diagrams for the documentation paper.",
        "commands": "uv run antstack-build --validate-only",
    },
    "papers/documentation/assets/tmp_images": {
        "role": "Temporary image staging area used during documentation rendering.",
        "commands": "uv run antstack-build --validate-only",
    },
    "scripts": {
        "role": "Thin operational wrappers around antstack_core and paper-runner APIs.",
        "commands": "uv run antstack-build --validate-only; uv run run-all-antstack --config configs/run_all_antstack.example.yaml --validate-only; uv run pytest -q tests/complexity_energetics/test_orchestrators.py",
    },
    "scripts/ant_stack": {
        "role": "Ant Stack manuscript script namespace retained for paper-specific wrapper scripts.",
        "commands": "uv run antstack-build --validate-only",
    },
    "scripts/common_pipeline": {
        "role": "Shared validation, build, tracing, formatting, and key-number update wrappers.",
        "commands": "uv run antstack-build --validate-only",
    },
    "scripts/complexity_energetics": {
        "role": "Complexity energetics analysis and figure-generation wrappers.",
        "commands": "uv run antstack-ce papers/complexity_energetics/manifest.example.yaml --out papers/complexity_energetics/out",
    },
    "tests": {
        "role": "Pytest suite for package APIs, complexity energetics workflows, and core rendering behavior.",
        "commands": "uv run pytest --collect-only -q; uv run pytest -q",
    },
    "tests/antstack_core": {
        "role": "Unit and integration tests for antstack_core package modules.",
        "commands": "uv run pytest -q tests/antstack_core",
    },
    "tests/complexity_energetics": {
        "role": "Tests for complexity energetics runners, orchestrators, generated key numbers, and integration workflows.",
        "commands": "uv run pytest -q tests/complexity_energetics",
    },
    "tests/core_rendering": {
        "role": "Tests for rendering and core refactor compatibility contracts.",
        "commands": "uv run pytest -q tests/core_rendering",
    },
    "tools": {
        "role": "Repository maintenance tools, shell helpers, and Pandoc filters.",
        "commands": "uv run python tools/ensure_folder_docs.py --check",
    },
    "tools/filters": {
        "role": "Pandoc Lua filters for cross-references, code links, and Unicode-to-TeX normalization.",
        "commands": "uv run antstack-build --validate-only",
    },
}


README_TEMPLATE = """# {title}

## Purpose
{role}

## Local commands
```bash
{commands}
```

## Notes
- Keep generated files in this directory only when they are intentional, reproducible, and documented by a command above.
- Prefer package-owned logic in `antstack_core`; scripts and CLIs should stay thin wrappers.
- Update this README and the local `AGENTS.md` whenever the directory contract changes.
"""


AGENTS_TEMPLATE = """# AGENTS.md

## Scope
This file applies to `{path}`.

## Directory contract
{role}

## Working rules
- Use `uv` for Python commands and dependency-managed execution.
- Keep changes local to this directory's responsibility unless a shared API contract requires a coordinated edit.
- Preserve generated artifacts only when they are intentional and reproducible; remove bytecode, caches, and OS metadata.
- When editing manuscript prose, keep scientific claims source-backed and soften or cite claims that are not directly supported.

## Validation
Run:
```bash
{commands}
```
"""


def title_for(path: str) -> str:
    return "Ant Stack" if path == "." else path


def render_readme(path: str, meta: dict[str, str]) -> str:
    return README_TEMPLATE.format(title=title_for(path), role=meta["role"], commands=meta["commands"])


def render_agents(path: str, meta: dict[str, str]) -> str:
    return AGENTS_TEMPLATE.format(path=path, role=meta["role"], commands=meta["commands"])


def _gitignored_directory_patterns() -> list[str]:
    """Directory ignore patterns from the repo .gitignore (line comments only)."""
    patterns: list[str] = []
    gitignore = ROOT / ".gitignore"
    if not gitignore.is_file():
        return patterns
    for line in gitignore.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        patterns.append(stripped.rstrip("/"))
    return patterns


def _is_ignored_directory(rel: str, patterns: list[str]) -> bool:
    """True when a .gitignore pattern excludes this directory path."""
    for pattern in patterns:
        if "/" in pattern:
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(rel, pattern + "/*"):
                return True
        elif fnmatch.fnmatch(rel.split("/")[-1], pattern):
            return True
    return False


def missing_docs() -> list[Path]:
    missing: list[Path] = []
    ignore_patterns = _gitignored_directory_patterns()
    for rel in DIRECTORIES:
        if _is_ignored_directory(rel, ignore_patterns):
            continue
        directory = ROOT / rel
        if not directory.is_dir():
            missing.append(directory)
            continue
        for filename in ("README.md", "AGENTS.md"):
            candidate = directory / filename
            if not candidate.is_file():
                missing.append(candidate)
    return missing


def write_missing() -> list[Path]:
    written: list[Path] = []
    for rel, meta in DIRECTORIES.items():
        directory = ROOT / rel
        directory.mkdir(parents=True, exist_ok=True)
        readme = directory / "README.md"
        agents = directory / "AGENTS.md"
        if not readme.exists():
            readme.write_text(render_readme(rel, meta), encoding="utf-8")
            written.append(readme)
        if not agents.exists():
            agents.write_text(render_agents(rel, meta), encoding="utf-8")
            written.append(agents)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Create missing README.md and AGENTS.md files.")
    parser.add_argument("--check", action="store_true", help="Fail if any intentional directory lacks docs.")
    args = parser.parse_args()

    if args.write:
        written = write_missing()
        for path in written:
            print(path.relative_to(ROOT))

    missing = missing_docs()
    if args.check or not args.write:
        if missing:
            print("Missing directory documentation:")
            for path in missing:
                print(path.relative_to(ROOT))
            return 1
        print("All intentional directories have README.md and AGENTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
