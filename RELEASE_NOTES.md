# Ant Stack v2.0.0 Release Notes

Release date: 2026-05-18

Ant Stack v2.0.0 is a major repository restructure and validation release. It turns the project into a modern `uv` Python package with `bun`-managed Mermaid tooling, package-owned orchestration, canonical generated outputs, and executable architecture contracts.

## Highlights

- Rebuilt the repository around installable `antstack_core` package modules for analysis, cohereants, figures, mathematics, orchestration, publishing, and CLI entrypoints.
- Added `run-all-antstack`, a package-owned orchestrator that validates one YAML config and generates canonical data, statistics, visualizations, animations, reports, paper projections, logs, manifests, and provenance under `outputs/<run_id>/`.
- Added current console commands: `antstack-build`, `antstack-ce`, and `run-all-antstack`.
- Added executable architecture contracts through `antstack_core.architecture`, including module/folder validation, public export checks, Mermaid rendering, and tests against placeholder method bodies.
- Added real Cohereants helper APIs for physical conversions, spectroscopy, behavioral analysis, statistics, power analysis, and plotting hooks.
- Added folder-level `README.md` and `AGENTS.md` coverage for every intentional directory and a checker at `tools/ensure_folder_docs.py`.
- Added Mermaid diagrams and corrected command contracts across the curated documentation set.
- Preserved compatibility outputs for the complexity energetics paper while establishing canonical outputs under `outputs/<run_id>/`.
- Removed tracked caches, Python bytecode, OS metadata, and the old `package-lock.json`; added `bun.lock`, `uv.lock`, and project-level ignore policy.

## Validation

The release was validated with:

```bash
uv sync --extra dev
bun install
uv run pytest --collect-only -q
uv run pytest -q
uv run ruff check .
uv run python tools/ensure_folder_docs.py --check
uv run antstack-build --validate-only
uv run run-all-antstack --config configs/run_all_antstack.example.yaml --validate-only
bun run validate
bun run test:collect
bun run lint
```

Verified suite result:

```text
676 passed, 10 subtests passed
```

## Compatibility Notes

- New canonical generated artifacts should go under `outputs/<run_id>/`.
- Complexity energetics compatibility artifacts still project to `papers/complexity_energetics/out`, `papers/complexity_energetics/assets`, and `papers/complexity_energetics/Generated.md` unless disabled in config.
- Scripts and CLIs should remain thin wrappers over package APIs.
- Public package contracts are enforced through `__all__` exports and pytest contract tests.
