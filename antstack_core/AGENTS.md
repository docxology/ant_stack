# AGENTS.md

## Scope
This file applies to `antstack_core`.

## Directory contract
Installable Python package surface for analysis, executable architecture contracts, figures, orchestration, publishing helpers, CLI wrappers, cohereants helpers, and math utilities.

## Working rules
- Use `uv` for Python commands and dependency-managed execution.
- Keep changes local to this directory's responsibility unless a shared API contract requires a coordinated edit.
- Preserve generated artifacts only when they are intentional and reproducible; remove bytecode, caches, and OS metadata.
- When editing manuscript prose, keep scientific claims source-backed and soften or cite claims that are not directly supported.

## Validation
Run:
```bash
uv run pytest -q tests/antstack_core; uv run pytest -q tests/antstack_core/test_architecture_contract.py; uv run ruff check antstack_core; uv run run-all-antstack --config configs/run_all_antstack.example.yaml --validate-only
```
