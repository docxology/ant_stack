# AGENTS.md

## Scope
This file applies to `scripts`.

## Directory contract
Thin operational wrappers around antstack_core and paper-runner APIs.

## Working rules
- Use `uv` for Python commands and dependency-managed execution.
- Keep changes local to this directory's responsibility unless a shared API contract requires a coordinated edit.
- Preserve generated artifacts only when they are intentional and reproducible; remove bytecode, caches, and OS metadata.
- When editing manuscript prose, keep scientific claims source-backed and soften or cite claims that are not directly supported.

## Validation
Run:
```bash
uv run antstack-build --validate-only; uv run run-all-antstack --config configs/run_all_antstack.example.yaml --validate-only; uv run pytest -q tests/complexity_energetics/test_orchestrators.py
```
