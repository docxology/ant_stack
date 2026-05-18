# AGENTS.md

## Scope
This file applies to `antstack_core/orchestration`.

## Directory contract
Package-owned orchestration implementation for canonical Ant Stack generated outputs.

## Working rules
- Keep business logic here; scripts and console entrypoints should delegate.
- Validate configs before creating run artifacts.
- Record every intentional generated artifact with a checksum, producing task, source config, and provenance path.
- Preserve compatibility projections for existing paper-local complexity energetics outputs unless config disables them.

## Validation
Run:
```bash
uv run pytest -q tests/antstack_core/test_run_all_antstack.py
uv run run-all-antstack --config configs/run_all_antstack.example.yaml --validate-only
```
