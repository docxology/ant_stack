# AGENTS.md

## Scope
This file applies to `outputs`.

## Directory contract
Default canonical root for generated Ant Stack run artifacts and provenance.

## Working rules
- Use `uv run run-all-antstack --config configs/run_all_antstack.example.yaml` to regenerate canonical outputs.
- Keep generated runs organized under `outputs/<run_id>/`.
- Preserve generated files only when they are intentional, reproducible, and listed by a run manifest.
- Do not place caches, bytecode, virtual environments, or ad hoc scratch files here.

## Validation
Run:
```bash
uv run run-all-antstack --config configs/run_all_antstack.example.yaml --validate-only
```
