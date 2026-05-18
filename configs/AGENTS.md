# AGENTS.md

## Scope
This file applies to `configs`.

## Directory contract
Validated orchestration config files for canonical Ant Stack output generation.

## Working rules
- Use `uv` for Python commands and dependency-managed execution.
- Keep scientific workload schemas in `ExperimentManifest` files; do not duplicate those contracts here.
- Keep example configs deterministic and runnable from a clean checkout.
- Update this file and `README.md` when the run-all config schema changes.

## Validation
Run:
```bash
uv run run-all-antstack --config configs/run_all_antstack.example.yaml --validate-only
```
