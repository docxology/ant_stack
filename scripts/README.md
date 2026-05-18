# Scripts

This directory contains thin operational wrappers for Ant Stack paper builds, validation, tracing, key-number updates, run-all orchestration, and complexity energetics analyses. Package-owned logic belongs in `antstack_core`; scripts should delegate to package APIs or paper-local runner APIs.

## Layout

| Path | Purpose |
| --- | --- |
| `common_pipeline/` | Shared build, validation, tracing, formatting, and key-number wrapper scripts. |
| `complexity_energetics/` | Complexity energetics analysis and figure-generation wrappers. |
| `ant_stack/` | Namespace for Ant Stack manuscript-specific wrappers. |
| `run_all_antstack.py` | Thin script wrapper over `antstack_core.orchestration.run_all`. |
| `validate_rendering_system.py` | Rendering-system validation entrypoint. |
| `validate_unified_config.py` | Unified configuration validation helper. |
| `demonstrate_improvements.py` | Demonstration script for package-level improvements. |

## Commands

```bash
uv run antstack-build --validate-only
uv run antstack-ce papers/complexity_energetics/manifest.example.yaml --out papers/complexity_energetics/out
uv run run-all-antstack --config configs/run_all_antstack.example.yaml --validate-only
uv run pytest -q tests/complexity_energetics/test_orchestrators.py
```

Use `uv run python <script>` only when there is no console entrypoint for the workflow. Prefer adding a package-backed CLI in `antstack_core/cli` over adding business logic directly in this directory.

## Generated Artifacts

Scripts may write intentional figures, tables, and key-number files only to documented output roots. Canonical run artifacts belong under `outputs/<run_id>/`; paper compatibility artifacts belong under documented paper-local roots. Scripts must not create or retain `__pycache__`, local caches, or OS metadata.
