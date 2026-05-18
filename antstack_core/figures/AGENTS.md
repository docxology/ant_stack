# AGENTS.md

## Scope
This file applies to `antstack_core/figures`.

## Directory contract
Matplotlib, Mermaid, asset-management, and figure-reference utilities for publication rendering.

## Working rules
- Use `uv` for Python commands and dependency-managed execution.
- Keep changes local to this directory's responsibility unless a shared API contract requires a coordinated edit.
- Preserve generated artifacts only when they are intentional and reproducible; remove bytecode, caches, and OS metadata.
- When editing manuscript prose, keep scientific claims source-backed and soften or cite claims that are not directly supported.

## Validation
Run:
```bash
uv run pytest -q tests/antstack_core/test_figures_plots.py tests/antstack_core/test_visualization.py
```
