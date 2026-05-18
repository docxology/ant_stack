# AGENTS.md

## Scope
This file applies to `papers/documentation/assets`.

## Directory contract
Assets and render intermediates for the documentation paper.

## Working rules
- Use `uv` for Python commands and dependency-managed execution.
- Keep changes local to this directory's responsibility unless a shared API contract requires a coordinated edit.
- Preserve generated artifacts only when they are intentional and reproducible; remove bytecode, caches, and OS metadata.
- When editing manuscript prose, keep scientific claims source-backed and soften or cite claims that are not directly supported.

## Validation
Run:
```bash
uv run antstack-build --validate-only
```
