# AGENTS.md

## Scope
This file applies to `scripts/complexity_energetics`.

## Directory contract
Complexity energetics analysis and figure-generation wrappers.

## Working rules
- Use `uv` for Python commands and dependency-managed execution.
- Keep changes local to this directory's responsibility unless a shared API contract requires a coordinated edit.
- Preserve generated artifacts only when they are intentional and reproducible; remove bytecode, caches, and OS metadata.
- When editing manuscript prose, keep scientific claims source-backed and soften or cite claims that are not directly supported.

## Validation
Run:
```bash
uv run antstack-ce papers/complexity_energetics/manifest.example.yaml --out papers/complexity_energetics/out
```
