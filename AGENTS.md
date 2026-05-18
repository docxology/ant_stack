# AGENTS.md

## Scope
This file applies to `.`.

## Directory contract
Repository root for the Ant Stack Python package, paper sources, canonical outputs, compatibility paper artifacts, validation scripts, and development tooling.

## Working rules
- If `skills/PAI/SKILL.md` is present in this checkout, read it before large architecture or documentation changes.
- Use `uv` for Python commands and dependency-managed execution.
- Keep changes local to this directory's responsibility unless a shared API contract requires a coordinated edit.
- Preserve generated artifacts only when they are intentional and reproducible; remove bytecode, caches, and OS metadata.
- When editing manuscript prose, keep scientific claims source-backed and soften or cite claims that are not directly supported.

## Validation
Run:
```bash
uv sync --extra dev; bun install; uv run pytest -q; uv run ruff check .; uv run python tools/ensure_folder_docs.py --check
```
