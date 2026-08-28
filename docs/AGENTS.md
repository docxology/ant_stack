# AGENTS.md

## Scope
This file applies to `docs`.

## Directory contract
General project documentation, validation guidance, theory notes, and empirical ant-data integration guidance.

## Working rules
- Use `uv` for Python commands and dependency-managed execution.
- Keep changes local to this directory's responsibility unless a shared API contract requires a coordinated edit.
- Preserve generated artifacts only when they are intentional and reproducible; remove bytecode, caches, and OS metadata.
- When editing manuscript prose, keep scientific claims source-backed and soften or cite claims that are not directly supported.

## Validation
Run:
```bash
uv run pytest -q tests/antstack_core/test_docs_contract.py; uv run python tools/ensure_folder_docs.py --check
```

`ensure_folder_docs.py --check` passes: gitignored staging directories such as
`papers/*/assets/tmp_images/` are excluded via `.gitignore` patterns; see `docs/README.md`.
