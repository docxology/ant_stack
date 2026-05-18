# antstack_core/publishing

## Purpose
Build orchestration, PDF generation, quality validation, reference management, and template helpers.

## Local commands
```bash
uv run antstack-build --validate-only; uv run pytest -q tests/core_rendering
```

## Notes
- Keep generated files in this directory only when they are intentional, reproducible, and documented by a command above.
- Prefer package-owned logic in `antstack_core`; scripts and CLIs should stay thin wrappers.
- Update this README and the local `AGENTS.md` whenever the directory contract changes.
