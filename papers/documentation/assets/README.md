# papers/documentation/assets

## Purpose
Assets and render intermediates for the documentation paper.

## Local commands
```bash
uv run antstack-build --validate-only
```

## Notes
- Keep generated files in this directory only when they are intentional, reproducible, and documented by a command above.
- Prefer package-owned logic in `antstack_core`; scripts and CLIs should stay thin wrappers.
- Update this README and the local `AGENTS.md` whenever the directory contract changes.
