# antstack_core/mathematics

## Purpose
Small publishing-safe math helpers, including Unicode math normalization and LaTeX label extraction.

## Local commands
```bash
uv run pytest -q tests/antstack_core/test_core_package.py
```

## Notes
- Keep generated files in this directory only when they are intentional, reproducible, and documented by a command above.
- Prefer package-owned logic in `antstack_core`; scripts and CLIs should stay thin wrappers.
- Update this README and the local `AGENTS.md` whenever the directory contract changes.
