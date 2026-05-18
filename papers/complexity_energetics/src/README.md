# papers/complexity_energetics/src

## Purpose
Paper-local runner implementation for complexity energetics workflows.

## Local commands
```bash
uv run pytest -q tests/complexity_energetics/test_ce.py tests/complexity_energetics/test_orchestrators.py
```

## Notes
- Keep generated files in this directory only when they are intentional, reproducible, and documented by a command above.
- Prefer package-owned logic in `antstack_core`; scripts and CLIs should stay thin wrappers.
- Update this README and the local `AGENTS.md` whenever the directory contract changes.
