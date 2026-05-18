# complexity_energetics

## Purpose
Legacy top-level complexity energetics generated figures and tables retained for comparison and provenance.

## Local commands
```bash
uv run pytest -q tests/complexity_energetics
```

## Notes
- Keep generated files in this directory only when they are intentional, reproducible, and documented by a command above.
- Prefer `outputs/<run_id>/` for new canonical generated artifacts.
- Prefer package-owned logic in `antstack_core`; scripts and CLIs should stay thin wrappers.
- Update this README and the local `AGENTS.md` whenever the directory contract changes.
