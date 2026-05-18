# antstack_core/cohereants

## Purpose
Packaged Cohereants scientific helper APIs for spectral, behavioral, and core physical calculations.

## Local commands
```bash
uv run pytest -q tests/antstack_core/test_cohereants_core.py tests/antstack_core/test_cohereants_behavioral.py tests/antstack_core/test_spectroscopy.py
```

## Notes
- Keep generated files in this directory only when they are intentional, reproducible, and documented by a command above.
- Prefer package-owned logic in `antstack_core`; scripts and CLIs should stay thin wrappers.
- Update this README and the local `AGENTS.md` whenever the directory contract changes.
