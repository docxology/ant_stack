# papers/complexity_energetics/out

## Purpose
Canonical output directory for complexity energetics runner artifacts.

## Local commands
```bash
uv run antstack-ce papers/complexity_energetics/manifest.example.yaml --out papers/complexity_energetics/out
```

## Notes
- Keep generated files in this directory only when they are intentional, reproducible, and documented by a command above.
- Prefer package-owned logic in `antstack_core`; scripts and CLIs should stay thin wrappers.
- Update this README and the local `AGENTS.md` whenever the directory contract changes.
- `summary.json` and `provenance.json` record the manifest path, command, generated outputs, dependency versions, Python version, and git state for each regenerated run.
