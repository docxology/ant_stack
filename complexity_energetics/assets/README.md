# complexity_energetics/assets

## Purpose
Versioned generated image artifacts for the legacy complexity energetics layout.

## Local commands
```bash
uv run antstack-ce papers/complexity_energetics/manifest.example.yaml --out papers/complexity_energetics/out
```

## Notes
- Keep generated files in this directory only when they are intentional, reproducible, and documented by a command above.
- Prefer package-owned logic in `antstack_core`; scripts and CLIs should stay thin wrappers.
- Update this README and the local `AGENTS.md` whenever the directory contract changes.
