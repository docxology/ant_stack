# papers/complexity_energetics/out

## Purpose
Compatibility projection and retained historical CE results, not a fresh standalone generation destination.

## Local commands
```bash
uv run antstack-ce papers/complexity_energetics/manifest.example.yaml --validate-only
```

## Notes
- Keep generated files in this directory only when they are intentional, reproducible, and documented by a command above.
- Prefer package-owned logic in `antstack_core`; scripts and CLIs should stay thin wrappers.
- Update this README and the local `AGENTS.md` whenever the directory contract changes.
- Use `antstack-ce MANIFEST --out NEW_EMPTY_DIRECTORY` for a new standalone family, or the configured canonical run with an owned paper projection. Existing retained bytes do not establish current completion.
