# antstack_core/cli

## Purpose
Thin console-script entrypoints that delegate to package or paper-runner APIs.

## Local commands
```bash
uv run antstack-build --validate-only; uv run antstack-ce papers/complexity_energetics/manifest.example.yaml --out papers/complexity_energetics/out; uv run run-all-antstack --config configs/run_all_antstack.example.yaml --validate-only
```

## Notes
- Current installed commands are `antstack-build`, `antstack-ce`, and `run-all-antstack`.
- Keep implementation logic out of CLI wrappers; wrappers should delegate to package APIs or the paper runner.
- Update this README and the local `AGENTS.md` whenever the directory contract changes.
