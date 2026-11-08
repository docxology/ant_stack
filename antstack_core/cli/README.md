# antstack_core/cli

## Purpose
Thin console-script entrypoints that delegate to installed package-owned build and analysis APIs.

## Local commands
```bash
uv run antstack-build --validate-only; uv run antstack-ce papers/complexity_energetics/manifest.example.yaml --validate-only; uv run run-all-antstack --config configs/run_all_antstack.example.yaml --validate-only
```

## Notes
- Current installed commands are `antstack-build`, `antstack-ce`, and `run-all-antstack`.
- Keep implementation logic out of CLI wrappers; the builder and complexity energetics runner ship in the Python distribution. Paper inputs and `tools/render_pdf.sh` belong to the selected working project.
- Update this README and the local `AGENTS.md` whenever the directory contract changes.
