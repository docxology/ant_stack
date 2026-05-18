# antstack_core/orchestration

## Purpose
Package-owned run orchestration for canonical Ant Stack data, statistics, visualization, animation, report, paper, logging, and provenance outputs.

## Local commands
```bash
uv run pytest -q tests/antstack_core/test_run_all_antstack.py
uv run run-all-antstack --config configs/run_all_antstack.example.yaml --validate-only
```

## Notes
- Public APIs are exported from `antstack_core.orchestration`.
- `run_all` owns orchestration, validation, manifests, checksums, and provenance; `scripts/run_all_antstack.py` and `run-all-antstack` are thin wrappers.
- Complexity energetics scientific parameters are loaded through the existing `ExperimentManifest`; this package adds only run-wide orchestration fields.
