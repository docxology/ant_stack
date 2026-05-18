# outputs

## Purpose
Default canonical root for generated Ant Stack run artifacts.

## Local commands
```bash
uv run run-all-antstack --config configs/run_all_antstack.example.yaml
```

## Notes
- Each `run-all-antstack` run writes to `outputs/<run_id>/` by default.
- Expected run subfolders are `data/raw`, `data/derived`, `statistics`, `visualizations/static`, `visualizations/animations`, `reports`, `papers`, `logs`, and `provenance`.
- Each complete run includes a root `manifest.json` and provenance files under `provenance/`.
- Keep this directory for intentional, reproducible outputs only; remove transient caches and bytecode.
