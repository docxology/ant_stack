# configs

## Purpose
Validated YAML configuration files for package-owned Ant Stack workflows.

## Local commands
```bash
uv run run-all-antstack --config configs/run_all_antstack.example.yaml --validate-only
uv run run-all-antstack --config configs/run_all_antstack.example.yaml
```

## Notes
- `run_all_antstack.example.yaml` is the canonical single-file orchestration config for generating data, statistics, visualizations, animations, reports, paper projections, logs, and provenance under `outputs/<run_id>/`.
- Keep scientific workload parameters in the complexity energetics `ExperimentManifest`; this directory only owns orchestration-level options.
- Add new config files only when they are reproducible and documented by a command above.
