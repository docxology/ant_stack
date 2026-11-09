# Canonical Run Outputs and Reproducibility Evidence

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

## Qualified 3.0 Analytical Capsules

- [CI capsule `ci-42548062547-1`](ci-42548062547-1/manifest.json): source `422c71e5a402d31e7b2a2676423a2651277509ce`, all seven configured tasks and 52 declared artifacts.
- [Native capsule `release-v3-0-0-20261109`](release-v3-0-0-20261109/manifest.json): the independently qualified native source `47fc8bcbbb2fbc99e46828101c1eb9a09a820e65`. Its analytical CSV was compared with the CI capsule across all 300 rows, with zero observed numeric differences.

Each capsule remains immutable and retains its own source/runtime provenance. A capsule's explicitly labelled copied historical manuscript is not a new full-paper render. Current reviewed full manuscripts and their input/resource identities are identified separately by [manuscript provenance](../papers/manuscripts.provenance.json); deterministic analytical replay is not empirical replication or byte-identical PDF/plot output.

