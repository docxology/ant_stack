# tests/antstack_core

Unit and integration tests for the installable `antstack_core` package, CLI wrappers, documentation contracts, and canonical output orchestration.

## Local Commands

```bash
uv run pytest -q tests/antstack_core
uv run pytest -q tests/antstack_core/test_docs_contract.py
uv run pytest -q tests/antstack_core/test_public_contracts.py tests/antstack_core/test_cli_entrypoints.py
uv run pytest -q tests/antstack_core/test_run_all_antstack.py
```

## Coverage Areas

- `analysis`: energy models, workloads, statistics, manifests, power meters, key numbers, empirical reports, and theoretical limits.
- `cohereants`: physical helpers, spectroscopy, CHC analysis, behavioral datasets, response statistics, and plots.
- `figures`: plotting, publication figures, Mermaid preprocessing, cross-reference checks, and asset organization.
- `publishing`: build helpers, provenance, templates, validation, and references.
- `orchestration`: run-all config validation, output layout, manifests, checksums, provenance, and paper projection.
- `cli`: installed command wrappers for `antstack-build`, `antstack-ce`, and `run-all-antstack`.
- `docs`: curated guide Mermaid coverage and folder-level signposting.

## Test Principles

- Exercise real package methods and CLI contracts.
- Keep generated artifacts in temporary directories unless a workflow test explicitly targets versioned outputs.
- Assert manifests, checksums, and provenance when validating output generation.
- Keep the default `uv run pytest -q` suite free of skipped tests and runtime-warning summaries.
