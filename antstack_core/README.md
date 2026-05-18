# antstack_core

## Purpose
Installable Python package surface for analysis, executable architecture contracts, figures, orchestration, publishing helpers, CLI wrappers, cohereants helpers, and math utilities.

## Local commands
```bash
uv run pytest -q tests/antstack_core; uv run pytest -q tests/antstack_core/test_architecture_contract.py; uv run ruff check antstack_core; uv run run-all-antstack --config configs/run_all_antstack.example.yaml --validate-only
```

## Notes
- Keep generated files outside the package tree; canonical generated artifacts belong under `outputs/<run_id>/`.
- Keep nested modules represented by `antstack_core.architecture` when they become public contracts.
- Prefer package-owned logic in `antstack_core`; scripts and CLIs should stay thin wrappers.
- Update this README and the local `AGENTS.md` whenever the directory contract changes.
