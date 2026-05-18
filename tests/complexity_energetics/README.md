# tests/complexity_energetics

Tests for the complexity energetics runner, analysis workflows, generated key numbers, and paper-local compatibility outputs.

## Local Commands

```bash
uv run pytest -q tests/complexity_energetics
uv run antstack-ce papers/complexity_energetics/manifest.example.yaml --out papers/complexity_energetics/out
```

## Coverage Areas

- Energy estimation and workload scaling.
- Statistical summaries and theoretical limits.
- Contact dynamics, neural network, and active-inference orchestrators.
- Generated key-number loading and placeholder replacement.
- Compatibility with `papers/complexity_energetics/out`, `assets`, and `Generated.md`.

Canonical run-all behavior is tested under `tests/antstack_core/test_run_all_antstack.py`.
