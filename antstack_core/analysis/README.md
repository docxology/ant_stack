# antstack_core/analysis

## Purpose
Energy, workload, complexity, statistics, experiment configuration, key-number, power-meter, and theoretical-limit APIs.

## Local commands
```bash
uv run pytest -q tests/antstack_core/test_analysis_energy.py tests/antstack_core/test_statistical_analysis.py tests/antstack_core/test_theoretical_limits.py
```

## Notes
- Keep generated files in this directory only when they are intentional, reproducible, and documented by a command above.
- Prefer package-owned logic in `antstack_core`; scripts and CLIs should stay thin wrappers.
- Update this README and the local `AGENTS.md` whenever the directory contract changes.
