# tests/core_rendering

Tests for rendering, figure generation, paper configuration, and core package integration.

## Local Commands

```bash
uv run pytest -q tests/core_rendering
uv run antstack-build --validate-only
```

## Coverage Areas

- Bar, line, and scatter plot generation.
- Energy and scaling helper integration.
- Paper configuration loading.
- Analysis-to-visualization workflows.
- Rendering readiness without requiring a full PDF build.
