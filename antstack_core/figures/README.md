# antstack_core/figures

## Purpose
Configurable Matplotlib, Mermaid, asset-management, and figure-reference utilities for publication rendering.

## Local commands
```bash
uv run pytest -q tests/antstack_core/test_figures_plots.py tests/antstack_core/test_visualization.py
```

## Public Surface
- `PlotConfig` controls basic plot figure size, DPI, statistical annotations, automatic log scaling, and save-time cleanup.
- `PublicationPlotConfig` controls publication plot figure size, DPI, statistical annotations, density coloring, and save-time cleanup.
- Basic helpers: `bar_plot`, `line_plot`, `scatter_plot`.
- Publication helpers: `FigureManager`, `publication_bar_plot`, `publication_line_plot`, `publication_scatter_plot`.
- Document helpers: Mermaid preprocessing, figure ID validation, cross-reference repair, and asset copying.

## Numerical Edge Cases
- Empty agent populations and single-point plots are valid inputs and should not emit NumPy, SciPy, or Matplotlib warnings.
- Regression and correlation overlays are skipped when data have fewer than two informative points, zero variance, or non-finite values.
- Log-log scaling overlays require positive, finite, non-constant x and y values.
- Tests close Matplotlib figures after each case to keep visualization workflows deterministic in long suites.

## Notes
- Keep generated files in this directory only when they are intentional, reproducible, and documented by a command above.
- Prefer package-owned logic in `antstack_core`; scripts and CLIs should stay thin wrappers.
- Update this README and the local `AGENTS.md` whenever the directory contract changes.
