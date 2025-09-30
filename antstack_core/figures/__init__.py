"""Figure management and visualization for scientific publications.

This module provides comprehensive figure handling including:
- Cross-reference validation and management
- Mermaid diagram preprocessing and rendering
- Publication-quality plotting with consistent styling
- Asset management and file organization

Following .cursorrules specifications for:
- Proper figure format: ## Figure: Title {#fig:id}
- LaTeX macro usage for math symbols
- Descriptive captions with ** emphasis
- Pandoc-compatible cross-reference format
"""

from .plots import bar_plot, line_plot, scatter_plot
from .publication_plots import (
    FigureManager, publication_bar_plot, publication_line_plot, publication_scatter_plot
)
from .mermaid import preprocess_mermaid_diagrams, validate_mermaid_syntax
from .references import validate_cross_references, fix_figure_ids
from .assets import organize_figure_assets, copy_figure_files

__all__ = [
    "bar_plot",
    "line_plot", 
    "scatter_plot",
    "FigureManager",
    "publication_bar_plot",
    "publication_line_plot", 
    "publication_scatter_plot",
    "preprocess_mermaid_diagrams",
    "validate_mermaid_syntax",
    "validate_cross_references",
    "fix_figure_ids",
    "organize_figure_assets",
    "copy_figure_files"
]
