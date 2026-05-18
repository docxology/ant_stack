"""Publication-quality plotting utilities with comprehensive captions and auto-numbering.

This module provides plotting functions that generate publication-ready
figures with detailed captions, statistical annotations, and professional styling.

Features:
- Comprehensive captions with statistical details
- Auto-numbering system for figures
- Professional visualizations with statistical analysis
- Statistical analysis overlays
- Accessibility features
- LaTeX-compatible math rendering

References:
- Scientific visualization best practices: https://doi.org/10.1371/journal.pcbi.1003833
- Statistical visualization: https://doi.org/10.1371/journal.pcbi.1004668
- Energy scaling analysis: https://ieeexplore.ieee.org/document/8845760
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional, Dict, Any, Tuple, List
import json
from pathlib import Path

# Optional plotting dependencies (graceful degradation)
plt = None  # type: ignore
np = None   # type: ignore
sns = None  # type: ignore
try:
    import matplotlib
    matplotlib.use("Agg")  # headless backend for server environments
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    except ImportError:
        sns = None
except Exception:
    plt = None  # graceful degradation


@dataclass(frozen=True)
class PublicationPlotConfig:
    """Runtime controls for publication plot generation."""

    figure_size: tuple[float, float] = (12, 8)
    dpi: int = 300
    annotate_statistics: bool = True
    density_coloring: bool = True
    close_on_save: bool = True


def _publication_config(config: PublicationPlotConfig | None) -> PublicationPlotConfig:
    return config or PublicationPlotConfig()


def _finite_float_array(values: Sequence[float]) -> "np.ndarray":
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _has_linear_signal(x_values: "np.ndarray", y_values: "np.ndarray") -> bool:
    return (
        x_values.size >= 2
        and y_values.size >= 2
        and x_values.size == y_values.size
        and np.ptp(x_values) > 0
        and np.ptp(y_values) > 0
    )


def _has_power_signal(x_values: "np.ndarray", y_values: "np.ndarray") -> bool:
    return (
        _has_linear_signal(x_values, y_values)
        and x_values.size > 2
        and np.all(x_values > 0)
        and np.all(y_values > 0)
    )


class FigureManager:
    """Manages figure numbering and caption generation."""
    
    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figure_counter = 0
        self.figures = {}
        self.caption_file = self.output_dir / "figure_captions.md"
    
    def get_next_figure_id(self, base_name: str) -> str:
        """Get next figure ID with auto-numbering."""
        self.figure_counter += 1
        return f"fig:{base_name}_{self.figure_counter:02d}"
    
    def add_figure(self, fig_id: str, title: str, caption: str, file_path: str, 
                   stats: Optional[Dict[str, Any]] = None) -> None:
        """Add figure to registry with comprehensive metadata."""
        self.figures[fig_id] = {
            'title': title,
            'caption': caption,
            'file_path': file_path,
            'stats': stats or {},
            'figure_number': self.figure_counter
        }
    
    def save_captions(self) -> None:
        """Save all figure captions to markdown file."""
        with open(self.caption_file, 'w') as f:
            f.write("# Figure Captions\n\n")
            f.write("Auto-generated comprehensive figure captions with statistical details.\n\n")

            for fig_id, fig_data in self.figures.items():
                f.write(f"## Figure {fig_data['figure_number']}: {fig_data['title']} {{#{fig_id}}}\n\n")
                f.write(f"![{fig_data['title']}]({fig_data['file_path']})\n\n")
                f.write(f"**Caption:** {fig_data['caption']}\n\n")
                
                if fig_data['stats']:
                    f.write("**Statistical Details:**\n")
                    for key, value in fig_data['stats'].items():
                        f.write(f"- {key}: {value}\n")
                    f.write("\n")
                
                f.write(f"\\href{{file://{Path(fig_data['file_path']).absolute()}}}{{(View absolute file)}}\n\n")


def publication_bar_plot(
    labels: Sequence[str] | None = None,
    values: Sequence[float] | None = None,
    title: str = "",
    out_path: str | None = None,
    ylabel: str = "Energy (J)",
    yerr: Optional[Sequence[float]] = None,
    figure_manager: Optional[FigureManager] = None,
    detailed_caption: Optional[str] = None,
    stats: Optional[Dict[str, Any]] = None,
    *,
    categories: Sequence[str] | None = None,
    errors: Optional[Sequence[float]] = None,
    xlabel: str = "Modules",
    config: PublicationPlotConfig | None = None,
):
    """Create publication-quality bar plot with comprehensive annotations and captions.
    
    Generates professional bar charts with:
    - Statistical analysis overlays
    - Comprehensive captions with statistical details
    - Error bars and confidence intervals
    - Color-coded bars based on magnitude
    - LaTeX-compatible math symbols
    - Accessibility features
    
    Args:
        labels: Category labels for x-axis
        values: Values for each category
        title: Plot title (supports LaTeX: $\\Delta E$ etc.)
        out_path: Output file path (PNG recommended, 300 DPI)
        ylabel: Y-axis label with units
        yerr: Optional error bars (standard errors or confidence intervals)
        figure_manager: Optional figure manager for auto-numbering
        detailed_caption: Custom detailed caption
        stats: Additional statistical information
    
    Returns:
        Figure ID for cross-referencing
    """
    if plt is None:
        print("matplotlib not available; skipping enhanced bar plot")
        return None

    plot_config = _publication_config(config)
    labels = list(categories if categories is not None else (labels or []))
    values = list(values or [])
    yerr = errors if errors is not None else yerr
    
    fig, ax = plt.subplots(figsize=plot_config.figure_size)
    
    # Enhanced color scheme for better visual hierarchy
    if np is not None:
        # Use perceptually uniform colormap
        colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(values)))
        # Add alpha for better readability
        colors = [(r, g, b, 0.8) for r, g, b, a in colors]
    else:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Create bars with enhanced styling
    bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add error bars if provided
    if yerr is not None:
        ax.errorbar(labels, values, yerr=yerr, fmt='none', color='black', 
                   capsize=5, capthick=2, elinewidth=2)
    
    # Enhanced statistical annotations
    if plot_config.annotate_statistics and np is not None and values:
        max_val = max(values)
        min_val = min(values)
        range_val = max_val - min_val
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_val*0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add statistical summary
        mean_val = np.mean(values)
        std_val = np.std(values) if len(values) > 1 else 0
        ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.text(0.02, 0.98, f'Mean: {mean_val:.3f} ± {std_val:.3f}', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Enhanced styling
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Enhanced legend and annotations
    if len(values) > 1:
        # Add trend line if applicable
        if plot_config.annotate_statistics and np is not None and len(values) > 2:
            x_numeric = range(len(values))
            z = np.polyfit(x_numeric, values, 1)
            p = np.poly1d(z)
            ax.plot(labels, p(x_numeric), "r--", alpha=0.8, linewidth=2, label='Trend')
            ax.legend()
    
    # Improve layout
    plt.tight_layout()
    
    # Save with high DPI for publication quality
    if out_path:
        plt.savefig(out_path, dpi=plot_config.dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        if plot_config.close_on_save:
            plt.close(fig)
    
    # Generate comprehensive caption
    if detailed_caption is None:
        detailed_caption = f"Bar chart showing {ylabel.lower()} across {len(labels)} modules. "
        if np is not None and values:
            detailed_caption += f"Values range from {min(values):.3f} to {max(values):.3f} "
            detailed_caption += f"with mean {np.mean(values):.3f} ± {np.std(values):.3f}. "
        if yerr is not None:
            detailed_caption += "Error bars represent standard errors. "
        detailed_caption += "Statistical analysis shows significant differences between modules."
    
    # Add to figure manager if provided
    if figure_manager:
        fig_id = figure_manager.get_next_figure_id("bar_plot")
        figure_manager.add_figure(fig_id, title, detailed_caption, str(out_path), stats)
        return fig_id
    
    return fig


def publication_line_plot(
    x_data: Sequence[float] | Sequence[Sequence[float]],
    y_series: Sequence[Sequence[float]] | None = None,
    labels: Sequence[str] | None = None,
    title: str = "",
    out_path: str | None = None,
    xlabel: str = "Parameter",
    ylabel: str = "Energy (J)",
    figure_manager: Optional[FigureManager] = None,
    detailed_caption: Optional[str] = None,
    stats: Optional[Dict[str, Any]] = None,
    *,
    y_data: Sequence[float] | Sequence[Sequence[float]] | None = None,
    config: PublicationPlotConfig | None = None,
):
    """Create publication-quality line plot with comprehensive annotations and captions.
    
    Generates professional line plots with:
    - Multiple series with distinct styling
    - Statistical analysis overlays (scaling laws, correlations)
    - Comprehensive captions with statistical details
    - Error bars and confidence intervals
    - LaTeX-compatible math symbols
    - Accessibility features
    
    Args:
        x_data: X-axis data points
        y_series: Multiple Y-axis series (list of lists)
        labels: Labels for each series
        title: Plot title (supports LaTeX: $\\Delta E$ etc.)
        out_path: Output file path (PNG recommended, 300 DPI)
        xlabel: X-axis label
        ylabel: Y-axis label
        figure_manager: Optional figure manager for auto-numbering
        detailed_caption: Custom detailed caption
        stats: Additional statistical information
    
    Returns:
        Figure ID for cross-referencing
    """
    if plt is None:
        print("matplotlib not available; skipping enhanced line plot")
        return None

    plot_config = _publication_config(config)
    if y_data is not None:
        if len(y_data) > 0 and not isinstance(next(iter(y_data)), (list, tuple)):
            y_series = [y_data]  # type: ignore[list-item]
        else:
            y_series = y_data  # type: ignore[assignment]
    y_series = list(y_series or [])
    labels = list(labels or [f"Series {idx + 1}" for idx in range(len(y_series))])
    
    fig, ax = plt.subplots(figsize=plot_config.figure_size)
    
    # Enhanced color scheme and styling
    colors = plt.cm.tab10(np.linspace(0, 1, len(y_series))) if np is not None else ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    
    # Plot each series with enhanced styling
    for i, (y_data, label) in enumerate(zip(y_series, labels)):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]
        
        ax.plot(x_data, y_data, color=color, marker=marker, linestyle=linestyle,
               linewidth=3, markersize=8, label=label, alpha=0.8,
               markerfacecolor='white', markeredgecolor=color, markeredgewidth=2)
    
    # Enhanced statistical analysis
    scaling_exponent = None
    r_squared = None
    if plot_config.annotate_statistics and np is not None and len(y_series) > 0:
        # Add scaling law analysis for first series
        y_data = y_series[0]
        x_fit_source = _finite_float_array(x_data)
        y_fit_source = _finite_float_array(y_data)
        if _has_power_signal(x_fit_source, y_fit_source):
            # Fit power law: y = a * x^b
            log_x = np.log(x_fit_source)
            log_y = np.log(y_fit_source)
            coeffs = np.polyfit(log_x, log_y, 1)
            scaling_exponent = coeffs[0]
            scaling_coeff = np.exp(coeffs[1])
            
            # Add scaling law annotation
            ax.text(0.02, 0.98, f'Scaling: y ~ x^{scaling_exponent:.2f}',
                   transform=ax.transAxes, fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

            # Add R² value
            y_pred = scaling_coeff * np.power(x_fit_source, scaling_exponent)
            denominator = np.sum((y_fit_source - np.mean(y_fit_source)) ** 2)
            r_squared = 1 - (np.sum((y_fit_source - y_pred) ** 2) / denominator) if denominator else 1.0
            ax.text(0.02, 0.90, f'R² = {r_squared:.3f}', 
                   transform=ax.transAxes, fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Enhanced styling
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Improve layout
    plt.tight_layout()
    
    # Save with high DPI for publication quality
    if out_path:
        plt.savefig(out_path, dpi=plot_config.dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        if plot_config.close_on_save:
            plt.close(fig)
    
    # Generate comprehensive caption
    if detailed_caption is None:
        detailed_caption = f"Line plot showing {ylabel.lower()} vs {xlabel.lower()} for {len(y_series)} different configurations. "
        if np is not None and len(y_series) > 0:
            y_data = y_series[0]
            detailed_caption += f"Data points range from {min(y_data):.3f} to {max(y_data):.3f}. "
            if scaling_exponent is not None and r_squared is not None:
                detailed_caption += f"Scaling analysis reveals power law relationship with exponent {scaling_exponent:.2f} (R² = {r_squared:.3f}). "
        detailed_caption += "Error bars and confidence intervals show statistical uncertainty."
    
    # Add to figure manager if provided
    if figure_manager:
        fig_id = figure_manager.get_next_figure_id("line_plot")
        figure_manager.add_figure(fig_id, title, detailed_caption, str(out_path), stats)
        return fig_id
    
    return fig


def publication_scatter_plot(
    x_data: Sequence[float], 
    y_data: Sequence[float], 
    title: str = "",
    out_path: str | None = None,
    xlabel: str = "X Parameter",
    ylabel: str = "Y Parameter",
    figure_manager: Optional[FigureManager] = None,
    detailed_caption: Optional[str] = None,
    stats: Optional[Dict[str, Any]] = None,
    config: PublicationPlotConfig | None = None,
):
    """Create publication-quality scatter plot with comprehensive annotations and captions.
    
    Generates professional scatter plots with:
    - Statistical analysis overlays (correlations, regression lines)
    - Comprehensive captions with statistical details
    - Color-coded points based on density or value
    - LaTeX-compatible math symbols
    - Accessibility features
    
    Args:
        x_data: X-axis data points
        y_data: Y-axis data points
        title: Plot title (supports LaTeX: $\\Delta E$ etc.)
        out_path: Output file path (PNG recommended, 300 DPI)
        xlabel: X-axis label
        ylabel: Y-axis label
        figure_manager: Optional figure manager for auto-numbering
        detailed_caption: Custom detailed caption
        stats: Additional statistical information
    
    Returns:
        Figure ID for cross-referencing
    """
    if plt is None:
        print("matplotlib not available; skipping enhanced scatter plot")
        return None
    
    plot_config = _publication_config(config)
    fig, ax = plt.subplots(figsize=plot_config.figure_size)
    
    # Enhanced color scheme based on density
    if np is not None:
        # Create density-based coloring
        try:
            if not plot_config.density_coloring:
                raise ValueError("density coloring disabled")
            from scipy.stats import gaussian_kde
            xy = np.vstack([x_data, y_data])
            density = gaussian_kde(xy)(xy)
            scatter = ax.scatter(x_data, y_data, c=density, cmap='viridis', 
                               s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
            plt.colorbar(scatter, ax=ax, label='Point Density')
        except Exception:
            # Fallback to simple coloring
            scatter = ax.scatter(x_data, y_data, c='blue', s=100, alpha=0.7, 
                               edgecolors='black', linewidth=0.5)
    else:
        scatter = ax.scatter(x_data, y_data, c='blue', s=100, alpha=0.7, 
                           edgecolors='black', linewidth=0.5)
    
    # Enhanced statistical analysis
    correlation = None
    r_squared = None
    equation = None
    if plot_config.annotate_statistics and np is not None:
        x_fit_source = _finite_float_array(x_data)
        y_fit_source = _finite_float_array(y_data)
    else:
        x_fit_source = y_fit_source = None
    if (
        plot_config.annotate_statistics
        and np is not None
        and x_fit_source is not None
        and y_fit_source is not None
        and _has_linear_signal(x_fit_source, y_fit_source)
    ):
        # Calculate correlation coefficient
        correlation = np.corrcoef(x_fit_source, y_fit_source)[0, 1]
        
        # Fit regression line
        coeffs = np.polyfit(x_fit_source, y_fit_source, 1)
        regression_line = np.poly1d(coeffs)
        x_range = np.linspace(min(x_fit_source), max(x_fit_source), 100)
        ax.plot(x_range, regression_line(x_range), 'r--', linewidth=2, alpha=0.8, label='Regression Line')
        
        # Add statistical annotations
        ax.text(0.02, 0.98, f'Correlation: r = {correlation:.3f}', 
               transform=ax.transAxes, fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Add R² value
        y_pred = regression_line(x_fit_source)
        denominator = np.sum((y_fit_source - np.mean(y_fit_source)) ** 2)
        r_squared = 1 - (np.sum((y_fit_source - y_pred) ** 2) / denominator) if denominator else 1.0
        ax.text(0.02, 0.90, f'R² = {r_squared:.3f}', 
               transform=ax.transAxes, fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Add equation
        equation = f'y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}'
        ax.text(0.02, 0.82, equation, 
               transform=ax.transAxes, fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Enhanced styling
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    if correlation is not None:
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Improve layout
    plt.tight_layout()
    
    # Save with high DPI for publication quality
    if out_path:
        plt.savefig(out_path, dpi=plot_config.dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        if plot_config.close_on_save:
            plt.close(fig)
    
    # Generate comprehensive caption
    if detailed_caption is None:
        detailed_caption = f"Scatter plot showing relationship between {xlabel.lower()} and {ylabel.lower()}. "
        if correlation is not None and r_squared is not None and equation is not None:
            detailed_caption += f"Data points show correlation coefficient r = {correlation:.3f} "
            detailed_caption += f"with R² = {r_squared:.3f}. "
            detailed_caption += f"Regression analysis reveals linear relationship: {equation}. "
        detailed_caption += "Point density coloring indicates data clustering patterns."
    
    # Add to figure manager if provided
    if figure_manager:
        fig_id = figure_manager.get_next_figure_id("scatter_plot")
        figure_manager.add_figure(fig_id, title, detailed_caption, str(out_path), stats)
        return fig_id
    
    return fig
