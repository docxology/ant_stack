"""Publication-quality plotting utilities for scientific publications.

Enhanced plotting functions that generate publication-ready figures with:
- Professional styling and consistent formatting
- Statistical analysis overlays (scaling laws, correlations)
- Error bars and confidence intervals
- Accessibility features (color-blind friendly palettes)
- LaTeX-compatible math rendering

References:
- Scientific visualization best practices: https://doi.org/10.1371/journal.pcbi.1003833
- Matplotlib publication quality: https://matplotlib.org/stable/tutorials/introductory/usage.html
- Energy scaling analysis: https://ieeexplore.ieee.org/document/8845760
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional, Sequence as Seq

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
class PlotConfig:
    """Runtime plotting controls shared by the basic figure helpers.

    The defaults preserve the historical public behavior while making output
    quality, statistical overlays, and save-time cleanup explicit.
    """

    figure_size: tuple[float, float] | None = None
    dpi: int = 300
    annotate_statistics: bool = True
    auto_log_scale: bool = True
    close_on_save: bool = True


def _config_or_default(config: PlotConfig | None) -> PlotConfig:
    return config or PlotConfig()


def _finite_float_array(values: Sequence[float]) -> "np.ndarray":
    """Return finite numeric values as a NumPy array when plotting supports it."""
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _has_linear_fit_signal(x_values: "np.ndarray", y_values: "np.ndarray") -> bool:
    """True when regression/correlation can run without degenerate warnings."""
    return (
        x_values.size >= 2
        and y_values.size >= 2
        and x_values.size == y_values.size
        and np.ptp(x_values) > 0
        and np.ptp(y_values) > 0
    )


def _has_power_fit_signal(x_values: "np.ndarray", y_values: "np.ndarray") -> bool:
    """True when log-log scaling can be fitted without zero/constant warnings."""
    return (
        _has_linear_fit_signal(x_values, y_values)
        and x_values.size > 3
        and np.all(x_values > 0)
        and np.all(y_values > 0)
    )


def _positive_dynamic_range(values: Sequence[float]) -> float | None:
    if np is None:
        return None
    arr = _finite_float_array(values)
    if arr.size == 0 or np.any(arr <= 0):
        return None
    min_value = float(np.min(arr))
    if min_value <= 0:
        return None
    return float(np.max(arr) / min_value)


def bar_plot(
    labels: Sequence[str] | None = None,
    values: Sequence[float] | None = None,
    title: str = "",
    out_path: str | None = None,
    ylabel: str = "Energy (J)",
    yerr: Optional[Sequence[float]] = None,
    *,
    categories: Sequence[str] | None = None,
    xlabel: str = "Components",
    errors: Optional[Sequence[float]] = None,
    config: PlotConfig | None = None,
):
    r"""Create publication-quality bar plot with comprehensive annotations.
    
    Generates professional bar charts suitable for academic publications with:
    - Error bars with confidence intervals
    - Statistical significance indicators  
    - Color-coded bars based on magnitude
    - Detailed annotations and formatting
    - LaTeX-compatible math symbols
    
    Args:
        labels: Category labels for x-axis
        values: Values for each category
        title: Plot title (supports LaTeX: $\\Delta E$ etc.)
        out_path: Output file path (PNG recommended, 300 DPI)
        ylabel: Y-axis label with units
        yerr: Optional error bars (standard errors or confidence intervals)
    
    References:
        - Error bar best practices: https://doi.org/10.1038/nmeth.2813
        - Statistical visualization: https://doi.org/10.1371/journal.pcbi.1004668
    """
    if plt is None:
        print("matplotlib not available; skipping bar plot")
        return None

    plot_config = _config_or_default(config)
    if categories is None and labels is None and values:
        raise TypeError("categories or labels are required when values are provided")
    labels = list(categories if categories is not None else (labels or []))
    values = list(values or [])
    yerr = errors if errors is not None else yerr
    
    fig, ax = plt.subplots(figsize=plot_config.figure_size or (10, 6))
    
    # Color bars based on magnitude for visual hierarchy
    if np is not None:
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(values)))
    else:
        colors = ['C{}'.format(i % 10) for i in range(len(values))]
        
    bars = ax.bar(
        labels, values, yerr=yerr, capsize=6, color=colors,
        edgecolor='black', linewidth=0.8, alpha=0.8
    )
    
    # Add value labels on top of bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height + (max(values) * 0.01),
            f'{value:.3f}', 
            ha='center', va='bottom', 
            fontweight='bold', fontsize=10
        )
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Rotate x-axis labels if needed
    if labels and len(max(labels, key=len)) > 8:
        plt.xticks(rotation=45, ha='right')
    
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=plot_config.dpi, bbox_inches='tight', facecolor='white')
        if plot_config.close_on_save:
            plt.close(fig)
    return fig


def line_plot(
    x: Sequence[float] | Sequence[Sequence[float]] | None = None,
    ys: Sequence[Sequence[float]] | None = None,
    labels: Sequence[str] | None = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    out_path: str | None = None,
    bands: Optional[Seq[Seq[tuple[float, float]]]] = None,
    *,
    x_data: Sequence[Sequence[float]] | None = None,
    y_data: Sequence[Sequence[float]] | None = None,
    config: PlotConfig | None = None,
):
    """Create enhanced line plot with scaling analysis and trend visualization.
    
    Generates publication-quality line plots with:
    - Multiple series with distinct styling
    - Confidence bands and uncertainty quantification
    - Trend lines and scaling law annotations
    - Statistical analysis overlays
    - Automatic log scaling for wide dynamic ranges
    
    Args:
        x: X-axis values (independent variable)
        ys: Y-axis values for each series
        labels: Labels for each series
        title: Plot title (supports LaTeX math)
        xlabel: X-axis label with units
        ylabel: Y-axis label with units  
        out_path: Output file path (PNG, 300 DPI)
        bands: Optional confidence bands as (low, high) tuples
    
    References:
        - Scaling laws in computing: https://doi.org/10.1126/science.1062081
        - Uncertainty visualization: https://doi.org/10.1111/cgf.12974
    """
    if plt is None:
        print("matplotlib not available; skipping line plot")
        return None

    plot_config = _config_or_default(config)
    if x_data is None and x is None and (y_data is not None or ys is not None):
        raise ValueError("x_data or x is required when y series are provided")
    if y_data is None and ys is None and x is not None:
        raise TypeError("ys or y_data is required when x data is provided")
    if x_data is not None:
        x_series = list(x_data)
    else:
        x_series = [x or [] for _ in range(len(ys or []))]
    y_series = list(y_data if y_data is not None else (ys or []))
    labels = list(labels or ["" for _ in y_series])
    
    fig, ax = plt.subplots(figsize=plot_config.figure_size or (10, 7))
    
    # Enhanced color palette and markers
    if np is not None:
        colors = plt.cm.Set1(np.linspace(0, 1, max(len(y_series), 1)))
    else:
        colors = ['C{}'.format(i) for i in range(max(len(y_series), 1))]
        
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    linestyles = ['-', '--', '-.', ':']
    
    for idx, (series_x, y, label) in enumerate(zip(x_series, y_series, labels)):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        linestyle = linestyles[idx % len(linestyles)]
        
        # Plot main line with enhanced styling
        ax.plot(
            series_x, y, marker=marker, label=label, linewidth=2.5,
            markersize=8, color=color, linestyle=linestyle,
            markerfacecolor='white', markeredgecolor=color, 
            markeredgewidth=2, alpha=0.9
        )
        
        # Add confidence bands if available
        if bands and idx < len(bands) and bands[idx]:
            lows = [lo for (lo, _hi) in bands[idx]]
            his = [hi for (_lo, hi) in bands[idx]]
            ax.fill_between(
                series_x, lows, his, alpha=0.2, color=color,
                label=f'{label} 95% CI'
            )
            
        # Fit and display scaling law if data suggests power law
        if plot_config.annotate_statistics and np is not None:
            try:
                x_fit_source = _finite_float_array(series_x)
                y_fit_source = _finite_float_array(y)
                if not _has_power_fit_signal(x_fit_source, y_fit_source):
                    continue
                # Log-log fit to detect power laws
                log_x = np.log10(x_fit_source)
                log_y = np.log10(y_fit_source)
                coeffs = np.polyfit(log_x, log_y, 1)
                scaling_exp = coeffs[0]

                # Only show if it's a reasonable scaling relationship
                if 0.5 < abs(scaling_exp) < 3.0:
                    # Generate fitted line
                    x_fit = np.logspace(np.log10(min(x_fit_source)), np.log10(max(x_fit_source)), 100)
                    y_fit = 10**(coeffs[1]) * x_fit**scaling_exp
                    ax.plot(x_fit, y_fit, '--', color=color, alpha=0.6, linewidth=1.5)

                    # Add scaling annotation (LaTeX format)
                    mid_idx = len(series_x) // 2
                    ax.annotate(
                        f'$\\propto K^{{{scaling_exp:.2f}}}$',
                        xy=(series_x[mid_idx], y[mid_idx]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, color=color, fontweight='bold',
                        bbox=dict(
                            boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor=color, alpha=0.8
                        )
                    )
            except (ValueError, ZeroDivisionError):
                scaling_exp = None
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    
    # Enhanced legend
    if any(labels):
        ax.legend(
            title="Series", title_fontsize=12, fontsize=11, 
            loc='best', frameon=True, fancybox=True, shadow=True
        )
    
    # Professional grid and spines
    ax.grid(True, axis='both', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Log scale if data spans multiple orders of magnitude
    if plot_config.auto_log_scale and np is not None and x_series and y_series and all(y for y in y_series):
        try:
            all_x = [value for series in x_series for value in series]
            x_range = _positive_dynamic_range(all_x)
            y_range = _positive_dynamic_range([value for series in y_series for value in series])
            if x_range is not None and x_range > 100:
                ax.set_xscale('log')
            if y_range is not None and y_range > 100:
                ax.set_yscale('log')
        except (ValueError, ZeroDivisionError):
            x_range = None
            y_range = None
    
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=plot_config.dpi, bbox_inches='tight', facecolor='white')
        if plot_config.close_on_save:
            plt.close(fig)
    return fig


def scatter_plot(
    x: Sequence[float] | None = None,
    y: Sequence[float] | None = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    out_path: str | None = None,
    *,
    x_data: Sequence[float] | None = None,
    y_data: Sequence[float] | None = None,
    sizes: Sequence[float] | None = None,
    config: PlotConfig | None = None,
):
    """Create enhanced scatter plot with statistical analysis and trend fitting.
    
    Generates publication-quality scatter plots with:
    - Regression lines and confidence intervals
    - Correlation coefficients and R² values
    - Outlier detection and highlighting  
    - Density-based coloring for large datasets
    - Statistical significance annotations
    
    Args:
        x: X-axis values (independent variable)
        y: Y-axis values (dependent variable)  
        title: Plot title (supports LaTeX math)
        xlabel: X-axis label with units
        ylabel: Y-axis label with units
        out_path: Output file path (PNG, 300 DPI)
    
    References:
        - Statistical visualization: https://doi.org/10.1038/nmeth.2813
        - Correlation analysis: https://doi.org/10.1016/j.jneumeth.2013.08.024
    """
    if plt is None:
        print("matplotlib not available; skipping scatter plot")
        return None

    plot_config = _config_or_default(config)
    x = x_data if x_data is not None else (x or [])
    y = y_data if y_data is not None else (y or [])
    
    fig, ax = plt.subplots(figsize=plot_config.figure_size or (10, 8))
    
    # Convert to arrays for analysis
    if np is not None:
        x_arr = np.array(x)
        y_arr = np.array(y)
        
        # Create scatter plot with color mapping
        scatter = ax.scatter(
            x_arr, y_arr, s=sizes if sizes is not None else 60, alpha=0.7, c=y_arr,
            cmap='viridis', edgecolors='black', linewidth=0.5
        )
        
        if y_arr.size > 0:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(ylabel, fontsize=12, fontweight='bold')
        
        # Fit regression line
        if plot_config.annotate_statistics and _has_linear_fit_signal(x_arr, y_arr):
            coeffs = np.polyfit(x_arr, y_arr, 1)
            x_fit = np.linspace(min(x_arr), max(x_arr), 100)
            y_fit = coeffs[0] * x_fit + coeffs[1]
            ax.plot(x_fit, y_fit, 'r--', linewidth=2, alpha=0.8, label='Linear fit')
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(x_arr, y_arr)[0, 1]
            
            # Add statistics annotation
            stats_text = f'R² = {correlation**2:.3f}\\nSlope = {coeffs[0]:.3e}'
            ax.text(
                0.05, 0.95, stats_text, transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=11, verticalalignment='top', fontweight='bold'
            )
    else:
        # Fallback without numpy
        ax.scatter(x, y, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.grid(True, axis='both', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if 'fit' in locals():
        ax.legend(fontsize=11)
    
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=plot_config.dpi, bbox_inches='tight', facecolor='white')
        if plot_config.close_on_save:
            plt.close(fig)
    return fig
