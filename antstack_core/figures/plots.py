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
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    try:
        import seaborn as sns
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


def _paired_float_arrays(*values: Sequence[float]):
    """Keep aligned, finite observations; never pair independently filtered data."""
    arrays = tuple(np.asarray(value, dtype=float) for value in values)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("Plot data must be one-dimensional")
    if arrays and any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("Paired plot data must have equal lengths")
    mask = np.logical_and.reduce([np.isfinite(array) for array in arrays])
    return tuple(array[mask] for array in arrays), mask


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


def _save_figure(figure, output_path, *, dpi=300, close=True):
    """Publish only a complete image; keep the prior output on producer failure."""
    from pathlib import Path
    import tempfile
    from ..publishing.provenance import publication_path, publish_file
    destination = publication_path(output_path)
    if not destination.suffix:
        from matplotlib import rcParams
        destination = destination.with_suffix("." + rcParams["savefig.format"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=".antstack-figure-", dir=destination.parent) as directory:
            staged = Path(directory) / destination.name
            figure.savefig(staged, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
            publish_file(staged, destination)
    finally:
        if close:
            plt.close(figure)


def interval_plot(
    labels: Sequence[str], values: Sequence[float], title: str = "",
    out_path: str | None = None, xlabel: str = "Energy (J)", *,
    confidence_intervals: Sequence[tuple[float, float]] | None = None,
    config: PlotConfig | None = None,
):
    """Categorical estimates with supplied bounds, not zero-baseline log bars.

    Positive observations spanning over two decades use an explicitly labelled
    logarithmic axis. Zero/negative values or bounds retain a linear axis.
    Bounds are shown as supplied even when they exclude the point estimate.
    """
    if plt is None or np is None:
        raise ImportError("interval_plot requires matplotlib and numpy")
    labels, numeric = list(labels), np.asarray(values, dtype=float)
    if numeric.ndim != 1 or not numeric.size or len(labels) != numeric.size or not np.all(np.isfinite(numeric)):
        raise ValueError("Interval plot labels and finite values need equal nonzero lengths")
    bounds = None
    if confidence_intervals is not None:
        bounds = np.asarray(confidence_intervals, dtype=float)
        if bounds.shape != (len(labels), 2) or not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError("Intervals require one finite ordered lower/upper pair per estimate")
    settings = _config_or_default(config)
    figure, axis = plt.subplots(figsize=settings.figure_size or (9, 5))
    positions = np.arange(len(labels))
    axis.scatter(numeric, positions, color="#2563a6", s=65, zorder=3, label="Model estimate")
    if bounds is not None:
        axis.hlines(positions, bounds[:, 0], bounds[:, 1], color="#334155", linewidth=2, label="Supplied interval")
        axis.vlines(bounds[:, 0], positions - .08, positions + .08, color="#334155")
        axis.vlines(bounds[:, 1], positions - .08, positions + .08, color="#334155")
    observed = numeric if bounds is None else np.concatenate((numeric, bounds.ravel()))
    dynamic_range = _positive_dynamic_range(observed)
    logarithmic = settings.auto_log_scale and dynamic_range is not None and dynamic_range > 100
    if logarithmic:
        axis.set_xscale("log")
    axis.set_xlabel(xlabel + (" (log scale)" if logarithmic else ""), fontsize=12)
    axis.set_yticks(positions, [str(label) for label in labels], fontsize=11)
    axis.set_ylim(len(labels) - .5, -.5)
    axis.margins(x=.24)
    for position, value in zip(positions, numeric):
        axis.annotate(f"{value:.4g}", (value, position), xytext=(8, 7), textcoords="offset points", fontsize=10)
    axis.set_title(title, fontsize=14, pad=18)
    axis.grid(axis="x", which="both", alpha=.25, linestyle="--")
    axis.spines[["top", "right"]].set_visible(False)
    if bounds is not None:
        axis.legend(loc="best", fontsize=9)
    figure.tight_layout()
    if out_path:
        _save_figure(figure, out_path, dpi=settings.dpi, close=settings.close_on_save)
    return figure


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
    confidence_intervals: Optional[Sequence[tuple[float, float]]] = None,
):
    r"""Plot categorical values with optional error magnitudes or explicit interval bounds.

    ``confidence_intervals`` contains one (lower, upper) pair per value. These
    ranges are drawn at their actual coordinates, even when they exclude the
    displayed mean. No significance or confidence level is inferred from values.
    
    Args:
        labels: Category labels for x-axis
        values: Values for each category
        title: Plot title (supports LaTeX: $\\Delta E$ etc.)
        out_path: Output file path (PNG recommended, 300 DPI)
        ylabel: Y-axis label with units
        yerr: Optional error bars (standard errors or confidence intervals)
        confidence_intervals: Explicit lower/upper bounds, mutually exclusive with yerr
    
    References:
        - Error bar best practices: https://doi.org/10.1038/nmeth.2813
        - Statistical visualization: https://doi.org/10.1371/journal.pcbi.1004668
    """
    if plt is None or np is None:
        raise ImportError("bar_plot requires matplotlib and numpy")

    plot_config = _config_or_default(config)
    if categories is None and labels is None and values is not None and len(values):
        raise TypeError("categories or labels are required when values are provided")
    labels = list(categories if categories is not None else ([] if labels is None else labels))
    values = list([] if values is None else values)
    yerr = errors if errors is not None else yerr
    numeric = np.asarray(values, dtype=float)
    if numeric.ndim != 1 or not numeric.size or len(labels) != numeric.size or not np.all(np.isfinite(numeric)):
        raise ValueError("Bar labels and finite values must have the same nonzero length")
    bounds = None
    if confidence_intervals is not None:
        if yerr is not None:
            raise ValueError("Specify either error magnitudes or interval bounds, not both")
        bounds = np.asarray(confidence_intervals, dtype=float)
        if bounds.shape != (len(values), 2) or not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError("Intervals require one finite ordered lower/upper pair per bar")
    
    fig, ax = plt.subplots(figsize=plot_config.figure_size or (10, 6))
    
    # Color bars based on magnitude for visual hierarchy
    if np is not None:
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(values)))
    else:
        colors = ['C{}'.format(i % 10) for i in range(len(values))]
        
    bars = ax.bar(
        range(len(labels)), numeric, yerr=yerr, capsize=6, color=colors,
        edgecolor='black', linewidth=0.8, alpha=0.8
    )
    ax.set_xticks(range(len(labels)), labels)
    if bounds is not None:
        positions = np.arange(len(values))
        ax.vlines(positions, bounds[:, 0], bounds[:, 1], colors='black')
        for endpoints in (bounds[:, 0], bounds[:, 1]):
            ax.hlines(endpoints, positions - 0.08, positions + 0.08, colors='black')
    
    # Add value labels on top of bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height + (max(values) * 0.01),
            f'{value:.4g}',
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
        _save_figure(fig, out_path, dpi=plot_config.dpi, close=plot_config.close_on_save)
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
    scaling_symbol: str = "x",
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
    if plt is None or np is None:
        raise ImportError("Plotting requires matplotlib and numpy")

    plot_config = _config_or_default(config)
    if x_data is None and x is None and (y_data is not None or ys is not None):
        raise ValueError("x_data or x is required when y series are provided")
    if y_data is None and ys is None and x is not None:
        raise TypeError("ys or y_data is required when x data is provided")
    if x_data is not None:
        x_series = list(x_data)
    else:
        x_series = [[] if x is None else x for _ in range(len([] if ys is None else ys))]
    y_series = list(y_data if y_data is not None else ([] if ys is None else ys))
    labels = list(labels) if labels is not None else ["" for _ in y_series]
    if len(x_series) != len(y_series) or len(labels) != len(y_series):
        raise ValueError("Each series requires one x sequence and one label")
    prepared = []
    for idx, (series_x, series_y) in enumerate(zip(x_series, y_series)):
        if bands is not None and idx < len(bands) and len(bands[idx]):
            bounds = np.asarray(bands[idx], dtype=float)
            if bounds.ndim != 2 or bounds.shape != (len(series_x), 2):
                raise ValueError("Bands require one (low, high) pair per observation")
            (clean_x, clean_y, lows, highs), _ = _paired_float_arrays(series_x, series_y, bounds[:, 0], bounds[:, 1])
            if np.any(lows > highs):
                raise ValueError("Band lower bounds must not exceed upper bounds")
            prepared.append((clean_x, clean_y, (lows, highs)))
        else:
            (clean_x, clean_y), _ = _paired_float_arrays(series_x, series_y)
            prepared.append((clean_x, clean_y, None))
    
    fig, ax = plt.subplots(figsize=plot_config.figure_size or (10, 7))
    
    # Enhanced color palette and markers
    if np is not None:
        colors = plt.cm.Set1(np.linspace(0, 1, max(len(y_series), 1)))
    else:
        colors = ['C{}'.format(i) for i in range(max(len(y_series), 1))]
        
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    linestyles = ['-', '--', '-.', ':']
    
    for idx, ((series_x, y, band), label) in enumerate(zip(prepared, labels)):
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
        if band is not None:
            ax.fill_between(series_x, band[0], band[1], alpha=0.2,
                            color=color, label=f'{label} supplied interval')
            
        # Fit and display scaling law if data suggests power law
        if plot_config.annotate_statistics and np is not None:
            try:
                x_fit_source, y_fit_source = series_x, y
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
                        f'$\\propto {scaling_symbol}^{{{scaling_exp:.2f}}}$',
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
    if plot_config.auto_log_scale and np is not None and prepared:
        try:
            all_x = [value for series_x, _y, _band in prepared for value in series_x]
            x_range = _positive_dynamic_range(all_x)
            y_range = _positive_dynamic_range([value for _x, series_y, _band in prepared for value in series_y])
            if x_range is not None and x_range > 100:
                ax.set_xscale('log')
            if y_range is not None and y_range > 100:
                ax.set_yscale('log')
        except (ValueError, ZeroDivisionError):
            x_range = None
            y_range = None
    
    fig.tight_layout()
    if out_path:
        _save_figure(fig, out_path, dpi=plot_config.dpi, close=plot_config.close_on_save)
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
    if plt is None or np is None:
        raise ImportError("Plotting requires matplotlib and numpy")

    plot_config = _config_or_default(config)
    x = x_data if x_data is not None else ([] if x is None else x)
    y = y_data if y_data is not None else ([] if y is None else y)
    (x_arr, y_arr), mask = _paired_float_arrays(x, y)
    if sizes is not None:
        sizes = np.asarray(sizes, dtype=float)
        if sizes.ndim != 1 or sizes.shape != mask.shape or not np.all(np.isfinite(sizes)) or np.any(sizes < 0):
            raise ValueError("Sizes must be finite, nonnegative and match paired data")
        sizes = sizes[mask]
    drew_regression = False
    
    fig, ax = plt.subplots(figsize=plot_config.figure_size or (10, 8))
    
    # Convert to arrays for analysis
    if np is not None:
        
        # Create scatter plot with color mapping
        scatter = ax.scatter(
            x_arr, y_arr, s=sizes if sizes is not None else 60, alpha=0.7, c=y_arr,
            cmap='viridis', edgecolors='black', linewidth=0.5
        )
        
        if y_arr.size > 0:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(ylabel, fontsize=12, fontweight='bold')
        
        # Fit regression line
        drew_regression = False
        if plot_config.annotate_statistics and _has_linear_fit_signal(x_arr, y_arr):
            coeffs = np.polyfit(x_arr, y_arr, 1)
            x_fit = np.linspace(min(x_arr), max(x_arr), 100)
            y_fit = coeffs[0] * x_fit + coeffs[1]
            ax.plot(x_fit, y_fit, 'r--', linewidth=2, alpha=0.8, label='Linear fit')
            drew_regression = True
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(x_arr, y_arr)[0, 1]
            
            # Add statistics annotation
            stats_text = f'R² = {correlation**2:.3f}\nSlope = {coeffs[0]:.3e}'
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
    
    if drew_regression:
        ax.legend(fontsize=11)
    
    fig.tight_layout()
    if out_path:
        _save_figure(fig, out_path, dpi=plot_config.dpi, close=plot_config.close_on_save)
    return fig
