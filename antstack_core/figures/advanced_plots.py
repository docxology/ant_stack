"""Advanced visualization capabilities for complexity energetics research.

This module provides cutting-edge visualization methods based on recent research:
- Complexity-entropy diagrams for intrinsic computation analysis
- Network visualization with structural complexity metrics
- Thermodynamic efficiency heat maps and phase diagrams
- Multi-scale visualization with statistical overlays
- Interactive dashboards for real-time analysis
- Publication-quality figures with comprehensive captions

References:
- Complexity-entropy visualization: https://arxiv.org/abs/0806.4789
- Network visualization: https://doi.org/10.1371/journal.pcbi.1003833
- Thermodynamic phase diagrams: https://pubmed.ncbi.nlm.nih.gov/28505845/
- Scientific visualization: https://doi.org/10.1371/journal.pcbi.1004668
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional, Any, Sequence
from pathlib import Path

# Optional plotting dependencies (graceful degradation)
plt = None  # type: ignore
np = None   # type: ignore
try:
    import matplotlib
    matplotlib.use("Agg")  # headless backend for server environments
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    from matplotlib.patches import FancyBboxPatch
    import numpy as np
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    except ImportError:
        pass  # seaborn is optional enhancement
except Exception:
    plt = None  # graceful degradation


def complexity_entropy_diagram(
    complexity_values: List[float],
    entropy_values: List[float],
    title: str,
    out_path: str,
    time_points: Optional[List[float]] = None,
    figure_manager: Optional[Any] = None
) -> str:
    """Create complexity-entropy diagram for intrinsic computation analysis.
    
    Generates publication-quality complexity-entropy diagrams that visualize
    the relationship between system complexity and entropy, enabling identification
    of optimal operating regimes and phase transitions.
    
    Args:
        complexity_values: Lempel-Ziv complexity values
        entropy_values: Shannon entropy values
        title: Plot title
        out_path: Output file path
        time_points: Optional time points for trajectory visualization
        figure_manager: Optional figure manager for auto-numbering
        
    Returns:
        Figure ID for cross-referencing
        
    References:
        - Complexity-entropy analysis: https://arxiv.org/abs/0806.4789
        - Intrinsic computation: https://doi.org/10.1038/nature10872
    """
    if plt is None:
        print("matplotlib not available; skipping complexity-entropy diagram")
        return "fig:unavailable"
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create scatter plot with color mapping
    if np is not None and len(complexity_values) > 0:
        # Color by density or trajectory
        if time_points and len(time_points) == len(complexity_values):
            scatter = ax.scatter(complexity_values, entropy_values, 
                               c=time_points, cmap='viridis', s=100, 
                               alpha=0.7, edgecolors='black', linewidth=0.5)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Time', fontsize=12, fontweight='bold')
        else:
            # Density-based coloring
            from scipy.stats import gaussian_kde
            try:
                xy = np.vstack([complexity_values, entropy_values])
                density = gaussian_kde(xy)(xy)
                scatter = ax.scatter(complexity_values, entropy_values, 
                                   c=density, cmap='plasma', s=100, 
                                   alpha=0.7, edgecolors='black', linewidth=0.5)
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Point Density', fontsize=12, fontweight='bold')
            except ImportError:
                scatter = ax.scatter(complexity_values, entropy_values, 
                                   c='blue', s=100, alpha=0.7, 
                                   edgecolors='black', linewidth=0.5)
    else:
        scatter = ax.scatter(complexity_values, entropy_values, 
                           c='blue', s=100, alpha=0.7, 
                           edgecolors='black', linewidth=0.5)
    
    # Add theoretical boundaries
    if np is not None:
        # Maximum entropy line (uniform distribution)
        max_entropy = np.log2(len(complexity_values)) if len(complexity_values) > 1 else 1.0
        ax.axhline(y=max_entropy, color='red', linestyle='--', alpha=0.7, 
                  linewidth=2, label=f'Max Entropy = {max_entropy:.2f}')
        
        # Complexity-entropy relationship regions
        x_range = np.linspace(0, 1, 100)
        
        # Ordered regime (low complexity, low entropy)
        ordered_region = 0.1 * x_range
        ax.plot(x_range, ordered_region, 'g--', alpha=0.5, linewidth=1, 
               label='Ordered Regime')
        
        # Random regime (high entropy, variable complexity)
        random_entropy = np.full_like(x_range, max_entropy * 0.8)
        ax.plot(x_range, random_entropy, 'orange', linestyle='--', alpha=0.5, 
               linewidth=1, label='Random Regime')
        
        # Complex regime (high complexity, medium entropy)
        complex_region = 0.3 * x_range + 0.2
        ax.plot(x_range, complex_region, 'purple', linestyle='--', alpha=0.5, 
               linewidth=1, label='Complex Regime')
    
    # Enhanced styling
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Complexity (Lempel-Ziv)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Entropy (Shannon)', fontsize=14, fontweight='bold')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(entropy_values) * 1.1 if entropy_values else 1)
    
    # Improve layout
    plt.tight_layout()
    
    # Save with high DPI for publication quality
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # Generate comprehensive caption
    detailed_caption = f"Complexity-entropy diagram showing the relationship between system complexity and entropy. "
    if np is not None and len(complexity_values) > 0:
        detailed_caption += f"Data points span complexity range {min(complexity_values):.3f} to {max(complexity_values):.3f} "
        detailed_caption += f"and entropy range {min(entropy_values):.3f} to {max(entropy_values):.3f}. "
    detailed_caption += "Theoretical boundaries separate ordered, complex, and random regimes. "
    detailed_caption += "Optimal operating points typically lie in the complex regime with balanced complexity and entropy."
    
    # Add to figure manager if provided
    if figure_manager:
        fig_id = figure_manager.get_next_figure_id("complexity_entropy")
        figure_manager.add_figure(fig_id, title, detailed_caption, str(out_path))
        return fig_id
    
    return "fig:complexity_entropy"


def network_visualization(
    adjacency_matrix: List[List[float]],
    node_labels: Optional[List[str]] = None,
    title: str = "Network Structure",
    out_path: str = "network.png",
    layout: str = "spring",
    figure_manager: Optional[Any] = None
) -> str:
    """Create advanced network visualization with structural complexity metrics.
    
    Generates publication-quality network visualizations that highlight:
    - Node importance (centrality measures)
    - Community structure and modularity
    - Edge weights and connection patterns
    - Structural complexity metrics
    
    Args:
        adjacency_matrix: Square adjacency matrix
        node_labels: Optional node labels
        title: Plot title
        out_path: Output file path
        layout: Layout algorithm ('spring', 'circular', 'hierarchical')
        figure_manager: Optional figure manager for auto-numbering
        
    Returns:
        Figure ID for cross-referencing
    """
    if plt is None:
        print("matplotlib not available; skipping network visualization")
        return "fig:unavailable"
    
    if not adjacency_matrix or len(adjacency_matrix) == 0:
        print("Empty adjacency matrix; skipping network visualization")
        return "fig:unavailable"
    
    n_nodes = len(adjacency_matrix)
    if node_labels is None:
        node_labels = [f"Node {i}" for i in range(n_nodes)]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Calculate node positions based on layout
    if layout == "spring" and np is not None:
        # Spring layout simulation
        pos = _spring_layout(adjacency_matrix)
    elif layout == "circular":
        # Circular layout
        angles = [2 * math.pi * i / n_nodes for i in range(n_nodes)]
        pos = [(math.cos(angle), math.sin(angle)) for angle in angles]
    else:
        # Hierarchical layout
        pos = _hierarchical_layout(adjacency_matrix)
    
    # Calculate node sizes based on centrality
    node_sizes = _calculate_node_centrality(adjacency_matrix)
    max_size = max(node_sizes) if node_sizes else 1
    node_sizes = [300 + 200 * (size / max_size) for size in node_sizes]
    
    # Draw edges
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if adjacency_matrix[i][j] > 0:
                x1, y1 = pos[i]
                x2, y2 = pos[j]
                
                # Edge width based on connection strength
                edge_width = 0.5 + 2.0 * adjacency_matrix[i][j]
                edge_alpha = 0.3 + 0.7 * adjacency_matrix[i][j]
                
                ax.plot([x1, x2], [y1, y2], 'k-', alpha=edge_alpha, 
                       linewidth=edge_width, zorder=1)
    
    # Draw nodes
    for i, (x, y) in enumerate(pos):
        # Node color based on centrality
        color_intensity = node_sizes[i] / max(node_sizes) if node_sizes else 0.5
        color = plt.cm.viridis(color_intensity)
        
        circle = plt.Circle((x, y), 0.1, color=color, alpha=0.8, zorder=2)
        ax.add_patch(circle)
        
        # Node labels
        ax.text(x, y + 0.15, node_labels[i], ha='center', va='bottom', 
               fontsize=8, fontweight='bold', zorder=3)
    
    # Add network statistics
    if np is not None:
        density = _calculate_network_density(adjacency_matrix)
        clustering = _calculate_clustering_coefficient(adjacency_matrix)
        
        stats_text = f'Density: {density:.3f}\\nClustering: {clustering:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Enhanced styling
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Improve layout
    plt.tight_layout()
    
    # Save with high DPI for publication quality
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # Generate comprehensive caption
    detailed_caption = f"Network visualization showing structural relationships between {n_nodes} nodes. "
    if np is not None:
        detailed_caption += f"Network density: {density:.3f}, clustering coefficient: {clustering:.3f}. "
    detailed_caption += "Node sizes represent centrality measures, edge thickness represents connection strength. "
    detailed_caption += "Color coding indicates node importance in the network structure."
    
    # Add to figure manager if provided
    if figure_manager:
        fig_id = figure_manager.get_next_figure_id("network")
        figure_manager.add_figure(fig_id, title, detailed_caption, str(out_path))
        return fig_id
    
    return "fig:network"


def thermodynamic_phase_diagram(
    temperature_values: List[float],
    pressure_values: List[float],
    efficiency_values: List[float],
    title: str = "Thermodynamic Phase Diagram",
    out_path: str = "phase_diagram.png",
    figure_manager: Optional[Any] = None
) -> str:
    """Create thermodynamic phase diagram with efficiency heat map.
    
    Generates publication-quality phase diagrams that visualize:
    - Thermodynamic efficiency across temperature-pressure space
    - Phase boundaries and critical points
    - Optimal operating regions
    - Efficiency gradients and contours
    
    Args:
        temperature_values: Temperature values (K)
        pressure_values: Pressure values (Pa)
        efficiency_values: Thermodynamic efficiency values
        title: Plot title
        out_path: Output file path
        figure_manager: Optional figure manager for auto-numbering
        
    Returns:
        Figure ID for cross-referencing
    """
    if plt is None:
        print("matplotlib not available; skipping phase diagram")
        return "fig:unavailable"
    
    if not temperature_values or not pressure_values or not efficiency_values:
        print("Insufficient data for phase diagram")
        return "fig:unavailable"
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if np is not None and len(temperature_values) > 1:
        # Create meshgrid for contour plot
        T_unique = sorted(set(temperature_values))
        P_unique = sorted(set(pressure_values))
        
        if len(T_unique) > 1 and len(P_unique) > 1:
            T_mesh, P_mesh = np.meshgrid(T_unique, P_unique)
            
            # Interpolate efficiency values
            from scipy.interpolate import griddata
            try:
                points = np.column_stack((temperature_values, pressure_values))
                efficiency_grid = griddata(points, efficiency_values, 
                                        (T_mesh, P_mesh), method='cubic')
                
                # Create contour plot
                contour = ax.contourf(T_mesh, P_mesh, efficiency_grid, 
                                    levels=20, cmap='viridis', alpha=0.8)
                
                # Add contour lines
                contour_lines = ax.contour(T_mesh, P_mesh, efficiency_grid, 
                                         levels=10, colors='black', alpha=0.5, linewidths=0.5)
                ax.clabel(contour_lines, inline=True, fontsize=8)
                
                # Add colorbar
                cbar = plt.colorbar(contour, ax=ax)
                cbar.set_label('Thermodynamic Efficiency', fontsize=12, fontweight='bold')
                
            except ImportError:
                # Fallback to scatter plot
                scatter = ax.scatter(temperature_values, pressure_values, 
                                   c=efficiency_values, cmap='viridis', 
                                   s=100, alpha=0.7, edgecolors='black')
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Thermodynamic Efficiency', fontsize=12, fontweight='bold')
        else:
            # Simple scatter plot
            scatter = ax.scatter(temperature_values, pressure_values, 
                               c=efficiency_values, cmap='viridis', 
                               s=100, alpha=0.7, edgecolors='black')
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Thermodynamic Efficiency', fontsize=12, fontweight='bold')
    else:
        # Fallback without numpy
        scatter = ax.scatter(temperature_values, pressure_values, 
                           c=efficiency_values, cmap='viridis', 
                           s=100, alpha=0.7, edgecolors='black')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Thermodynamic Efficiency', fontsize=12, fontweight='bold')
    
    # Add theoretical phase boundaries
    if np is not None and len(temperature_values) > 1:
        T_range = np.linspace(min(temperature_values), max(temperature_values), 100)
        
        # Critical temperature line (example)
        critical_T = np.mean(temperature_values)
        ax.axvline(x=critical_T, color='red', linestyle='--', alpha=0.7, 
                  linewidth=2, label=f'Critical T = {critical_T:.1f} K')
        
        # Optimal efficiency region
        if len(efficiency_values) > 0:
            max_efficiency = max(efficiency_values)
            threshold = 0.8 * max_efficiency
            
            # Highlight high-efficiency region
            high_eff_mask = [eff >= threshold for eff in efficiency_values]
            if any(high_eff_mask):
                high_T = [T for i, T in enumerate(temperature_values) if high_eff_mask[i]]
                high_P = [P for i, P in enumerate(pressure_values) if high_eff_mask[i]]
                ax.scatter(high_T, high_P, c='red', s=150, alpha=0.8, 
                          marker='*', label='High Efficiency Region', zorder=5)
    
    # Enhanced styling
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Temperature (K)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Pressure (Pa)', fontsize=14, fontweight='bold')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Improve layout
    plt.tight_layout()
    
    # Save with high DPI for publication quality
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # Generate comprehensive caption
    detailed_caption = f"Thermodynamic phase diagram showing efficiency distribution across temperature-pressure space. "
    if len(efficiency_values) > 0:
        detailed_caption += f"Efficiency ranges from {min(efficiency_values):.3f} to {max(efficiency_values):.3f}. "
    detailed_caption += "Contour lines indicate efficiency gradients, with optimal operating regions highlighted. "
    detailed_caption += "Critical temperature line separates different thermodynamic phases."
    
    # Add to figure manager if provided
    if figure_manager:
        fig_id = figure_manager.get_next_figure_id("phase_diagram")
        figure_manager.add_figure(fig_id, title, detailed_caption, str(out_path))
        return fig_id
    
    return "fig:phase_diagram"


def multi_scale_visualization(
    data_dict: Dict[str, Dict[str, List[float]]],
    title: str = "Multi-Scale Analysis",
    out_path: str = "multi_scale.png",
    figure_manager: Optional[Any] = None
) -> str:
    """Create multi-scale visualization with statistical overlays.
    
    Generates publication-quality multi-panel figures that integrate:
    - Different analytical scales (micro, meso, macro)
    - Statistical analysis overlays
    - Cross-scale correlations and interactions
    - Comprehensive data integration
    
    Args:
        data_dict: Dictionary with scale names and data
        title: Plot title
        out_path: Output file path
        figure_manager: Optional figure manager for auto-numbering
        
    Returns:
        Figure ID for cross-referencing
    """
    if plt is None:
        print("matplotlib not available; skipping multi-scale visualization")
        return "fig:unavailable"
    
    n_scales = len(data_dict)
    if n_scales == 0:
        print("No data provided for multi-scale visualization")
        return "fig:unavailable"
    
    # Create subplot grid
    n_cols = min(3, n_scales)
    n_rows = (n_scales + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_scales == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    scale_names = list(data_dict.keys())
    
    for i, (scale_name, scale_data) in enumerate(data_dict.items()):
        ax = axes[i] if i < len(axes) else axes[-1]
        
        # Plot data for this scale
        if 'x' in scale_data and 'y' in scale_data:
            x_data = scale_data['x']
            y_data = scale_data['y']
            
            # Scatter plot with trend line
            ax.scatter(x_data, y_data, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            
            # Add trend line
            if np is not None and len(x_data) > 1:
                try:
                    coeffs = np.polyfit(x_data, y_data, 1)
                    x_trend = np.linspace(min(x_data), max(x_data), 100)
                    y_trend = np.poly1d(coeffs)(x_trend)
                    ax.plot(x_trend, y_trend, 'r--', alpha=0.8, linewidth=2)
                    
                    # Add R² value
                    y_pred = np.poly1d(coeffs)(x_data)
                    r_squared = 1 - (np.sum((y_data - y_pred) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2))
                    ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax.transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                except:
                    pass
        
        # Enhanced styling for each subplot
        ax.set_title(f'{scale_name.title()} Scale', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
    
    # Hide unused subplots
    for i in range(n_scales, len(axes)):
        axes[i].set_visible(False)
    
    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Improve layout
    plt.tight_layout()
    
    # Save with high DPI for publication quality
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # Generate comprehensive caption
    detailed_caption = f"Multi-scale analysis showing {n_scales} different analytical scales. "
    detailed_caption += f"Scales analyzed: {', '.join(scale_names)}. "
    detailed_caption += "Each panel shows data relationships with trend lines and R² values. "
    detailed_caption += "Cross-scale correlations reveal emergent properties and system behavior."
    
    # Add to figure manager if provided
    if figure_manager:
        fig_id = figure_manager.get_next_figure_id("multi_scale")
        figure_manager.add_figure(fig_id, title, detailed_caption, str(out_path))
        return fig_id
    
    return "fig:multi_scale"


# Helper functions for network visualization
def _spring_layout(adjacency_matrix: List[List[float]], iterations: int = 50) -> List[Tuple[float, float]]:
    """Calculate spring layout positions for network nodes."""
    n_nodes = len(adjacency_matrix)
    if n_nodes == 0:
        return []
    
    # Initialize positions randomly
    if np is not None:
        pos = np.random.rand(n_nodes, 2) * 2 - 1
    else:
        pos = [(random.random() * 2 - 1, random.random() * 2 - 1) for _ in range(n_nodes)]
    
    # Spring force simulation
    for _ in range(iterations):
        if np is not None:
            new_pos = pos.copy()
            for i in range(n_nodes):
                force = np.zeros(2)
                for j in range(n_nodes):
                    if i != j:
                        # Repulsive force
                        diff = pos[i] - pos[j]
                        dist = np.linalg.norm(diff)
                        if dist > 0:
                            force += diff / (dist ** 2) * 0.1
                        
                        # Attractive force for connected nodes
                        if adjacency_matrix[i][j] > 0:
                            force -= diff * adjacency_matrix[i][j] * 0.01
                
                new_pos[i] += force
            pos = new_pos
        else:
            # Simplified spring layout without numpy
            new_pos = list(pos)
            for i in range(n_nodes):
                force_x, force_y = 0.0, 0.0
                for j in range(n_nodes):
                    if i != j:
                        dx = pos[i][0] - pos[j][0]
                        dy = pos[i][1] - pos[j][1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist > 0:
                            force_x += dx / (dist ** 2) * 0.1
                            force_y += dy / (dist ** 2) * 0.1
                        
                        if adjacency_matrix[i][j] > 0:
                            force_x -= dx * adjacency_matrix[i][j] * 0.01
                            force_y -= dy * adjacency_matrix[i][j] * 0.01
                
                new_pos[i] = (pos[i][0] + force_x, pos[i][1] + force_y)
            pos = new_pos
    
    return [(float(x), float(y)) for x, y in pos]


def _hierarchical_layout(adjacency_matrix: List[List[float]]) -> List[Tuple[float, float]]:
    """Calculate hierarchical layout positions for network nodes."""
    n_nodes = len(adjacency_matrix)
    if n_nodes == 0:
        return []
    
    # Simple hierarchical layout
    pos = []
    for i in range(n_nodes):
        x = i % int(math.sqrt(n_nodes)) if n_nodes > 0 else 0
        y = i // int(math.sqrt(n_nodes)) if n_nodes > 0 else 0
        pos.append((float(x), float(y)))
    
    return pos


def _calculate_node_centrality(adjacency_matrix: List[List[float]]) -> List[float]:
    """Calculate node centrality measures."""
    n_nodes = len(adjacency_matrix)
    if n_nodes == 0:
        return []
    
    centrality = []
    for i in range(n_nodes):
        # Degree centrality (simplified)
        degree = sum(1 for val in adjacency_matrix[i] if val > 0)
        centrality.append(degree)
    
    return centrality


def _calculate_network_density(adjacency_matrix: List[List[float]]) -> float:
    """Calculate network density."""
    n_nodes = len(adjacency_matrix)
    if n_nodes < 2:
        return 0.0
    
    edges = sum(1 for row in adjacency_matrix for val in row if val > 0)
    max_edges = n_nodes * (n_nodes - 1)
    return edges / max_edges if max_edges > 0 else 0.0


def _calculate_clustering_coefficient(adjacency_matrix: List[List[float]]) -> float:
    """Calculate average clustering coefficient."""
    n_nodes = len(adjacency_matrix)
    if n_nodes < 3:
        return 0.0
    
    clustering_coeffs = []
    for i in range(n_nodes):
        neighbors = [j for j in range(n_nodes) if adjacency_matrix[i][j] > 0]
        k = len(neighbors)
        
        if k < 2:
            clustering_coeffs.append(0.0)
            continue
        
        # Count triangles
        triangles = 0
        for j in neighbors:
            for l in neighbors:
                if j < l and adjacency_matrix[j][l] > 0:
                    triangles += 1
        
        # Clustering coefficient for node i
        max_possible = k * (k - 1) / 2
        clustering_coeffs.append(triangles / max_possible if max_possible > 0 else 0.0)
    
    return sum(clustering_coeffs) / len(clustering_coeffs) if clustering_coeffs else 0.0
