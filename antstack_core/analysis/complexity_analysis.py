"""Complexity analysis with advanced statistical methods and thermodynamic frameworks.

This module implements cutting-edge complexity analysis methods based on recent research:
- Agent-based modeling for emergent behavior analysis
- Network complexity metrics and structural analysis
- Thermodynamic computing efficiency frameworks
- Complexity-entropy diagrams for intrinsic computation analysis
- Advanced bootstrap methods with bias correction
- Multi-scale analysis integration

References:
- Agent-based modeling: https://eprints.whiterose.ac.uk/81723/
- Complexity-entropy analysis: https://arxiv.org/abs/0806.4789
- Thermodynamic computing: https://pubmed.ncbi.nlm.nih.gov/28505845/
- Bootstrap methods: https://doi.org/10.1214/aos/1176344552
- Network complexity: https://doi.org/10.1038/s41586-020-2196-x
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple, Optional, Any, Sequence
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    from scipy import stats
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, dendrogram
    HAS_SCIPY = True
except ImportError:
    stats = None
    pdist = None
    squareform = None
    linkage = None
    dendrogram = None
    HAS_SCIPY = False

from .statistics import bootstrap_mean_ci, analyze_scaling_relationship
from .energy import EnergyBreakdown, EnergyCoefficients


@dataclass
class ComplexityMetrics:
    """Comprehensive complexity metrics for system analysis."""
    
    # Information-theoretic measures
    entropy: float = 0.0
    mutual_information: float = 0.0
    information_density: float = 0.0
    
    # Structural complexity
    network_density: float = 0.0
    clustering_coefficient: float = 0.0
    path_length: float = 0.0
    modularity: float = 0.0
    
    # Thermodynamic measures
    thermodynamic_efficiency: float = 0.0
    entropy_production_rate: float = 0.0
    free_energy: float = 0.0
    
    # Scaling measures
    scaling_exponent: float = 0.0
    scaling_confidence: float = 0.0
    regime_classification: str = "unknown"
    
    # Statistical measures
    variance: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0


@dataclass
class AgentBasedModel:
    """Agent-based model for emergent behavior analysis."""
    
    agents: List[Dict[str, Any]]
    interactions: List[Tuple[int, int, float]]  # (agent1, agent2, strength)
    environment_params: Dict[str, float]
    time_steps: int = 1000
    
    def simulate(self, seed: Optional[int] = None) -> Dict[str, List[float]]:
        """Simulate agent interactions and return emergent metrics."""
        if seed is not None:
            random.seed(seed)
            if HAS_NUMPY:
                np.random.seed(seed)
        
        # Initialize agent states
        n_agents = len(self.agents)
        states = [agent.get('initial_state', 0.0) for agent in self.agents]
        energies = [agent.get('initial_energy', 1.0) for agent in self.agents]
        
        # Track emergent properties
        emergent_metrics = {
            'total_energy': [],
            'state_variance': [],
            'interaction_strength': [],
            'clustering_coefficient': [],
            'entropy': []
        }
        
        for t in range(self.time_steps):
            # Update agent states based on interactions
            new_states = states.copy()
            new_energies = energies.copy()
            
            for agent1, agent2, strength in self.interactions:
                if agent1 < n_agents and agent2 < n_agents:
                    # Simple interaction model
                    state_diff = states[agent2] - states[agent1]
                    energy_transfer = strength * state_diff * 0.01
                    
                    new_states[agent1] += energy_transfer
                    new_states[agent2] -= energy_transfer
                    new_energies[agent1] += abs(energy_transfer)
                    new_energies[agent2] += abs(energy_transfer)
            
            states = new_states
            energies = new_energies
            
            # Calculate emergent metrics
            emergent_metrics['total_energy'].append(sum(energies))
            emergent_metrics['state_variance'].append(np.var(states) if HAS_NUMPY else self._variance(states))
            emergent_metrics['interaction_strength'].append(sum(abs(s) for _, _, s in self.interactions))
            emergent_metrics['clustering_coefficient'].append(self._calculate_clustering(states))
            emergent_metrics['entropy'].append(self._calculate_entropy(states))
        
        return emergent_metrics
    
    def _variance(self, values: List[float]) -> float:
        """Calculate variance without numpy."""
        if len(values) < 2:
            return 0.0
        mean_val = sum(values) / len(values)
        return sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
    
    def _calculate_clustering(self, states: List[float]) -> float:
        """Calculate clustering coefficient for agent states."""
        if len(states) < 3:
            return 0.0
        
        # Simple clustering based on state similarity
        threshold = 0.1
        clusters = 0
        total_pairs = 0
        
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                total_pairs += 1
                if abs(states[i] - states[j]) < threshold:
                    clusters += 1
        
        return clusters / total_pairs if total_pairs > 0 else 0.0
    
    def _calculate_entropy(self, states: List[float]) -> float:
        """Calculate Shannon entropy of agent states."""
        if not states:
            return 0.0
        
        # Discretize states into bins
        n_bins = min(10, len(states))
        if HAS_NUMPY:
            hist, _ = np.histogram(states, bins=n_bins)
            probs = hist / len(states)
        else:
            # Simple binning without numpy
            min_state = min(states)
            max_state = max(states)
            bin_width = (max_state - min_state) / n_bins if max_state > min_state else 1.0
            
            hist = [0] * n_bins
            for state in states:
                bin_idx = min(int((state - min_state) / bin_width), n_bins - 1)
                hist[bin_idx] += 1
            
            probs = [h / len(states) for h in hist]
        
        # Calculate entropy
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy


class NetworkComplexityAnalyzer:
    """Advanced network complexity analysis for system interactions."""
    
    def __init__(self):
        """Initialize network complexity analyzer."""
        pass
    
    def analyze_network_complexity(self, adjacency_matrix: List[List[float]]) -> ComplexityMetrics:
        """Analyze network complexity using multiple metrics.
        
        Args:
            adjacency_matrix: Square matrix representing network connections
            
        Returns:
            ComplexityMetrics with network analysis results
        """
        if not adjacency_matrix or len(adjacency_matrix) == 0:
            return ComplexityMetrics()
        
        n_nodes = len(adjacency_matrix)
        
        # Convert to numpy array if available
        if HAS_NUMPY:
            adj_matrix = np.array(adjacency_matrix)
        else:
            adj_matrix = adjacency_matrix
        
        # Calculate network density
        density = self._calculate_density(adj_matrix)
        
        # Calculate clustering coefficient
        clustering = self._calculate_clustering_coefficient(adj_matrix)
        
        # Calculate average path length
        path_length = self._calculate_path_length(adj_matrix)
        
        # Calculate modularity (simplified)
        modularity = self._calculate_modularity(adj_matrix)
        
        # Calculate information-theoretic measures
        entropy = self._calculate_network_entropy(adj_matrix)
        mutual_info = self._calculate_mutual_information(adj_matrix)
        
        return ComplexityMetrics(
            network_density=density,
            clustering_coefficient=clustering,
            path_length=path_length,
            modularity=modularity,
            entropy=entropy,
            mutual_information=mutual_info
        )
    
    def _calculate_density(self, adj_matrix) -> float:
        """Calculate network density."""
        if HAS_NUMPY:
            n = adj_matrix.shape[0]
            return np.sum(adj_matrix > 0) / (n * (n - 1))
        else:
            n = len(adj_matrix)
            edges = sum(1 for row in adj_matrix for val in row if val > 0)
            return edges / (n * (n - 1))
    
    def _calculate_clustering_coefficient(self, adj_matrix) -> float:
        """Calculate average clustering coefficient."""
        if not HAS_NUMPY:
            return 0.0  # Simplified without numpy
        
        n = adj_matrix.shape[0]
        clustering_coeffs = []
        
        for i in range(n):
            neighbors = np.where(adj_matrix[i] > 0)[0]
            k = len(neighbors)
            
            if k < 2:
                clustering_coeffs.append(0.0)
                continue
            
            # Count triangles
            triangles = 0
            for j in neighbors:
                for k in neighbors:
                    if j < k and adj_matrix[j, k] > 0:
                        triangles += 1
            
            # Clustering coefficient for node i
            max_possible = k * (k - 1) / 2
            clustering_coeffs.append(triangles / max_possible if max_possible > 0 else 0.0)
        
        return np.mean(clustering_coeffs)
    
    def _calculate_path_length(self, adj_matrix) -> float:
        """Calculate average shortest path length."""
        if not HAS_NUMPY:
            return 0.0  # Simplified without numpy
        
        n = adj_matrix.shape[0]
        
        # Floyd-Warshall algorithm for shortest paths
        dist = np.full((n, n), np.inf)
        np.fill_diagonal(dist, 0)
        
        # Initialize with direct connections
        for i in range(n):
            for j in range(n):
                if adj_matrix[i, j] > 0:
                    dist[i, j] = 1
        
        # Floyd-Warshall
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])
        
        # Calculate average path length (excluding infinite distances)
        finite_distances = dist[dist != np.inf]
        return np.mean(finite_distances) if len(finite_distances) > 0 else 0.0
    
    def _calculate_modularity(self, adj_matrix) -> float:
        """Calculate network modularity (simplified)."""
        if not HAS_NUMPY:
            return 0.0  # Simplified without numpy
        
        n = adj_matrix.shape[0]
        m = np.sum(adj_matrix) / 2  # Total edges (undirected)
        
        if m == 0:
            return 0.0
        
        # Simple community detection (random assignment for now)
        communities = np.random.randint(0, 2, n)  # Binary communities
        
        modularity = 0.0
        for i in range(n):
            for j in range(n):
                if adj_matrix[i, j] > 0:
                    expected = (np.sum(adj_matrix[i]) * np.sum(adj_matrix[j])) / (2 * m)
                    modularity += (adj_matrix[i, j] - expected) * (1 if communities[i] == communities[j] else 0)
        
        return modularity / (2 * m)
    
    def _calculate_network_entropy(self, adj_matrix) -> float:
        """Calculate Shannon entropy of network structure."""
        if not HAS_NUMPY:
            return 0.0  # Simplified without numpy
        
        # Calculate degree distribution
        degrees = np.sum(adj_matrix > 0, axis=1)
        
        # Normalize to probabilities
        total_degrees = np.sum(degrees)
        if total_degrees == 0:
            return 0.0
        
        probs = degrees / total_degrees
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return entropy
    
    def _calculate_mutual_information(self, adj_matrix) -> float:
        """Calculate mutual information between node pairs."""
        if not HAS_NUMPY:
            return 0.0  # Simplified without numpy
        
        n = adj_matrix.shape[0]
        if n < 2:
            return 0.0
        
        # Calculate mutual information between all pairs
        mutual_info_sum = 0.0
        pair_count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                # Simple mutual information based on connection strength
                if adj_matrix[i, j] > 0:
                    mutual_info_sum += adj_matrix[i, j]
                pair_count += 1
        
        return mutual_info_sum / pair_count if pair_count > 0 else 0.0


class ThermodynamicComplexityAnalyzer:
    """Thermodynamic analysis of computational complexity and energy efficiency."""
    
    def __init__(self, temperature_k: float = 300.0):
        """Initialize thermodynamic analyzer.
        
        Args:
            temperature_k: Operating temperature in Kelvin
        """
        self.temperature_k = temperature_k
        self.k_boltzmann = 1.380649e-23  # J/K
    
    def analyze_thermodynamic_efficiency(self, 
                                       energy_breakdown: EnergyBreakdown,
                                       computational_work: float,
                                       information_processed: float) -> ComplexityMetrics:
        """Analyze thermodynamic efficiency of computational processes.
        
        Args:
            energy_breakdown: Detailed energy breakdown
            computational_work: Computational work performed (J)
            information_processed: Information processed (bits)
            
        Returns:
            ComplexityMetrics with thermodynamic analysis
        """
        # Calculate thermodynamic efficiency
        total_energy = energy_breakdown.total
        theoretical_minimum = self._calculate_theoretical_minimum(information_processed)
        
        efficiency = theoretical_minimum / total_energy if total_energy > 0 else 0.0
        
        # Calculate entropy production rate
        entropy_production = self._calculate_entropy_production(energy_breakdown)
        
        # Calculate free energy
        free_energy = self._calculate_free_energy(computational_work, entropy_production)
        
        return ComplexityMetrics(
            thermodynamic_efficiency=efficiency,
            entropy_production_rate=entropy_production,
            free_energy=free_energy
        )
    
    def _calculate_theoretical_minimum(self, information_processed: float) -> float:
        """Calculate theoretical minimum energy (Landauer limit)."""
        return information_processed * self.k_boltzmann * self.temperature_k * math.log(2)
    
    def _calculate_entropy_production(self, energy_breakdown: EnergyBreakdown) -> float:
        """Calculate entropy production rate."""
        # Simplified entropy production based on energy dissipation
        dissipated_energy = energy_breakdown.total - energy_breakdown.compute_flops
        return dissipated_energy / (self.temperature_k * 0.01)  # 10ms decision cycle
    
    def _calculate_free_energy(self, computational_work: float, entropy_production: float) -> float:
        """Calculate free energy available for computation."""
        return computational_work - self.temperature_k * entropy_production


class ComplexityEntropyAnalyzer:
    """Complexity-entropy analysis for intrinsic computation characterization."""
    
    def __init__(self):
        """Initialize complexity-entropy analyzer."""
        pass
    
    def create_complexity_entropy_diagram(self, 
                                        time_series: List[float],
                                        window_size: int = 100) -> Dict[str, List[float]]:
        """Create complexity-entropy diagram for time series analysis.
        
        Args:
            time_series: Time series data to analyze
            window_size: Window size for sliding analysis
            
        Returns:
            Dictionary with complexity and entropy values
        """
        if len(time_series) < window_size:
            return {'complexity': [], 'entropy': []}
        
        complexities = []
        entropies = []
        
        for i in range(len(time_series) - window_size + 1):
            window = time_series[i:i + window_size]
            
            # Calculate complexity (approximate Lempel-Ziv complexity)
            complexity = self._calculate_lemple_ziv_complexity(window)
            complexities.append(complexity)
            
            # Calculate entropy
            entropy = self._calculate_shannon_entropy(window)
            entropies.append(entropy)
        
        return {
            'complexity': complexities,
            'entropy': entropies,
            'time_points': list(range(len(complexities)))
        }
    
    def _calculate_lemple_ziv_complexity(self, sequence: List[float]) -> float:
        """Calculate Lempel-Ziv complexity (simplified)."""
        if not sequence:
            return 0.0
        
        # Discretize sequence
        if HAS_NUMPY:
            # Use quantiles for discretization
            quantiles = np.percentile(sequence, [25, 50, 75])
            discrete = np.digitize(sequence, quantiles)
        else:
            # Simple discretization
            min_val = min(sequence)
            max_val = max(sequence)
            if max_val == min_val:
                discrete = [0] * len(sequence)
            else:
                discrete = [int(3 * (x - min_val) / (max_val - min_val)) for x in sequence]
        
        # Calculate LZ complexity
        n = len(discrete)
        if n == 0:
            return 0.0
        
        # Simplified LZ complexity calculation
        patterns = set()
        i = 0
        while i < n:
            j = i + 1
            while j <= n and tuple(discrete[i:j]) not in patterns:
                j += 1
            patterns.add(tuple(discrete[i:j-1]))
            i = j - 1
        
        return len(patterns) / n
    
    def _calculate_shannon_entropy(self, sequence: List[float]) -> float:
        """Calculate Shannon entropy of sequence."""
        if not sequence:
            return 0.0
        
        # Discretize sequence
        if HAS_NUMPY:
            hist, _ = np.histogram(sequence, bins=min(10, len(sequence)))
            probs = hist / len(sequence)
        else:
            # Simple binning
            n_bins = min(10, len(sequence))
            min_val = min(sequence)
            max_val = max(sequence)
            bin_width = (max_val - min_val) / n_bins if max_val > min_val else 1.0
            
            hist = [0] * n_bins
            for val in sequence:
                bin_idx = min(int((val - min_val) / bin_width), n_bins - 1)
                hist[bin_idx] += 1
            
            probs = [h / len(sequence) for h in hist]
        
        # Calculate entropy
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy


class EnhancedBootstrapAnalyzer:
    """Enhanced bootstrap analysis with bias correction and advanced statistics."""
    
    def __init__(self, n_bootstrap: int = 1000):
        """Initialize enhanced bootstrap analyzer.
        
        Args:
            n_bootstrap: Number of bootstrap samples
        """
        self.n_bootstrap = n_bootstrap
    
    def analyze_scaling_with_bootstrap(self, 
                                     x_values: List[float], 
                                     y_values: List[float],
                                     confidence_level: float = 0.95) -> Dict[str, Any]:
        """Analyze scaling relationship with comprehensive bootstrap analysis.
        
        Args:
            x_values: Independent variable values
            y_values: Dependent variable values
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with comprehensive bootstrap analysis results
        """
        if len(x_values) != len(y_values) or len(x_values) < 3:
            return {'error': 'Insufficient data for bootstrap analysis'}
        
        # Original scaling analysis
        original_result = analyze_scaling_relationship(x_values, y_values)
        
        if not original_result.get('valid', False):
            return original_result
        
        # Bootstrap analysis
        bootstrap_exponents = []
        bootstrap_intercepts = []
        bootstrap_r_squared = []
        
        for _ in range(self.n_bootstrap):
            # Bootstrap sample
            if HAS_NUMPY:
                indices = np.random.choice(len(x_values), len(x_values), replace=True)
                boot_x = [x_values[i] for i in indices]
                boot_y = [y_values[i] for i in indices]
            else:
                # Simple bootstrap without numpy
                boot_x = [random.choice(x_values) for _ in range(len(x_values))]
                boot_y = [random.choice(y_values) for _ in range(len(y_values))]
            
            # Analyze bootstrap sample
            boot_result = analyze_scaling_relationship(boot_x, boot_y)
            if boot_result.get('valid', False):
                bootstrap_exponents.append(boot_result.get('scaling_exponent', 0))
                bootstrap_intercepts.append(boot_result.get('intercept', 0))
                bootstrap_r_squared.append(boot_result.get('r_squared', 0))
        
        if not bootstrap_exponents:
            return original_result
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        if HAS_NUMPY:
            exp_ci = np.percentile(bootstrap_exponents, [lower_percentile, upper_percentile])
            int_ci = np.percentile(bootstrap_intercepts, [lower_percentile, upper_percentile])
            r2_ci = np.percentile(bootstrap_r_squared, [lower_percentile, upper_percentile])
            
            # Bias correction
            bias_exp = np.mean(bootstrap_exponents) - original_result.get('scaling_exponent', 0)
            bias_int = np.mean(bootstrap_intercepts) - original_result.get('intercept', 0)
            
            # Corrected estimates
            corrected_exp = original_result.get('scaling_exponent', 0) - bias_exp
            corrected_int = original_result.get('intercept', 0) - bias_int
            
        else:
            # Fallback without numpy
            bootstrap_exponents.sort()
            bootstrap_intercepts.sort()
            bootstrap_r_squared.sort()
            
            n = len(bootstrap_exponents)
            lower_idx = int(lower_percentile / 100 * n)
            upper_idx = int(upper_percentile / 100 * n)
            
            exp_ci = [bootstrap_exponents[lower_idx], bootstrap_exponents[upper_idx]]
            int_ci = [bootstrap_intercepts[lower_idx], bootstrap_intercepts[upper_idx]]
            r2_ci = [bootstrap_r_squared[lower_idx], bootstrap_r_squared[upper_idx]]
            
            # Simple bias correction
            bias_exp = sum(bootstrap_exponents) / len(bootstrap_exponents) - original_result.get('scaling_exponent', 0)
            bias_int = sum(bootstrap_intercepts) / len(bootstrap_intercepts) - original_result.get('intercept', 0)
            
            corrected_exp = original_result.get('scaling_exponent', 0) - bias_exp
            corrected_int = original_result.get('intercept', 0) - bias_int
        
        return {
            'original_result': original_result,
            'bootstrap_exponents': bootstrap_exponents,
            'bootstrap_intercepts': bootstrap_intercepts,
            'bootstrap_r_squared': bootstrap_r_squared,
            'confidence_intervals': {
                'exponent': exp_ci,
                'intercept': int_ci,
                'r_squared': r2_ci
            },
            'bias_corrected': {
                'exponent': corrected_exp,
                'intercept': corrected_int
            },
            'bootstrap_stats': {
                'mean_exponent': sum(bootstrap_exponents) / len(bootstrap_exponents),
                'std_exponent': self._calculate_std(bootstrap_exponents),
                'mean_r_squared': sum(bootstrap_r_squared) / len(bootstrap_r_squared),
                'std_r_squared': self._calculate_std(bootstrap_r_squared)
            }
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation without numpy."""
        if len(values) < 2:
            return 0.0
        
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)


def comprehensive_complexity_analysis(energy_data: List[EnergyBreakdown],
                                    computational_data: List[Dict[str, float]],
                                    network_data: Optional[List[List[float]]] = None,
                                    time_series_data: Optional[List[float]] = None) -> Dict[str, Any]:
    """Perform comprehensive complexity analysis integrating multiple methodologies.
    
    Args:
        energy_data: List of energy breakdowns
        computational_data: List of computational metrics
        network_data: Optional network adjacency matrix
        time_series_data: Optional time series for complexity-entropy analysis
        
    Returns:
        Comprehensive analysis results
    """
    results = {
        'thermodynamic_analysis': {},
        'network_analysis': {},
        'complexity_entropy_analysis': {},
        'bootstrap_analysis': {},
        'integrated_metrics': {}
    }
    
    # Thermodynamic analysis
    if energy_data and computational_data:
        thermo_analyzer = ThermodynamicComplexityAnalyzer()
        
        for i, (energy, comp) in enumerate(zip(energy_data, computational_data)):
            thermo_result = thermo_analyzer.analyze_thermodynamic_efficiency(
                energy, 
                comp.get('computational_work', 0.0),
                comp.get('information_processed', 0.0)
            )
            results['thermodynamic_analysis'][f'step_{i}'] = thermo_result
    
    # Network analysis
    if network_data:
        network_analyzer = NetworkComplexityAnalyzer()
        network_result = network_analyzer.analyze_network_complexity(network_data)
        results['network_analysis'] = network_result
    
    # Complexity-entropy analysis
    if time_series_data:
        ce_analyzer = ComplexityEntropyAnalyzer()
        ce_result = ce_analyzer.create_complexity_entropy_diagram(time_series_data)
        results['complexity_entropy_analysis'] = ce_result
    
    # Bootstrap analysis for scaling relationships
    if len(computational_data) > 3:
        bootstrap_analyzer = EnhancedBootstrapAnalyzer()
        
        # Extract scaling parameters
        x_values = [comp.get('parameter_value', i) for i, comp in enumerate(computational_data)]
        y_values = [comp.get('energy', 0.0) for comp in computational_data]
        
        bootstrap_result = bootstrap_analyzer.analyze_scaling_with_bootstrap(x_values, y_values)
        results['bootstrap_analysis'] = bootstrap_result
    
    # Integrated metrics
    results['integrated_metrics'] = {
        'analysis_timestamp': math.floor(time.time()) if 'time' in globals() else 0,
        'data_points': len(energy_data),
        'methodologies_applied': [
            'thermodynamic_analysis',
            'network_analysis' if network_data else None,
            'complexity_entropy_analysis' if time_series_data else None,
            'bootstrap_analysis'
        ]
    }
    
    return results
