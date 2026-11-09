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
import time
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
    """Optional quantities: None is unavailable, not a computed or supplied zero."""
    
    # Information-theoretic measures
    entropy: Optional[float] = None
    mutual_information: Optional[float] = None
    information_density: Optional[float] = None
    
    # Structural complexity
    network_density: Optional[float] = None
    clustering_coefficient: Optional[float] = None
    path_length: Optional[float] = None
    modularity: Optional[float] = None
    
    # Thermodynamic measures
    thermodynamic_efficiency: Optional[float] = None
    entropy_production_rate: Optional[float] = None
    entropy_change: Optional[float] = None  # J/K per cycle
    free_energy: Optional[float] = None
    
    # Scaling measures
    scaling_exponent: Optional[float] = None
    scaling_confidence: Optional[float] = None
    regime_classification: str = "unknown"
    
    # Statistical measures
    variance: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None


@dataclass
class AgentBasedModel:
    """Deterministic scalar-relaxation toy, not an embodied or neural simulator.

    ``total_energy`` is an arbitrary-unit accumulated interaction-cost proxy,
    not physical joules. ``environment_params`` is descriptive metadata only.
    State-pair similarity is not a graph clustering coefficient.
    """

    agents: List[Dict[str, Any]]
    interactions: List[Tuple[int, int, float]]
    environment_params: Dict[str, float]
    time_steps: int = 1000

    def simulate(self) -> Dict[str, List[float]]:
        """Evolve synchronous pair interactions without touching caller RNGs."""
        from numbers import Integral
        if isinstance(self.time_steps, bool) or not isinstance(self.time_steps, Integral) or self.time_steps < 0:
            raise ValueError("time_steps must be a nonnegative integer")
        if any(not isinstance(agent, dict) for agent in self.agents):
            raise ValueError("agents must be state/energy mappings")
        try:
            states = [float(agent.get("initial_state", 0.0)) for agent in self.agents]
            energies = [float(agent.get("initial_energy", 1.0)) for agent in self.agents]
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Initial states and costs must be finite numbers") from error
        if not all(math.isfinite(value) for value in states + energies) or any(value < 0 for value in energies):
            raise ValueError("States must be finite and initial costs finite/nonnegative")
        n_agents = len(states)
        interactions = []
        for interaction in self.interactions:
            if len(interaction) != 3:
                raise ValueError("Interactions need two indices and a finite strength")
            first, second, strength = interaction
            if any(isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < n_agents
                   for index in (first, second)):
                raise ValueError("Interaction indices must identify existing agents")
            try:
                strength = float(strength)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("Interaction strengths must be finite") from error
            if not math.isfinite(strength):
                raise ValueError("Interaction strengths must be finite")
            interactions.append((int(first), int(second), strength))
        metrics = {name: [] for name in ("total_energy", "state_variance", "interaction_strength",
                                         "state_pair_similarity", "entropy")}
        strength_total = sum(abs(weight) for _, _, weight in interactions)
        for _ in range(self.time_steps):
            new_states, new_energies = states.copy(), energies.copy()
            for first, second, strength in interactions:
                delta = strength * (states[second] - states[first]) * .01
                new_states[first] += delta
                new_states[second] -= delta
                new_energies[first] += abs(delta)
                new_energies[second] += abs(delta)
            states, energies = new_states, new_energies
            try:
                total_cost, variance = sum(energies), self._variance(states)
            except OverflowError as error:
                raise ValueError("Toy state/cost aggregation overflowed") from error
            if not all(math.isfinite(value) for value in states + energies + [total_cost, variance]):
                raise ValueError("Toy state/cost aggregation must remain finite")
            metrics["total_energy"].append(total_cost)
            metrics["state_variance"].append(variance)
            metrics["interaction_strength"].append(strength_total)
            metrics["state_pair_similarity"].append(self._state_pair_similarity(states))
            metrics["entropy"].append(self._calculate_entropy(states))
        return metrics

    def _variance(self, values: List[float]) -> float:
        """Population variance of the complete modeled state, in either backend."""
        if not values:
            return 0.0
        mean = sum(value / len(values) for value in values)
        return sum((value - mean) ** 2 for value in values) / len(values)

    def _state_pair_similarity(self, states: List[float]) -> float:
        pairs = len(states) * (len(states) - 1) // 2
        if not pairs:
            return 0.0
        similar = sum(abs(first - second) < .1 for index, first in enumerate(states) for second in states[index + 1:])
        return similar / pairs

    def _calculate_entropy(self, states: List[float]) -> float:
        """Shannon entropy of the toy's equal-width state histogram, in bits."""
        if not states:
            return 0.0
        n_bins = min(10, len(states))
        if HAS_NUMPY:
            histogram, _ = np.histogram(states, bins=n_bins)
        else:
            low, high = min(states), max(states)
            width = (high - low) / n_bins if high > low else 1.0
            histogram = [0] * n_bins
            for state in states:
                histogram[min(int((state - low) / width), n_bins - 1)] += 1
        probabilities = (count / len(states) for count in histogram)
        return -sum(probability * math.log2(probability) for probability in probabilities if probability > 0)




class NetworkComplexityAnalyzer:
    """Metrics for finite nonnegative undirected graphs; self-loops are ignored.

    Paths are unweighted distances between distinct nodes (infinity if disconnected).
    Default modularity uses connected components, not an inferred optimal partition.
    Information measures sample an oriented edge in proportion to its weight:
    entropy is H(source), mutual_information is I(source; destination), in bits.
    """

    def __init__(self):
        self.backend_capabilities = {"numpy": HAS_NUMPY, "scipy": HAS_SCIPY}
        self.last_metrics: Optional[ComplexityMetrics] = None

    def _matrix(self, adjacency_matrix):
        matrix = [[float(value) for value in row] for row in adjacency_matrix]
        n = len(matrix)
        if any(len(row) != n for row in matrix):
            raise ValueError("Adjacency matrix must be square")
        if any(not math.isfinite(value) or value < 0 for row in matrix for value in row):
            raise ValueError("Edge weights must be finite and nonnegative")
        if any(matrix[i][j] != matrix[j][i] for i in range(n) for j in range(i)):
            raise ValueError("Undirected adjacency matrix must be symmetric")
        for i in range(n):
            matrix[i][i] = 0.0
        return matrix

    def analyze_network_complexity(self, adjacency_matrix: List[List[float]],
                                   communities: Optional[Sequence[Any]] = None) -> ComplexityMetrics:
        matrix = self._matrix(adjacency_matrix)
        self.last_metrics = ComplexityMetrics(
            network_density=self._calculate_density(matrix),
            clustering_coefficient=self._calculate_clustering_coefficient(matrix),
            path_length=self._calculate_path_length(matrix),
            modularity=self._calculate_modularity(matrix, communities),
            entropy=self._calculate_network_entropy(matrix),
            mutual_information=self._calculate_mutual_information(matrix),
        )
        return self.last_metrics

    def _calculate_density(self, adj_matrix) -> float:
        matrix = self._matrix(adj_matrix)
        n = len(matrix)
        return sum(value > 0 for row in matrix for value in row) / (n * (n - 1)) if n > 1 else 0.0

    def _calculate_clustering_coefficient(self, adj_matrix) -> float:
        matrix = self._matrix(adj_matrix)
        coefficients = []
        for row in matrix:
            neighbors = [j for j, value in enumerate(row) if value > 0]
            degree = len(neighbors)
            links = sum(matrix[j][k] > 0 for index, j in enumerate(neighbors) for k in neighbors[index + 1:])
            coefficients.append(2 * links / (degree * (degree - 1)) if degree > 1 else 0.0)
        return sum(coefficients) / len(matrix) if matrix else 0.0

    def _calculate_path_length(self, adj_matrix) -> float:
        matrix = self._matrix(adj_matrix)
        n = len(matrix)
        if n < 2:
            return 0.0
        total = 0
        for source in range(n):
            distances = {source: 0}
            queue = [source]
            for node in queue:
                for target, weight in enumerate(matrix[node]):
                    if weight > 0 and target not in distances:
                        distances[target] = distances[node] + 1
                        queue.append(target)
            if len(distances) != n:
                return math.inf
            total += sum(distances.values())
        return total / (n * (n - 1))

    def _calculate_modularity(self, adj_matrix, communities=None) -> float:
        """Weighted Newman modularity, including nonedges and diagonal terms."""
        matrix = self._matrix(adj_matrix)
        n = len(matrix)
        if communities is None:
            communities = [None] * n
            for source in range(n):
                if communities[source] is not None:
                    continue
                communities[source] = source
                queue = [source]
                for node in queue:
                    for target, weight in enumerate(matrix[node]):
                        if weight > 0 and communities[target] is None:
                            communities[target] = source
                            queue.append(target)
        if len(communities) != n:
            raise ValueError("Partition must label every node")
        degrees = [sum(row) for row in matrix]
        total = sum(degrees)
        if total == 0:
            return 0.0
        return sum(matrix[i][j] - degrees[i] * degrees[j] / total
                   for i in range(n) for j in range(n)
                   if communities[i] == communities[j]) / total

    def _calculate_network_entropy(self, adj_matrix) -> float:
        degrees = [sum(row) for row in self._matrix(adj_matrix)]
        total = sum(degrees)
        return -sum((degree / total) * math.log2(degree / total)
                    for degree in degrees if degree > 0) if total else 0.0

    def _calculate_mutual_information(self, adj_matrix) -> Optional[float]:
        matrix = self._matrix(adj_matrix)
        degrees = [sum(row) for row in matrix]
        total = sum(degrees)
        if not total:
            return None  # No edge-sampling probability distribution is defined.
        return sum((weight / total) * math.log2(weight * total / (degrees[i] * degrees[j]))
                   for i, row in enumerate(matrix) for j, weight in enumerate(row) if weight > 0)


class ThermodynamicComplexityAnalyzer:
    """Cycle-energy proxies, not a state-function thermodynamic model.

    Non-FLOP energy is treated as dissipated heat. Entropy change is Q/T (J/K),
    rate is Q/(T*duration_s) (J/K/s), and the free-energy proxy is work - T*Q/T (J).
    """

    def __init__(self, temperature_k: float = 300.0):
        if not math.isfinite(temperature_k) or temperature_k <= 0:
            raise ValueError("Temperature must be finite and positive")
        self.temperature_k = temperature_k
        self.k_boltzmann = 1.380649e-23

    def analyze_thermodynamic_efficiency(self, energy_breakdown: EnergyBreakdown,
                                       computational_work: float,
                                       information_processed: float,
                                       duration_s: float = 0.01) -> ComplexityMetrics:
        """Analyze a cycle; work and breakdown in J, information in bits, duration in s."""
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("Cycle duration must be finite and positive")
        if not math.isfinite(computational_work) or computational_work < 0:
            raise ValueError("Computational work must be finite and nonnegative joules")
        total = energy_breakdown.total
        if not math.isfinite(total) or total <= 0:
            raise ValueError("Efficiency requires finite positive total energy")
        entropy_change = self._calculate_entropy_production(energy_breakdown)
        return ComplexityMetrics(
            thermodynamic_efficiency=self._calculate_theoretical_minimum(information_processed) / total,
            entropy_production_rate=entropy_change / duration_s,
            entropy_change=entropy_change,
            free_energy=self._calculate_free_energy(computational_work, entropy_change),
        )

    def _calculate_theoretical_minimum(self, information_processed: float) -> float:
        if not math.isfinite(information_processed) or information_processed < 0:
            raise ValueError("Information must be finite and nonnegative bits")
        return information_processed * self.k_boltzmann * self.temperature_k * math.log(2)

    def _calculate_entropy_production(self, energy_breakdown: EnergyBreakdown) -> float:
        """Return cycle entropy-change proxy in J/K, not a rate."""
        dissipated = energy_breakdown.total - energy_breakdown.compute_flops
        if not math.isfinite(dissipated) or dissipated < 0:
            raise ValueError("Dissipated energy must be finite and nonnegative")
        return dissipated / self.temperature_k

    def _calculate_entropy_production_rate(self, energy_values: Sequence[float],
                                           duration_s: float) -> float:
        """Sum dissipated sample energies (J) over an explicit observation duration."""
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("Duration must be finite and positive")
        if any(not math.isfinite(value) or value < 0 for value in energy_values):
            raise ValueError("Dissipated energies must be finite and nonnegative")
        return sum(energy_values) / (self.temperature_k * duration_s)

    def _calculate_free_energy(self, computational_work: float, entropy_change: float) -> float:
        """Return work (J) minus temperature (K) times entropy change (J/K)."""
        if not math.isfinite(computational_work) or not math.isfinite(entropy_change):
            raise ValueError("Work and entropy change must be finite")
        return computational_work - self.temperature_k * entropy_change


class ComplexityEntropyAnalyzer:
    """Complexity-entropy analysis for intrinsic computation characterization."""
    
    def __init__(self):
        """Initialize complexity-entropy analyzer."""
        self.last_diagram: Optional[Dict[str, List[float]]] = None
    
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
            self.last_diagram = {'complexity': [], 'entropy': [], 'time_points': []}
            return self.last_diagram
        
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
        
        diagram = {
            'complexity': complexities,
            'entropy': entropies,
            'time_points': list(range(len(complexities)))
        }
        self.last_diagram = diagram
        return diagram
    
    def _calculate_lemple_ziv_complexity(self, sequence: List[float]) -> float:
        """LZ78 phrase count / sample count after four equal-width amplitude bins.

        Parse the shortest phrase not already in the dictionary, starting with the
        empty phrase. Count a remaining terminal phrase even if already known.
        References: 0000 -> 0|00|0 (3/4); 0123 -> 0|1|2|3 (1).
        This finite-length normalized count is not an entropy-rate estimator.
        """
        if not sequence:
            return 0.0
        if any(not math.isfinite(value) for value in sequence):
            raise ValueError("LZ input must be finite")
        low, high = min(sequence), max(sequence)
        symbols = [min(3, int(4 * (value - low) / (high - low))) if high > low else 0
                   for value in sequence]
        dictionary = {()}
        phrase = ()
        count = 0
        for symbol in symbols:
            phrase += (symbol,)
            if phrase not in dictionary:
                dictionary.add(phrase)
                count += 1
                phrase = ()
        if phrase:
            count += 1
        return count / len(symbols)
    
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

    def _calculate_conditional_entropy(self, sequence: List[float]) -> float:
        """Calculate a simple first-order conditional entropy estimate."""
        if len(sequence) < 2:
            return 0.0
        if HAS_NUMPY:
            quantiles = np.percentile(sequence, [25, 50, 75])
            discrete = [int(x) for x in np.digitize(sequence, quantiles)]
        else:
            min_val = min(sequence)
            max_val = max(sequence)
            width = (max_val - min_val) / 4 if max_val > min_val else 1.0
            discrete = [min(int((x - min_val) / width), 3) for x in sequence]

        transitions: Dict[int, Dict[int, int]] = {}
        for prev, current in zip(discrete, discrete[1:]):
            transitions.setdefault(prev, {})
            transitions[prev][current] = transitions[prev].get(current, 0) + 1

        total_pairs = len(discrete) - 1
        conditional = 0.0
        for counts in transitions.values():
            state_total = sum(counts.values())
            state_entropy = 0.0
            for count in counts.values():
                p = count / state_total
                state_entropy -= p * math.log2(p)
            conditional += (state_total / total_pairs) * state_entropy
        return conditional


class EnhancedBootstrapAnalyzer:
    """Paired-observation percentile bootstrap with a local reproducible RNG."""

    def __init__(self, n_bootstrap: int = 1000, seed: Optional[int] = 0):
        if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap < 1:
            raise ValueError("n_bootstrap must be a positive integer")
        self.n_bootstrap = n_bootstrap
        self.seed = seed

    def analyze_scaling_with_bootstrap(self, x_values: List[float], y_values: List[float],
                                     confidence_level: float = 0.95) -> Dict[str, Any]:
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must lie strictly between zero and one")
        if len(x_values) != len(y_values) or len(x_values) < 3:
            return {'valid': False, 'error': 'Insufficient data for bootstrap analysis'}
        original = analyze_scaling_relationship(x_values, y_values)
        if not original.get('valid', False):
            return original
        rng = random.Random(self.seed)
        exponents, intercepts, r_squared = [], [], []
        for _ in range(self.n_bootstrap):
            indices = [rng.randrange(len(x_values)) for _ in x_values]
            fit = analyze_scaling_relationship([x_values[i] for i in indices],
                                               [y_values[i] for i in indices])
            if fit.get('valid', False):
                exponents.append(fit['scaling_exponent'])
                intercepts.append(fit['intercept'])
                if fit['r_squared'] is not None:
                    r_squared.append(fit['r_squared'])
        if not exponents:
            return {**original, 'valid': False, 'error': 'No valid paired bootstrap fits',
                    'bootstrap_attempted': self.n_bootstrap, 'bootstrap_valid': 0}

        def quantile(values, probability):
            ordered = sorted(values)
            position = probability * (len(ordered) - 1)
            low = math.floor(position)
            high = math.ceil(position)
            return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

        alpha = (1 - confidence_level) / 2
        intervals = {name: (quantile(values, alpha), quantile(values, 1 - alpha)) if values else None
                     for name, values in [('exponent', exponents), ('intercept', intercepts),
                                          ('r_squared', r_squared)]}
        return {
            **original, 'valid': True, 'original_result': original,
            'confidence_interval': intervals['exponent'],
            'confidence_intervals': intervals,
            'bootstrap_exponents': exponents, 'bootstrap_intercepts': intercepts,
            'bootstrap_r_squared': r_squared,
            'bootstrap_attempted': self.n_bootstrap, 'bootstrap_valid': len(exponents),
            'bias_corrected': {
                'exponent': 2 * original['scaling_exponent'] - sum(exponents) / len(exponents),
                'intercept': 2 * original['intercept'] - sum(intercepts) / len(intercepts),
            },
            'bootstrap_stats': {
                'mean_exponent': sum(exponents) / len(exponents),
                'std_exponent': self._calculate_std(exponents),
                'mean_r_squared': sum(r_squared) / len(r_squared) if r_squared else None,
                'std_r_squared': self._calculate_std(r_squared) if r_squared else None,
            },
        }

    def _generate_bootstrap_sample(self, values: List[float]) -> List[float]:
        rng = random.Random(self.seed)
        return [values[rng.randrange(len(values))] for _ in values]

    def _calculate_std(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


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
