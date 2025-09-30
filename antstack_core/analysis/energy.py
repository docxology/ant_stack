"""Energy estimation and analysis for computational systems.

Comprehensive energy modeling framework supporting:
- Multi-component energy breakdown (compute, memory, actuation, sensing)
- Device-specific energy coefficients with uncertainty quantification
- Theoretical energy limits (Landauer limit, thermodynamic bounds)
- Energy efficiency metrics and cost-of-transport calculations

References:
- Landauer limit: https://doi.org/10.1147/rd.53.0183
- Energy efficiency of computation: https://ieeexplore.ieee.org/document/8845760
- Neuromorphic energy scaling: https://doi.org/10.1038/s41928-017-0002-z
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple
import math


@dataclass(frozen=True)
class EnergyCoefficients:
    """Device- and system-level energy coefficients for comprehensive modeling.
    
    Provides energy costs for different computational and physical operations:
    - FLOP operations (digital computation)
    - Memory accesses (SRAM/DRAM hierarchy)  
    - Neuromorphic spikes (event-driven computation)
    - Physical actuation and sensing
    - System baseline/idle power
    
    References:
        - FLOP energy survey: https://ieeexplore.ieee.org/document/8845760
        - Memory energy analysis: https://doi.org/10.1109/ISSCC.2012.6176927  
        - Neuromorphic benchmarks: https://doi.org/10.1038/s41928-017-0002-z
    """
    # Digital computation energy costs
    flops_pj: float = 1.0                    # pJ per FLOP (modern processors)
    sram_pj_per_byte: float = 0.1            # pJ per SRAM byte access
    dram_pj_per_byte: float = 20.0           # pJ per DRAM byte access
    
    # Neuromorphic computation
    spike_aj: float = 1.0                    # aJ per spike (advanced neuromorphic)
    
    # Physical system energy costs
    body_per_joint_w: float = 2.0            # W per joint (average under gait)
    body_sensor_w_per_channel: float = 0.005 # W per sensor channel (duty-cycled)
    
    # System baseline power
    baseline_w: float = 0.50  # Matches paper config (was 0.0)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization and analysis."""
        return asdict(self)
    
    def scale_by_technology(self, node_nm: float) -> 'EnergyCoefficients':
        """Scale coefficients by technology node (rough approximation).
        
        Args:
            node_nm: Technology node in nanometers (e.g., 7, 14, 28)
            
        Returns:
            New EnergyCoefficients with scaled values
            
        Note: This is a rough approximation. Actual scaling depends on
        circuit design, voltage scaling, and process variations.
        """
        # Rough scaling based on technology trends
        scale_factor = (node_nm / 7.0) ** 1.5  # Empirical approximation
        
        return EnergyCoefficients(
            flops_pj=self.flops_pj * scale_factor,
            sram_pj_per_byte=self.sram_pj_per_byte * scale_factor,
            dram_pj_per_byte=self.dram_pj_per_byte * scale_factor,
            spike_aj=self.spike_aj * scale_factor,
            body_per_joint_w=self.body_per_joint_w,  # Physical doesn't scale
            body_sensor_w_per_channel=self.body_sensor_w_per_channel,
            baseline_w=self.baseline_w
        )


@dataclass(frozen=True)
class ComputeLoad:
    """Computational workload specification for energy analysis.
    
    Defines the computational requirements of a task:
    - FLOPs: Floating point operations
    - Memory: SRAM and DRAM byte transfers
    - Spikes: Neuromorphic spike events
    - Additional metrics: Memory bandwidth, cache hits, etc.
    """
    flops: float = 0.0           # Total floating point operations
    sram_bytes: float = 0.0      # SRAM bytes transferred  
    dram_bytes: float = 0.0      # DRAM bytes transferred
    spikes: float = 0.0          # Neuromorphic spike events
    
    # Additional optional metrics
    memory_bandwidth_gb_s: float = 0.0  # Memory bandwidth utilization
    cache_hit_rate: float = 0.95        # Cache hit rate (0.0 to 1.0)
    
    def scale(self, factor: float) -> 'ComputeLoad':
        """Scale all load components by a constant factor."""
        return ComputeLoad(
            flops=self.flops * factor,
            sram_bytes=self.sram_bytes * factor,
            dram_bytes=self.dram_bytes * factor,
            spikes=self.spikes * factor,
            memory_bandwidth_gb_s=self.memory_bandwidth_gb_s * factor,
            cache_hit_rate=self.cache_hit_rate  # Hit rate doesn't scale
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for analysis and serialization."""
        return asdict(self)


@dataclass
class EnergyBreakdown:
    """Detailed energy breakdown with component analysis and uncertainty.
    
    Provides comprehensive energy accounting across all system components:
    - Actuation: Physical movement and control
    - Sensing: Data acquisition and processing
    - Compute: Digital and neuromorphic computation
    - Memory: Data storage and transfer
    - Baseline: System idle power
    """
    actuation: float = 0.0       # Physical actuation energy (J)
    sensing: float = 0.0         # Sensing and data acquisition (J)
    compute_flops: float = 0.0   # Digital computation (J)
    compute_memory: float = 0.0  # Memory access energy (J)  
    compute_spikes: float = 0.0  # Neuromorphic computation (J)
    baseline: float = 0.0        # Baseline/idle energy (J)
    
    # Uncertainty quantification
    total_uncertainty: float = 0.0  # Total uncertainty estimate (J)
    
    @property
    def total_compute(self) -> float:
        """Total compute energy across all compute components."""
        return self.compute_flops + self.compute_memory + self.compute_spikes
    
    @property
    def total(self) -> float:
        """Total energy across all components."""
        return (self.actuation + self.sensing + self.total_compute + 
                self.baseline)
    
    @property  
    def compute_fraction(self) -> float:
        """Fraction of total energy used for computation."""
        total = self.total
        return self.total_compute / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for analysis and plotting."""
        return {
            'Actuation': self.actuation,
            'Sensing': self.sensing,
            'Compute (FLOPs)': self.compute_flops,
            'Compute (Memory)': self.compute_memory,
            'Compute (Spikes)': self.compute_spikes,
            'Baseline': self.baseline,
            'Total': self.total
        }
    
    def dominant_component(self) -> str:
        """Identify the dominant energy component."""
        components = {
            'Actuation': self.actuation,
            'Sensing': self.sensing,
            'Compute (FLOPs)': self.compute_flops,
            'Compute (Memory)': self.compute_memory,
            'Compute (Spikes)': self.compute_spikes,
            'Baseline': self.baseline
        }
        return max(components.items(), key=lambda x: x[1])[0]


def estimate_detailed_energy(
    load: ComputeLoad, 
    coeffs: EnergyCoefficients, 
    duration_s: float = 0.01,
    actuation_energy: float = 0.0,
    sensing_energy: float = 0.0
) -> EnergyBreakdown:
    """Estimate detailed energy breakdown for a computational workload.
    
    Performs comprehensive energy estimation across all system components
    with uncertainty quantification and efficiency analysis.
    
    Args:
        load: Computational workload specification
        coeffs: Energy coefficients for the target system
        duration_s: Time duration for the workload
        actuation_energy: Additional actuation energy (J)
        sensing_energy: Additional sensing energy (J)
        
    Returns:
        Detailed energy breakdown with component analysis
        
    References:
        - Energy modeling methodology: https://doi.org/10.1145/2485922.2485949
        - System-level energy analysis: https://doi.org/10.1109/MC.2003.1250885
    """
    # Digital computation energy
    compute_flops_j = pj_to_j(load.flops * coeffs.flops_pj)
    
    # Memory access energy (hierarchy-aware)
    sram_energy_j = pj_to_j(load.sram_bytes * coeffs.sram_pj_per_byte)
    dram_energy_j = pj_to_j(load.dram_bytes * coeffs.dram_pj_per_byte)
    compute_memory_j = sram_energy_j + dram_energy_j
    
    # Neuromorphic computation energy  
    compute_spikes_j = load.spikes * coeffs.spike_aj * 1e-18  # aJ to J
    
    # Baseline energy over duration
    baseline_j = coeffs.baseline_w * duration_s
    
    # Uncertainty estimation (rough approximation)
    # Typically 10-20% for energy measurements
    total_compute = compute_flops_j + compute_memory_j + compute_spikes_j
    total_energy = (actuation_energy + sensing_energy + total_compute + 
                   baseline_j)
    uncertainty = total_energy * 0.15  # 15% typical uncertainty
    
    return EnergyBreakdown(
        actuation=actuation_energy,
        sensing=sensing_energy,
        compute_flops=compute_flops_j,
        compute_memory=compute_memory_j,
        compute_spikes=compute_spikes_j,
        baseline=baseline_j,
        total_uncertainty=uncertainty
    )


def pj_to_j(pj: float) -> float:
    """Convert picojoules to joules.
    
    Args:
        pj: Energy in picojoules
        
    Returns:
        Energy in joules
    """
    return pj * 1e-12


def aj_to_j(aj: float) -> float:
    """Convert attojoules to joules.
    
    Args:
        aj: Energy in attojoules
        
    Returns:
        Energy in joules
    """
    return aj * 1e-18


# Physical constants
GRAVITY_M_S2: float = 9.80665


# Additional unit conversion functions
def j_to_mj(joules: float) -> float:
    """Convert joules to millijoules.
    
    Args:
        joules: Energy in joules
        
    Returns:
        Energy in millijoules
    """
    return joules * 1e3


def j_to_kj(joules: float) -> float:
    """Convert joules to kilojoules.
    
    Args:
        joules: Energy in joules
        
    Returns:
        Energy in kilojoules
    """
    return joules * 1e-3


def j_to_wh(joules: float) -> float:
    """Convert joules to watt-hours.
    
    Args:
        joules: Energy in joules
        
    Returns:
        Energy in watt-hours
    """
    return joules / 3600.0


def wh_to_j(wh: float) -> float:
    """Convert watt-hours to joules.
    
    Args:
        wh: Energy in watt-hours
        
    Returns:
        Energy in joules
    """
    return wh * 3600.0


def w_to_mw(watts: float) -> float:
    """Convert watts to milliwatts.
    
    Args:
        watts: Power in watts
        
    Returns:
        Power in milliwatts
    """
    return watts * 1e3


def mw_to_w(milliwatts: float) -> float:
    """Convert milliwatts to watts.
    
    Args:
        milliwatts: Power in milliwatts
        
    Returns:
        Power in watts
    """
    return milliwatts * 1e-3


def s_to_ms(seconds: float) -> float:
    """Convert seconds to milliseconds.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Time in milliseconds
    """
    return seconds * 1e3


def ms_to_s(milliseconds: float) -> float:
    """Convert milliseconds to seconds.
    
    Args:
        milliseconds: Time in milliseconds
        
    Returns:
        Time in seconds
    """
    return milliseconds * 1e-3


def integrate_power_to_energy(power_watts: list[float], timestamps_s: list[float]) -> float:
    """Compute energy in Joules by trapezoidal integration of power over time.

    Lists must be the same length (≥ 2) and timestamps strictly increasing.
    Returns 0.0 for invalid inputs.
    
    Args:
        power_watts: List of power measurements in watts
        timestamps_s: List of timestamps in seconds
        
    Returns:
        Total energy in joules
    """
    n = len(power_watts)
    if n < 2 or n != len(timestamps_s):
        return 0.0
    e = 0.0
    for i in range(n - 1):
        dt = timestamps_s[i + 1] - timestamps_s[i]
        if dt <= 0:
            continue
        p_avg = max(0.0, (power_watts[i] + power_watts[i + 1]) / 2.0)
        e += p_avg * dt
    return e


def estimate_compute_energy(load: ComputeLoad, coeffs: EnergyCoefficients) -> float:
    """Translate counters to estimated energy in Joules using device coefficients.
    
    Args:
        load: Computational workload specification
        coeffs: Energy coefficients for different operations
        
    Returns:
        Total energy in joules
    """
    e = 0.0
    e += pj_to_j(coeffs.flops_pj) * load.flops
    e += pj_to_j(coeffs.sram_pj_per_byte) * load.sram_bytes
    e += pj_to_j(coeffs.dram_pj_per_byte) * load.dram_bytes
    e += aj_to_j(coeffs.spike_aj) * load.spikes
    return e


def add_baseline_energy(energy_j: float, duration_s: float, coeffs: EnergyCoefficients) -> float:
    """Add baseline/idle energy for the measured duration (J = W · s).
    
    Args:
        energy_j: Base energy consumption in joules
        duration_s: Duration in seconds
        coeffs: Energy coefficients containing baseline power
        
    Returns:
        Total energy including baseline
    """
    if coeffs.baseline_w <= 0.0 or duration_s <= 0.0:
        return energy_j
    return energy_j + coeffs.baseline_w * duration_s


def cost_of_transport(energy_j: float, mass_kg: float, distance_m: float) -> float:
    """Compute dimensionless Cost of Transport: E / (m g d).
    
    Cost of Transport is a dimensionless metric for locomotion efficiency,
    commonly used in robotics and biomechanics.
    
    Args:
        energy_j: Energy consumed in joules
        mass_kg: System mass in kilograms
        distance_m: Distance traveled in meters
        
    Returns:
        Cost of Transport (dimensionless), or 0 for invalid inputs
    """
    if energy_j <= 0.0 or mass_kg <= 0.0 or distance_m <= 0.0:
        return 0.0
    return energy_j / (mass_kg * GRAVITY_M_S2 * distance_m)


def calculate_landauer_limit(bits_erased: float, temperature_k: float = 300.0) -> float:
    """Calculate the Landauer limit for irreversible computation.
    
    The fundamental thermodynamic limit for erasing information.
    
    Args:
        bits_erased: Number of bits irreversibly erased
        temperature_k: Temperature in Kelvin (default: room temperature)
        
    Returns:
        Minimum energy required (Joules)
        
    References:
        - Landauer's principle: https://doi.org/10.1147/rd.53.0183
        - Quantum limits of computation: https://doi.org/10.1103/PhysRevLett.85.441
    """
    k_boltzmann = 1.380649e-23  # J/K (Boltzmann constant)
    return bits_erased * k_boltzmann * temperature_k * math.log(2)


