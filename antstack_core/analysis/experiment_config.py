"""Experiment configuration and manifest management.

Comprehensive experiment configuration framework supporting:
- Workload specification and parameterization
- Energy coefficient configuration
- Scaling analysis parameters
- Power measurement configuration
- Experiment provenance tracking

Following .cursorrules principles:
- Real configuration management (no mocks)
- Professional, well-documented implementation
- Comprehensive validation and error handling
- Clear separation of concerns

References:
- Reproducible research practices: https://doi.org/10.1038/s41586-020-2196-x
- Experiment design methodology: https://doi.org/10.1371/journal.pcbi.1004668
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List, Union
from pathlib import Path
import yaml
import json


@dataclass
class WorkloadConfig:
    """Configuration for a computational workload experiment.
    
    Defines the parameters and execution settings for a specific workload
    in an energy analysis experiment.
    
    Attributes:
        name: Human-readable name for the workload
        duration_s: Duration of each workload execution in seconds
        repeats: Number of times to repeat the workload
        params: Workload-specific parameters (e.g., J, C, S for body workloads)
        mode: Execution mode ("loop" for wall-time, "closed_form" for deterministic)
    """
    name: str
    duration_s: float = 2.0
    repeats: int = 5
    params: Dict[str, Any] = None
    mode: Optional[str] = None  # "loop" (default) or "closed_form"
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.params is None:
            self.params = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation of the workload configuration
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkloadConfig':
        """Create WorkloadConfig from dictionary.
        
        Args:
            data: Dictionary containing workload configuration
            
        Returns:
            WorkloadConfig instance
        """
        return cls(
            name=data.get("name", ""),
            duration_s=float(data.get("duration_s", 2.0)),
            repeats=int(data.get("repeats", 5)),
            params=data.get("params", {}) or {},
            mode=data.get("mode")
        )


@dataclass
class EnergyCoefficientsConfig:
    """Configuration for energy coefficients used in analysis.
    
    Defines the energy costs for different computational operations
    and physical processes in the system.
    
    Attributes:
        flops_pj: Energy per FLOP in picojoules
        sram_pj_per_byte: Energy per SRAM byte access in picojoules
        dram_pj_per_byte: Energy per DRAM byte access in picojoules
        spike_aj: Energy per neuromorphic spike in attojoules
        body_per_joint_w: Power per joint for actuation in watts
        body_sensor_w_per_channel: Power per sensor channel in watts
        baseline_w: Baseline system power in watts
    """
    flops_pj: float = 1.0
    sram_pj_per_byte: float = 0.1
    dram_pj_per_byte: float = 20.0
    spike_aj: float = 1.0
    body_per_joint_w: float = 2.0
    body_sensor_w_per_channel: float = 0.005
    baseline_w: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation of energy coefficients
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnergyCoefficientsConfig':
        """Create EnergyCoefficientsConfig from dictionary.
        
        Args:
            data: Dictionary containing energy coefficients
            
        Returns:
            EnergyCoefficientsConfig instance
        """
        return cls(
            flops_pj=float(data.get("flops_pj", 1.0)),
            sram_pj_per_byte=float(data.get("sram_pj_per_byte", 0.1)),
            dram_pj_per_byte=float(data.get("dram_pj_per_byte", 20.0)),
            spike_aj=float(data.get("spike_aj", 1.0)),
            body_per_joint_w=float(data.get("body_per_joint_w", 2.0)),
            body_sensor_w_per_channel=float(data.get("body_sensor_w_per_channel", 0.005)),
            baseline_w=float(data.get("baseline_w", 0.0))
        )


@dataclass
class ScalingConfig:
    """Configuration for scaling analysis parameters.

    Defines the parameter ranges and analysis settings for
    computational scaling studies.

    Attributes:
        parameter_ranges: Dictionary mapping parameter names to value ranges
        analysis_methods: List of scaling analysis methods to apply
        confidence_level: Confidence level for statistical analysis (0.0 to 1.0)
        bootstrap_samples: Number of bootstrap samples for uncertainty quantification
        n_points: Number of points for parameter ranges (backward compatibility)
        scaling_type: Type of scaling analysis (backward compatibility)
    """
    parameter_ranges: Dict[str, List[float]] = None
    analysis_methods: List[str] = None
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    n_points: int = 20  # For backward compatibility
    scaling_type: str = "power_law"  # For backward compatibility

    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.parameter_ranges is None:
            self.parameter_ranges = {}
        if self.analysis_methods is None:
            self.analysis_methods = ["power_law", "exponential", "polynomial"]

    def __init__(self, parameter_ranges=None, analysis_methods=None, confidence_level=0.95,
                 bootstrap_samples=1000, parameter_range=None, n_points=20, scaling_type="power_law"):
        """Initialize ScalingConfig with backward compatibility."""
        # Handle backward compatibility for parameter_range
        if parameter_range is not None and parameter_ranges is None:
            parameter_ranges = {"default": parameter_range}

        self.parameter_ranges = parameter_ranges or {}
        self.analysis_methods = analysis_methods or ["power_law", "exponential", "polynomial"]
        self.confidence_level = confidence_level
        self.bootstrap_samples = bootstrap_samples
        self.n_points = n_points
        self.scaling_type = scaling_type

    def to_energy_coefficients(self) -> 'EnergyCoefficientsConfig':
        """Convert to EnergyCoefficientsConfig for backward compatibility.

        Returns:
            EnergyCoefficientsConfig instance
        """
        # This is a placeholder implementation for backward compatibility
        # In a real implementation, this would extract energy coefficients from scaling config
        return EnergyCoefficientsConfig()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation of scaling configuration
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScalingConfig':
        """Create ScalingConfig from dictionary.
        
        Args:
            data: Dictionary containing scaling configuration
            
        Returns:
            ScalingConfig instance
        """
        return cls(
            parameter_ranges=data.get("parameter_ranges", {}),
            analysis_methods=data.get("analysis_methods", ["power_law", "exponential", "polynomial"]),
            confidence_level=float(data.get("confidence_level", 0.95)),
            bootstrap_samples=int(data.get("bootstrap_samples", 1000))
        )


@dataclass
class MeterConfig:
    """Configuration for power measurement devices.

    Defines the settings for power measurement and energy monitoring
    during experiments.

    Attributes:
        meter_type: Type of power meter ("null", "rapl", "nvml")
        device_index: Device index for multi-device systems
        sampling_rate_hz: Sampling rate for power measurements
        measurement_duration_s: Duration of measurement period
        channels: List of measurement channels
        calibration_file: Path to calibration file
        baseline_measurement: Whether to perform baseline measurements
        energy_path: Path to energy measurement file (for RAPL)
        log_measurements: Whether to log power measurements
    """
    meter_type: str = "null"
    device_index: int = 0
    sampling_rate_hz: float = 1000.0
    measurement_duration_s: float = 10.0
    channels: List[str] = None
    calibration_file: Optional[str] = None
    baseline_measurement: bool = False
    energy_path: Optional[str] = None
    log_measurements: bool = False

    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.channels is None:
            self.channels = ["cpu", "memory", "disk"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation of meter configuration
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MeterConfig':
        """Create MeterConfig from dictionary.

        Args:
            data: Dictionary containing meter configuration

        Returns:
            MeterConfig instance
        """
        return cls(
            meter_type=data.get("meter_type", "null"),
            device_index=int(data.get("device_index", 0)),
            sampling_rate_hz=float(data.get("sampling_rate_hz", 1000.0)),
            measurement_duration_s=float(data.get("measurement_duration_s", 10.0)),
            channels=data.get("channels", ["cpu", "memory", "disk"]),
            calibration_file=data.get("calibration_file"),
            baseline_measurement=bool(data.get("baseline_measurement", False)),
            energy_path=data.get("energy_path"),
            log_measurements=bool(data.get("log_measurements", False))
        )


@dataclass
class ExperimentManifest:
    """Complete experiment configuration manifest.

    Comprehensive configuration for energy analysis experiments including
    workload specifications, energy coefficients, scaling parameters,
    power measurement settings, and provenance information.

    Attributes:
        experiment_name: Name of the experiment (backward compatibility)
        seed: Random seed for reproducible experiments
        workloads: Dictionary of workload configurations
        coefficients: Energy coefficients configuration
        scaling: Scaling analysis configuration
        meter: Power measurement configuration
        provenance: Experiment provenance and metadata
        mass_kg: System mass for cost-of-transport calculations
        distance_m: Distance for cost-of-transport calculations
    """
    experiment_name: str = ""  # For backward compatibility
    seed: int = 0
    workloads: Dict[str, WorkloadConfig] = None
    coefficients: EnergyCoefficientsConfig = None
    scaling: ScalingConfig = None
    meter: MeterConfig = None
    provenance: Dict[str, Any] = None
    mass_kg: Optional[float] = None
    distance_m: Optional[float] = None
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.workloads is None:
            self.workloads = {}
        if self.coefficients is None:
            self.coefficients = EnergyCoefficientsConfig()
        if self.scaling is None:
            self.scaling = ScalingConfig()
        if self.meter is None:
            self.meter = MeterConfig()
        if self.provenance is None:
            self.provenance = {}
    
    @staticmethod
    def load(path: Union[str, Path]) -> 'ExperimentManifest':
        """Load experiment manifest from YAML file.
        
        Args:
            path: Path to YAML manifest file
            
        Returns:
            ExperimentManifest instance
            
        Raises:
            FileNotFoundError: If manifest file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        # Parse workloads
        workloads = {}
        for k, v in (data.get("workloads", {}) or {}).items():
            workloads[k] = WorkloadConfig.from_dict(v)
        
        # Parse coefficients
        coefficients = EnergyCoefficientsConfig.from_dict(
            data.get("coefficients", {}) or {}
        )
        
        # Parse scaling
        scaling = ScalingConfig.from_dict(
            data.get("scaling", {}) or {}
        )
        
        # Parse meter
        meter = MeterConfig.from_dict(
            data.get("meter", {}) or {}
        )
        
        return ExperimentManifest(
            experiment_name=data.get("experiment_name", ""),
            seed=int(data.get("seed", 0)),
            workloads=workloads,
            coefficients=coefficients,
            scaling=scaling,
            meter=meter,
            provenance=data.get("provenance", {}) or {},
            mass_kg=float(data.get("mass_kg")) if data.get("mass_kg") is not None else None,
            distance_m=float(data.get("distance_m")) if data.get("distance_m") is not None else None,
        )
    
    def save(self, path: Union[str, Path]) -> None:
        """Save experiment manifest to YAML file.
        
        Args:
            path: Path to save YAML manifest file
            
        Raises:
            OSError: If file cannot be written
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "workloads": {k: v.to_dict() for k, v in self.workloads.items()},
            "coefficients": self.coefficients.to_dict(),
            "scaling": self.scaling.to_dict(),
            "meter": self.meter.to_dict(),
            "provenance": self.provenance,
        }
        
        if self.mass_kg is not None:
            data["mass_kg"] = self.mass_kg
        if self.distance_m is not None:
            data["distance_m"] = self.distance_m
        
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the experiment manifest
        """
        return {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "workloads": {k: v.to_dict() for k, v in self.workloads.items()},
            "coefficients": self.coefficients.to_dict(),
            "scaling": self.scaling.to_dict(),
            "meter": self.meter.to_dict(),
            "provenance": self.provenance,
            "mass_kg": self.mass_kg,
            "distance_m": self.distance_m,
        }
    
    def validate(self) -> List[str]:
        """Validate experiment manifest configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate seed
        if self.seed < 0:
            errors.append("Seed must be non-negative")
        
        # Validate workloads
        for name, workload in self.workloads.items():
            if workload.duration_s <= 0:
                errors.append(f"Workload '{name}' duration must be positive")
            if workload.repeats <= 0:
                errors.append(f"Workload '{name}' repeats must be positive")
        
        # Validate coefficients
        if self.coefficients.flops_pj <= 0:
            errors.append("FLOP energy coefficient must be positive")
        if self.coefficients.sram_pj_per_byte <= 0:
            errors.append("SRAM energy coefficient must be positive")
        if self.coefficients.dram_pj_per_byte <= 0:
            errors.append("DRAM energy coefficient must be positive")
        
        # Validate scaling
        if not 0 < self.scaling.confidence_level < 1:
            errors.append("Confidence level must be between 0 and 1")
        if self.scaling.bootstrap_samples <= 0:
            errors.append("Bootstrap samples must be positive")
        
        # Validate physical parameters for CoT
        if self.mass_kg is not None and self.mass_kg <= 0:
            errors.append("Mass must be positive")
        if self.distance_m is not None and self.distance_m <= 0:
            errors.append("Distance must be positive")
        
        return errors


# Convenience aliases for backward compatibility
WorkloadConfig = WorkloadConfig
ExperimentManifest = ExperimentManifest
