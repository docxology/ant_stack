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
import math

from .energy import _require_positive
from .workloads import _require_integer, validate_workload_parameters


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
            repeats=data.get("repeats", 5),
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

    def to_energy_coefficients(self):
        """Convert configuration values to the runtime EnergyCoefficients type."""
        from .energy import EnergyCoefficients

        return EnergyCoefficients(
            flops_pj=self.flops_pj,
            sram_pj_per_byte=self.sram_pj_per_byte,
            dram_pj_per_byte=self.dram_pj_per_byte,
            spike_aj=self.spike_aj,
            body_per_joint_w=self.body_per_joint_w,
            body_sensor_w_per_channel=self.body_sensor_w_per_channel,
            baseline_w=self.baseline_w,
        )
    
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
                 bootstrap_samples=1000, parameter_range=None, n_points=10, scaling_type="power_law"):
        """Initialize ScalingConfig with backward compatibility."""
        # Handle backward compatibility for parameter_range
        if parameter_range is not None and parameter_ranges is None:
            parameter_ranges = {"default": parameter_range}

        self.parameter_ranges = parameter_ranges or {}
        self.parameter_range = tuple(self.parameter_ranges.get("default", parameter_range or (1, 10)))
        self.analysis_methods = analysis_methods or ["power_law", "exponential", "polynomial"]
        self.confidence_level = confidence_level
        self.bootstrap_samples = bootstrap_samples
        self.n_points = n_points
        self.scaling_type = scaling_type

    def generate_parameter_values(self) -> List[float]:
        """Generate parameter values for the configured range and spacing."""
        start, stop = self.parameter_range
        if self.n_points <= 1:
            return [float(start)]
        if self.scaling_type == "logarithmic":
            if start <= 0 or stop <= 0:
                raise ValueError("Logarithmic scaling requires positive range bounds")
            import math

            log_start = math.log10(start)
            log_stop = math.log10(stop)
            step = (log_stop - log_start) / (self.n_points - 1)
            return [10 ** (log_start + i * step) for i in range(self.n_points)]
        step = (stop - start) / (self.n_points - 1)
        return [start + i * step for i in range(self.n_points)]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation of scaling configuration
        """
        return {
            "parameter_range": list(self.parameter_range),
            "parameter_ranges": {k: list(v) for k, v in self.parameter_ranges.items()},
            "analysis_methods": self.analysis_methods,
            "confidence_level": self.confidence_level,
            "bootstrap_samples": self.bootstrap_samples,
            "n_points": self.n_points,
            "scaling_type": self.scaling_type,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScalingConfig':
        """Create ScalingConfig from dictionary.
        
        Args:
            data: Dictionary containing scaling configuration
            
        Returns:
            ScalingConfig instance
        """
        return cls(
            parameter_ranges=data.get("parameter_ranges"),
            parameter_range=data.get("parameter_range"),
            analysis_methods=data.get("analysis_methods", ["power_law", "exponential", "polynomial"]),
            confidence_level=float(data.get("confidence_level", 0.95)),
            bootstrap_samples=int(data.get("bootstrap_samples", 1000)),
            n_points=int(data.get("n_points", 10)),
            scaling_type=data.get("scaling_type", "power_law"),
        )


@dataclass
class MeterConfig:
    """Only options actually consumed by the supported meter backends.

    Device availability is checked by the meter, not by config construction.
    No calibration, baseline-subtraction, sampler or logging service is implied.
    """
    meter_type: str = "null"
    device_index: int = 0
    energy_path: Optional[str] = None

    def __post_init__(self):
        from numbers import Integral
        self.meter_type = "null" if self.meter_type is None else self.meter_type
        if not isinstance(self.meter_type, str) or self.meter_type not in {"null", "rapl", "nvml"}:
            raise ValueError("Meter type must be null, rapl, or nvml")
        if isinstance(self.device_index, bool) or not isinstance(self.device_index, Integral) or self.device_index < 0:
            raise ValueError("device_index must be a nonnegative integer")
        self.device_index = int(self.device_index)
        if self.device_index and self.meter_type != "nvml":
            raise ValueError("device_index selects NVML devices only")
        if self.energy_path is not None:
            if self.meter_type != "rapl" or not isinstance(self.energy_path, str) or not self.energy_path.strip():
                raise ValueError("energy_path must be a nonempty RAPL counter path")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MeterConfig':
        if not isinstance(data, dict):
            raise ValueError("meter must be a mapping")
        unsupported = set(data) - {"meter_type", "type", "device_index", "energy_path"}
        if unsupported:
            raise ValueError("Unsupported meter controls: " + ", ".join(sorted(unsupported)))
        kind = data.get("meter_type", data.get("type", "null"))
        if "meter_type" in data and "type" in data:
            normalized = ["null" if value is None else value for value in (data["meter_type"], data["type"])]
            if normalized[0] != normalized[1]:
                raise ValueError("Conflicting meter type declarations")
        return cls(meter_type=kind, device_index=data.get("device_index", 0), energy_path=data.get("energy_path"))




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
    experiment_name: str = "Ant Stack experiment"
    seed: int = 0
    workloads: Dict[str, WorkloadConfig] = None
    coefficients: EnergyCoefficientsConfig = None
    scaling: ScalingConfig = None
    meter: MeterConfig = None
    provenance: Dict[str, Any] = None
    mass_kg: Optional[float] = None
    distance_m: Optional[float] = None

    def __init__(
        self,
        experiment_name: str = "Ant Stack experiment",
        seed: int = 0,
        workloads: Optional[Dict[str, WorkloadConfig]] = None,
        coefficients: Optional[EnergyCoefficientsConfig] = None,
        scaling: Optional[ScalingConfig] = None,
        meter: Optional[MeterConfig] = None,
        provenance: Optional[Dict[str, Any]] = None,
        mass_kg: Optional[float] = None,
        distance_m: Optional[float] = None,
        description: str = "",
        workload_configs: Optional[List[Union[WorkloadConfig, Dict[str, Any]]]] = None,
        energy_coefficients: Optional[EnergyCoefficientsConfig] = None,
        scaling_config: Optional[ScalingConfig] = None,
        meter_config: Optional[MeterConfig] = None,
        output_directory: str = "./output",
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize canonical and legacy manifest fields together."""
        self.experiment_name = experiment_name
        self.description = description
        self.seed = int(seed if random_seed is None else random_seed)
        self.random_seed = self.seed
        self.coefficients = coefficients or energy_coefficients or EnergyCoefficientsConfig()
        self.energy_coefficients = self.coefficients
        self.scaling = scaling or scaling_config or ScalingConfig()
        self.scaling_config = self.scaling
        self.meter = meter or meter_config or MeterConfig()
        self.meter_config = self.meter
        self.provenance = provenance or {}
        self.mass_kg = mass_kg
        self.distance_m = distance_m
        self.output_directory = output_directory

        if workload_configs is not None:
            normalized_configs = [
                item if isinstance(item, WorkloadConfig) else WorkloadConfig.from_dict(item)
                for item in workload_configs
            ]
            self.workload_configs = normalized_configs
            self.workloads = {cfg.name or f"workload_{i}": cfg for i, cfg in enumerate(normalized_configs)}
        else:
            self.workloads = workloads or {}
            self.workload_configs = list(self.workloads.values())

        self.__post_init__()
    
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
        self.workload_configs = list(self.workloads.values())
        self.energy_coefficients = self.coefficients
        self.scaling_config = self.scaling
        self.meter_config = self.meter
        self.random_seed = self.seed
    
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
        if not isinstance(data, dict):
            raise ValueError("Experiment manifest must be a mapping")
        
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
            experiment_name=data.get("experiment_name", path.stem),
            description=data.get("description", ""),
            seed=int(data.get("seed", 0)),
            workloads=workloads,
            coefficients=coefficients,
            scaling=scaling,
            meter=meter,
            provenance=data.get("provenance", {}) or {},
            output_directory=data.get("output_directory", "./output"),
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
            "description": self.description,
            "seed": self.seed,
            "random_seed": self.random_seed,
            "workloads": {k: v.to_dict() for k, v in self.workloads.items()},
            "workload_configs": [v.to_dict() for v in self.workload_configs],
            "coefficients": self.coefficients.to_dict(),
            "energy_coefficients": self.energy_coefficients.to_dict(),
            "scaling": self.scaling.to_dict(),
            "scaling_config": self.scaling_config.to_dict(),
            "meter": self.meter.to_dict(),
            "meter_config": self.meter_config.to_dict(),
            "provenance": self.provenance,
            "output_directory": self.output_directory,
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
            "description": self.description,
            "seed": self.seed,
            "random_seed": self.random_seed,
            "workloads": {k: v.to_dict() for k, v in self.workloads.items()},
            "workload_configs": [v.to_dict() for v in self.workload_configs],
            "coefficients": self.coefficients.to_dict(),
            "energy_coefficients": self.energy_coefficients.to_dict(),
            "scaling": self.scaling.to_dict(),
            "scaling_config": self.scaling_config.to_dict(),
            "meter": self.meter.to_dict(),
            "meter_config": self.meter_config.to_dict(),
            "provenance": self.provenance,
            "output_directory": self.output_directory,
            "mass_kg": self.mass_kg,
            "distance_m": self.distance_m,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentManifest':
        """Create ExperimentManifest from canonical or legacy dictionaries."""
        workload_configs = data.get("workload_configs")
        workloads = None
        if workload_configs is None:
            workloads = {
                name: WorkloadConfig.from_dict(value)
                for name, value in (data.get("workloads", {}) or {}).items()
            }
        return cls(
            experiment_name=data.get("experiment_name", "Ant Stack experiment"),
            description=data.get("description", ""),
            workload_configs=[WorkloadConfig.from_dict(v) for v in workload_configs] if workload_configs is not None else None,
            workloads=workloads,
            energy_coefficients=EnergyCoefficientsConfig.from_dict(data.get("energy_coefficients", data.get("coefficients", {})) or {}),
            scaling_config=ScalingConfig.from_dict(data.get("scaling_config", data.get("scaling", {})) or {}),
            meter_config=MeterConfig.from_dict(data.get("meter_config", data.get("meter", {})) or {}),
            output_directory=data.get("output_directory", "./output"),
            random_seed=int(data.get("random_seed", data.get("seed", 0))),
            provenance=data.get("provenance", {}) or {},
            mass_kg=data.get("mass_kg"),
            distance_m=data.get("distance_m"),
        )
    
    def validate(self) -> tuple[bool, List[str]]:
        """Validate experiment manifest configuration.
        
        Returns:
            Tuple of validity and validation errors.
        """
        errors = []

        def check(label, function, *args):
            try:
                function(*args)
            except (TypeError, ValueError, OverflowError) as exc:
                errors.append(f"{label}: {exc}")

        if not isinstance(self.experiment_name, str) or not self.experiment_name.strip():
            errors.append("Experiment name is required")
        if not self.workloads:
            errors.append("At least one workload is required")
        check("Seed", _require_integer, "seed", self.seed)
        for name, workload in self.workloads.items():
            check(f"Workload '{name}' duration", _require_positive, "duration_s", workload.duration_s)
            check(f"Workload '{name}' repeats", _require_integer, "repeats", workload.repeats, 1)
            if workload.mode not in {None, "loop", "closed_form"}:
                errors.append(f"Workload '{name}' mode must be loop or closed_form")
            check(f"Workload '{name}' parameters", validate_workload_parameters, name, workload.duration_s, workload.params)
        check("Energy coefficients", self.coefficients.to_energy_coefficients)
        if not 0 < self.scaling.confidence_level < 1:
            errors.append("Confidence level must be between 0 and 1")
        check("Bootstrap samples", _require_integer, "bootstrap_samples", self.scaling.bootstrap_samples, 1)
        check("Scaling points", _require_integer, "n_points", self.scaling.n_points, 1)
        for name, values in self.scaling.parameter_ranges.items():
            if not all(math.isfinite(v) for v in values):
                errors.append(f"Scaling range '{name}' must be finite")
        if self.meter.meter_type not in {"null", "rapl", "nvml"}:
            errors.append("Meter type must be null, rapl, or nvml")
        check("Meter device", _require_integer, "device_index", self.meter.device_index)
        if self.mass_kg is not None:
            check("Mass", _require_positive, "mass_kg", self.mass_kg)
        if self.distance_m is not None:
            check("Distance", _require_positive, "distance_m", self.distance_m)
        return not errors, errors

    def save_to_yaml(self, path: Union[str, Path]) -> None:
        """Backward-compatible YAML save alias."""
        self.save(path)

    @classmethod
    def load_from_yaml(cls, path: Union[str, Path]) -> 'ExperimentManifest':
        """Backward-compatible YAML load alias."""
        return cls.load(path)

    @classmethod
    def create_from_workloads(
        cls,
        experiment_name: str,
        workloads: List[Union[WorkloadConfig, Dict[str, Any]]],
        description: str = "",
        **kwargs: Any,
    ) -> 'ExperimentManifest':
        """Create a manifest from a workload list."""
        workload_configs = [
            item if isinstance(item, WorkloadConfig) else WorkloadConfig.from_dict(item)
            for item in workloads
        ]
        return cls(
            experiment_name=experiment_name,
            description=description,
            workload_configs=workload_configs,
            **kwargs,
        )
