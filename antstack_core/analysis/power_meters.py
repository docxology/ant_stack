"""Power measurement and energy monitoring for computational systems.

Comprehensive power measurement framework supporting:
- Multiple power measurement backends (RAPL, NVML, null)
- Context managers for energy measurement
- Cross-platform compatibility with graceful fallbacks
- Integration with energy analysis workflows

References:
- Intel RAPL: https://www.intel.com/content/www/us/en/developer/articles/technical/rapl-power-metering-and-the-linux-kernel.html
- NVIDIA NVML: https://developer.nvidia.com/nvidia-management-library-nvml
- Power measurement best practices: https://doi.org/10.1145/2485922.2485949
"""

from __future__ import annotations

import time
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Union
from abc import ABC, abstractmethod

try:
    import pynvml  # type: ignore
except ImportError:
    pynvml = None  # Optional dependency


@dataclass
class PowerSample:
    """Power measurement sample with timestamp and power reading.
    
    Attributes:
        timestamp: Unix timestamp of the measurement
        watts: Power consumption in watts
    """
    timestamp: float
    watts: float


class PowerMeter(ABC):
    """Abstract base class for power measurement devices."""
    
    @abstractmethod
    def read(self) -> PowerSample:
        """Read current power consumption.
        
        Returns:
            PowerSample with current power reading
        """
        pass


class NullPowerMeter(PowerMeter):
    """Fallback power meter returning 0 W. Useful for CI and offline estimation.
    
    This meter is used when no hardware power measurement is available
    or when running in environments where power measurement is not possible.
    """
    
    def read(self) -> PowerSample:
        """Return zero power reading.
        
        Returns:
            PowerSample with 0 watts
        """
        return PowerSample(timestamp=time.time(), watts=0.0)


class RaplPowerMeter(PowerMeter):
    """Linux RAPL-backed cumulative energy meter (CPU package).
    
    Uses Intel RAPL (Running Average Power Limit) to measure cumulative
    energy consumption. Energy is measured in microjoules and converted
    to power by taking the derivative over time.
    
    Defaults to ``/sys/class/powercap/intel-rapl:0/energy_uj``.
    
    References:
        - Intel RAPL overview: https://www.intel.com/content/www/us/en/developer/articles/technical/rapl-power-metering-and-the-linux-kernel.html
        - RAPL accuracy analysis: https://doi.org/10.1145/2485922.2485949
    """
    
    def __init__(self, energy_uj_path: Optional[str] = None) -> None:
        """Initialize RAPL power meter.
        
        Args:
            energy_uj_path: Path to RAPL energy file. Defaults to standard location.
        """
        default_path = "/sys/class/powercap/intel-rapl:0/energy_uj"
        self.energy_uj_path = energy_uj_path or default_path
        self._last_energy_uj = 0.0
        self._last_timestamp = 0.0
    
    def read(self) -> PowerSample:
        """Read current power consumption.
        
        Note: RAPL provides cumulative energy, not instantaneous power.
        This method returns 0 watts as instantaneous power is not directly available.
        Use the context manager for energy measurement.
        
        Returns:
            PowerSample with 0 watts (instantaneous power not available)
        """
        return PowerSample(timestamp=time.time(), watts=0.0)
    
    def read_energy_uj(self) -> float:
        """Read cumulative energy consumption in microjoules.
        
        Returns:
            Cumulative energy in microjoules since system boot
        """
        try:
            with open(self.energy_uj_path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
            return float(int(txt))
        except Exception:
            return 0.0


class NvmlPowerMeter(PowerMeter):
    """NVIDIA NVML-based power meter (GPU). Reports instantaneous power in Watts.
    
    Uses NVIDIA Management Library (NVML) to measure GPU power consumption.
    Energy is integrated by the ``measure_energy`` context manager.
    
    References:
        - NVIDIA NVML: https://developer.nvidia.com/nvidia-management-library-nvml
        - GPU power measurement: https://doi.org/10.1109/HPCA.2018.00014
    """
    
    def __init__(self, index: int = 0) -> None:
        """Initialize NVML power meter.
        
        Args:
            index: GPU device index (0 for first GPU)
        """
        self.index = index
        self.handle = None
        
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            except Exception:
                self.handle = None
        else:
            print("Warning: pynvml not available. NVML power meter will return 0 watts.")
    
    def read(self) -> PowerSample:
        """Read current GPU power consumption.
        
        Returns:
            PowerSample with current GPU power in watts
        """
        watts = 0.0
        if self.handle is not None:
            try:
                mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                watts = max(0.0, float(mw) / 1000.0)  # Convert mW to W
            except Exception:
                watts = 0.0
        return PowerSample(timestamp=time.time(), watts=watts)


@contextmanager
def measure_energy(meter: Optional[PowerMeter] = None):
    """Context manager for measuring energy consumption.
    
    Measures energy consumption over the duration of the context.
    Supports both instantaneous power measurement and cumulative energy measurement.
    
    Args:
        meter: Power meter to use. Defaults to NullPowerMeter.
        
    Yields:
        None
        
    Example:
        >>> with measure_energy(RaplPowerMeter()) as m:
        ...     # Your computation here
        ...     result = expensive_computation()
        >>> # Energy measurement is automatically logged
    """
    meter = meter or NullPowerMeter()
    t0 = time.time()
    p0 = 0.0
    
    # Check if meter supports direct energy reading
    energy_mode = hasattr(meter, "read_energy_uj")
    
    if not energy_mode:
        # Use instantaneous power measurement
        p0 = meter.read().watts
    else:
        # Use cumulative energy measurement
        try:
            e0_uj = float(getattr(meter, "read_energy_uj")())
        except Exception:
            e0_uj = 0.0
    
    try:
        yield
    finally:
        t1 = time.time()
        duration = max(0.0, t1 - t0)
        
        if energy_mode:
            # Calculate energy from cumulative readings
            try:
                e1_uj = float(getattr(meter, "read_energy_uj")())
            except Exception:
                e1_uj = e0_uj
            
            energy_j = max(0.0, (e1_uj - e0_uj) * 1e-6)  # Convert μJ to J
            avg_p = (energy_j / duration) if duration > 0 else 0.0
        else:
            # Calculate energy from power integration
            p1 = meter.read().watts
            # Trapezoidal integration with two samples (best-effort)
            avg_p = max(0.0, (p0 + p1) / 2.0)
            energy_j = avg_p * duration
        
        # Log measurement if enabled
        if str(os.environ.get("CE_METER_LOG", "0")) == "1":
            src = meter.__class__.__name__
            print(f"meter={src} energy_j={energy_j:.6f} duration_s={duration:.6f} avg_w={avg_p:.3f}")


def create_power_meter(meter_type: str = "null", **kwargs) -> PowerMeter:
    """Factory function to create power meters.
    
    Args:
        meter_type: Type of power meter ("null", "rapl", "nvml")
        **kwargs: Additional arguments passed to meter constructor
        
    Returns:
        Configured power meter instance
        
    Raises:
        ValueError: If meter_type is not supported
    """
    if meter_type == "null":
        return NullPowerMeter()
    elif meter_type == "rapl":
        return RaplPowerMeter(**kwargs)
    elif meter_type == "nvml":
        return NvmlPowerMeter(**kwargs)
    else:
        raise ValueError(f"Unsupported meter type: {meter_type}")


# Convenience aliases for backward compatibility
PowerSample = PowerSample
NullPowerMeter = NullPowerMeter
RaplPowerMeter = RaplPowerMeter
NvmlPowerMeter = NvmlPowerMeter
measure_energy = measure_energy
