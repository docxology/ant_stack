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
from pathlib import Path
from .energy import _require_nonnegative, _require_positive


class PowerMeasurementUnavailable(RuntimeError):
    """The requested hardware measurement could not be obtained."""

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
    watts: Optional[float]
    available: bool = True
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.available:
            _require_nonnegative("watts", self.watts)
        elif self.watts is not None:
            raise ValueError("Unavailable samples must not carry a power value")


class PowerMeter(ABC):
    """Abstract base class for power measurement devices."""
    
    @abstractmethod
    def read(self) -> PowerSample:
        """Read current power consumption.
        
        Returns:
            PowerSample with current power reading
        """
        raise NotImplementedError


class NullPowerMeter(PowerMeter):
    """Explicitly disabled meter, not a hardware fallback or empirical zero."""
    
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
    
    def __init__(self, energy_uj_path: Optional[str] = None, *, max_energy_range_uj: Optional[float] = None) -> None:
        """Initialize RAPL power meter.
        
        Args:
            energy_uj_path: Path to RAPL energy file. Defaults to standard location.
        """
        default_path = "/sys/class/powercap/intel-rapl:0/energy_uj"
        self.energy_uj_path = energy_uj_path or default_path
        self._last_energy_uj: Optional[float] = None
        self._last_timestamp: Optional[float] = None
        if max_energy_range_uj is not None:
            _require_positive("max_energy_range_uj", max_energy_range_uj)
        self.max_energy_range_uj = max_energy_range_uj
    
    def read(self) -> PowerSample:
        """Read current power consumption.
        
        Returns:
            PowerSample with estimated watts from cumulative energy deltas.
        """
        now = time.monotonic()
        energy_uj = self.read_energy_uj()
        if self._last_energy_uj is None or self._last_timestamp is None:
            self._last_energy_uj = energy_uj
            self._last_timestamp = now
            return PowerSample(time.time(), None, available=False, error="Two counter readings are required")
        elapsed = now - self._last_timestamp
        if elapsed <= 0:
            raise PowerMeasurementUnavailable("RAPL elapsed time must be positive")
        delta_uj = self.energy_delta_uj(self._last_energy_uj, energy_uj)
        watts = delta_uj * 1e-6 / elapsed
        self._last_energy_uj = energy_uj
        self._last_timestamp = now
        return PowerSample(timestamp=time.time(), watts=watts)

    def energy_delta_uj(self, start: float, end: float) -> float:
        """Counter difference, allowing at most one wrap with a known range.

        Endpoints alone cannot detect multiple complete wraps; callers must
        choose a measurement interval shorter than the counter wrap period.
        """
        _require_nonnegative("start counter", start)
        _require_nonnegative("end counter", end)
        counter_range = self.max_energy_range_uj
        if end < start and counter_range is None:
            range_path = Path(self.energy_uj_path).with_name("max_energy_range_uj")
            try:
                counter_range = float(range_path.read_text(encoding="utf-8").strip())
                _require_positive("max_energy_range_uj", counter_range)
            except (OSError, ValueError) as exc:
                raise PowerMeasurementUnavailable("RAPL counter decreased without a known wrap range") from exc
        if counter_range is not None and (start >= counter_range or end >= counter_range):
            raise PowerMeasurementUnavailable("RAPL counter is outside its declared range")
        return end - start if end >= start else counter_range - start + end
    
    def read_energy_uj(self) -> float:
        """Read cumulative energy consumption in microjoules.
        
        Returns:
            Cumulative energy in microjoules since system boot
        """
        try:
            with open(self.energy_uj_path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
            value = float(int(txt))
            _require_nonnegative("RAPL counter", value)
            return value
        except (OSError, ValueError) as exc:
            raise PowerMeasurementUnavailable(f"Cannot read RAPL counter: {self.energy_uj_path}") from exc


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
        self.pynvml = pynvml
        self._initialized = False
        self._available = False

    def _backend(self):
        """Return the active NVML backend, honoring runtime module replacement."""
        if pynvml is not None and pynvml is not self.pynvml:
            return pynvml
        return self.pynvml

    def _ensure_initialized(self) -> None:
        """Initialize NVML lazily."""
        backend = self._backend()
        if backend is None:
            raise PowerMeasurementUnavailable("NVML backend is unavailable")
        if self._initialized:
            if not self._available:
                raise PowerMeasurementUnavailable("NVML initialization failed")
            return
        try:
            backend.nvmlInit()
            self.handle = backend.nvmlDeviceGetHandleByIndex(self.index)
            self.pynvml = backend
            self._initialized = True
            self._available = True
        except Exception as exc:
            self.handle = None
            self._initialized = True
            self._available = False
            raise PowerMeasurementUnavailable("NVML initialization failed") from exc
    
    def read(self) -> PowerSample:
        """Read current GPU power consumption.
        
        Returns:
            PowerSample with current GPU power in watts
        """
        self._ensure_initialized()
        backend = self._backend()
        try:
            watts = float(backend.nvmlDeviceGetPowerUsage(self.handle)) / 1000.0
            return PowerSample(timestamp=time.time(), watts=watts)
        except Exception as exc:
            raise PowerMeasurementUnavailable("NVML power reading failed") from exc


@contextmanager
def measure_energy(meter: Optional[PowerMeter] = None):
    """Context manager for measuring energy consumption.
    
    Measures energy consumption over the duration of the context.
    Supports both instantaneous power measurement and cumulative energy measurement.
    
    Args:
        meter: Power meter to use. Defaults to NullPowerMeter.
        
    Yields:
        Measurement dictionary with explicit status and measured flag.
        
    Example:
        >>> with measure_energy(RaplPowerMeter()) as m:
        ...     # Your computation here
        ...     result = expensive_computation()
        >>> # Energy measurement is automatically logged
    """
    meter = meter or NullPowerMeter()
    disabled = isinstance(meter, NullPowerMeter)
    measurement = {
        "energy_joules": None,
        "duration_seconds": None,
        "average_power_watts": None,
        "samples": [],
        "status": "disabled" if disabled else "pending",
        "measured": False,
    }
    t0 = time.monotonic()
    energy_mode = isinstance(meter, RaplPowerMeter)
    if energy_mode:
        e0_uj = meter.read_energy_uj()
    elif not disabled:
        sample0 = meter.read()
        if not sample0.available:
            raise PowerMeasurementUnavailable(sample0.error or "Initial power sample unavailable")
        measurement["samples"].append(sample0)
        p0 = sample0.watts
    try:
        yield measurement
    finally:
        duration = time.monotonic() - t0
        _require_nonnegative("measurement duration", duration)
        if disabled:
            energy_j = avg_p = 0.0  # Explicitly disabled, never an observed zero.
        elif energy_mode:
            energy_j = meter.energy_delta_uj(e0_uj, meter.read_energy_uj()) * 1e-6
            if duration == 0 and energy_j > 0:
                raise PowerMeasurementUnavailable("Positive energy with zero elapsed time")
            avg_p = energy_j / duration if duration > 0 else 0.0
        else:
            sample1 = meter.read()
            if not sample1.available:
                raise PowerMeasurementUnavailable(sample1.error or "Final power sample unavailable")
            measurement["samples"].append(sample1)
            avg_p = (p0 + sample1.watts) / 2.0
            energy_j = avg_p * duration
        _require_nonnegative("measured energy", energy_j)
        _require_nonnegative("average power", avg_p)
        measurement.update({
            "energy_joules": energy_j,
            "duration_seconds": duration,
            "average_power_watts": avg_p,
            "status": "disabled" if disabled else "measured",
            "measured": not disabled,
        })
        if str(os.environ.get("CE_METER_LOG", "0")) == "1":
            print(f"meter={meter.__class__.__name__} status={measurement['status']} energy_j={energy_j:.6f} duration_s={duration:.6f} avg_w={avg_p:.3f}")


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
