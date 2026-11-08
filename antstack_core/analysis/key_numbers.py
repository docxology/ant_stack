"""Consume the separate enhanced analytical model, never empirical evidence.

Only the complete schema emitted by ``ce_tables_and_numbers`` is accepted.
Input coherence does not establish source freshness, calibration or scientific
validity; callers retain custody of the producer and its generation.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Union

from antstack_core.publishing.provenance import (
    atomic_json, atomic_text, file_identity, ordinary_path,
)
from .energy import EnergyCoefficients


_ENHANCED_PATH = Path("papers/complexity_energetics/generated_content/key_numbers.json")
_MODULES = ("body", "brain", "mind")


def _section(value: Any, keys: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} must contain exactly: {', '.join(sorted(keys))}")


def _number(value: Any, name: str, *, positive: bool = False) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")


def _equal(actual: float, expected: float, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"Incoherent {name}: {actual} != {expected}")


@dataclass
class KeyNumbersData:
    """Supplied enhanced estimates and their required, unverified model context."""

    schema_version: int
    model_family: str
    provenance: str
    assumed_energy_coefficients: Dict[str, float]
    per_decision_energy: Dict[str, float]
    computational_load: Dict[str, Union[float, int]]
    scaling_exponents: Dict[str, Any]
    system_parameters: Dict[str, Any]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1 or self.model_family != "enhanced_analytical":
            raise ValueError("Unsupported enhanced-model schema or family")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("Enhanced-model provenance notes are required")
        coefficients = self.assumed_energy_coefficients
        _section(coefficients, {item.name for item in fields(EnergyCoefficients)}, "assumed_energy_coefficients")
        for key, value in coefficients.items():
            _number(value, f"assumed_energy_coefficients.{key}")
        # The producer explicitly declares default assumptions. A changed
        # coefficient cannot retain that declaration and claim coherence.
        if coefficients != asdict(EnergyCoefficients()):
            raise ValueError("Coefficients contradict the declared default-model provenance")
        energy = self.per_decision_energy
        _section(energy, {f"{module}_mj" for module in (*_MODULES, "total")}, "per_decision_energy")
        for key, value in energy.items():
            _number(value, f"per_decision_energy.{key}")
        _equal(energy["total_mj"], sum(energy[f"{module}_mj"] for module in _MODULES), "total energy (mJ)")
        loads = self.computational_load
        _section(loads, {f"{module}_{metric}" for module in _MODULES for metric in ("flops", "memory_kb")}, "computational_load")
        for key, value in loads.items():
            _number(value, f"computational_load.{key}")
        scaling = self.scaling_exponents
        _section(scaling, {"body_energy", "body_r_squared", "body_regime"}, "scaling_exponents")
        exponent, r_squared = scaling["body_energy"], scaling["body_r_squared"]
        if exponent is not None and (isinstance(exponent, bool) or not isinstance(exponent, (int, float)) or not math.isfinite(exponent)):
            raise ValueError("body_energy exponent must be finite or unavailable (null)")
        if r_squared is not None:
            if isinstance(r_squared, bool) or not isinstance(r_squared, (int, float)) or not math.isfinite(r_squared) or not 0 <= r_squared <= 1:
                raise ValueError("body_r_squared must be in [0, 1] or undefined (null)")
            if exponent is None:
                raise ValueError("An unavailable fit cannot supply an R-squared")
        if not isinstance(scaling["body_regime"], str) or not scaling["body_regime"].strip():
            raise ValueError("body_regime must state the supplied fit regime")
        if exponent is None and scaling["body_regime"] != "unavailable":
            raise ValueError("An unavailable exponent must have an unavailable regime")
        system = self.system_parameters
        _section(system, {"control_frequency_hz", "decision_period_ms", "baseline_power_mw", "total_power_w", "total_power_boundary"}, "system_parameters")
        for key in ("control_frequency_hz", "decision_period_ms"):
            _number(system[key], f"system_parameters.{key}", positive=True)
        for key in ("baseline_power_mw", "total_power_w"):
            _number(system[key], f"system_parameters.{key}")
        if system["total_power_boundary"] != "sum of isolated module compute-plus-idle estimates, not whole-device power":
            raise ValueError("Missing or unsupported isolated-module power boundary")
        _equal(system["control_frequency_hz"] * system["decision_period_ms"], 1000, "decision interval")
        _equal(system["baseline_power_mw"], coefficients["baseline_w"] * 1000, "baseline power")
        _equal(system["total_power_w"], energy["total_mj"] / system["decision_period_ms"], "isolated-module power sum")
        for module in _MODULES:
            # SRAM/DRAM split and spike counts are not supplied in this schema;
            # only a necessary lower bound can be checked, not model reproduction.
            lower_j = (coefficients["baseline_w"] * system["decision_period_ms"] / 1000
                       + loads[f"{module}_flops"] * coefficients["flops_pj"] * 1e-12
                       + loads[f"{module}_memory_kb"] * 1024
                       * min(coefficients["sram_pj_per_byte"], coefficients["dram_pj_per_byte"]) * 1e-12)
            actual_j = energy[f"{module}_mj"] / 1000
            if actual_j < lower_j and not math.isclose(actual_j, lower_j, rel_tol=1e-10, abs_tol=1e-15):
                raise ValueError(f"{module} energy is below its declared compute-plus-idle lower bound")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KeyNumbersData":
        """Reject incomplete or mixed-family input instead of inventing context."""
        _section(data, {item.name for item in fields(cls)}, "key numbers")
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize every supplied context and numerical field without loss."""
        self.__post_init__()
        return asdict(self)

    def model_context(self) -> Dict[str, Any]:
        """Expose assumptions separately from the evidence they do not supply."""
        self.__post_init__()
        return {
            "schema_version": self.schema_version,
            "model_family": self.model_family,
            "provenance": self.provenance,
            "assumed_energy_coefficients": dict(self.assumed_energy_coefficients),
            "empirically_validated": False,
            "canonical_base_evidence": False,
            "source_freshness_verified": False,
        }


class KeyNumbersLoader:
    """Load caller-owned enhanced input; never fall back to another family."""

    def __init__(self, json_path: Optional[Union[str, Path]] = None):
        if json_path is None:
            # Checkout convenience only, relative to the caller's project. An
            # installed distribution has no bundled paper evidence to discover.
            if not _ENHANCED_PATH.is_file():
                raise FileNotFoundError("Provide an enhanced-model JSON path; no caller-owned checkout input was found")
            json_path = _ENHANCED_PATH
        self.json_path = ordinary_path(json_path)
        self._cache: Optional[KeyNumbersData] = None
        self._identity: Optional[Dict[str, Any]] = None

    def load(self, force_reload: bool = False) -> KeyNumbersData:
        """Load complete coherent input; content identity, not mtime, binds cache."""
        source = ordinary_path(self.json_path)
        if not source.is_file():
            raise FileNotFoundError(f"Key numbers file not found: {source}")
        identity = file_identity(source)
        if force_reload or self._cache is None or identity != self._identity:
            with source.open(encoding="utf-8") as handle:
                supplied = json.load(handle)
            data = KeyNumbersData.from_dict(supplied)
            self._cache, self._identity = data, identity
        # Never expose the cached mutable dictionaries to consumer mutation.
        return KeyNumbersData.from_dict(self._cache.to_dict())

    def get_value(self, key_path: str) -> Any:
        """Read a supplied value; missing paths raise, explicit null stays null."""
        value: Any = self.load().to_dict()
        for key in key_path.split("."):
            if not isinstance(value, dict) or key not in value:
                raise KeyError(f"Key path not found: {key_path}")
            value = value[key]
        return value

    def get_formatted_value(self, key_path: str, format_spec: str = ".3f") -> str:
        value = self.get_value(key_path)
        if value is None:
            raise ValueError(f"Key-number estimate is unavailable: {key_path}")
        return format(value, format_spec) if isinstance(value, (int, float)) else str(value)

    def get_all_values(self, format_spec: str = ".3f") -> Dict[str, Any]:
        """Format supplied numbers while retaining provenance and null estimates."""
        def format_leaf(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: format_leaf(item) for key, item in value.items()}
            return format(value, format_spec) if isinstance(value, (int, float)) else value
        return {key: value if key == "schema_version" else format_leaf(value)
                for key, value in self.load().to_dict().items()}

    def get_summary(self) -> Dict[str, Any]:
        data = self.load()
        return {
            **data.model_context(),
            "total_energy_mj": data.per_decision_energy["total_mj"],
            "dominant_module": max(_MODULES, key=lambda module: data.per_decision_energy[f"{module}_mj"]),
            "body_scaling_regime": data.scaling_exponents["body_regime"],
            "total_power_boundary": data.system_parameters["total_power_boundary"],
        }

    def get_energy_value(self, component: str, unit: str = "mj") -> float:
        """Read the producer's explicit mJ values, optionally converting to J."""
        if unit.lower() not in {"j", "mj"}:
            raise ValueError("Supported energy units are J and mJ")
        value = self.load().per_decision_energy[f"{component}_mj"]
        return value / 1000 if unit.lower() == "j" else value

    def get_computational_load(self, component: str, metric: str) -> Union[float, int]:
        return self.load().computational_load[f"{component}_{metric}"]

    def get_scaling_info(self, component: str) -> Dict[str, Any]:
        scaling = self.load().scaling_exponents
        return {
            "exponent": scaling[f"{component}_energy"],
            "r_squared": scaling[f"{component}_r_squared"],
            "regime": scaling[f"{component}_regime"],
        }


class KeyNumbersManager:
    """Render and edit enhanced inputs without claiming empirical validity."""

    def __init__(self, loader: Union[KeyNumbersLoader, str, Path]):
        self.loader = loader if isinstance(loader, KeyNumbersLoader) else KeyNumbersLoader(loader)

    @staticmethod
    def resolve_key_path(key_path: str) -> str:
        """Resolve explicitly authored semantic placeholders, not numeric literals."""
        parts = key_path.split(".")
        if len(parts) == 2 and parts[0] == "energy" and parts[1] in {*_MODULES, "total"}:
            return f"per_decision_energy.{parts[1]}_mj"
        if len(parts) == 3 and parts[0] == "scaling" and parts[1] in _MODULES:
            metrics = {"exponent": "energy", "r_squared": "r_squared", "regime": "regime"}
            return f"scaling_exponents.{parts[1]}_{metrics[parts[2]]}"
        return key_path

    def replace_placeholders(self, text: str) -> str:
        """Render exact supplied metrics; missing/unavailable data is an error."""
        def replace_match(match):
            parts = match.group(1).split(":", 1)
            path = self.resolve_key_path(parts[0].strip())
            return self.loader.get_formatted_value(path, parts[1].strip() if len(parts) == 2 else ".3f")
        return re.sub(r"\{\{key_numbers\.([^}]+)\}\}", replace_match, text)

    def update_paper_section(self, file_path: Union[str, Path], dry_run: bool = False) -> bool:
        source = ordinary_path(file_path)
        original = source.read_text(encoding="utf-8")
        updated = self.replace_placeholders(original)
        if updated == original:
            return False
        if not dry_run:
            atomic_text(source, updated)
        return True

    def get_value(self, key_path: str) -> Any:
        return self.loader.get_value(key_path)

    def get_formatted_value(self, key_path: str, format_spec: str = ".3f") -> str:
        return self.loader.get_formatted_value(key_path, format_spec)

    def update_value(self, key_path: str, value: Any) -> None:
        self.update_multiple_values({key_path: value})

    def update_multiple_values(self, updates: Dict[str, Any]) -> None:
        """Publish all explicit edits together only if the complete input coheres.

        No related total, coefficient or power is silently recalculated. Callers
        must supply every affected value, or regenerate the enhanced family.
        """
        data = self.loader.load(force_reload=True).to_dict()
        for key_path, value in updates.items():
            cursor = data
            parts = key_path.split(".")
            for key in parts[:-1]:
                if not isinstance(cursor, dict) or key not in cursor:
                    raise KeyError(key_path)
                cursor = cursor[key]
            if not isinstance(cursor, dict) or parts[-1] not in cursor:
                raise KeyError(key_path)
            cursor[parts[-1]] = value
        checked = KeyNumbersData.from_dict(data)
        atomic_json(self.loader.json_path, checked.to_dict())
        self.loader.load(force_reload=True)

    def calculate_total_energy(self) -> float:
        """Return the supplied, sum-checked total in mJ, not a guessed total."""
        return self.loader.get_energy_value("total", "mj")

    def get_scaling_analysis(self) -> Dict[str, Any]:
        data = self.loader.load()
        return {**data.model_context(), "scaling_exponents": dict(data.scaling_exponents)}

    def generate_report(self) -> str:
        data = self.loader.load()
        return ("Key Numbers Report — Enhanced Analytical Estimates\n"
                + data.provenance + "\nInput coherence only; source freshness is not verified.\n"
                + json.dumps(data.to_dict(), indent=2, allow_nan=False))

    def validate_data(self) -> tuple[bool, list[str]]:
        """Check complete input coherence only; not scientific validation."""
        result = self.validate_key_numbers()
        return result["valid"], result["issues"]

    def validate_key_numbers(self) -> Dict[str, Any]:
        """Report schema/accounting coherence, never empirical certification."""
        result = {
            "valid": False,
            "validation_scope": "enhanced_model_input_coherence",
            "empirically_validated": False,
            "canonical_base_evidence": False,
            "source_freshness_verified": False,
            "issues": [],
            "warnings": ["Supplied deterministic estimates are not calibrated or empirically validated; source freshness is not verified."],
        }
        try:
            data = self.loader.load(force_reload=True)
        except (OSError, ValueError, TypeError) as error:
            result["issues"].append(str(error))
        else:
            result.update(data.model_context())
            result["valid"] = True
        return result


def get_key_numbers_loader() -> KeyNumbersLoader:
    """Create a loader bound to this caller's checkout, without global custody."""
    return KeyNumbersLoader()


def get_key_numbers_manager() -> KeyNumbersManager:
    return KeyNumbersManager(get_key_numbers_loader())
