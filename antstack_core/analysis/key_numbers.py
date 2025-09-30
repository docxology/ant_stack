"""
Key Numbers Integration Module

Provides comprehensive integration of key_numbers.json data into paper sections
with dynamic updating, validation, and formatting capabilities.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Union, List
from pathlib import Path


@dataclass
class KeyNumbersData:
    """Structured representation of key numbers data."""

    per_decision_energy: Dict[str, float]
    computational_load: Dict[str, Union[float, int]]
    scaling_exponents: Dict[str, Union[float, str]]
    system_parameters: Dict[str, Union[float, int]]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KeyNumbersData":
        """Create KeyNumbersData from dictionary."""
        return cls(
            per_decision_energy=data.get("per_decision_energy", {}),
            computational_load=data.get("computational_load", {}),
            scaling_exponents=data.get("scaling_exponents", {}),
            system_parameters=data.get("system_parameters", {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)


class KeyNumbersLoader:
    """Loads and manages key numbers data from JSON files.

    Provides caching, validation, and formatted access to key numbers
    for integration into paper sections.
    """

    def __init__(self, json_path: Optional[Union[str, Path]] = None):
        """Initialize loader with path to key_numbers.json.

        Args:
            json_path: Path to key_numbers.json file. If None, searches in standard locations.
        """
        if json_path is None:
            # Search in standard locations
            candidates = [
                Path("papers/complexity_energetics/generated_content/key_numbers.json"),
                Path("generated_content/key_numbers.json"),
                Path("key_numbers.json")
            ]
            for candidate in candidates:
                if candidate.exists():
                    json_path = candidate
                    break

        if json_path is None:
            raise FileNotFoundError("Could not find key_numbers.json in standard locations")

        self.json_path = Path(json_path)
        self._cache: Optional[KeyNumbersData] = None
        self._last_modified: Optional[float] = None

    def load(self, force_reload: bool = False) -> KeyNumbersData:
        """Load key numbers data with caching.

        Args:
            force_reload: Force reload even if cached

        Returns:
            KeyNumbersData object with current values
        """
        if not self.json_path.exists():
            raise FileNotFoundError(f"Key numbers file not found: {self.json_path}")

        current_mtime = self.json_path.stat().st_mtime

        if (not force_reload and
            self._cache is not None and
            self._last_modified == current_mtime):
            return self._cache

        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self._cache = KeyNumbersData.from_dict(data)
        self._last_modified = current_mtime
        return self._cache

    def get_formatted_value(self, key_path: str, format_spec: str = ".3f") -> str:
        """Get formatted value from key numbers data.

        Args:
            key_path: Dot-separated path to value (e.g., "per_decision_energy.body_mj")
            format_spec: Python format specification

        Returns:
            Formatted string representation
        """
        data = self.load()
        keys = key_path.split('.')
        value = data

        for key in keys:
            if hasattr(value, key):
                value = getattr(value, key)
            elif isinstance(value, dict) and key in value:
                value = value[key]
            else:
                raise KeyError(f"Key path not found: {key_path}")

        if isinstance(value, (int, float)):
            return f"{value:{format_spec}}"
        return str(value)

    def get_energy_value(self, component: str, unit: str = "mj") -> float:
        """Get energy value for specific component.

        Args:
            component: Component name ("body", "brain", "mind", "total")
            unit: Unit ("mj" for millijoules, "j" for joules)

        Returns:
            Energy value in specified units
        """
        data = self.load()
        key = f"{component}_{unit}"
        return data.per_decision_energy.get(key, 0.0)

    def get_computational_load(self, component: str, metric: str) -> Union[float, int]:
        """Get computational load metric.

        Args:
            component: Component name ("body", "brain", "mind")
            metric: Metric name ("flops", "memory_kb")

        Returns:
            Metric value
        """
        data = self.load()
        key = f"{component}_{metric}"
        return data.computational_load.get(key, 0)

    def get_scaling_info(self, component: str) -> Dict[str, Union[float, str]]:
        """Get scaling information for component.

        Args:
            component: Component name

        Returns:
            Dictionary with scaling exponent, r_squared, and regime
        """
        data = self.load()
        prefix = f"{component}_"
        return {
            "exponent": data.scaling_exponents.get(f"{prefix}energy", 0),
            "r_squared": data.scaling_exponents.get(f"{prefix}r_squared", 0),
            "regime": data.scaling_exponents.get(f"{prefix}regime", "unknown")
        }


class KeyNumbersManager:
    """Manages key numbers integration into paper sections.

    Provides methods to replace placeholders in markdown files with
    current key numbers values.
    """

    def __init__(self, loader: KeyNumbersLoader):
        """Initialize manager with loader.

        Args:
            loader: KeyNumbersLoader instance
        """
        self.loader = loader

    def replace_placeholders(self, text: str) -> str:
        """Replace key numbers placeholders in text.

        Supports placeholders like:
        - {{key_numbers.per_decision_energy.body_mj:.2e}}
        - {{key_numbers.energy.body}}
        - {{key_numbers.scaling.brain.exponent}}

        Args:
            text: Text containing placeholders

        Returns:
            Text with placeholders replaced
        """
        import re

        def replace_match(match):
            placeholder = match.group(1)
            parts = placeholder.split(':', 1)
            key_path = parts[0].strip()
            format_spec = parts[1].strip() if len(parts) > 1 else ".3f"

            try:
                if key_path.startswith("energy."):
                    # Handle energy shortcuts
                    component = key_path.split('.')[1]
                    return self.loader.get_formatted_value(
                        f"per_decision_energy.{component}_mj", format_spec
                    )
                elif key_path.startswith("scaling."):
                    # Handle scaling shortcuts
                    parts = key_path.split('.')
                    if len(parts) >= 3:
                        component = parts[1]
                        metric = parts[2]
                        scaling_info = self.loader.get_scaling_info(component)
                        return self.loader.get_formatted_value(
                            f"scaling_exponents.{component}_{metric}", format_spec
                        )
                else:
                    # Handle full key paths
                    return self.loader.get_formatted_value(key_path, format_spec)
            except (KeyError, ValueError):
                return match.group(0)  # Return original if replacement fails

        # Replace {{key_numbers.*}} patterns
        pattern = r'\{\{key_numbers\.([^}]+)\}\}'
        return re.sub(pattern, replace_match, text)

    def update_paper_section(self, file_path: Union[str, Path], dry_run: bool = False) -> bool:
        """Update paper section file with current key numbers.

        Args:
            file_path: Path to markdown file
            dry_run: If True, don't write changes

        Returns:
            True if file was updated
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return False

        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        updated_content = self.replace_placeholders(original_content)

        if updated_content != original_content:
            if not dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
            return True

        return False

    def validate_key_numbers(self) -> Dict[str, Any]:
        """Validate key numbers data for completeness and consistency.

        Returns:
            Dictionary with validation results
        """
        try:
            data = self.loader.load()

            validation_results = {
                "valid": True,
                "issues": [],
                "warnings": []
            }

            # Check required energy values
            required_energy_keys = ["body_mj", "brain_mj", "mind_mj", "total_mj"]
            for key in required_energy_keys:
                if key not in data.per_decision_energy:
                    validation_results["issues"].append(f"Missing energy key: {key}")
                    validation_results["valid"] = False

            # Check computational load values
            required_load_keys = ["body_flops", "brain_flops", "mind_flops"]
            for key in required_load_keys:
                if key not in data.computational_load:
                    validation_results["issues"].append(f"Missing load key: {key}")
                    validation_results["valid"] = False

            # Check scaling exponents
            if not data.scaling_exponents:
                validation_results["warnings"].append("No scaling exponents found")

            # Check system parameters
            required_system_keys = ["control_frequency_hz", "decision_period_ms"]
            for key in required_system_keys:
                if key not in data.system_parameters:
                    validation_results["issues"].append(f"Missing system key: {key}")
                    validation_results["valid"] = False

            return validation_results

        except Exception as e:
            return {
                "valid": False,
                "issues": [f"Validation failed: {str(e)}"],
                "warnings": []
            }


# Global instance for convenience
_default_loader = None
_default_manager = None

def get_key_numbers_loader() -> KeyNumbersLoader:
    """Get default key numbers loader instance."""
    global _default_loader
    if _default_loader is None:
        _default_loader = KeyNumbersLoader()
    return _default_loader

def get_key_numbers_manager() -> KeyNumbersManager:
    """Get default key numbers manager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = KeyNumbersManager(get_key_numbers_loader())
    return _default_manager
