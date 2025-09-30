"""
Configuration loader for unified paper-level configuration.

This module provides a single source of truth for all configuration values
used across the complexity and energetics analysis pipeline. It loads
configuration from paper_config.yaml and provides it to all analysis components.

Key Features:
- Single source of truth for energy coefficients
- Unified parameter values across all analysis modules
- Validation of configuration consistency
- Support for both paper-level and analysis-specific configuration
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
import sys
sys.path.insert(0, str(project_root))

from antstack_core.analysis import EnergyCoefficients


@dataclass
class PaperConfig:
    """Unified paper configuration loaded from paper_config.yaml."""
    
    # Energy coefficients (SINGLE SOURCE OF TRUTH)
    energy_coefficients: Dict[str, float]
    
    # Workload parameters
    workload_params: Dict[str, Any]
    
    # Analysis configuration
    analysis: Dict[str, Any]
    
    # Build configuration
    build: Dict[str, Any]
    
    @classmethod
    def load(cls, paper_dir: str = "papers/complexity_energetics") -> 'PaperConfig':
        """Load configuration from paper_config.yaml.
        
        Args:
            paper_dir: Path to paper directory containing paper_config.yaml
            
        Returns:
            PaperConfig instance with loaded configuration
            
        Raises:
            FileNotFoundError: If paper_config.yaml not found
            ValueError: If configuration is invalid
        """
        config_path = Path(paper_dir) / "paper_config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Paper configuration not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = ['energy_coefficients', 'workload_params', 'analysis', 'build']
        missing_sections = [s for s in required_sections if s not in config_data]
        
        if missing_sections:
            raise ValueError(f"Missing required configuration sections: {missing_sections}")
        
        return cls(
            energy_coefficients=config_data['energy_coefficients'],
            workload_params=config_data['workload_params'],
            analysis=config_data['analysis'],
            build=config_data['build']
        )
    
    def get_energy_coefficients(self) -> EnergyCoefficients:
        """Get energy coefficients as EnergyCoefficients object.
        
        Returns:
            EnergyCoefficients instance with values from paper config
        """
        return EnergyCoefficients(
            flops_pj=float(self.energy_coefficients['e_flop_pj']),
            sram_pj_per_byte=float(self.energy_coefficients['e_sram_pj_per_byte']),
            dram_pj_per_byte=float(self.energy_coefficients['e_dram_pj_per_byte']),
            spike_aj=float(self.energy_coefficients['e_spike_aj']),
            baseline_w=float(self.energy_coefficients['baseline_w']),
            body_per_joint_w=float(self.energy_coefficients['body_per_joint_w']),
            body_sensor_w_per_channel=float(self.energy_coefficients['body_sensor_w_per_channel']),
        )
    
    def get_workload_params(self, workload: str) -> Dict[str, Any]:
        """Get workload-specific parameters.
        
        Args:
            workload: Workload name ('body', 'brain', 'mind')
            
        Returns:
            Dictionary of workload parameters
        """
        return self.workload_params.get(workload, {})
    
    def validate_consistency(self) -> bool:
        """Validate configuration consistency.
        
        Returns:
            True if configuration is consistent, False otherwise
        """
        try:
            # Check that energy coefficients are valid
            coeffs = self.get_energy_coefficients()
            
            # Check that all required energy coefficients are present
            required_coeffs = ['e_flop_pj', 'e_sram_pj_per_byte', 'e_dram_pj_per_byte', 
                             'e_spike_aj', 'baseline_w', 'body_per_joint_w', 'body_sensor_w_per_channel']
            
            for coeff in required_coeffs:
                if coeff not in self.energy_coefficients:
                    print(f"Missing energy coefficient: {coeff}")
                    return False
                
                if not isinstance(self.energy_coefficients[coeff], (int, float)):
                    print(f"Invalid energy coefficient type for {coeff}: {type(self.energy_coefficients[coeff])}")
                    return False
                
                if self.energy_coefficients[coeff] < 0:
                    print(f"Negative energy coefficient for {coeff}: {self.energy_coefficients[coeff]}")
                    return False
            
            # Check that workload parameters are valid
            for workload in ['body', 'brain', 'mind']:
                if workload not in self.workload_params:
                    print(f"Missing workload parameters for: {workload}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False


# Global configuration instance
_paper_config: Optional[PaperConfig] = None


def get_paper_config() -> PaperConfig:
    """Get the global paper configuration instance.

    Returns:
        PaperConfig instance (loads on first call)

    Raises:
        FileNotFoundError: If paper configuration file is not found
        ValueError: If configuration is invalid
        RuntimeError: If configuration loading fails
    """
    global _paper_config

    if _paper_config is None:
        try:
            _paper_config = PaperConfig.load()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Paper configuration file not found: {e}") from e
        except ValueError as e:
            raise ValueError(f"Invalid paper configuration: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to load paper configuration: {e}") from e

    return _paper_config


def get_energy_coefficients() -> EnergyCoefficients:
    """Get energy coefficients from paper configuration.

    Returns:
        EnergyCoefficients instance with unified values

    Raises:
        RuntimeError: If energy coefficients cannot be retrieved
    """
    try:
        return get_paper_config().get_energy_coefficients()
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve energy coefficients: {e}") from e


def get_workload_params(workload: str) -> Dict[str, Any]:
    """Get workload parameters from paper configuration.

    Args:
        workload: Workload name ('body', 'brain', 'mind')

    Returns:
        Dictionary of workload parameters

    Raises:
        ValueError: If workload name is invalid
        RuntimeError: If workload parameters cannot be retrieved
    """
    try:
        return get_paper_config().get_workload_params(workload)
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve workload parameters for '{workload}': {e}") from e


def validate_configuration() -> bool:
    """Validate the paper configuration.
    
    Returns:
        True if configuration is valid, False otherwise
    """
    return get_paper_config().validate_consistency()


if __name__ == "__main__":
    """Test configuration loading and validation."""
    try:
        config = get_paper_config()
        print("✓ Paper configuration loaded successfully")
        
        # Validate configuration
        if config.validate_consistency():
            print("✓ Configuration validation passed")
        else:
            print("✗ Configuration validation failed")
            exit(1)
        
        # Print energy coefficients
        coeffs = config.get_energy_coefficients()
        print(f"\nEnergy Coefficients (SINGLE SOURCE OF TRUTH):")
        print(f"  FLOP energy: {coeffs.flops_pj:.2f} pJ/FLOP (matches Appendices.md: $e_\\text{{FLOP}}$)")
        print(f"  SRAM energy: {coeffs.sram_pj_per_byte:.2f} pJ/byte (matches Appendices.md: $e_\\text{{SRAM}}$)")
        print(f"  DRAM energy: {coeffs.dram_pj_per_byte:.2f} pJ/byte (matches Appendices.md: $e_\\text{{DRAM}}$)")
        print(f"  Spike energy: {coeffs.spike_aj:.2f} aJ/spike (matches Appendices.md: $E_\\text{{spk}}$)")
        print(f"  Baseline power: {coeffs.baseline_w:.3f} W (matches Appendices.md: $P_\\text{{idle}}$)")
        print(f"  Body per joint: {coeffs.body_per_joint_w:.3f} W")
        print(f"  Sensor per channel: {coeffs.body_sensor_w_per_channel:.6f} W")
        
        # Print workload parameters
        print(f"\nWorkload Parameters:")
        for workload in ['body', 'brain', 'mind']:
            params = config.get_workload_params(workload)
            print(f"  {workload}: {params}")
        
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        exit(1)
