#!/usr/bin/env python3
"""
Unified Configuration Validation Script

This script validates that all components of the Ant Stack complexity and energetics
analysis use the same configuration values, ensuring coherent maximal usage of
unified paper-level configuration.

Validates:
- Energy coefficients consistency across all sources
- Parameter values alignment between paper config and manifest
- Test configuration consistency
- Analysis pipeline configuration usage
"""

import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from antstack_core.analysis import EnergyCoefficients


class ConfigValidator:
    """Validates unified configuration across all components."""
    
    def __init__(self):
        self.project_root = project_root
        self.paper_dir = self.project_root / "papers" / "complexity_energetics"
        self.errors = []
        self.warnings = []
    
    def validate_energy_coefficients(self) -> bool:
        """Validate energy coefficients consistency across all sources."""
        print("🔍 Validating energy coefficients consistency...")
        
        # Load paper config
        paper_config_path = self.paper_dir / "paper_config.yaml"
        with open(paper_config_path, 'r') as f:
            paper_config = yaml.safe_load(f)
        
        paper_coeffs = paper_config['energy_coefficients']
        
        # Load manifest config
        manifest_path = self.paper_dir / "manifest.example.yaml"
        with open(manifest_path, 'r') as f:
            manifest_config = yaml.safe_load(f)
        
        manifest_coeffs = manifest_config['coefficients']
        
        # Check consistency
        required_coeffs = [
            'e_flop_pj', 'e_sram_pj_per_byte', 'e_dram_pj_per_byte',
            'e_spike_aj', 'baseline_w', 'body_per_joint_w', 'body_sensor_w_per_channel'
        ]
        
        consistent = True
        for coeff in required_coeffs:
            if coeff in paper_coeffs and coeff in manifest_coeffs:
                if paper_coeffs[coeff] != manifest_coeffs[coeff]:
                    self.errors.append(f"Energy coefficient {coeff} mismatch: paper={paper_coeffs[coeff]}, manifest={manifest_coeffs[coeff]}")
                    consistent = False
            elif coeff not in paper_coeffs:
                self.errors.append(f"Missing energy coefficient {coeff} in paper config")
                consistent = False
            elif coeff not in manifest_coeffs:
                self.errors.append(f"Missing energy coefficient {coeff} in manifest")
                consistent = False
        
        if consistent:
            print("✓ Energy coefficients are consistent across all sources")
        else:
            print("✗ Energy coefficients have inconsistencies")
        
        return consistent
    
    def validate_antstack_core_defaults(self) -> bool:
        """Validate that antstack_core defaults don't conflict with paper config."""
        print("🔍 Validating antstack_core defaults...")
        
        # Load paper config
        paper_config_path = self.paper_dir / "paper_config.yaml"
        with open(paper_config_path, 'r') as f:
            paper_config = yaml.safe_load(f)
        
        paper_coeffs = paper_config['energy_coefficients']
        
        # Get antstack_core defaults
        core_coeffs = EnergyCoefficients()
        
        # Check for conflicts (warnings, not errors)
        conflicts = []
        if core_coeffs.baseline_w != paper_coeffs['baseline_w']:
            conflicts.append(f"baseline_w: core={core_coeffs.baseline_w}, paper={paper_coeffs['baseline_w']}")
        
        if conflicts:
            for conflict in conflicts:
                self.warnings.append(f"antstack_core default conflict: {conflict}")
            print("⚠ antstack_core defaults have conflicts with paper config (warnings only)")
        else:
            print("✓ antstack_core defaults are compatible with paper config")
        
        return True  # Warnings don't fail validation
    
    def validate_test_consistency(self) -> bool:
        """Validate that tests use consistent configuration values."""
        print("🔍 Validating test configuration consistency...")
        
        # Check test files for hardcoded values that should match paper config
        test_files = [
            "tests/core_rendering/test_core_refactor.py",
            "tests/complexity_energetics/test_enhanced_ce.py",
            "tests/complexity_energetics/test_key_numbers.py"
        ]
        
        paper_config_path = self.paper_dir / "paper_config.yaml"
        with open(paper_config_path, 'r') as f:
            paper_config = yaml.safe_load(f)
        
        paper_coeffs = paper_config['energy_coefficients']
        
        consistent = True
        for test_file in test_files:
            test_path = self.project_root / test_file
            if test_path.exists():
                with open(test_path, 'r') as f:
                    content = f.read()
                
                # Check for hardcoded baseline_w values
                if "baseline_w=0.05" in content:
                    self.warnings.append(f"Test file {test_file} uses old baseline_w=0.05, should use {paper_coeffs['baseline_w']}")
                    consistent = False
                
                if "baseline_w=0.0" in content:
                    self.warnings.append(f"Test file {test_file} uses default baseline_w=0.0, should use {paper_coeffs['baseline_w']}")
                    consistent = False
        
        if consistent:
            print("✓ Test configuration is consistent with paper config")
        else:
            print("⚠ Test configuration has inconsistencies (warnings)")
        
        return True  # Warnings don't fail validation
    
    def validate_runner_configuration(self) -> bool:
        """Validate that runner uses unified configuration."""
        print("🔍 Validating runner configuration usage...")
        
        runner_path = self.paper_dir / "src" / "runner.py"
        if not runner_path.exists():
            self.errors.append("Main runner file not found")
            return False
        
        with open(runner_path, 'r') as f:
            content = f.read()
        
        # Check that runner uses manifest configuration
        if "manifest.coefficients" not in content:
            self.errors.append("Runner doesn't use manifest coefficients")
            return False
        
        if "SINGLE SOURCE OF TRUTH" not in content:
            self.warnings.append("Runner doesn't explicitly reference single source of truth")
        
        # Check that legacy runner is removed
        legacy_runner = self.paper_dir / "src" / "ce" / "runner.py"
        if legacy_runner.exists():
            self.errors.append("Legacy runner file still exists, should be removed")
            return False
        
        print("✓ Runner configuration is properly unified")
        return True
    
    def validate_paper_config_completeness(self) -> bool:
        """Validate that paper config contains all necessary parameters."""
        print("🔍 Validating paper config completeness...")
        
        paper_config_path = self.paper_dir / "paper_config.yaml"
        with open(paper_config_path, 'r') as f:
            paper_config = yaml.safe_load(f)
        
        required_sections = [
            'paper', 'content', 'build', 'analysis', 'energy_coefficients', 
            'workload_params', 'validation'
        ]
        
        missing_sections = [s for s in required_sections if s not in paper_config]
        if missing_sections:
            self.errors.append(f"Missing required sections in paper config: {missing_sections}")
            return False
        
        # Check energy coefficients completeness
        required_coeffs = [
            'e_flop_pj', 'e_sram_pj_per_byte', 'e_dram_pj_per_byte',
            'e_spike_aj', 'baseline_w', 'body_per_joint_w', 'body_sensor_w_per_channel'
        ]
        
        missing_coeffs = [c for c in required_coeffs if c not in paper_config['energy_coefficients']]
        if missing_coeffs:
            self.errors.append(f"Missing energy coefficients in paper config: {missing_coeffs}")
            return False
        
        print("✓ Paper config is complete")
        return True
    
    def run_validation(self) -> bool:
        """Run all validation checks."""
        print("🚀 Starting unified configuration validation...\n")
        
        checks = [
            self.validate_paper_config_completeness,
            self.validate_energy_coefficients,
            self.validate_antstack_core_defaults,
            self.validate_test_consistency,
            self.validate_runner_configuration,
        ]
        
        all_passed = True
        for check in checks:
            try:
                if not check():
                    all_passed = False
            except Exception as e:
                self.errors.append(f"Validation check failed: {e}")
                all_passed = False
            print()  # Add spacing between checks
        
        # Print summary
        print("=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        if self.errors:
            print(f"❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")
            print()
        
        if self.warnings:
            print(f"⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")
            print()
        
        if all_passed and not self.errors:
            print("✅ All validation checks passed! Configuration is unified and consistent.")
        else:
            print("❌ Validation failed. Please fix the errors above.")
        
        return all_passed and not self.errors


def main():
    """Main entry point for configuration validation."""
    validator = ConfigValidator()
    success = validator.run_validation()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
