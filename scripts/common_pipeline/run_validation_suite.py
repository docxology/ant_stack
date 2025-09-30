#!/usr/bin/env python3
"""Validation suite for enhanced complexity_energetics methods.

This script runs comprehensive validation including:
- Unit tests for all enhanced methods
- Integration tests for complete workflows
- Performance benchmarks and scaling validation
- Numerical accuracy verification
- Statistical validation of analysis methods

Usage:
    python scripts/run_validation_suite.py [--verbose] [--benchmark] [--coverage]

References:
- Software testing best practices: https://doi.org/10.1371/journal.pcbi.1004668
- Scientific software validation: https://doi.org/10.1109/MCSE.2009.139
"""

import argparse
import sys
import os
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import unittest
    from antstack_core.analysis import (
        analyze_scaling_relationship, calculate_energy_efficiency_metrics,
        estimate_theoretical_limits, calculate_contact_complexity, 
        calculate_sparse_neural_complexity, enhanced_body_workload_closed_form
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def run_unit_tests(verbose: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """Run unit tests and return results summary.
    
    Args:
        verbose: Whether to show detailed test output
    
    Returns:
        Tuple of (success, results_dict)
    """
    print("Running unit tests...")
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), '..', 'tests')
    
    try:
        suite = loader.discover(start_dir, pattern='test_*.py')
        runner = unittest.TextTestRunner(verbosity=2 if verbose else 1, stream=sys.stdout)
        
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()
        
        success = result.wasSuccessful()
        
        results = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped),
            'success': success,
            'duration': end_time - start_time
        }
        
        if verbose:
            print(f"\nTest Results Summary:")
            print(f"  Tests run: {results['tests_run']}")
            print(f"  Failures: {results['failures']}")
            print(f"  Errors: {results['errors']}")
            print(f"  Skipped: {results['skipped']}")
            print(f"  Duration: {results['duration']:.2f}s")
            print(f"  Success: {success}")
        
        return success, results
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return False, {'error': str(e)}


def run_performance_benchmarks(verbose: bool = False) -> Dict[str, Any]:
    """Run performance benchmarks for key methods.
    
    Args:
        verbose: Whether to show detailed benchmark output
    
    Returns:
        Dictionary containing benchmark results
    """
    print("Running performance benchmarks...")
    
    benchmarks = {}
    
    # Benchmark contact complexity calculation
    print("  Benchmarking contact complexity...")
    contact_times = []
    for num_contacts in [10, 50, 100, 200]:
        start_time = time.time()
        for _ in range(100):  # Multiple runs for averaging
            calculate_contact_complexity(num_contacts, "pgs")
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        contact_times.append((num_contacts, avg_time))
        
        if verbose:
            print(f"    {num_contacts} contacts: {avg_time*1000:.3f}ms")
    
    benchmarks['contact_complexity'] = contact_times
    
    # Benchmark sparse neural complexity
    print("  Benchmarking sparse neural complexity...")
    neural_times = []
    for N_total in [1000, 10000, 50000, 100000]:
        start_time = time.time()
        for _ in range(10):  # Fewer runs for larger networks
            calculate_sparse_neural_complexity(N_total, 0.02, "biological")
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        neural_times.append((N_total, avg_time))
        
        if verbose:
            print(f"    {N_total} neurons: {avg_time*1000:.3f}ms")
    
    benchmarks['neural_complexity'] = neural_times
    
    # Benchmark scaling analysis
    print("  Benchmarking scaling analysis...")
    x_values = list(range(1, 101))  # 100 data points
    y_values = [x**1.5 + 0.1*x for x in x_values]  # Power law with noise
    
    start_time = time.time()
    for _ in range(100):
        analyze_scaling_relationship(x_values, y_values)
    end_time = time.time()
    
    scaling_time = (end_time - start_time) / 100
    benchmarks['scaling_analysis'] = scaling_time
    
    if verbose:
        print(f"    Scaling analysis (100 points): {scaling_time*1000:.3f}ms")
    
    # Benchmark workload calculation
    print("  Benchmarking workload calculations...")
    params = {'J': 18, 'C': 12, 'S': 256, 'hz': 100, 'contact_solver': 'pgs'}
    
    start_time = time.time()
    for _ in range(1000):
        enhanced_body_workload_closed_form(0.01, params)
    end_time = time.time()
    
    workload_time = (end_time - start_time) / 1000
    benchmarks['workload_calculation'] = workload_time
    
    if verbose:
        print(f"    Body workload calculation: {workload_time*1000:.3f}ms")
    
    return benchmarks


def validate_numerical_accuracy(verbose: bool = False) -> Dict[str, Any]:
    """Validate numerical accuracy of key calculations.
    
    Args:
        verbose: Whether to show detailed validation output
    
    Returns:
        Dictionary containing validation results
    """
    print("Validating numerical accuracy...")
    
    validation_results = {}
    
    # Test scaling analysis accuracy
    print("  Validating scaling analysis...")
    
    # Perfect linear relationship
    x_linear = [1, 2, 4, 8, 16]
    y_linear = [10, 20, 40, 80, 160]
    
    result = analyze_scaling_relationship(x_linear, y_linear)
    linear_accuracy = abs(result.get('scaling_exponent', 0) - 1.0) < 0.01
    linear_r2 = result.get('r_squared', 0) > 0.99
    
    validation_results['linear_scaling'] = {
        'exponent_accurate': linear_accuracy,
        'r_squared_high': linear_r2,
        'exponent': result.get('scaling_exponent', 0),
        'r_squared': result.get('r_squared', 0)
    }
    
    # Perfect quadratic relationship
    x_quad = [1, 2, 3, 4, 5]
    y_quad = [1, 4, 9, 16, 25]
    
    result = analyze_scaling_relationship(x_quad, y_quad)
    quad_accuracy = abs(result.get('scaling_exponent', 0) - 2.0) < 0.01
    quad_r2 = result.get('r_squared', 0) > 0.99
    
    validation_results['quadratic_scaling'] = {
        'exponent_accurate': quad_accuracy,
        'r_squared_high': quad_r2,
        'exponent': result.get('scaling_exponent', 0),
        'r_squared': result.get('r_squared', 0)
    }
    
    if verbose:
        print(f"    Linear scaling: exponent={validation_results['linear_scaling']['exponent']:.4f}, R²={validation_results['linear_scaling']['r_squared']:.4f}")
        print(f"    Quadratic scaling: exponent={validation_results['quadratic_scaling']['exponent']:.4f}, R²={validation_results['quadratic_scaling']['r_squared']:.4f}")
    
    # Test theoretical limits calculation
    print("  Validating theoretical limits...")
    
    # Known values for validation
    kT = 4.1e-21  # Thermal energy at room temperature
    test_bits = 1000
    expected_landauer = test_bits * kT * 0.693147  # ln(2) ≈ 0.693147
    
    limits = estimate_theoretical_limits({
        'flops': 100,
        'bits_processed': test_bits,
        'mechanical_work_j': 0.001
    })
    
    landauer_accuracy = abs(limits['landauer_limit_j'] - expected_landauer) / expected_landauer < 0.01
    
    validation_results['theoretical_limits'] = {
        'landauer_accurate': landauer_accuracy,
        'calculated': limits['landauer_limit_j'],
        'expected': expected_landauer,
        'relative_error': abs(limits['landauer_limit_j'] - expected_landauer) / expected_landauer
    }
    
    if verbose:
        print(f"    Landauer limit: calculated={limits['landauer_limit_j']:.2e}, expected={expected_landauer:.2e}")
    
    # Test efficiency metrics
    print("  Validating efficiency metrics...")
    
    energy_vals = [0.1, 0.1, 0.1, 0.1, 0.1]  # Constant energy
    perf_vals = [1.0, 1.0, 1.0, 1.0, 1.0]    # Constant performance
    
    metrics = calculate_energy_efficiency_metrics(energy_vals, perf_vals)
    
    # Should have zero variance for constant values
    variance_accurate = metrics.get('efficiency_variance', 1.0) < 1e-10
    
    validation_results['efficiency_metrics'] = {
        'variance_accurate': variance_accurate,
        'variance': metrics.get('efficiency_variance', 0),
        'energy_per_performance': metrics.get('energy_per_performance', 0)
    }
    
    if verbose:
        print(f"    Efficiency variance: {metrics.get('efficiency_variance', 0):.2e}")
    
    return validation_results


def run_integration_tests(verbose: bool = False) -> Dict[str, Any]:
    """Run integration tests for complete workflows.
    
    Args:
        verbose: Whether to show detailed test output
    
    Returns:
        Dictionary containing integration test results
    """
    print("Running integration tests...")
    
    integration_results = {}
    
    # Test complete analysis pipeline
    print("  Testing complete analysis pipeline...")
    
    try:
        from antstack_core.analysis import EnergyCoefficients, estimate_detailed_energy
        
        # Create test configuration
        coeffs = EnergyCoefficients()
        params = {'J': 18, 'C': 12, 'S': 256, 'hz': 100}
        
        # Run complete pipeline
        load = enhanced_body_workload_closed_form(0.01, params)
        breakdown = estimate_detailed_energy(load, coeffs, 0.01)
        
        # Validate results
        pipeline_success = (
            breakdown.total > 0 and
            breakdown.total_compute >= 0 and
            load.flops > 0
        )
        
        integration_results['analysis_pipeline'] = {
            'success': pipeline_success,
            'total_energy': breakdown.total,
            'compute_energy': breakdown.total_compute,
            'flops': load.flops
        }
        
        if verbose:
            print(f"    Pipeline success: {pipeline_success}")
            print(f"    Total energy: {breakdown.total:.2e} J")
            print(f"    FLOPs: {load.flops:.0f}")
        
    except Exception as e:
        integration_results['analysis_pipeline'] = {
            'success': False,
            'error': str(e)
        }
        if verbose:
            print(f"    Pipeline failed: {e}")
    
    # Test parameter sensitivity
    print("  Testing parameter sensitivity...")
    
    try:
        base_params = {'J': 18, 'C': 12, 'S': 256, 'hz': 100}
        
        # Test with different joint counts
        energies = []
        for J in [6, 12, 18, 24]:
            params = dict(base_params, J=J)
            load = enhanced_body_workload_closed_form(0.01, params)
            breakdown = estimate_detailed_energy(load, coeffs, 0.01)
            energies.append(breakdown.total)
        
        # Energy should increase with joint count
        sensitivity_correct = all(energies[i] <= energies[i+1] for i in range(len(energies)-1))
        
        integration_results['parameter_sensitivity'] = {
            'success': sensitivity_correct,
            'energies': energies,
            'monotonic_increase': sensitivity_correct
        }
        
        if verbose:
            print(f"    Sensitivity test: {sensitivity_correct}")
            print(f"    Energy progression: {energies}")
        
    except Exception as e:
        integration_results['parameter_sensitivity'] = {
            'success': False,
            'error': str(e)
        }
    
    return integration_results


def generate_validation_report(results: Dict[str, Any], output_path: str) -> None:
    """Generate comprehensive validation report.
    
    Args:
        results: Complete validation results
        output_path: Path for output report
    """
    with open(output_path, 'w') as f:
        f.write("# Validation Suite Report\n\n")
        f.write("Comprehensive validation results for enhanced complexity_energetics methods.\n\n")
        
        # Unit Tests Summary
        if 'unit_tests' in results:
            unit_results = results['unit_tests']
            f.write("## Unit Tests\n\n")
            f.write(f"- **Tests Run**: {unit_results.get('tests_run', 0)}\n")
            f.write(f"- **Failures**: {unit_results.get('failures', 0)}\n")
            f.write(f"- **Errors**: {unit_results.get('errors', 0)}\n")
            f.write(f"- **Success**: {unit_results.get('success', False)}\n")
            f.write(f"- **Duration**: {unit_results.get('duration', 0):.2f}s\n\n")
        
        # Performance Benchmarks
        if 'benchmarks' in results:
            benchmarks = results['benchmarks']
            f.write("## Performance Benchmarks\n\n")
            
            if 'contact_complexity' in benchmarks:
                f.write("### Contact Complexity Calculation\n\n")
                for contacts, time_ms in benchmarks['contact_complexity']:
                    f.write(f"- {contacts} contacts: {time_ms*1000:.3f}ms\n")
                f.write("\n")
            
            if 'scaling_analysis' in benchmarks:
                time_ms = benchmarks['scaling_analysis'] * 1000
                f.write(f"### Scaling Analysis\n\n")
                f.write(f"- 100 data points: {time_ms:.3f}ms\n\n")
        
        # Numerical Accuracy
        if 'numerical_validation' in results:
            validation = results['numerical_validation']
            f.write("## Numerical Accuracy Validation\n\n")
            
            if 'linear_scaling' in validation:
                linear = validation['linear_scaling']
                f.write("### Linear Scaling Detection\n\n")
                f.write(f"- **Exponent Accuracy**: {linear['exponent_accurate']}\n")
                f.write(f"- **R² Quality**: {linear['r_squared_high']}\n")
                f.write(f"- **Measured Exponent**: {linear['exponent']:.4f}\n")
                f.write(f"- **R²**: {linear['r_squared']:.4f}\n\n")
            
            if 'theoretical_limits' in validation:
                limits = validation['theoretical_limits']
                f.write("### Theoretical Limits Calculation\n\n")
                f.write(f"- **Landauer Limit Accuracy**: {limits['landauer_accurate']}\n")
                f.write(f"- **Relative Error**: {limits['relative_error']:.2e}\n\n")
        
        # Integration Tests
        if 'integration_tests' in results:
            integration = results['integration_tests']
            f.write("## Integration Tests\n\n")
            
            if 'analysis_pipeline' in integration:
                pipeline = integration['analysis_pipeline']
                f.write("### Complete Analysis Pipeline\n\n")
                f.write(f"- **Success**: {pipeline['success']}\n")
                if pipeline['success']:
                    f.write(f"- **Total Energy**: {pipeline['total_energy']:.2e} J\n")
                    f.write(f"- **FLOPs**: {pipeline['flops']:.0f}\n")
                f.write("\n")
        
        # Overall Assessment
        f.write("## Overall Assessment\n\n")
        
        # Count successes
        success_count = 0
        total_count = 0
        
        if 'unit_tests' in results:
            total_count += 1
            if results['unit_tests'].get('success', False):
                success_count += 1
        
        if 'numerical_validation' in results:
            validation = results['numerical_validation']
            for test_name, test_result in validation.items():
                if isinstance(test_result, dict):
                    total_count += 1
                    # Check if all accuracy flags are True
                    accuracy_flags = [v for k, v in test_result.items() if k.endswith('_accurate')]
                    if accuracy_flags and all(accuracy_flags):
                        success_count += 1
        
        if 'integration_tests' in results:
            integration = results['integration_tests']
            for test_name, test_result in integration.items():
                if isinstance(test_result, dict):
                    total_count += 1
                    if test_result.get('success', False):
                        success_count += 1
        
        success_rate = success_count / total_count if total_count > 0 else 0
        
        f.write(f"- **Overall Success Rate**: {success_rate:.1%} ({success_count}/{total_count})\n")
        
        if success_rate >= 0.9:
            f.write("- **Status**: ✅ PASS - All critical validations successful\n")
        elif success_rate >= 0.7:
            f.write("- **Status**: ⚠️ PARTIAL - Some validations failed, review required\n")
        else:
            f.write("- **Status**: ❌ FAIL - Multiple validation failures, investigation needed\n")
        
        f.write("\n")
        f.write("---\n")
        f.write("*Report generated by validation suite*\n")


def main():
    """Main validation orchestration function."""
    parser = argparse.ArgumentParser(description="Run comprehensive validation suite")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed output")
    parser.add_argument("--benchmark", "-b", action="store_true",
                       help="Run performance benchmarks")
    parser.add_argument("--coverage", "-c", action="store_true",
                       help="Generate test coverage report")
    parser.add_argument("--output", "-o", default="validation_report.md",
                       help="Output file for validation report")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Enhanced Complexity_Energetics Validation Suite")
    print("=" * 60)
    
    all_results = {}
    
    # Run unit tests
    unit_success, unit_results = run_unit_tests(args.verbose)
    all_results['unit_tests'] = unit_results
    
    if not unit_success:
        print("❌ Unit tests failed! Continuing with other validations...")
    else:
        print("✅ Unit tests passed!")
    
    # Run numerical accuracy validation
    numerical_results = validate_numerical_accuracy(args.verbose)
    all_results['numerical_validation'] = numerical_results
    
    # Run integration tests
    integration_results = run_integration_tests(args.verbose)
    all_results['integration_tests'] = integration_results
    
    # Run performance benchmarks if requested
    if args.benchmark:
        benchmark_results = run_performance_benchmarks(args.verbose)
        all_results['benchmarks'] = benchmark_results
    
    # Generate coverage report if requested
    if args.coverage:
        print("Generating test coverage report...")
        try:
            subprocess.run([
                "python", "-m", "coverage", "run", "-m", "unittest", "discover", "tests"
            ], check=True, cwd=os.path.dirname(__file__) + "/..")
            subprocess.run([
                "python", "-m", "coverage", "report"
            ], cwd=os.path.dirname(__file__) + "/..")
            subprocess.run([
                "python", "-m", "coverage", "html"
            ], cwd=os.path.dirname(__file__) + "/..")
            print("✅ Coverage report generated in htmlcov/")
        except subprocess.CalledProcessError:
            print("⚠️ Coverage report generation failed (coverage package may not be installed)")
        except FileNotFoundError:
            print("⚠️ Coverage package not found")
    
    # Generate validation report
    generate_validation_report(all_results, args.output)
    
    print("\n" + "=" * 60)
    print("Validation Suite Complete")
    print("=" * 60)
    print(f"Report saved to: {args.output}")
    
    # Exit with appropriate code
    overall_success = (
        unit_success and
        all(test.get('success', False) for test in integration_results.values() if isinstance(test, dict))
    )
    
    if overall_success:
        print("✅ All validations passed!")
        sys.exit(0)
    else:
        print("❌ Some validations failed - see report for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
