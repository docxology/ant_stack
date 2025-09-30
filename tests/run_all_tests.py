#!/usr/bin/env python3
"""
Run All Tests Script

Comprehensive test runner for the Ant Stack test suite, executing all tests
across the three subfolders with detailed reporting and coverage analysis.

This script provides:
- Unified test execution across all test categories
- Detailed reporting with coverage analysis
- Error handling and debugging information
- Parallel execution capabilities
- CI/CD integration support

Usage:
    python tests/run_all_tests.py                    # Run all tests
    python tests/run_all_tests.py --component ce    # Run complexity energetics only
    python tests/run_all_tests.py --coverage        # Run with coverage analysis
    python tests/run_all_tests.py --verbose         # Verbose output
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class TestRunner:
    """Unified test runner for the Ant Stack test suite."""

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize test runner with project root."""
        if project_root is None:
            project_root = Path(__file__).parent.parent
        self.project_root = project_root
        self.tests_dir = project_root / "tests"

        # Define test components
        self.test_components = {
            "ant_stack": {
                "path": self.tests_dir / "ant_stack",
                "description": "Ant Stack framework tests"
            },
            "complexity_energetics": {
                "path": self.tests_dir / "complexity_energetics",
                "description": "Computational complexity and energy analysis tests"
            },
            "core_rendering": {
                "path": self.tests_dir / "core_rendering",
                "description": "Core rendering and system architecture tests"
            },
            "antstack_core": {
                "path": self.tests_dir / "antstack_core",
                "description": "AntStack Core scientific methods tests"
            }
        }

    def find_test_files(self, component: str) -> List[Path]:
        """Find all test files in a component directory."""
        component_path = self.test_components[component]["path"]
        if not component_path.exists():
            return []

        test_files = []
        for file_path in component_path.glob("test_*.py"):
            test_files.append(file_path)

        return test_files

    def run_component_tests(self, component: str, verbose: bool = False,
                          coverage: bool = False, parallel: bool = False) -> Tuple[int, str]:
        """Run tests for a specific component."""
        test_files = self.find_test_files(component)

        if not test_files:
            return 0, f"No test files found for {component}"

        # Build pytest command with system Python
        cmd = [sys.executable, "-m", "pytest"]

        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")

        if coverage:
            cmd.extend([
                "--cov=antstack_core",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-fail-under=80"
            ])

        if parallel:
            try:
                import pytest_xdist
                cmd.extend(["-n", "auto"])
            except ImportError:
                print("Warning: pytest-xdist not available, running sequentially")

        # Add test file paths
        for test_file in test_files:
            cmd.append(str(test_file))

        try:
            result = subprocess.run(cmd, cwd=self.project_root,
                                  capture_output=True, text=True, timeout=300)

            return result.returncode, result.stdout + result.stderr

        except subprocess.TimeoutExpired:
            return 1, f"Tests for {component} timed out after 300 seconds"
        except Exception as e:
            return 1, f"Error running tests for {component}: {str(e)}"

    def run_all_tests(self, components: Optional[List[str]] = None,
                     verbose: bool = False, coverage: bool = False,
                     parallel: bool = False) -> Dict[str, Tuple[int, str]]:
        """Run tests for all specified components."""
        if components is None:
            components = list(self.test_components.keys())

        results = {}

        print("🧪 Running Ant Stack Test Suite")
        print("=" * 50)

        for component in components:
            if component not in self.test_components:
                print(f"⚠️  Warning: Unknown component '{component}', skipping")
                continue

            description = self.test_components[component]["description"]
            print(f"\n📁 Running {component} tests: {description}")

            returncode, output = self.run_component_tests(
                component, verbose, coverage, parallel
            )

            results[component] = (returncode, output)

            if returncode == 0:
                print(f"✅ {component}: PASSED")
            else:
                print(f"❌ {component}: FAILED (exit code {returncode})")

            if verbose or returncode != 0:
                print(f"Output for {component}:")
                print(output)

        return results

    def generate_summary_report(self, results: Dict[str, Tuple[int, str]]) -> str:
        """Generate a summary report of test results."""
        total_components = len(results)
        passed_components = sum(1 for code, _ in results.values() if code == 0)
        failed_components = total_components - passed_components

        report = []
        report.append("📊 Test Summary Report")
        report.append("=" * 30)
        report.append(f"Total components tested: {total_components}")
        report.append(f"Passed: {passed_components}")
        report.append(f"Failed: {failed_components}")
        report.append("")

        if failed_components > 0:
            report.append("❌ Failed Components:")
            for component, (code, _) in results.items():
                if code != 0:
                    report.append(f"   - {component} (exit code {code})")
        else:
            report.append("✅ All components passed!")

        return "\n".join(report)


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="Run Ant Stack test suite")
    parser.add_argument(
        "--component", "-c",
        choices=["ant_stack", "complexity_energetics", "core_rendering", "antstack_core"],
        help="Run tests for specific component only"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose test output"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate summary report without running tests"
    )

    args = parser.parse_args()

    # Initialize test runner
    runner = TestRunner()

    # Determine which components to test
    components = None
    if args.component:
        components = [args.component]

    if args.report_only:
        # Just show what would be tested
        print("📋 Test Components Available:")
        for name, info in runner.test_components.items():
            test_files = runner.find_test_files(name)
            print(f"  {name}: {len(test_files)} test files")
            for test_file in test_files:
                print(f"    - {test_file.name}")
        return

    # Run tests
    try:
        results = runner.run_all_tests(
            components=components,
            verbose=args.verbose,
            coverage=args.coverage,
            parallel=args.parallel
        )

        # Generate summary
        summary = runner.generate_summary_report(results)
        print(f"\n{summary}")

        # Exit with appropriate code
        failed_count = sum(1 for code, _ in results.values() if code != 0)
        sys.exit(0 if failed_count == 0 else 1)

    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error running tests: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
