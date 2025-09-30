#!/usr/bin/env python3
"""Unified Build and Validation System for Ant Stack Papers.

This script provides a comprehensive, harmonized approach to building, validating,
and managing both Ant Stack papers with integrated quality assurance.

Features:
- Pre-build validation and dependency checks
- Integrated figure generation and cross-reference validation
- Post-build quality assurance with detailed reporting
- Consistent formatting and documentation standards
- Comprehensive test integration
- Streamlined workflow for both development and production builds
"""

import os
import sys
import subprocess
import tempfile
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime

class UnifiedBuildSystem:
    """Comprehensive build and validation system for Ant Stack papers."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.papers = {
            "ant_stack": {
                "title": "The Ant Stack",
                "output": "1_ant_stack.pdf",
                "has_generated": False,
                "analysis_script": None
            },
            "complexity_energetics": {
                "title": "Computational Complexity and Energetics of the Ant Stack", 
                "output": "2_complexity_energetics.pdf",
                "has_generated": True,
                "analysis_script": "complexity_energetics/src/runner.py"
            },
            "cohereAnts": {
                "title": "Infrared Vibrational Detection in Insect Olfaction",
                "output": "3_cohereAnts.pdf",
                "has_generated": True,
                "analysis_script": "cohereAnts/scripts/run_all_case_studies.py"
            }
        }
        self.validation_results = {}
        
    def validate_environment(self) -> Dict[str, bool]:
        """Validate build environment and dependencies."""
        checks = {}
        
        # Check essential tools
        tools = ["pandoc", "xelatex", "python3", "grep", "pdftotext"]
        for tool in tools:
            checks[f"tool_{tool}"] = subprocess.run(
                ["which", tool], capture_output=True
            ).returncode == 0
            
        # Check Python packages
        python_packages = ["matplotlib", "numpy", "pandas", "pathlib", "yaml"]
        for package in python_packages:
            try:
                __import__(package)
                checks[f"python_{package}"] = True
            except ImportError:
                checks[f"python_{package}"] = False
                
        # Check project structure
        required_dirs = ["tools", "complexity_energetics", "ant_stack", "tests"]
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            checks[f"dir_{dir_name}"] = dir_path.exists() and dir_path.is_dir()
            
        # Check critical files
        critical_files = [
            "tools/render_pdf.sh",
            ".cursorrules", 
            "README.md",
            "complexity_energetics/src/runner.py"
        ]
        for file_name in critical_files:
            file_path = self.project_root / file_name
            checks[f"file_{file_name.replace('/', '_')}"] = file_path.exists()
            
        return checks
        
    def validate_paper_structure(self, paper_name: str) -> Dict[str, any]:
        """Validate paper structure and content."""
        paper_dir = self.project_root / paper_name
        results = {
            "exists": paper_dir.exists(),
            "markdown_files": [],
            "assets_dir": False,
            "figure_issues": [],
            "reference_issues": []
        }
        
        if not results["exists"]:
            return results
            
        # Find markdown files
        for md_file in paper_dir.glob("*.md"):
            results["markdown_files"].append(md_file.name)
            
        # Check assets directory
        assets_dir = paper_dir / "assets"
        results["assets_dir"] = assets_dir.exists()
        
        # Validate cross-references
        ref_issues = self.validate_cross_references(paper_dir)
        results["figure_issues"] = ref_issues.get("figure_mismatches", [])
        results["reference_issues"] = ref_issues.get("broken_refs", [])
        
        return results
        
    def validate_cross_references(self, paper_dir: Path) -> Dict[str, List]:
        """Validate cross-references within a paper."""
        figure_defs = set()
        figure_refs = set()
        issues = {"figure_mismatches": [], "broken_refs": []}
        
        # Scan all markdown files
        for md_file in paper_dir.glob("*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Find figure definitions
                def_pattern = r'## Figure:.*?\{#fig:([^}]+)\}|!\[[^\]]*\]\([^)]+\)\{#fig:([^}]+)\}'
                for match in re.finditer(def_pattern, content):
                    fig_id = match.group(1) or match.group(2)
                    if fig_id:
                        figure_defs.add(fig_id)
                        
                # Find figure references  
                ref_pattern = r'Figure~\\ref\{fig:([^}]+)\}'
                for match in re.finditer(ref_pattern, content):
                    fig_id = match.group(1)
                    figure_refs.add(fig_id)
                    
            except Exception as e:
                issues["broken_refs"].append(f"Error reading {md_file}: {e}")
                
        # Find mismatches
        missing_defs = figure_refs - figure_defs
        unused_defs = figure_defs - figure_refs
        
        if missing_defs:
            issues["figure_mismatches"].extend([
                f"Missing definition for fig:{fig_id}" for fig_id in missing_defs
            ])
            
        if unused_defs:
            issues["figure_mismatches"].extend([
                f"Unused definition fig:{fig_id}" for fig_id in unused_defs  
            ])
            
        return issues
        
    def run_analysis_pipeline(self, paper_name: str) -> bool:
        """Run analysis pipeline for papers that have generated content."""
        paper_info = self.papers.get(paper_name)
        if not paper_info or not paper_info["has_generated"]:
            return True
            
        analysis_script = paper_info.get("analysis_script")
        if not analysis_script:
            return True
            
        print(f"🔬 Running analysis pipeline for {paper_name}...")
        
        try:
            # Run the analysis script
            if paper_name == "complexity_energetics":
                manifest_path = self.project_root / "complexity_energetics" / "manifest.example.yaml"
                output_dir = self.project_root / "complexity_energetics" / "out"
                
                cmd = [
                    "python3", "-m", "complexity_energetics.src.ce.runner",
                    str(manifest_path), "--out", str(output_dir)
                ]
                
                result = subprocess.run(
                    cmd, cwd=self.project_root, 
                    capture_output=True, text=True
                )
                
                if result.returncode != 0:
                    print(f"❌ Analysis pipeline failed: {result.stderr}")
                    return False
                    
                print(f"✅ Analysis pipeline completed successfully")
                return True
                
        except Exception as e:
            print(f"❌ Analysis pipeline error: {e}")
            return False
            
        return True
        
    def build_paper(self, paper_name: str) -> Tuple[bool, Dict[str, any]]:
        """Build a single paper with comprehensive validation."""
        print(f"\n📄 Building paper: {paper_name}")
        
        # Pre-build validation
        paper_validation = self.validate_paper_structure(paper_name)
        if not paper_validation["exists"]:
            return False, {"error": f"Paper directory {paper_name} not found"}
            
        # Run analysis pipeline if needed
        if not self.run_analysis_pipeline(paper_name):
            return False, {"error": "Analysis pipeline failed"}
            
        # Build the PDF
        render_script = self.project_root / "tools" / "render_pdf.sh"
        cmd = ["bash", str(render_script), paper_name]
        
        print(f"🏗️  Executing: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd, cwd=self.project_root,
                capture_output=True, text=True, timeout=300
            )
            
            build_result = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "paper_validation": paper_validation
            }
            
            if result.returncode == 0:
                # Post-build validation
                pdf_path = self.project_root / self.papers[paper_name]["output"]
                if pdf_path.exists():
                    pdf_validation = self.validate_pdf_quality(pdf_path)
                    build_result["pdf_validation"] = pdf_validation
                    
                    print(f"✅ Successfully built {paper_name}")
                    return True, build_result
                else:
                    print(f"❌ PDF file not created for {paper_name}")
                    return False, build_result
            else:
                print(f"❌ Build failed for {paper_name}")
                print(f"Error: {result.stderr}")
                return False, build_result
                
        except subprocess.TimeoutExpired:
            print(f"❌ Build timed out for {paper_name}")
            return False, {"error": "Build timeout"}
        except Exception as e:
            print(f"❌ Build error for {paper_name}: {e}")
            return False, {"error": str(e)}
            
    def validate_pdf_quality(self, pdf_path: Path) -> Dict[str, any]:
        """Validate PDF quality and content."""
        validation = {
            "exists": pdf_path.exists(),
            "size_bytes": 0,
            "broken_references": 0,
            "figure_count": 0,
            "page_count": 0,
            "warnings": []
        }
        
        if not validation["exists"]:
            return validation
            
        try:
            validation["size_bytes"] = pdf_path.stat().st_size
            
            # Extract text for analysis
            result = subprocess.run(
                ["pdftotext", "-layout", str(pdf_path), "-"],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                text = result.stdout
                
                # Count broken references
                broken_refs = re.findall(r'Figure~\?\?', text)
                validation["broken_references"] = len(broken_refs)
                
                # Count figures
                figures = re.findall(r'Figure \d+', text)
                validation["figure_count"] = len(figures)
                
                # Count pages (approximate)
                pages = text.count('\f')  # Form feed characters
                validation["page_count"] = max(1, pages)
                
                # Check for common issues
                if validation["broken_references"] > 0:
                    validation["warnings"].append(
                        f"{validation['broken_references']} broken figure references"
                    )
                    
                if validation["figure_count"] == 0:
                    validation["warnings"].append("No figures found in PDF")
                    
                if validation["size_bytes"] < 50000:  # Less than 50KB
                    validation["warnings"].append("PDF seems unusually small")
                    
        except Exception as e:
            validation["warnings"].append(f"PDF validation error: {e}")
            
        return validation
        
    def run_tests(self) -> Dict[str, bool]:
        """Run the test suite."""
        print("\n🧪 Running test suite...")
        
        test_results = {}
        
        # Run Python tests
        try:
            result = subprocess.run(
                ["python3", "-m", "pytest", "tests/", "-v"],
                cwd=self.project_root,
                capture_output=True, text=True
            )
            
            test_results["pytest"] = result.returncode == 0
            if result.returncode != 0:
                print(f"❌ pytest failed:\n{result.stdout}\n{result.stderr}")
            else:
                print("✅ pytest passed")
                
        except Exception as e:
            test_results["pytest"] = False
            print(f"❌ pytest error: {e}")
            
        # Run validation suite if available
        validation_script = self.project_root / "scripts" / "run_validation_suite.py"
        if validation_script.exists():
            try:
                result = subprocess.run(
                    ["python3", str(validation_script)],
                    cwd=self.project_root,
                    capture_output=True, text=True
                )
                
                test_results["validation_suite"] = result.returncode == 0
                if result.returncode != 0:
                    print(f"⚠️  Validation suite had issues (non-critical)")
                else:
                    print("✅ Validation suite passed")
                    
            except Exception as e:
                test_results["validation_suite"] = False
                print(f"⚠️  Validation suite error: {e}")
        
        return test_results
        
    def generate_build_report(self, results: Dict[str, any]) -> str:
        """Generate a comprehensive build report."""
        report_lines = [
            "# Ant Stack Build Report",
            f"Generated: {datetime.now().isoformat()}",
            f"Project Root: {self.project_root}",
            "",
            "## Environment Validation"
        ]
        
        env_results = results.get("environment", {})
        for check, passed in env_results.items():
            status = "✅" if passed else "❌"
            report_lines.append(f"- {check}: {status}")
            
        report_lines.extend(["", "## Papers"])
        
        for paper_name, paper_result in results.get("papers", {}).items():
            success, data = paper_result
            status = "✅" if success else "❌"
            
            report_lines.extend([
                f"### {paper_name} {status}",
                ""
            ])
            
            if success and "pdf_validation" in data:
                pdf_val = data["pdf_validation"]
                report_lines.extend([
                    f"- PDF Size: {pdf_val['size_bytes']:,} bytes",
                    f"- Pages: {pdf_val['page_count']}",
                    f"- Figures: {pdf_val['figure_count']}",
                    f"- Broken References: {pdf_val['broken_references']}",
                ])
                
                if pdf_val["warnings"]:
                    report_lines.append("- Warnings:")
                    for warning in pdf_val["warnings"]:
                        report_lines.append(f"  - {warning}")
            elif not success:
                error = data.get("error", "Unknown error")
                report_lines.append(f"- Error: {error}")
                
        # Test results
        test_results = results.get("tests", {})
        if test_results:
            report_lines.extend(["", "## Test Results"])
            for test_name, passed in test_results.items():
                status = "✅" if passed else "❌"
                report_lines.append(f"- {test_name}: {status}")
                
        return "\n".join(report_lines)
        
    def build_all(self, run_tests: bool = True, generate_report: bool = True) -> Dict[str, any]:
        """Build all papers with comprehensive validation."""
        print("🚀 Starting unified build process...")
        
        results = {
            "start_time": datetime.now().isoformat(),
            "environment": {},
            "papers": {},
            "tests": {},
            "success": False
        }
        
        # Environment validation
        print("\n🔍 Validating environment...")
        env_results = self.validate_environment()
        results["environment"] = env_results
        
        failed_env_checks = [k for k, v in env_results.items() if not v]
        if failed_env_checks:
            print(f"❌ Environment validation failed: {failed_env_checks}")
            return results
            
        print("✅ Environment validation passed")
        
        # Build papers
        paper_success = True
        for paper_name in self.papers.keys():
            success, data = self.build_paper(paper_name)
            results["papers"][paper_name] = (success, data)
            if not success:
                paper_success = False
                
        # Run tests
        if run_tests and paper_success:
            test_results = self.run_tests()
            results["tests"] = test_results
        elif not paper_success:
            print("⚠️  Skipping tests due to build failures")
            
        results["success"] = paper_success and all(results.get("tests", {}).values())
        results["end_time"] = datetime.now().isoformat()
        
        # Generate report
        if generate_report:
            report = self.generate_build_report(results)
            report_path = self.project_root / "build_report.md"
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"\n📋 Build report saved to: {report_path}")
            
        # Summary
        if results["success"]:
            print("\n🎉 BUILD SUCCESSFUL!")
        else:
            print("\n❌ BUILD FAILED - Check report for details")
            
        return results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified Build System for Ant Stack Papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/unified_build.py                    # Build all papers
  python3 scripts/unified_build.py --paper ant_stack  # Build specific paper
  python3 scripts/unified_build.py --no-tests         # Skip tests
  python3 scripts/unified_build.py --validate-only    # Only validate, don't build
        """
    )
    
    parser.add_argument(
        "--paper", 
        choices=["ant_stack", "complexity_energetics", "cohereAnts"],
        help="Build specific paper only"
    )
    
    parser.add_argument(
        "--no-tests",
        action="store_true", 
        help="Skip test execution"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation, don't build"
    )
    
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory"
    )
    
    args = parser.parse_args()
    
    build_system = UnifiedBuildSystem(args.project_root)
    
    if args.validate_only:
        print("🔍 Running validation only...")
        env_results = build_system.validate_environment()
        
        print("\nEnvironment Validation:")
        for check, passed in env_results.items():
            status = "✅" if passed else "❌"
            print(f"  {check}: {status}")
            
        for paper_name in build_system.papers.keys():
            paper_results = build_system.validate_paper_structure(paper_name)
            print(f"\n{paper_name} Validation:")
            print(f"  Exists: {'✅' if paper_results['exists'] else '❌'}")
            print(f"  Markdown files: {len(paper_results['markdown_files'])}")
            print(f"  Assets dir: {'✅' if paper_results['assets_dir'] else '❌'}")
            
            if paper_results['figure_issues']:
                print(f"  Figure issues: {len(paper_results['figure_issues'])}")
                for issue in paper_results['figure_issues']:
                    print(f"    - {issue}")
                    
        return 0
    
    if args.paper:
        # Build single paper
        success, data = build_system.build_paper(args.paper)
        return 0 if success else 1
    else:
        # Build all papers
        results = build_system.build_all(run_tests=not args.no_tests)
        return 0 if results["success"] else 1

if __name__ == "__main__":
    sys.exit(main())
