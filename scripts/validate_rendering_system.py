#!/usr/bin/env python3
"""Comprehensive validation of the PDF rendering system.

This script validates all aspects of the rendering system:
- Paper configuration validation
- Cross-reference consistency
- Figure format compliance
- Math symbol formatting
- Hyperlink validation
- Build system integration

Usage:
    python3 scripts/validate_rendering_system.py [--paper PAPER_NAME] [--verbose]
"""

import argparse
import os
import re
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from antstack_core.figures import validate_cross_references
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False


class RenderingSystemValidator:
    """Comprehensive validation of the PDF rendering system."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.papers_dir = self.project_root / "papers"
        self.validation_results = {}
        
    def discover_papers(self) -> List[str]:
        """Discover all papers with valid configurations."""
        papers = []
        
        if self.papers_dir.exists():
            for paper_dir in self.papers_dir.iterdir():
                if paper_dir.is_dir():
                    config_file = paper_dir / "paper_config.yaml"
                    if config_file.exists():
                        papers.append(paper_dir.name)
        
        return papers
    
    def validate_paper_config(self, paper_name: str) -> Dict[str, Any]:
        """Validate paper configuration file."""
        config_file = self.papers_dir / paper_name / "paper_config.yaml"
        results = {"valid": False, "errors": [], "warnings": []}
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            required_sections = ['paper', 'content', 'build', 'latex']
            missing_sections = [s for s in required_sections if s not in config]
            
            if missing_sections:
                results["errors"].append(f"Missing required sections: {missing_sections}")
                return results
            
            # Validate paper metadata
            paper_info = config['paper']
            required_fields = ['name', 'title', 'author', 'output_filename']
            missing_fields = [f for f in required_fields if f not in paper_info]
            
            if missing_fields:
                results["errors"].append(f"Missing paper fields: {missing_fields}")
            
            # Validate content structure
            content = config['content']
            if 'files' not in content or not isinstance(content['files'], list):
                results["errors"].append("Content files must be a list")
            
            # Validate build configuration
            build = config['build']
            if not isinstance(build.get('cross_reference_validation', True), bool):
                results["warnings"].append("cross_reference_validation should be boolean")
            
            # Check if all content files exist
            missing_files = []
            for file_name in content.get('files', []):
                file_path = self.papers_dir / paper_name / file_name
                if not file_path.exists():
                    missing_files.append(file_name)
            
            if missing_files:
                results["errors"].append(f"Missing content files: {missing_files}")
            
            results["valid"] = len(results["errors"]) == 0
            
        except yaml.YAMLError as e:
            results["errors"].append(f"YAML syntax error: {e}")
        except Exception as e:
            results["errors"].append(f"Configuration error: {e}")
        
        return results
    
    def validate_figure_format(self, paper_name: str) -> Dict[str, Any]:
        """Validate figure format compliance."""
        results = {"valid": False, "errors": [], "warnings": [], "figures": []}
        
        paper_dir = self.papers_dir / paper_name
        
        # Find all markdown files
        md_files = list(paper_dir.glob("*.md"))
        
        for md_file in md_files:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find figure definitions
            figure_pattern = r'## Figure: ([^{]+) \{#fig:([^}]+)\}'
            figures = re.findall(figure_pattern, content)
            
            for title, fig_id in figures:
                figure_info = {
                    "file": md_file.name,
                    "title": title.strip(),
                    "id": fig_id,
                    "has_caption": False,
                    "has_image": False
                }
                
                # Check for caption
                caption_pattern = rf'## Figure: {re.escape(title)} \{{#fig:{re.escape(fig_id)}\}}.*?\*\*Caption:\*\* ([^\n]+)'
                if re.search(caption_pattern, content, re.DOTALL):
                    figure_info["has_caption"] = True
                
                # Check for image
                image_pattern = rf'## Figure: {re.escape(title)} \{{#fig:{re.escape(fig_id)}\}}.*?!\[([^\]]*)\]\(([^)]+)\)'
                if re.search(image_pattern, content, re.DOTALL):
                    figure_info["has_image"] = True
                
                results["figures"].append(figure_info)
                
                if not figure_info["has_caption"]:
                    results["warnings"].append(f"Figure {fig_id} missing caption")
                
                if not figure_info["has_image"]:
                    results["warnings"].append(f"Figure {fig_id} missing image")
        
        # Check for inline figure definitions (should not exist)
        inline_pattern = r'!\[[^\]]*\]\([^)]+\)\{#fig:[^}]+\}'
        for md_file in md_files:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if re.search(inline_pattern, content):
                results["errors"].append(f"Inline figure definition found in {md_file.name}")
        
        results["valid"] = len(results["errors"]) == 0
        return results
    
    def validate_math_formatting(self, paper_name: str) -> Dict[str, Any]:
        """Validate math symbol formatting."""
        results = {"valid": False, "errors": [], "warnings": [], "issues": []}
        
        paper_dir = self.papers_dir / paper_name
        md_files = list(paper_dir.glob("*.md"))
        
        # Common Unicode symbols that should be LaTeX
        unicode_symbols = {
            'μ': '\\mu',
            'λ': '\\lambda', 
            'π': '\\pi',
            'ε': '\\epsilon',
            'Δ': '\\Delta',
            'ρ': '\\rho',
            'σ': '\\sigma',
            '±': '\\pm',
            '≤': '\\le',
            '≥': '\\ge',
            '≈': '\\approx',
            '∝': '\\propto'
        }
        
        for md_file in md_files:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for unicode_char, latex_cmd in unicode_symbols.items():
                if unicode_char in content:
                    results["issues"].append({
                        "file": md_file.name,
                        "unicode": unicode_char,
                        "should_be": latex_cmd,
                        "context": self._get_context(content, unicode_char)
                    })
                    results["warnings"].append(f"Unicode symbol '{unicode_char}' found in {md_file.name}, should use '{latex_cmd}'")
        
        # Check for proper math wrapping
        math_patterns = [
            (r'\\\([^)]*\\\)', "LaTeX inline math should use $...$"),
            (r'\\mu\\\\mathrm', "Math formatting issue"),
            (r'\\lambda\\\\mathrm', "Math formatting issue"),
        ]
        
        for md_file in md_files:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for pattern, message in math_patterns:
                if re.search(pattern, content):
                    results["errors"].append(f"{message} in {md_file.name}")
        
        results["valid"] = len(results["errors"]) == 0
        return results
    
    def validate_hyperlinks(self, paper_name: str) -> Dict[str, Any]:
        """Validate hyperlink formatting."""
        results = {"valid": False, "errors": [], "warnings": [], "links": []}
        
        paper_dir = self.papers_dir / paper_name
        md_files = list(paper_dir.glob("*.md"))
        
        for md_file in md_files:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find naked URLs
            url_pattern = r'https?://[^\s\)]+'
            urls = re.findall(url_pattern, content)
            
            for url in urls:
                # Check if it's properly wrapped in \href{}
                if f'\\href{{{url}}}' not in content:
                    results["warnings"].append(f"Naked URL found in {md_file.name}: {url}")
                    results["links"].append({
                        "file": md_file.name,
                        "url": url,
                        "type": "naked",
                        "should_be": f"\\href{{{url}}}{{descriptive text}}"
                    })
            
            # Find \href{} links
            href_pattern = r'\\href\{([^}]+)\}\{([^}]+)\}'
            href_links = re.findall(href_pattern, content)
            
            for url, text in href_links:
                results["links"].append({
                    "file": md_file.name,
                    "url": url,
                    "text": text,
                    "type": "proper"
                })
        
        results["valid"] = len(results["errors"]) == 0
        return results
    
    def validate_cross_references(self, paper_name: str) -> Dict[str, Any]:
        """Validate cross-reference consistency."""
        results = {"valid": False, "errors": [], "warnings": [], "references": []}
        
        if not CORE_AVAILABLE:
            results["warnings"].append("Core modules not available, skipping cross-reference validation")
            return results
        
        paper_dir = self.papers_dir / paper_name
        md_files = list(paper_dir.glob("*.md"))
        
        # Collect all content
        full_content = ""
        for md_file in md_files:
            with open(md_file, 'r', encoding='utf-8') as f:
                full_content += f.read() + "\n"
        
        # Use core validation
        try:
            validation_result = validate_cross_references(full_content, str(paper_dir))
            results.update(validation_result)
        except Exception as e:
            results["errors"].append(f"Cross-reference validation failed: {e}")
        
        return results
    
    def validate_build_system(self) -> Dict[str, Any]:
        """Validate build system integration."""
        results = {"valid": False, "errors": [], "warnings": []}
        
        # Check build scripts exist
        build_script = self.project_root / "scripts" / "common_pipeline" / "build_core.py"
        render_script = self.project_root / "tools" / "render_pdf.sh"
        
        if not build_script.exists():
            results["errors"].append("Build script not found: build_core.py")
        
        if not render_script.exists():
            results["errors"].append("Render script not found: render_pdf.sh")
        
        # Check core package
        core_package = self.project_root / "antstack_core"
        if not core_package.exists():
            results["errors"].append("Core package not found: antstack_core")
        
        # Check required directories
        required_dirs = ["papers", "scripts", "tools", "tests"]
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                results["errors"].append(f"Required directory not found: {dir_name}")
        
        results["valid"] = len(results["errors"]) == 0
        return results
    
    def _get_context(self, content: str, char: str, context_length: int = 50) -> str:
        """Get context around a character in content."""
        pos = content.find(char)
        if pos == -1:
            return ""
        
        start = max(0, pos - context_length)
        end = min(len(content), pos + context_length)
        return content[start:end]
    
    def validate_paper(self, paper_name: str) -> Dict[str, Any]:
        """Validate a single paper comprehensively."""
        print(f"Validating paper: {paper_name}")
        
        results = {
            "paper": paper_name,
            "overall_valid": False,
            "config": {},
            "figures": {},
            "math": {},
            "hyperlinks": {},
            "cross_references": {},
            "summary": {}
        }
        
        # Validate each component
        results["config"] = self.validate_paper_config(paper_name)
        results["figures"] = self.validate_figure_format(paper_name)
        results["math"] = self.validate_math_formatting(paper_name)
        results["hyperlinks"] = self.validate_hyperlinks(paper_name)
        results["cross_references"] = self.validate_cross_references(paper_name)
        
        # Overall validation
        component_valid = all([
            results["config"]["valid"],
            results["figures"]["valid"],
            results["math"]["valid"],
            results["hyperlinks"]["valid"],
            results["cross_references"]["valid"]
        ])
        
        results["overall_valid"] = component_valid
        
        # Summary
        total_errors = sum(len(r.get("errors", [])) for r in results.values() if isinstance(r, dict))
        total_warnings = sum(len(r.get("warnings", [])) for r in results.values() if isinstance(r, dict))
        
        results["summary"] = {
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "status": "PASS" if component_valid else "FAIL"
        }
        
        return results
    
    def validate_all_papers(self) -> Dict[str, Any]:
        """Validate all papers in the system."""
        papers = self.discover_papers()
        results = {
            "system_valid": False,
            "papers": {},
            "build_system": {},
            "summary": {}
        }
        
        print(f"Found {len(papers)} papers: {papers}")
        
        # Validate build system
        results["build_system"] = self.validate_build_system()
        
        # Validate each paper
        for paper in papers:
            results["papers"][paper] = self.validate_paper(paper)
        
        # Overall system validation
        paper_valid = all(p["overall_valid"] for p in results["papers"].values())
        build_valid = results["build_system"]["valid"]
        results["system_valid"] = paper_valid and build_valid
        
        # Summary
        total_errors = sum(p["summary"]["total_errors"] for p in results["papers"].values())
        total_warnings = sum(p["summary"]["total_warnings"] for p in results["papers"].values())
        
        results["summary"] = {
            "total_papers": len(papers),
            "valid_papers": sum(1 for p in results["papers"].values() if p["overall_valid"]),
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "system_status": "PASS" if results["system_valid"] else "FAIL"
        }
        
        return results


def main():
    """Main validation entry point."""
    parser = argparse.ArgumentParser(description="Validate PDF rendering system")
    parser.add_argument("--paper", help="Validate specific paper only")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    validator = RenderingSystemValidator()
    
    if args.paper:
        # Validate single paper
        results = validator.validate_paper(args.paper)
        
        print(f"\n=== Validation Results for {args.paper} ===")
        print(f"Overall Status: {results['summary']['status']}")
        print(f"Errors: {results['summary']['total_errors']}")
        print(f"Warnings: {results['summary']['total_warnings']}")
        
        if args.verbose:
            for component, data in results.items():
                if isinstance(data, dict) and 'errors' in data:
                    if data['errors']:
                        print(f"\n{component.upper()} ERRORS:")
                        for error in data['errors']:
                            print(f"  - {error}")
                    
                    if data['warnings']:
                        print(f"\n{component.upper()} WARNINGS:")
                        for warning in data['warnings']:
                            print(f"  - {warning}")
        
        return 0 if results['overall_valid'] else 1
    
    else:
        # Validate all papers
        results = validator.validate_all_papers()
        
        print(f"\n=== System Validation Results ===")
        print(f"System Status: {results['summary']['system_status']}")
        print(f"Papers: {results['summary']['valid_papers']}/{results['summary']['total_papers']} valid")
        print(f"Total Errors: {results['summary']['total_errors']}")
        print(f"Total Warnings: {results['summary']['total_warnings']}")
        
        if args.verbose:
            for paper, data in results['papers'].items():
                print(f"\n--- {paper} ---")
                print(f"Status: {data['summary']['status']}")
                print(f"Errors: {data['summary']['total_errors']}, Warnings: {data['summary']['total_warnings']}")
        
        return 0 if results['system_valid'] else 1


if __name__ == "__main__":
    sys.exit(main())
