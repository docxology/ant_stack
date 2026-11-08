"""Comprehensive validation of the PDF rendering system. Moved verbatim from scripts/validate_rendering_system.py; thin wrapper retained for CLI compatibility."""

import re

import yaml
from pathlib import Path
from typing import Dict, List, Any

try:
    from antstack_core.figures import validate_cross_references
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False


from ..figures.references import _without_code
from .paper_builder import ModularPaperBuilder
from .pdf_generator import validate_pdf_environment
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

    def _content_sources(self, paper_name: str) -> List[Path]:
        config = ModularPaperBuilder(self.project_root).load_paper_config(paper_name)
        return [self.papers_dir / paper_name / name for name in config['content']['files']]

    
    def validate_paper_config(self, paper_name: str) -> Dict[str, Any]:
        """Use the same strict schema and ordered content selection as the builder."""
        results = {"valid": False, "errors": [], "warnings": []}
        try:
            sources = self._content_sources(paper_name)
            results['errors'].extend(f'Missing content file: {path.name}' for path in sources if not path.is_file())
            results['valid'] = not results['errors']
        except (OSError, ValueError, TypeError, yaml.YAMLError) as error:
            results['errors'].append(str(error))
        return results

    
    def validate_figure_format(self, paper_name: str) -> Dict[str, Any]:
        """Validate figure format compliance."""
        results = {"valid": False, "errors": [], "warnings": [], "figures": []}
        
        paper_dir = self.papers_dir / paper_name
        
        # Find all markdown files
        md_files = self._content_sources(paper_name)
        
        for md_file in md_files:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = _without_code(f.read())
            
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
                content = _without_code(f.read())
            
            if re.search(inline_pattern, content):
                results["errors"].append(f"Inline figure definition found in {md_file.name}")
        
        results["valid"] = len(results["errors"]) == 0
        return results
    
    def validate_math_formatting(self, paper_name: str) -> Dict[str, Any]:
        """Validate math symbol formatting."""
        results = {"valid": False, "errors": [], "warnings": [], "issues": []}
        
        paper_dir = self.papers_dir / paper_name
        md_files = self._content_sources(paper_name)
        
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
                content = _without_code(f.read())
            
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
            (r'\\mu\\\\mathrm', "Math formatting issue"),
            (r'\\lambda\\\\mathrm', "Math formatting issue"),
        ]
        
        for md_file in md_files:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = _without_code(f.read())
            
            for pattern, message in math_patterns:
                if re.search(pattern, content):
                    results["errors"].append(f"{message} in {md_file.name}")
        
        results["valid"] = len(results["errors"]) == 0
        return results
    
    def validate_hyperlinks(self, paper_name: str) -> Dict[str, Any]:
        """Validate hyperlink formatting."""
        results = {"valid": False, "errors": [], "warnings": [], "links": []}
        
        paper_dir = self.papers_dir / paper_name
        md_files = self._content_sources(paper_name)
        
        for md_file in md_files:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = _without_code(f.read())
            
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
            results['errors'].append('Core cross-reference validator is unavailable')
            return results
        
        paper_dir = self.papers_dir / paper_name
        md_files = self._content_sources(paper_name)
        
        # Collect all content
        full_content = ""
        for md_file in md_files:
            with open(md_file, 'r', encoding='utf-8') as f:
                full_content += f.read() + "\n"
        
        # Use core validation
        try:
            validation_result = validate_cross_references(full_content, str(paper_dir))
            results.update(valid=validation_result['valid'], errors=validation_result['issues'],
                           warnings=validation_result['warnings'], references=validation_result['references'])
        except Exception as e:
            results["errors"].append(f"Cross-reference validation failed: {e}")
        
        return results
    
    def validate_build_system(self) -> Dict[str, Any]:
        """Observe installed rendering tools, not unshipped repository script/test folders."""
        environment = validate_pdf_environment()
        errors = [f'Missing rendering prerequisite: {key}' for key, available in environment['validation'].items() if not available]
        if not self.papers_dir.is_dir():
            errors.append('Configured project has no papers directory')
        return {'valid': not errors, 'errors': errors, 'warnings': [], 'checks': environment['validation'],
                'validation_scope': 'source/tool preflight; no PDF render was performed'}

    
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
        if results['config']['valid']:
            results["figures"] = self.validate_figure_format(paper_name)
            results["math"] = self.validate_math_formatting(paper_name)
            results["hyperlinks"] = self.validate_hyperlinks(paper_name)
            results["cross_references"] = self.validate_cross_references(paper_name)
        else:
            for component in ('figures', 'math', 'hyperlinks', 'cross_references'):
                results[component] = {'valid': False, 'errors': [], 'warnings': [], 'status': 'not_run'}
        
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


def run(args):
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
