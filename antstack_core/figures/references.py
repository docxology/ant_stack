"""Cross-reference validation and management for scientific publications.

Comprehensive cross-reference handling following .cursorrules specifications:
- Figure ID validation and consistency checking
- Reference-definition matching
- Broken reference detection and repair
- Pandoc-compatible cross-reference format enforcement

Design principles:
- Zero tolerance for broken references (Figure~??)
- Automated detection of figure/reference mismatches
- Validation before PDF generation
- Clear error reporting and suggestions

References:
- Pandoc cross-references: https://pandoc.org/MANUAL.html#extension-crossref
- Scientific writing best practices: https://doi.org/10.1371/journal.pcbi.1005619
"""

from __future__ import annotations
import re
from typing import Dict, List, Tuple, Set, Optional
from pathlib import Path


class CrossReferenceValidator:
    """Validator for cross-references in scientific documents."""
    
    def __init__(self):
        self.figure_definitions: Dict[str, Dict[str, str]] = {}
        self.figure_references: Dict[str, List[Dict[str, str]]] = {}
        self.validation_results: Dict[str, any] = {}
    
    def validate_document(self, content: str, file_path: Optional[str] = None) -> Dict[str, any]:
        """Validate all cross-references in a document.
        
        Args:
            content: Document content (Markdown)
            file_path: Optional file path for context
            
        Returns:
            Dictionary with validation results and issues
        """
        self.figure_definitions = self._extract_figure_definitions(content)
        self.figure_references = self._extract_figure_references(content)
        
        issues = []
        warnings = []
        
        # Check for broken references
        broken_refs = self._find_broken_references()
        if broken_refs:
            issues.extend(broken_refs)
        
        # Check for undefined figures
        undefined_figs = self._find_undefined_figures()
        if undefined_figs:
            issues.extend(undefined_figs)
        
        # Check for unused definitions
        unused_defs = self._find_unused_definitions()
        if unused_defs:
            warnings.extend(unused_defs)
        
        # Check figure format compliance
        format_issues = self._check_figure_format_compliance(content)
        if format_issues:
            issues.extend(format_issues)
        
        self.validation_results = {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "figure_definitions": len(self.figure_definitions),
            "figure_references": sum(len(refs) for refs in self.figure_references.values()),
            "file_path": file_path
        }
        
        return self.validation_results
    
    def _extract_figure_definitions(self, content: str) -> Dict[str, Dict[str, str]]:
        """Extract figure definitions in the format: ## Figure: Title {#fig:id}"""
        definitions = {}
        
        # Pattern for figure definitions following .cursorrules format
        pattern = r'^##\s+Figure:\s+([^{]+)\s+\{#fig:([^}]+)\}'
        
        for match in re.finditer(pattern, content, re.MULTILINE):
            title = match.group(1).strip()
            fig_id = match.group(2).strip()
            
            definitions[fig_id] = {
                "title": title,
                "full_match": match.group(0),
                "line_number": content[:match.start()].count('\n') + 1
            }
        
        return definitions
    
    def _extract_figure_references(self, content: str) -> Dict[str, List[Dict[str, str]]]:
        """Extract figure references in the format: Figure~\\ref{fig:id}"""
        references = {}
        
        # Pattern for figure references
        patterns = [
            r'Figure~\\ref\{fig:([^}]+)\}',
            r'\\cref\{fig:([^}]+)\}',
            r'\\ref\{fig:([^}]+)\}'
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, content):
                fig_id = match.group(1).strip()
                
                if fig_id not in references:
                    references[fig_id] = []
                
                references[fig_id].append({
                    "full_match": match.group(0),
                    "pattern": pattern,
                    "line_number": content[:match.start()].count('\n') + 1
                })
        
        return references
    
    def _find_broken_references(self) -> List[str]:
        """Find references that don't have corresponding definitions."""
        broken = []
        
        for ref_id, refs in self.figure_references.items():
            if ref_id not in self.figure_definitions:
                for ref in refs:
                    broken.append(
                        f"Broken reference: {ref['full_match']} on line {ref['line_number']} "
                        f"- no definition found for 'fig:{ref_id}'"
                    )
        
        return broken
    
    def _find_undefined_figures(self) -> List[str]:
        """Find figure definitions that are never referenced."""
        undefined = []
        
        for def_id in self.figure_definitions:
            if def_id not in self.figure_references:
                definition = self.figure_definitions[def_id]
                undefined.append(
                    f"Unused figure definition: 'fig:{def_id}' ('{definition['title']}') "
                    f"on line {definition['line_number']} - no references found"
                )
        
        return undefined
    
    def _find_unused_definitions(self) -> List[str]:
        """Find definitions that exist but are never referenced (warnings only)."""
        return self._find_undefined_figures()  # Same logic, but these are warnings
    
    def _check_figure_format_compliance(self, content: str) -> List[str]:
        """Check compliance with .cursorrules figure format requirements."""
        issues = []
        
        # Check for inline figure definitions (not allowed per .cursorrules)
        inline_pattern = r'!\[[^\]]*\]\([^)]+\)\{#fig:[^}]+\}'
        inline_matches = list(re.finditer(inline_pattern, content))
        
        for match in inline_matches:
            line_num = content[:match.start()].count('\n') + 1
            issues.append(
                f"Invalid inline figure definition on line {line_num}: {match.group(0)} "
                f"- use '## Figure: Title {{#fig:id}}' format instead"
            )
        
        # Check for figures without captions
        fig_def_pattern = r'^##\s+Figure:[^{]+\{#fig:([^}]+)\}'
        caption_pattern = r'\*\*Caption:\*\*'
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if re.match(fig_def_pattern, line):
                # Look for caption in next few lines
                caption_found = False
                for j in range(i + 1, min(i + 5, len(lines))):
                    if re.search(caption_pattern, lines[j]):
                        caption_found = True
                        break
                
                if not caption_found:
                    match = re.match(fig_def_pattern, line)
                    if match:
                        fig_id = match.group(1)
                        issues.append(
                            f"Figure 'fig:{fig_id}' on line {i + 1} missing "
                            f"'**Caption:**' (required per .cursorrules)"
                        )
        
        return issues


def validate_cross_references(content: str, file_path: Optional[str] = None) -> Dict[str, any]:
    """Validate cross-references in a document.
    
    Args:
        content: Document content (Markdown)
        file_path: Optional file path for context
        
    Returns:
        Dictionary with validation results
    """
    validator = CrossReferenceValidator()
    return validator.validate_document(content, file_path)


def fix_figure_ids(content: str, id_mapping: Optional[Dict[str, str]] = None) -> str:
    """Fix figure IDs and references in content.
    
    Args:
        content: Document content to fix
        id_mapping: Optional mapping of old_id -> new_id
        
    Returns:
        Fixed content with consistent IDs
    """
    if id_mapping is None:
        # Generate automatic mapping based on sequential numbering
        validator = CrossReferenceValidator()
        definitions = validator._extract_figure_definitions(content)
        
        id_mapping = {}
        for i, old_id in enumerate(sorted(definitions.keys()), 1):
            new_id = f"fig_{i:03d}"
            id_mapping[old_id] = new_id
    
    # Apply ID mapping to definitions
    def replace_definition(match):
        title = match.group(1).strip()
        old_id = match.group(2).strip()
        new_id = id_mapping.get(old_id, old_id)
        return f"## Figure: {title} {{#fig:{new_id}}}"
    
    content = re.sub(
        r'^##\s+Figure:\s+([^{]+)\s+\{#fig:([^}]+)\}',
        replace_definition,
        content,
        flags=re.MULTILINE
    )
    
    # Apply ID mapping to references
    for old_id, new_id in id_mapping.items():
        # Replace various reference formats
        patterns = [
            (f'Figure~\\ref{{fig:{old_id}}}', f'Figure~\\ref{{fig:{new_id}}}'),
            (f'\\cref{{fig:{old_id}}}', f'\\cref{{fig:{new_id}}}'),
            (f'\\ref{{fig:{old_id}}}', f'\\ref{{fig:{new_id}}}')
        ]
        
        for old_pattern, new_pattern in patterns:
            content = content.replace(old_pattern, new_pattern)
    
    return content


def generate_cross_reference_report(content: str, file_path: Optional[str] = None) -> str:
    """Generate a comprehensive cross-reference report.
    
    Args:
        content: Document content to analyze
        file_path: Optional file path for context
        
    Returns:
        Formatted report string
    """
    results = validate_cross_references(content, file_path)
    
    report = []
    report.append("# Cross-Reference Validation Report")
    
    if file_path:
        report.append(f"**File:** {file_path}")
    
    report.append(f"**Status:** {'✅ VALID' if results['valid'] else '❌ ISSUES FOUND'}")
    report.append(f"**Figure Definitions:** {results['figure_definitions']}")
    report.append(f"**Figure References:** {results['figure_references']}")
    
    if results['issues']:
        report.append("\n## Issues (Must Fix)")
        for issue in results['issues']:
            report.append(f"- ❌ {issue}")
    
    if results['warnings']:
        report.append("\n## Warnings (Recommendations)")
        for warning in results['warnings']:
            report.append(f"- ⚠️ {warning}")
    
    if results['valid']:
        report.append("\n✅ All cross-references are valid!")
    
    return '\n'.join(report)
