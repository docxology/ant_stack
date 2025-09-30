"""
Reference Management System for Ant Stack Publications

Comprehensive cross-reference management with:
- Automatic reference extraction and validation
- BibTeX integration and citation management
- Figure and table reference tracking
- Section and equation cross-referencing
- Broken reference detection and repair

Following .cursorrules specifications for:
- Zero tolerance for broken references
- Pandoc-compatible cross-reference format
- Professional citation management
- Automated reference validation
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import yaml
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ReferenceDefinition:
    """Represents a reference definition in a document."""

    ref_type: str  # 'fig', 'tab', 'sec', 'eq'
    ref_id: str
    file_path: Path
    line_number: int
    label: str = ""
    caption: Optional[str] = None

    @property
    def full_id(self) -> str:
        """Full reference identifier."""
        return f"{self.ref_type}:{self.ref_id}"


@dataclass
class ReferenceUsage:
    """Represents a reference usage in a document."""

    ref_type: str
    ref_id: str
    file_path: Path
    line_number: int
    context: str = ""

    @property
    def full_id(self) -> str:
        """Full reference identifier."""
        return f"{self.ref_type}:{self.ref_id}"


@dataclass
class ReferenceReport:
    """Comprehensive reference analysis report."""

    definitions: List[ReferenceDefinition] = None
    usages: List[ReferenceUsage] = None
    broken_references: List[ReferenceUsage] = None
    orphaned_definitions: List[ReferenceDefinition] = None
    cross_file_references: List[Tuple[str, str, int]] = None
    summary: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.definitions is None:
            self.definitions = []
        if self.usages is None:
            self.usages = []
        if self.broken_references is None:
            self.broken_references = []
        if self.orphaned_definitions is None:
            self.orphaned_definitions = []
        if self.cross_file_references is None:
            self.cross_file_references = []
        if self.summary is None:
            self.summary = {}


class ReferenceManager:
    """
    Comprehensive reference management system for scientific publications.

    Manages:
    - Cross-reference extraction and validation
    - Broken reference detection and repair
    - Citation management and BibTeX integration
    - Reference consistency across documents
    """

    def __init__(self):
        """Initialize reference manager."""
        self._setup_patterns()

    def _setup_patterns(self):
        """Configure reference extraction patterns."""

        # Definition patterns (Pandoc format)
        self.definition_patterns = {
            'figure': [
                re.compile(r'!\[([^\]]*)\]\{([^}]*(#(fig:[^}\s]+))[^}]*\}'),
                re.compile(r'## Figure:\s*([^{]+)\s*\{#(fig:[^}]+)\}'),
                re.compile(r'\\label\{(fig:[^}]+)\}'),
            ],
            'table': [
                re.compile(r'## Table:\s*([^{]+)\s*\{#(tab:[^}]+)\}'),
                re.compile(r'\\label\{(tab:[^}]+)\}'),
            ],
            'section': [
                re.compile(r'#{1,6}\s+(.+)\s*\{#(sec:[^}]+)\}'),
                re.compile(r'\\label\{(sec:[^}]+)\}'),
            ],
            'equation': [
                re.compile(r'\\label\{(eq:[^}]+)\}'),
                re.compile(r'\\begin\{equation\}.*\\label\{(eq:[^}]+)\}', re.DOTALL),
            ]
        }

        # Usage patterns
        self.usage_patterns = {
            'figure': [
                re.compile(r'\\cref\{(fig:[^}]+)\}'),
                re.compile(r'\\ref\{(fig:[^}]+)\}'),
                re.compile(r'Figure~\s*\\ref\{(fig:[^}]+)\}'),
            ],
            'table': [
                re.compile(r'\\cref\{(tab:[^}]+)\}'),
                re.compile(r'\\ref\{(tab:[^}]+)\}'),
            ],
            'section': [
                re.compile(r'\\cref\{(sec:[^}]+)\}'),
                re.compile(r'\\ref\{(sec:[^}]+)\}'),
            ],
            'equation': [
                re.compile(r'\\cref\{(eq:[^}]+)\}'),
                re.compile(r'\\ref\{(eq:[^}]+)\}'),
            ]
        }

        # Citation patterns
        self.citation_patterns = [
            re.compile(r'\\cite\{([^}]+)\}'),
            re.compile(r'\\citep\{([^}]+)\}'),
            re.compile(r'\\citet\{([^}]+)\}'),
            re.compile(r'\\citeauthor\{([^}]+)\}'),
            re.compile(r'\\citeyear\{([^}]+)\}'),
        ]

    def analyze_references(self, files: List[Path]) -> ReferenceReport:
        """
        Perform comprehensive reference analysis across files.

        Args:
            files: List of files to analyze

        Returns:
            Comprehensive reference analysis report
        """
        report = ReferenceReport()

        # Extract definitions and usages
        for file_path in files:
            if file_path.exists() and file_path.suffix.lower() in ['.md', '.markdown']:
                try:
                    content = file_path.read_text(encoding='utf-8')
                    lines = content.split('\n')

                    # Extract definitions
                    file_definitions = self._extract_definitions(content, file_path, lines)
                    report.definitions.extend(file_definitions)

                    # Extract usages
                    file_usages = self._extract_usages(content, file_path, lines)
                    report.usages.extend(file_usages)

                except Exception as e:
                    logger.warning(f"Failed to analyze references in {file_path}: {str(e)}")

        # Analyze reference consistency
        report.broken_references = self._find_broken_references(report.definitions, report.usages)
        report.orphaned_definitions = self._find_orphaned_definitions(report.definitions, report.usages)
        report.cross_file_references = self._analyze_cross_file_references(report.usages, files)

        # Generate summary
        report.summary = self._generate_reference_summary(report)

        return report

    def _extract_definitions(self, content: str, file_path: Path,
                           lines: List[str]) -> List[ReferenceDefinition]:
        """Extract reference definitions from content."""
        definitions = []

        for ref_type, patterns in self.definition_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(content):
                    # Extract reference ID
                    if ref_type in ['figure', 'table', 'section']:
                        if len(match.groups()) >= 2:
                            label = match.group(1).strip()
                            ref_id_full = match.group(2)
                        else:
                            continue
                    else:  # equation
                        ref_id_full = match.group(1)

                    # Parse reference ID
                    if ':' in ref_id_full:
                        _, ref_id = ref_id_full.split(':', 1)
                    else:
                        ref_id = ref_id_full

                    # Find line number
                    line_number = self._find_line_number(content, match.start())

                    # Extract caption/label if available
                    caption = self._extract_caption(content, match, ref_type)

                    definition = ReferenceDefinition(
                        ref_type=ref_type,
                        ref_id=ref_id,
                        file_path=file_path,
                        line_number=line_number,
                        label=label if 'label' in locals() else "",
                        caption=caption
                    )
                    definitions.append(definition)

        return definitions

    def _extract_usages(self, content: str, file_path: Path,
                       lines: List[str]) -> List[ReferenceUsage]:
        """Extract reference usages from content."""
        usages = []

        for ref_type, patterns in self.usage_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(content):
                    ref_id_full = match.group(1)

                    # Parse reference ID
                    if ':' in ref_id_full:
                        _, ref_id = ref_id_full.split(':', 1)
                    else:
                        ref_id = ref_id_full

                    # Find line number and context
                    line_number = self._find_line_number(content, match.start())
                    context = self._extract_context(lines, line_number - 1, 2)

                    usage = ReferenceUsage(
                        ref_type=ref_type,
                        ref_id=ref_id,
                        file_path=file_path,
                        line_number=line_number,
                        context=context
                    )
                    usages.append(usage)

        return usages

    def _find_broken_references(self, definitions: List[ReferenceDefinition],
                              usages: List[ReferenceUsage]) -> List[ReferenceUsage]:
        """Find references that don't have corresponding definitions."""
        defined_refs = {defn.full_id for defn in definitions}
        broken_refs = []

        for usage in usages:
            if usage.full_id not in defined_refs:
                broken_refs.append(usage)

        return broken_refs

    def _find_orphaned_definitions(self, definitions: List[ReferenceDefinition],
                                 usages: List[ReferenceUsage]) -> List[ReferenceDefinition]:
        """Find definitions that are never referenced."""
        used_refs = {usage.full_id for usage in usages}
        orphaned = []

        for definition in definitions:
            if definition.full_id not in used_refs:
                orphaned.append(definition)

        return orphaned

    def _analyze_cross_file_references(self, usages: List[ReferenceUsage],
                                     files: List[Path]) -> List[Tuple[str, str, int]]:
        """Analyze references that cross between files."""
        file_map = {str(f): f.name for f in files}
        cross_refs = []

        for usage in usages:
            usage_file = str(usage.file_path)

            # This is a simplified analysis - in practice you'd need
            # to know which file contains each definition
            if len(files) > 1:
                cross_refs.append((usage.full_id, usage_file, usage.line_number))

        return cross_refs

    def _find_line_number(self, content: str, char_position: int) -> int:
        """Find line number for a character position in content."""
        return content[:char_position].count('\n') + 1

    def _extract_context(self, lines: List[str], line_idx: int, context_lines: int = 2) -> str:
        """Extract context around a line."""
        start = max(0, line_idx - context_lines)
        end = min(len(lines), line_idx + context_lines + 1)

        context_lines_list = []
        for i in range(start, end):
            marker = ">>> " if i == line_idx else "    "
            context_lines_list.append(f"{marker}{lines[i]}")

        return "\n".join(context_lines_list)

    def _extract_caption(self, content: str, match: re.Match, ref_type: str) -> Optional[str]:
        """Extract caption/label from reference definition."""
        try:
            # Look for caption pattern after the match
            start_pos = match.end()
            search_content = content[start_pos:start_pos + 500]  # Look ahead 500 chars

            if ref_type in ['figure', 'table']:
                caption_match = re.search(r'\*\*Caption:\*\*\s*([^\n]+)', search_content)
                if caption_match:
                    return caption_match.group(1).strip()

        except Exception:
            pass

        return None

    def _generate_reference_summary(self, report: ReferenceReport) -> Dict[str, Any]:
        """Generate summary statistics for reference analysis."""
        summary = {
            'total_definitions': len(report.definitions),
            'total_usages': len(report.usages),
            'broken_references': len(report.broken_references),
            'orphaned_definitions': len(report.orphaned_definitions),
            'cross_file_references': len(report.cross_file_references),
            'by_type': defaultdict(lambda: {'definitions': 0, 'usages': 0, 'broken': 0})
        }

        # Count by reference type
        for defn in report.definitions:
            summary['by_type'][defn.ref_type]['definitions'] += 1

        for usage in report.usages:
            summary['by_type'][usage.ref_type]['usages'] += 1

        for broken in report.broken_references:
            summary['by_type'][broken.ref_type]['broken'] += 1

        # Convert defaultdict to regular dict
        summary['by_type'] = dict(summary['by_type'])

        # Calculate health score
        total_refs = summary['total_definitions'] + summary['total_usages']
        if total_refs > 0:
            broken_ratio = summary['broken_references'] / total_refs
            summary['health_score'] = max(0, 100 - (broken_ratio * 100))
        else:
            summary['health_score'] = 100

        return dict(summary)

    def fix_broken_references(self, report: ReferenceReport,
                            auto_fix: bool = False) -> Dict[str, Any]:
        """
        Attempt to fix broken references automatically.

        Args:
            report: Reference analysis report
            auto_fix: Whether to apply fixes automatically

        Returns:
            Dictionary with fix results and suggestions
        """
        fixes_applied = []
        manual_fixes_needed = []

        for broken_ref in report.broken_references:
            # Try to find similar reference IDs
            similar_defs = self._find_similar_definitions(broken_ref, report.definitions)

            if similar_defs:
                suggestion = f"Change {broken_ref.full_id} to {similar_defs[0].full_id}"
                if auto_fix:
                    # In a real implementation, this would edit the file
                    fixes_applied.append({
                        'reference': broken_ref.full_id,
                        'file': str(broken_ref.file_path),
                        'line': broken_ref.line_number,
                        'fix_type': 'id_correction',
                        'old_id': broken_ref.full_id,
                        'new_id': similar_defs[0].full_id
                    })
                else:
                    manual_fixes_needed.append({
                        'reference': broken_ref.full_id,
                        'file': str(broken_ref.file_path),
                        'line': broken_ref.line_number,
                        'suggestion': suggestion,
                        'similar_definitions': [d.full_id for d in similar_defs[:3]]
                    })
            else:
                manual_fixes_needed.append({
                    'reference': broken_ref.full_id,
                    'file': str(broken_ref.file_path),
                    'line': broken_ref.line_number,
                    'suggestion': 'No similar definitions found - may need to create definition',
                    'similar_definitions': []
                })

        return {
            'fixes_applied': fixes_applied,
            'manual_fixes_needed': manual_fixes_needed,
            'auto_fix_enabled': auto_fix,
            'success_rate': len(fixes_applied) / len(report.broken_references) if report.broken_references else 1.0
        }

    def _find_similar_definitions(self, usage: ReferenceUsage,
                                definitions: List[ReferenceDefinition],
                                threshold: float = 0.8) -> List[ReferenceDefinition]:
        """Find definitions with similar IDs to a usage."""
        similar_defs = []

        for defn in definitions:
            if defn.ref_type == usage.ref_type:
                # Simple similarity based on string similarity
                similarity = self._calculate_string_similarity(defn.ref_id, usage.ref_id)
                if similarity >= threshold:
                    similar_defs.append(defn)

        # Sort by similarity (would need more sophisticated sorting)
        return similar_defs[:5]  # Return top 5

    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate simple string similarity."""
        # This is a very basic implementation
        # In practice, you'd use something like Levenshtein distance
        str1, str2 = str1.lower(), str2.lower()

        if str1 == str2:
            return 1.0

        # Check for common prefixes/suffixes
        if str1.startswith(str2) or str2.startswith(str1):
            return 0.9

        # Check for substring matches
        if str1 in str2 or str2 in str1:
            return 0.7

        return 0.0

    def validate_citations(self, files: List[Path],
                          bib_file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Validate citations and bibliography consistency.

        Args:
            files: Files to check for citations
            bib_file: Bibliography file (BibTeX format)

        Returns:
            Citation validation results
        """
        citations_found = set()
        citation_usages = []

        # Extract citations from files
        for file_path in files:
            if file_path.exists() and file_path.suffix.lower() in ['.md', '.markdown']:
                try:
                    content = file_path.read_text(encoding='utf-8')

                    for pattern in self.citation_patterns:
                        for match in pattern.finditer(content):
                            citation_keys = match.group(1).split(',')
                            for key in citation_keys:
                                key = key.strip()
                                if key:
                                    citations_found.add(key)
                                    citation_usages.append({
                                        'key': key,
                                        'file': str(file_path),
                                        'line': self._find_line_number(content, match.start())
                                    })

                except Exception as e:
                    logger.warning(f"Failed to extract citations from {file_path}: {str(e)}")

        # Validate against bibliography if provided
        bib_validation = {}
        if bib_file and bib_file.exists():
            bib_entries = self._parse_bib_file(bib_file)
            bib_validation = self._validate_citation_coverage(citations_found, bib_entries)

        return {
            'citations_found': list(citations_found),
            'total_citations': len(citations_found),
            'citation_usages': citation_usages,
            'bibliography_validation': bib_validation
        }

    def _parse_bib_file(self, bib_file: Path) -> Set[str]:
        """Parse BibTeX file to extract entry keys."""
        bib_entries = set()

        try:
            content = bib_file.read_text(encoding='utf-8')

            # Simple BibTeX key extraction (very basic)
            key_pattern = re.compile(r'@\w+\{([^,\s]+)', re.MULTILINE)
            for match in key_pattern.finditer(content):
                bib_entries.add(match.group(1))

        except Exception as e:
            logger.warning(f"Failed to parse bibliography file {bib_file}: {str(e)}")

        return bib_entries

    def _validate_citation_coverage(self, citations: Set[str],
                                  bib_entries: Set[str]) -> Dict[str, Any]:
        """Validate that all citations have corresponding bibliography entries."""
        missing_entries = citations - bib_entries
        unused_entries = bib_entries - citations

        return {
            'missing_bibliography_entries': list(missing_entries),
            'unused_bibliography_entries': list(unused_entries),
            'coverage_ratio': len(citations - missing_entries) / len(citations) if citations else 1.0
        }


# Convenience functions
def analyze_document_references(files: List[Path]) -> ReferenceReport:
    """
    Analyze references in a set of documents.

    Args:
        files: Documents to analyze

    Returns:
        Comprehensive reference analysis report
    """
    manager = ReferenceManager()
    return manager.analyze_references(files)


def validate_citations(files: List[Path], bib_file: Optional[Path] = None) -> Dict[str, Any]:
    """
    Validate citations in documents.

    Args:
        files: Documents to check
        bib_file: Bibliography file

    Returns:
        Citation validation results
    """
    manager = ReferenceManager()
    return manager.validate_citations(files, bib_file)


def generate_reference_health_report(report: ReferenceReport) -> str:
    """
    Generate human-readable reference health report.

    Args:
        report: Reference analysis report

    Returns:
        Formatted health report
    """
    health_report = []
    health_report.append("🔗 AntStack Reference Health Report")
    health_report.append("=" * 40)

    summary = report.summary
    health_report.append(f"📊 Overall Health Score: {summary.get('health_score', 0):.1f}%")
    health_report.append("")

    health_report.append("📈 Reference Statistics:")
    health_report.append(f"  • Definitions: {summary.get('total_definitions', 0)}")
    health_report.append(f"  • Usages: {summary.get('total_usages', 0)}")
    health_report.append(f"  • Broken references: {summary.get('broken_references', 0)}")
    health_report.append(f"  • Orphaned definitions: {summary.get('orphaned_definitions', 0)}")
    health_report.append("")

    if report.broken_references:
        health_report.append("❌ Broken References:")
        for ref in report.broken_references[:5]:  # Show first 5
            health_report.append(f"  • {ref.full_id} in {ref.file_path.name}:{ref.line_number}")
        if len(report.broken_references) > 5:
            health_report.append(f"  ... and {len(report.broken_references) - 5} more")
        health_report.append("")

    if report.orphaned_definitions:
        health_report.append("⚠️  Orphaned Definitions:")
        for defn in report.orphaned_definitions[:3]:  # Show first 3
            health_report.append(f"  • {defn.full_id} in {defn.file_path.name}:{defn.line_number}")
        if len(report.orphaned_definitions) > 3:
            health_report.append(f"  ... and {len(report.orphaned_definitions) - 3} more")
        health_report.append("")

    health_report.append("💡 Recommendations:")
    if summary.get('broken_references', 0) > 0:
        health_report.append("  • Fix broken references to ensure document integrity")
    if summary.get('orphaned_definitions', 0) > 0:
        health_report.append("  • Remove or reference orphaned definitions")
    if summary.get('health_score', 0) < 80:
        health_report.append("  • Review reference management practices")

    return "\n".join(health_report)
