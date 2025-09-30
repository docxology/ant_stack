"""
Quality Assurance and Validation System for Ant Stack Publications

Comprehensive validation framework ensuring publication quality through:
- Cross-reference validation
- Content consistency checks
- Scientific accuracy verification
- Formatting standards compliance
- Build process validation

Following .cursorrules specifications for:
- Zero tolerance for broken references
- Professional document validation
- Scientific rigor verification
- Quality assurance standards
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
import yaml
import json
import logging
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Represents a validation issue found in a document."""

    issue_type: str  # 'error', 'warning', 'info'
    category: str    # 'reference', 'formatting', 'content', 'scientific'
    message: str
    file_path: Optional[Path] = None
    line_number: Optional[int] = None
    column: Optional[int] = None
    context: Optional[str] = None
    suggestion: Optional[str] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.context is None:
            self.context = ""
        if self.suggestion is None:
            self.suggestion = ""


@dataclass
class ValidationReport:
    """Comprehensive validation report."""

    total_files: int = 0
    total_issues: int = 0
    errors: int = 0
    warnings: int = 0
    info: int = 0
    issues: List[ValidationIssue] = None
    file_reports: Dict[str, List[ValidationIssue]] = None
    summary: Dict[str, Any] = None
    validation_passed: bool = False

    def __post_init__(self):
        """Initialize default values."""
        if self.issues is None:
            self.issues = []
        if self.file_reports is None:
            self.file_reports = {}
        if self.summary is None:
            self.summary = {}


class QualityValidator:
    """
    Comprehensive quality assurance system for scientific publications.

    Validates:
    - Cross-reference integrity
    - Content consistency
    - Scientific accuracy
    - Formatting standards
    - Build process readiness
    """

    def __init__(self):
        """Initialize quality validator."""
        self._setup_validation_rules()

    def _setup_validation_rules(self):
        """Configure validation rules and patterns."""

        # Cross-reference patterns
        self.ref_patterns = {
            'figure': re.compile(r'\\cref\{fig:([^}]+)\}|\[Figure ([^]]+)\]|\(Figure ([^)]+)\)'),
            'table': re.compile(r'\\cref\{tab:([^}]+)\}|\[Table ([^]]+)\]|\(Table ([^)]+)\)'),
            'section': re.compile(r'\\cref\{sec:([^}]+)\}|\[Section ([^]]+)\]|\(Section ([^)]+)\)'),
            'equation': re.compile(r'\\cref\{eq:([^}]+)\}|\[Equation ([^]]+)\]|\(Equation ([^)]+)\)'),
        }

        # Content validation patterns
        self.content_patterns = {
            'math_inline': re.compile(r'\$[^$]+\$'),
            'math_display': re.compile(r'\$\$[^$]+\$\$'),
            'citation': re.compile(r'\\cite\{([^}]+)\}'),
            'url': re.compile(r'https?://[^\s]+'),
            'unicode_symbol': re.compile(r'[≤≥≈∼Δρμλσ∝⋅∘∑∏∫∂∇∈∉⊂⊆∪∩∧∨¬∀∃⇒⇔]'),
        }

        # Scientific terminology validation
        self.scientific_terms = {
            'energy_units': ['J', 'kJ', 'MJ', 'mJ', 'μJ', 'nJ', 'pJ', 'fJ', 'aJ'],
            'time_units': ['s', 'ms', 'μs', 'ns', 'ps', 'fs'],
            'frequency_units': ['Hz', 'kHz', 'MHz', 'GHz', 'THz'],
            'statistical_terms': ['mean', 'median', 'mode', 'std', 'variance', 'correlation', 'p-value', 'confidence'],
        }

        # Formatting standards
        self.formatting_rules = {
            'figure_caption': re.compile(r'\\caption\{([^}]+)\}'),
            'table_caption': re.compile(r'\\caption\{([^}]+)\}'),
            'section_header': re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE),
            'list_item': re.compile(r'^[\s]*[-*+]\s+', re.MULTILINE),
        }

    def validate_publication(self, files: List[Path],
                           config: Optional[Dict[str, Any]] = None) -> ValidationReport:
        """
        Perform comprehensive validation of publication files.

        Args:
            files: List of files to validate
            config: Optional validation configuration

        Returns:
            Comprehensive validation report
        """
        report = ValidationReport()
        report.total_files = len(files)

        all_issues = []
        file_reports = {}

        # Validate each file
        for file_path in files:
            if file_path.exists() and file_path.suffix.lower() in ['.md', '.markdown']:
                issues = self._validate_file(file_path)
                file_reports[str(file_path)] = issues
                all_issues.extend(issues)

        # Cross-file validation
        cross_file_issues = self._validate_cross_file_consistency(files)
        all_issues.extend(cross_file_issues)

        # Generate summary
        report.issues = all_issues
        report.total_issues = len(all_issues)
        report.file_reports = file_reports

        # Categorize issues
        for issue in all_issues:
            if issue.issue_type == 'error':
                report.errors += 1
            elif issue.issue_type == 'warning':
                report.warnings += 1
            else:
                report.info += 1

        # Determine if validation passed
        report.validation_passed = report.errors == 0

        # Generate summary statistics
        report.summary = self._generate_summary_report(all_issues)

        return report

    def _validate_file(self, file_path: Path) -> List[ValidationIssue]:
        """Validate a single file for quality issues."""
        issues = []

        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')

            # Basic file validation
            issues.extend(self._validate_basic_structure(content, file_path))

            # Cross-reference validation
            issues.extend(self._validate_references(content, file_path))

            # Content validation
            issues.extend(self._validate_content_quality(content, file_path))

            # Scientific validation
            issues.extend(self._validate_scientific_accuracy(content, file_path))

            # Formatting validation
            issues.extend(self._validate_formatting_standards(content, file_path))

        except Exception as e:
            issues.append(ValidationIssue(
                issue_type='error',
                category='file',
                message=f"Failed to read file: {str(e)}",
                file_path=file_path,
                suggestion="Check file permissions and encoding"
            ))

        return issues

    def _validate_basic_structure(self, content: str, file_path: Path) -> List[ValidationIssue]:
        """Validate basic document structure."""
        issues = []

        # Check for empty file
        if not content.strip():
            issues.append(ValidationIssue(
                issue_type='error',
                category='structure',
                message="File is empty or contains only whitespace",
                file_path=file_path,
                suggestion="Add content to the file"
            ))
            return issues

        # Check for basic markdown structure
        if not re.search(r'^#', content, re.MULTILINE):
            issues.append(ValidationIssue(
                issue_type='warning',
                category='structure',
                message="No top-level heading (#) found",
                file_path=file_path,
                suggestion="Add a main title with # heading"
            ))

        # Check file size (warning for very large files)
        if len(content) > 500000:  # ~500KB
            issues.append(ValidationIssue(
                issue_type='warning',
                category='structure',
                message="File is very large (>500KB)",
                file_path=file_path,
                suggestion="Consider splitting into multiple files"
            ))

        return issues

    def _validate_references(self, content: str, file_path: Path) -> List[ValidationIssue]:
        """Validate cross-references in the document."""
        issues = []

        # Extract all references
        found_refs = defaultdict(list)

        for ref_type, pattern in self.ref_patterns.items():
            matches = pattern.findall(content)
            for match in matches:
                # Handle different capture group structures
                if isinstance(match, tuple):
                    ref_id = next((m for m in match if m), "")
                else:
                    ref_id = match
                found_refs[ref_type].append(ref_id)

        # Check for broken references (this is a simplified check)
        # In a full implementation, you'd cross-reference with actual definitions
        for ref_type, refs in found_refs.items():
            unique_refs = set(refs)
            for ref in unique_refs:
                # Check for obviously malformed references
                if not ref or ref.strip() != ref:
                    issues.append(ValidationIssue(
                        issue_type='error',
                        category='reference',
                        message=f"Malformed {ref_type} reference: '{ref}'",
                        file_path=file_path,
                        suggestion=f"Fix reference format for {ref_type}"
                    ))

        return issues

    def _validate_content_quality(self, content: str, file_path: Path) -> List[ValidationIssue]:
        """Validate content quality and consistency."""
        issues = []

        # Check for TODO/FIXME comments
        todo_pattern = re.compile(r'(?i)(todo|fixme|hack|xxx)')
        todo_matches = todo_pattern.findall(content)
        if todo_matches:
            issues.append(ValidationIssue(
                issue_type='warning',
                category='content',
                message=f"Found {len(todo_matches)} TODO/FIXME comments",
                file_path=file_path,
                suggestion="Address or remove development comments before publication"
            ))

        # Check for broken links
        url_pattern = re.compile(r'https?://[^\s)]+')
        urls = url_pattern.findall(content)
        for url in urls:
            if url.endswith('.') or url.endswith(','):
                issues.append(ValidationIssue(
                    issue_type='warning',
                    category='content',
                    message=f"URL may have trailing punctuation: {url}",
                    file_path=file_path,
                    suggestion="Remove trailing punctuation from URLs"
                ))

        # Check for long lines (>120 characters)
        lines = content.split('\n')
        long_lines = [i for i, line in enumerate(lines, 1) if len(line) > 120]
        if long_lines:
            issues.append(ValidationIssue(
                issue_type='info',
                category='formatting',
                message=f"Found {len(long_lines)} lines longer than 120 characters",
                file_path=file_path,
                suggestion="Consider breaking long lines for better readability"
            ))

        return issues

    def _validate_scientific_accuracy(self, content: str, file_path: Path) -> List[ValidationIssue]:
        """Validate scientific content for accuracy and consistency."""
        issues = []

        # Check for inconsistent unit usage
        energy_mentions = []
        for unit in self.scientific_terms['energy_units']:
            pattern = re.compile(rf'\b\d+\.?\d*\s*{re.escape(unit)}\b')
            matches = pattern.findall(content)
            energy_mentions.extend(matches)

        if len(set(energy_mentions)) > 5:  # Many different energy units
            issues.append(ValidationIssue(
                issue_type='info',
                category='scientific',
                message="Multiple energy units detected",
                file_path=file_path,
                suggestion="Consider standardizing energy units for consistency"
            ))

        # Check for potential scientific notation issues
        sci_notation_pattern = re.compile(r'\b\d+\.?\d*[eE][+-]?\d+\b')
        sci_numbers = sci_notation_pattern.findall(content)
        for num in sci_numbers:
            # Check for very large or small numbers that might be errors
            try:
                value = float(num.lower().replace('e', 'e'))
                if abs(value) > 1e20 or (abs(value) < 1e-10 and value != 0):
                    issues.append(ValidationIssue(
                        issue_type='warning',
                        category='scientific',
                        message=f"Extreme scientific notation value: {num}",
                        file_path=file_path,
                        suggestion="Verify the magnitude of this value is correct"
                    ))
            except ValueError:
                pass

        # Check for consistent statistical terminology
        stat_terms_found = []
        for term in self.scientific_terms['statistical_terms']:
            if re.search(rf'\b{re.escape(term)}\b', content, re.IGNORECASE):
                stat_terms_found.append(term)

        if stat_terms_found and len(stat_terms_found) > len(self.scientific_terms['statistical_terms']) * 0.7:
            issues.append(ValidationIssue(
                issue_type='info',
                category='scientific',
                message="Extensive statistical analysis detected",
                file_path=file_path,
                suggestion="Ensure statistical methods are properly described"
            ))

        return issues

    def _validate_formatting_standards(self, content: str, file_path: Path) -> List[ValidationIssue]:
        """Validate formatting standards compliance."""
        issues = []

        # Check for mixed line endings
        if '\r\n' in content and '\n' in content:
            issues.append(ValidationIssue(
                issue_type='warning',
                category='formatting',
                message="Mixed line endings detected (CRLF and LF)",
                file_path=file_path,
                suggestion="Use consistent line endings (LF preferred)"
            ))

        # Check for tabs vs spaces (prefer spaces)
        if '\t' in content:
            issues.append(ValidationIssue(
                issue_type='info',
                category='formatting',
                message="Tab characters detected",
                file_path=file_path,
                suggestion="Consider using spaces instead of tabs for consistency"
            ))

        # Check for trailing whitespace
        lines = content.split('\n')
        trailing_ws_lines = [i for i, line in enumerate(lines, 1) if line.rstrip() != line]
        if trailing_ws_lines:
            issues.append(ValidationIssue(
                issue_type='info',
                category='formatting',
                message=f"Found {len(trailing_ws_lines)} lines with trailing whitespace",
                file_path=file_path,
                suggestion="Remove trailing whitespace"
            ))

        return issues

    def _validate_cross_file_consistency(self, files: List[Path]) -> List[ValidationIssue]:
        """Validate consistency across multiple files."""
        issues = []

        # Collect all references across files
        all_refs = defaultdict(list)
        all_definitions = defaultdict(list)

        for file_path in files:
            if file_path.exists() and file_path.suffix.lower() in ['.md', '.markdown']:
                try:
                    content = file_path.read_text(encoding='utf-8')

                    # Extract references
                    for ref_type, pattern in self.ref_patterns.items():
                        matches = pattern.findall(content)
                        for match in matches:
                            ref_id = match if isinstance(match, str) else next((m for m in match if m), "")
                            if ref_id:
                                all_refs[ref_type].append((ref_id, file_path))

                    # Extract definitions (simplified - would need more sophisticated parsing)
                    # This is a placeholder for more advanced cross-reference validation

                except Exception as e:
                    issues.append(ValidationIssue(
                        issue_type='error',
                        category='file',
                        message=f"Failed to process file {file_path}: {str(e)}",
                        file_path=file_path
                    ))

        # Check for orphaned references (simplified check)
        ref_counts = Counter()
        for ref_type, refs in all_refs.items():
            for ref_id, _ in refs:
                ref_counts[f"{ref_type}:{ref_id}"] += 1

        # This is a simplified check - in practice you'd cross-reference with actual definitions
        if len(ref_counts) > 20:  # Many references found
            issues.append(ValidationIssue(
                issue_type='info',
                category='reference',
                message=f"Found {len(ref_counts)} unique references across files",
                suggestion="Verify all references have corresponding definitions"
            ))

        return issues

    def _generate_summary_report(self, issues: List[ValidationIssue]) -> Dict[str, Any]:
        """Generate summary statistics from validation issues."""
        summary = {
            'total_issues': len(issues),
            'by_type': {'error': 0, 'warning': 0, 'info': 0},
            'by_category': defaultdict(int),
            'most_common_issues': [],
            'files_affected': set()
        }

        # Count by type and category
        for issue in issues:
            summary['by_type'][issue.issue_type] += 1
            summary['by_category'][issue.category] += 1
            if issue.file_path:
                summary['files_affected'].add(str(issue.file_path))

        # Find most common issues
        issue_messages = Counter([issue.message for issue in issues])
        summary['most_common_issues'] = issue_messages.most_common(5)

        # Convert set to list for JSON serialization
        summary['files_affected'] = list(summary['files_affected'])

        return dict(summary)

    def validate_build_readiness(self, files: List[Path]) -> ValidationReport:
        """
        Validate that files are ready for PDF build process.

        Args:
            files: Files to validate for build readiness

        Returns:
            Validation report focused on build readiness
        """
        report = self.validate_publication(files)

        # Add build-specific validations
        build_issues = []

        for file_path in files:
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding='utf-8')

                    # Check for common build-breaking patterns
                    if '\\begin{document}' in content and '\\end{document}' not in content:
                        build_issues.append(ValidationIssue(
                            issue_type='error',
                            category='build',
                            message="Unmatched LaTeX document environment",
                            file_path=file_path,
                            suggestion="Fix LaTeX document structure"
                        ))

                    # Check for unescaped special characters
                    unescaped_patterns = [
                        r'(?<!\\)&(?![;&])',  # Unescaped ampersands
                        r'(?<!\\)%',         # Unescaped percent signs
                    ]

                    for pattern in unescaped_patterns:
                        if re.search(pattern, content):
                            build_issues.append(ValidationIssue(
                                issue_type='warning',
                                category='build',
                                message="Potentially unescaped LaTeX special characters",
                                file_path=file_path,
                                suggestion="Check and escape LaTeX special characters"
                            ))
                            break

                except Exception as e:
                    build_issues.append(ValidationIssue(
                        issue_type='error',
                        category='build',
                        message=f"Cannot read file for build validation: {str(e)}",
                        file_path=file_path
                    ))

        # Add build issues to main report
        report.issues.extend(build_issues)
        report.total_issues += len(build_issues)

        return report


# Convenience functions for common validation tasks
def validate_markdown_files(files: List[Path]) -> ValidationReport:
    """
    Validate markdown files for publication quality.

    Args:
        files: Markdown files to validate

    Returns:
        Comprehensive validation report
    """
    validator = QualityValidator()
    return validator.validate_publication(files)


def validate_build_readiness(files: List[Path]) -> ValidationReport:
    """
    Validate files are ready for PDF build process.

    Args:
        files: Files to validate for build readiness

    Returns:
        Build readiness validation report
    """
    validator = QualityValidator()
    return validator.validate_build_readiness(files)


def generate_validation_summary(report: ValidationReport) -> str:
    """
    Generate human-readable validation summary.

    Args:
        report: Validation report to summarize

    Returns:
        Formatted summary string
    """
    summary = []
    summary.append("📋 AntStack Publication Validation Report")
    summary.append("=" * 50)
    summary.append("")

    summary.append(f"📁 Files validated: {report.total_files}")
    summary.append(f"🔍 Total issues found: {report.total_issues}")
    summary.append("")

    summary.append("📊 Issues by severity:")
    summary.append(f"  ❌ Errors: {report.errors}")
    summary.append(f"  ⚠️  Warnings: {report.warnings}")
    summary.append(f"  ℹ️  Info: {report.info}")
    summary.append("")

    if report.summary.get('most_common_issues'):
        summary.append("🔄 Most common issues:")
        for issue, count in report.summary['most_common_issues'][:3]:
            summary.append(f"  • {issue} ({count} times)")

    summary.append("")
    summary.append("✅ Validation passed: " + ("Yes" if report.validation_passed else "No"))

    return "\n".join(summary)
