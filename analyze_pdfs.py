#!/usr/bin/env python3
"""PDF Analysis Script for Ant Stack Papers

This script analyzes the generated PDFs to identify issues with:
- Figure captions and references
- LaTeX math rendering (missing $ symbols)
- Bullet point formatting
- Citation hyperlinks
- Cross-references
- Missing or broken links
"""

import sys
import os
from pathlib import Path
import re
from typing import Dict, List, Set, Tuple
import PyPDF2

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text content from PDF using PyPDF2."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def find_latex_math_issues(text: str) -> List[str]:
    """Find LaTeX math rendering issues (missing $ symbols, etc.)."""
    issues = []

    # Find standalone math symbols that should be in math mode
    math_symbols = [
        r'\mathcal{O}', r'\propto', r'\approx', r'\times', r'\in', r'\subset',
        r'\subseteq', r'\forall', r'\exists', r'\leq', r'\geq', r'\neq',
        r'\sum', r'\prod', r'\int', r'\partial', r'\nabla', r'\alpha', r'\beta',
        r'\gamma', r'\delta', r'\epsilon', r'\lambda', r'\mu', r'\sigma',
        r'\theta', r'\pi', r'\omega', r'\Delta', r'\Omega', r'\Sigma'
    ]

    for symbol in math_symbols:
        # Look for symbol not preceded by $ or \
        pattern = rf'(?<!\$)(?<!\\){re.escape(symbol)}(?!\$)'
        matches = list(re.finditer(pattern, text))
        if matches:
            issues.append(f"LaTeX symbol '{symbol}' not in math mode at positions: {[m.start() for m in matches]}")

    # Find $ symbols that might indicate failed LaTeX rendering
    dollar_matches = list(re.finditer(r'\$', text))
    if dollar_matches:
        issues.append(f"Found {len(dollar_matches)} $ symbols that might indicate LaTeX rendering failures")

    # Check for common LaTeX formatting issues
    if r'\text' in text and r'\text{' in text:
        issues.append("Found \\text commands that might need proper LaTeX formatting")

    return issues

def find_caption_issues(text: str) -> List[str]:
    """Find issues with figure and table captions."""
    issues = []

    # Find figure references
    fig_refs = re.findall(r'Figure\s+(\d+)', text)
    if fig_refs:
        issues.append(f"Found {len(fig_refs)} figure references")

    # Check for incomplete captions (too short or missing)
    captions = re.findall(r'Figure\s+\d+:?\s*([^\n]+)', text, re.IGNORECASE)
    # Also look for **Caption:** format
    caption_patterns = re.findall(r'\*\*Caption:\*\*\s*([^\n]+)', text)
    captions.extend(caption_patterns)

    short_captions = [cap for cap in captions if len(cap.strip()) < 10]
    if short_captions:
        issues.append(f"Found {len(short_captions)} potentially incomplete captions")

    # Check for missing figure numbers
    if re.search(r'Figure\s+:', text):
        issues.append("Found 'Figure:' without number")

    return issues

def find_citation_issues(text: str) -> List[str]:
    """Find issues with citations and references."""
    issues = []

    # Look for citation patterns
    citations = re.findall(r'\([A-Za-z\s,]+\s+\d{4}[a-z]?\)', text)
    if citations:
        issues.append(f"Found {len(citations)} citation patterns")

    # Check for missing years in citations
    incomplete_citations = re.findall(r'\([A-Za-z\s,]+\s*\)', text)
    incomplete_citations = [c for c in incomplete_citations if len(c) > 10]  # Filter short ones
    if incomplete_citations:
        issues.append(f"Found {len(incomplete_citations)} potentially incomplete citations")

    # Check for web links that might not be hyperlinked
    urls = re.findall(r'https?://[^\s\)]+', text)
    if urls:
        issues.append(f"Found {len(urls)} URLs in text")

    return issues

def find_bullet_point_issues(text: str) -> List[str]:
    """Find issues with bullet point formatting."""
    issues = []

    # Check for inconsistent bullet points
    bullet_patterns = [
        r'^\s*[-*+]\s',  # Dash, asterisk, plus
        r'^\s*\d+\.\s',  # Numbered lists
        r'^\s*\([a-z]\)\s',  # Letter lists
        r'^\s*\([ivx]+\)\s',  # Roman numeral lists
    ]

    for pattern in bullet_patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        if matches:
            issues.append(f"Found {len(matches)} bullet points with pattern '{pattern}'")

    # Check for mixed bullet styles in same section
    sections_with_mixed_bullets = []
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if re.match(r'^\s*[-*+]\s', line):
            # Check nearby lines for different bullet styles
            for j in range(max(0, i-3), min(len(lines), i+4)):
                if j != i and re.match(r'^\s*\d+\.\s', lines[j]):
                    sections_with_mixed_bullets.append(f"Line {i+1} (dash) near line {j+1} (number)")
                    break

    if sections_with_mixed_bullets:
        issues.append(f"Found mixed bullet styles: {sections_with_mixed_bullets[:3]}...")

    return issues

def find_hyperlink_issues(text: str) -> List[str]:
    """Find issues with hyperlinks and cross-references."""
    issues = []

    # Check for broken cross-references
    ref_patterns = [
        r'Figure~\d+', r'Table~\d+', r'Section~\d+', r'Equation~\(\d+\)',
        r'Appendix~\w+', r'Chapter~\d+'
    ]

    for pattern in ref_patterns:
        matches = re.findall(pattern, text)
        if matches:
            issues.append(f"Found {len(matches)} cross-references with pattern '{pattern}'")

    # Check for potential hyperlink text that's not actually linked
    potential_links = re.findall(r'[A-Za-z\s]+(?:et al\.|et al|and|&)\s+\d{4}', text)
    if potential_links:
        issues.append(f"Found {len(potential_links)} potential citations that might not be hyperlinked")

    return issues

def analyze_pdf(pdf_path: Path) -> Dict[str, List[str]]:
    """Analyze a single PDF for various issues."""
    print(f"Analyzing {pdf_path.name}...")

    text = extract_text_from_pdf(pdf_path)
    if not text:
        return {"Error": [f"Could not extract text from {pdf_path}"]}

    issues = {
        "LaTeX Math Issues": find_latex_math_issues(text),
        "Caption Issues": find_caption_issues(text),
        "Citation Issues": find_citation_issues(text),
        "Bullet Point Issues": find_bullet_point_issues(text),
        "Hyperlink Issues": find_hyperlink_issues(text),
        "General Issues": []
    }

    # Check for common formatting problems
    caption_found = False
    # Look for various caption patterns
    caption_patterns = [
        r'\*\*Caption:\*\*',
        r'Caption:',
        r'Figure\s+\d+.*Caption',
        r'caption.*Figure'
    ]

    for pattern in caption_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            caption_found = True
            break

    if "Figure" in text and not caption_found:
        issues["General Issues"].append("Found 'Figure' references but no 'Caption' text")

    if "$" in text and len([x for x in text if x == "$"]) > 10:
        issues["General Issues"].append(f"Found {text.count('$')} $ symbols - potential LaTeX rendering issues")

    # Check for very long lines that might be formatting issues
    lines = text.split('\n')
    long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 200]
    if long_lines:
        issues["General Issues"].append(f"Found {len(long_lines)} very long lines (potential formatting issues)")

    return issues

def main():
    """Main analysis function."""
    # Find all PDF files
    pdf_dir = Path("/Users/4d/Documents/GitHub/ant")
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found!")
        return

    print(f"Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")

    print("\n" + "="*80)
    print("PDF ANALYSIS REPORT")
    print("="*80)

    all_issues = {}

    for pdf_file in pdf_files:
        print(f"\n📄 Analyzing {pdf_file.name}")
        print("-" * 50)

        issues = analyze_pdf(pdf_file)

        if issues and any(issue_list for issue_list in issues.values()):
            all_issues[pdf_file.name] = issues

            for category, issue_list in issues.items():
                if issue_list:
                    print(f"  {category}:")
                    for issue in issue_list[:3]:  # Show first 3 issues
                        print(f"    - {issue}")
                    if len(issue_list) > 3:
                        print(f"    ... and {len(issue_list) - 3} more")

        else:
            print("  ✅ No issues found")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if not all_issues:
        print("✅ No issues found in any PDF!")
    else:
        print(f"⚠️  Issues found in {len(all_issues)} PDFs:")
        for pdf_name, issues in all_issues.items():
            print(f"  - {pdf_name}: {sum(len(issue_list) for issue_list in issues.values())} total issues")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("1. Check LaTeX math rendering for missing $ symbols")
    print("2. Verify all figure captions are complete and properly formatted")
    print("3. Ensure citations are properly hyperlinked")
    print("4. Check for consistent bullet point formatting")
    print("5. Validate cross-references between sections, figures, and tables")
    print("6. Review long lines that might indicate formatting issues")

if __name__ == "__main__":
    main()
