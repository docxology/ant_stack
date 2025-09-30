#!/usr/bin/env python3
"""Comprehensive diagnostic script to identify cross-reference transformation issues.

This script creates minimal test cases and traces exactly where figure definitions
get transformed into references, breaking Pandoc anchors.
"""

import os
import tempfile
import subprocess
import re
from pathlib import Path

def create_test_markdown():
    """Create minimal test markdown with figure definitions and references."""
    return '''# Test Document

## Introduction

This document tests figure cross-referencing.

## Figure: Test Figure One {#fig:test_one}

![Test image one](test1.png)

**Caption:** This is the first test figure to verify cross-referencing works properly.

## Figure: Test Figure Two {#fig:test_two}

![Test image two](test2.png)

**Caption:** This is the second test figure to verify cross-referencing works properly.

## Results

We can see the results in Figure~\\ref{fig:test_one} and Figure~\\ref{fig:test_two}.

The scaling behavior is shown in Figure~\\ref{fig:test_one}.

## Figure: Complex Scaling Analysis {#fig:scaling_analysis}

![Scaling analysis](scaling.png)

**Caption:** This figure shows the scaling analysis results with detailed metrics.

## Discussion

The analysis from Figure~\\ref{fig:scaling_analysis} demonstrates the key findings.
'''

def test_unicode_replacement():
    """Test the unicode replacement function from render_pdf.sh in isolation."""
    
    # Replicate the smart_unicode_replace function from render_pdf.sh
    def smart_unicode_replace(text, replacements):
        import re
        math_blocks = []
        math_pattern = r'\$[^$]*\$|\\\[[^\]]*\\\]|\\\([^)]*\\\)'
        
        def protect_math(match):
            idx = len(math_blocks)
            math_blocks.append(match.group(0))
            return f'__MATH_BLOCK_{idx}__'
        
        # Protect existing math
        protected_text = re.sub(math_pattern, protect_math, text)
        
        # Replace unicode in non-math text
        for unicode_char, latex_replacement in replacements.items():
            protected_text = protected_text.replace(unicode_char, latex_replacement)
        
        # Restore math blocks
        for idx, math_block in enumerate(math_blocks):
            protected_text = protected_text.replace(f'__MATH_BLOCK_{idx}__', math_block)
        
        return protected_text

    # Test with the actual unicode_to_tex dictionary
    unicode_to_tex = {
        '∼': '$\\sim$', '≈': '$\\approx$', '≤': '$\\le$', '≥': '$\\ge$',
        '≠': '$\\ne$', '±': '$\\pm$', '×': '$\\times$', '÷': '$\\div$',
        '−': '-', '⋅': '$\\cdot$', '∘': '$\\circ$', '→': '$\\to$',
        '↔': '$\\leftrightarrow$', '⇒': '$\\Rightarrow$', '⇔': '$\\Leftrightarrow$',
        '∈': '$\\in$', '∉': '$\\notin$', '∩': '$\\cap$', '∪': '$\\cup$',
        'α': '$\\alpha$', 'β': '$\\beta$', 'γ': '$\\gamma$', 'δ': '$\\delta$',
        'Δ': '$\\Delta$', 'μ': '$\\mu$', 'µ': '$\\mu$', 'ρ': '$\\rho$',
        '𝛥': '$\\Delta$', '𝜇': '$\\mu$', '𝜂': '$\\eta$',
    }
    
    test_text = create_test_markdown()
    print("=== UNICODE REPLACEMENT TEST ===")
    print("Original text (first 500 chars):")
    print(test_text[:500])
    print("\n" + "="*50 + "\n")
    
    result = smart_unicode_replace(test_text, unicode_to_tex)
    print("After unicode replacement (first 500 chars):")
    print(result[:500])
    print("\n" + "="*50 + "\n")
    
    # Check if figure definitions are preserved
    fig_pattern = r'## Figure: ([^{]+)\{#fig:([^}]+)\}'
    original_figs = re.findall(fig_pattern, test_text)
    result_figs = re.findall(fig_pattern, result)
    
    print(f"Original figure definitions found: {len(original_figs)}")
    for title, fig_id in original_figs:
        print(f"  - {title.strip()}: #{fig_id}")
    
    print(f"Result figure definitions found: {len(result_figs)}")
    for title, fig_id in result_figs:
        print(f"  - {title.strip()}: #{fig_id}")
    
    if len(original_figs) != len(result_figs):
        print("❌ FIGURE DEFINITIONS WERE TRANSFORMED!")
        return False
    else:
        print("✅ Figure definitions preserved")
        return True

def test_texttt_replacement():
    """Test the texttt replacement in isolation."""
    test_text_with_texttt = '''# Test

Some text with \\texttt{filename.py} and \\texttt{another_file.txt}.

## Figure: Test {#fig:test}

More \\texttt{code.cpp} here.
'''
    
    print("=== TEXTTT REPLACEMENT TEST ===")
    print("Original:")
    print(test_text_with_texttt)
    
    # Apply the texttt replacement
    result = re.sub(r'\\texttt\{([^}]*)\}', r'`\1`', test_text_with_texttt)
    
    print("\nAfter texttt replacement:")
    print(result)
    
    # Check figure definitions
    fig_pattern = r'## Figure: ([^{]+)\{#fig:([^}]+)\}'
    original_figs = re.findall(fig_pattern, test_text_with_texttt)
    result_figs = re.findall(fig_pattern, result)
    
    print(f"\nFigure definitions: {len(original_figs)} -> {len(result_figs)}")
    
    if len(original_figs) != len(result_figs):
        print("❌ TEXTTT REPLACEMENT BROKE FIGURES!")
        return False
    else:
        print("✅ Texttt replacement preserved figures")
        return True

def test_mermaid_replacement():
    """Test if mermaid processing affects figure definitions."""
    test_text_with_mermaid = '''# Test

## Figure: Test {#fig:test}

Some content.

```mermaid
graph TD
    A[Start] --> B[Process]
    B --> C[End]
```

More content.

## Figure: Another {#fig:another}

Content here.
'''
    
    print("=== MERMAID REPLACEMENT SIMULATION ===")
    print("Original:")
    print(test_text_with_mermaid)
    
    # Simulate mermaid replacement (simplified)
    mermaid_pattern = re.compile(r"```mermaid[^\n]*\n(.*?)\n```", re.DOTALL)
    
    def repl_mermaid(match):
        return "![Mermaid Diagram](assets/mermaid/diagram_001.png){ width=70% }"
    
    result = mermaid_pattern.sub(repl_mermaid, test_text_with_mermaid)
    
    print("\nAfter mermaid replacement:")
    print(result)
    
    # Check figure definitions
    fig_pattern = r'## Figure: ([^{]+)\{#fig:([^}]+)\}'
    original_figs = re.findall(fig_pattern, test_text_with_mermaid)
    result_figs = re.findall(fig_pattern, result)
    
    print(f"\nFigure definitions: {len(original_figs)} -> {len(result_figs)}")
    
    if len(original_figs) != len(result_figs):
        print("❌ MERMAID REPLACEMENT BROKE FIGURES!")
        return False
    else:
        print("✅ Mermaid replacement preserved figures")
        return True

def full_pipeline_test():
    """Test the full processing pipeline to identify where the issue occurs."""
    print("=== FULL PIPELINE TEST ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.md")
        
        with open(test_file, 'w') as f:
            f.write(create_test_markdown())
        
        print(f"Created test file: {test_file}")
        
        # Try to extract the python script from render_pdf.sh and run it
        # This would require parsing the bash script, which is complex
        # Instead, let's create a minimal version
        
        print("Running minimal processing simulation...")
        
        # Read the file
        with open(test_file, 'r') as f:
            content = f.read()
        
        print("Step 1: Original content")
        fig_pattern = r'## Figure: ([^{]+)\{#fig:([^}]+)\}'
        figs = re.findall(fig_pattern, content)
        print(f"Figure definitions: {len(figs)}")
        
        # Step 2: Unicode replacement
        unicode_ok = test_unicode_replacement()
        
        # Step 3: Texttt replacement  
        texttt_ok = test_texttt_replacement()
        
        # Step 4: Mermaid replacement
        mermaid_ok = test_mermaid_replacement()
        
        return unicode_ok and texttt_ok and mermaid_ok

def main():
    """Run all diagnostic tests."""
    print("🔍 Cross-Reference Issue Diagnostic Tool")
    print("=" * 50)
    
    # Test individual components
    unicode_ok = test_unicode_replacement()
    print()
    
    texttt_ok = test_texttt_replacement()
    print()
    
    mermaid_ok = test_mermaid_replacement()
    print()
    
    # Full pipeline test
    pipeline_ok = full_pipeline_test()
    
    print("\n" + "=" * 50)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    print(f"Unicode replacement: {'✅' if unicode_ok else '❌'}")
    print(f"Texttt replacement: {'✅' if texttt_ok else '❌'}")
    print(f"Mermaid replacement: {'✅' if mermaid_ok else '❌'}")
    print(f"Full pipeline: {'✅' if pipeline_ok else '❌'}")
    
    if not (unicode_ok and texttt_ok and mermaid_ok):
        print("\n❌ Issues identified in processing pipeline!")
        print("The cross-reference transformation is happening during text processing.")
        return 1
    else:
        print("\n✅ All tests passed - issue may be elsewhere in pipeline")
        print("Further investigation needed in the bash script or Pandoc processing.")
        return 0

if __name__ == "__main__":
    exit(main())
