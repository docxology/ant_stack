#!/usr/bin/env python3
"""Trace the actual build pipeline step by step to find where cross-references break."""

import os
import tempfile
import subprocess
import shutil
from pathlib import Path
import re

def create_minimal_paper():
    """Create a minimal paper structure for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        paper_dir = os.path.join(temp_dir, "test_paper")
        os.makedirs(paper_dir)
        
        # Create minimal markdown files
        files = {
            "Abstract.md": "# Abstract\n\nThis is a test abstract.\n",
            "README.md": """# Test Paper

## Introduction

Testing cross-references.

## Figure: Overview {#fig:overview}

![Overview diagram](overview.png)

**Caption:** This is the overview diagram.
""",
            "Results.md": """# Results

The results are shown in Figure~\\ref{fig:overview} and Figure~\\ref{fig:analysis}.

## Figure: Analysis Results {#fig:analysis}

![Analysis results](analysis.png)

**Caption:** This shows the analysis results in detail.

## Discussion

As we can see from Figure~\\ref{fig:overview}, the approach works well.
"""
        }
        
        for filename, content in files.items():
            with open(os.path.join(paper_dir, filename), 'w') as f:
                f.write(content)
        
        return paper_dir

def test_concatenation_step():
    """Test the concatenation step specifically."""
    print("=== TESTING CONCATENATION STEP ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        paper_dir = os.path.join(temp_dir, "test_paper")
        os.makedirs(paper_dir)
        
        # Create test files
        files = {
            "file1.md": """# File 1

## Figure: First Figure {#fig:first}

![First](first.png)

**Caption:** First figure caption.
""",
            "file2.md": """# File 2

See Figure~\\ref{fig:first} and Figure~\\ref{fig:second}.

## Figure: Second Figure {#fig:second}

![Second](second.png)

**Caption:** Second figure caption.
"""
        }
        
        for filename, content in files.items():
            filepath = os.path.join(paper_dir, filename)
            with open(filepath, 'w') as f:
                f.write(content)
        
        # Simulate the concatenation process
        temp_file = os.path.join(temp_dir, "concatenated.md")
        with open(temp_file, 'w') as outf:
            for i, filename in enumerate(files.keys()):
                filepath = os.path.join(paper_dir, filename)
                with open(filepath, 'r') as inf:
                    content = inf.read()
                    outf.write(content)
                    if i < len(files) - 1:
                        outf.write('\f\n\n')  # Page break like in render_pdf.sh
        
        # Check the concatenated result
        with open(temp_file, 'r') as f:
            concatenated = f.read()
        
        print("Concatenated content:")
        print(concatenated)
        
        # Check figure definitions
        fig_pattern = r'## Figure: ([^{]+)\{#fig:([^}]+)\}'
        figs = re.findall(fig_pattern, concatenated)
        print(f"\nFigure definitions found: {len(figs)}")
        for title, fig_id in figs:
            print(f"  - {title.strip()}: #{fig_id}")
        
        # Check references
        ref_pattern = r'Figure~\\ref\{fig:([^}]+)\}'
        refs = re.findall(ref_pattern, concatenated)
        print(f"\nFigure references found: {len(refs)}")
        for ref_id in refs:
            print(f"  - \\ref{{fig:{ref_id}}}")
        
        return len(figs) >= 2 and len(refs) >= 2

def test_actual_render_script():
    """Test with the actual render script on a minimal example."""
    print("=== TESTING ACTUAL RENDER SCRIPT ===")
    
    original_dir = os.getcwd()
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy necessary files to temp directory
            temp_ant_dir = os.path.join(temp_dir, "ant")
            os.makedirs(temp_ant_dir)
            
            # Copy tools directory
            tools_src = os.path.join(original_dir, "tools")
            tools_dst = os.path.join(temp_ant_dir, "tools")
            shutil.copytree(tools_src, tools_dst)
            
            # Create minimal test paper
            paper_dir = os.path.join(temp_ant_dir, "test_paper")
            os.makedirs(paper_dir)
            os.makedirs(os.path.join(paper_dir, "assets", "mermaid"), exist_ok=True)
            
            files = {
                "Abstract.md": "# Abstract\n\nTest abstract.\n",
                "README.md": """# Test Paper

## Introduction

Testing cross-references.

## Figure: Overview Diagram {#fig:overview}

![Overview](overview.png)

**Caption:** This is the overview diagram showing the system architecture.
""",
                "Results.md": """# Results

The results are shown in Figure~\\ref{fig:overview} and Figure~\\ref{fig:analysis}.

## Figure: Analysis Results {#fig:analysis}

![Analysis](analysis.png)  

**Caption:** Detailed analysis results with statistical significance testing.

## Discussion

Figure~\\ref{fig:overview} demonstrates the effectiveness of our approach.
""",
                "references.bib": "@article{test2024,\n  title={Test},\n  author={Test Author},\n  year={2024}\n}\n"
            }
            
            for filename, content in files.items():
                with open(os.path.join(paper_dir, filename), 'w') as f:
                    f.write(content)
            
            # Modify render_pdf.sh to handle our test paper
            render_script = os.path.join(temp_ant_dir, "tools", "render_pdf.sh")
            
            # Add our test paper to the script
            with open(render_script, 'r') as f:
                script_content = f.read()
            
            # Insert test_paper case
            test_case = '''        test_paper)
            MARKDOWN_FILES=(
                "test_paper/Abstract.md"
                "test_paper/README.md"
                "test_paper/Results.md"
            )
            ;;'''
            
            script_content = script_content.replace(
                '        complexity_energetics)',
                f'        complexity_energetics)\n{test_case}\n        complexity_energetics)'
            )
            
            with open(render_script, 'w') as f:
                f.write(script_content)
            
            # Change to temp directory and run
            os.chdir(temp_ant_dir)
            
            # Run the render script
            try:
                result = subprocess.run([
                    'bash', 'tools/render_pdf.sh', 'test_paper'
                ], capture_output=True, text=True, timeout=60)
                
                print("Render script output:")
                print(result.stdout)
                if result.stderr:
                    print("Stderr:")
                    print(result.stderr)
                
                # Look for the temporary markdown file that was created
                temp_files = list(Path('/tmp').glob('tmp*.md'))
                for temp_file in temp_files:
                    if temp_file.stat().st_mtime > (os.path.getctime(temp_ant_dir)):
                        print(f"\nFound recent temp file: {temp_file}")
                        try:
                            with open(temp_file, 'r') as f:
                                temp_content = f.read()
                            
                            print("Temp file content (first 1000 chars):")
                            print(temp_content[:1000])
                            
                            # Check figure definitions in temp file
                            fig_pattern = r'## Figure: ([^{]+)\{#fig:([^}]+)\}'
                            figs = re.findall(fig_pattern, temp_content)
                            print(f"\nFigure definitions in temp file: {len(figs)}")
                            
                            # Look for problematic transformations
                            bold_ref_pattern = r'\*\*Figure~\\ref\{fig:([^}]+)\}:'
                            bold_refs = re.findall(bold_ref_pattern, temp_content)
                            if bold_refs:
                                print(f"❌ Found {len(bold_refs)} transformed figure definitions!")
                                for ref_id in bold_refs:
                                    print(f"  - **Figure~\\ref{{fig:{ref_id}}}:**")
                                return False
                            
                            break
                        except Exception as e:
                            print(f"Error reading temp file: {e}")
                
            except subprocess.TimeoutExpired:
                print("Render script timed out")
                return False
            except Exception as e:
                print(f"Error running render script: {e}")
                return False
                
    finally:
        os.chdir(original_dir)
    
    return True

def main():
    """Run pipeline tracing tests."""
    print("🔍 Pipeline Tracing Tool")
    print("=" * 50)
    
    # Test concatenation
    concat_ok = test_concatenation_step()
    print(f"\nConcatenation test: {'✅' if concat_ok else '❌'}")
    
    # Test actual render script
    render_ok = test_actual_render_script()
    print(f"Render script test: {'✅' if render_ok else '❌'}")
    
    print("\n" + "=" * 50)
    print("📊 PIPELINE TRACING SUMMARY")
    print("=" * 50)
    print(f"Concatenation: {'✅' if concat_ok else '❌'}")
    print(f"Render script: {'✅' if render_ok else '❌'}")
    
    return 0 if (concat_ok and render_ok) else 1

if __name__ == "__main__":
    exit(main())
