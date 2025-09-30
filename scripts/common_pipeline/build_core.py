#!/usr/bin/env python3
"""Core Build System for Modular Scientific Publications.

This is the new modular build system that works with the refactored antstack_core
package and the papers/ directory structure. It provides:

- Modular paper configuration via YAML
- Unified build pipeline using core methods
- Cross-reference validation and quality assurance
- Multi-backend support (Pandoc, XeLaTeX, etc.)
- Comprehensive error reporting and validation

Following .cursorrules specifications:
- Zero tolerance for broken references
- Professional, test-driven implementation
- Comprehensive validation and quality assurance
- Clear separation between core methods and paper content

Usage:
    python3 scripts/build_core.py                    # Build all papers
    python3 scripts/build_core.py --paper ant_stack  # Build specific paper
    python3 scripts/build_core.py --validate-only    # Only validate
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    import yaml
except ImportError:
    print("Error: PyYAML required. Install with: pip install PyYAML")
    sys.exit(1)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from antstack_core.figures import (
        validate_cross_references, preprocess_mermaid_diagrams,
        organize_figure_assets
    )
    from antstack_core.analysis import EnergyCoefficients
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Core modules not fully available: {e}")
    CORE_AVAILABLE = False


class ModularPaperBuilder:
    """Modular paper builder using antstack_core methods."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.papers_dir = self.project_root / "papers"
        self.scripts_dir = self.project_root / "scripts"
        self.tools_dir = self.project_root / "tools"
        
        # Build state
        self.build_results = {}
        self.validation_results = {}
        
    def discover_papers(self) -> List[str]:
        """Discover available papers by scanning papers directory."""
        papers = []
        
        if self.papers_dir.exists():
            for paper_dir in self.papers_dir.iterdir():
                if paper_dir.is_dir():
                    config_file = paper_dir / "paper_config.yaml"
                    if config_file.exists():
                        papers.append(paper_dir.name)
        
        return papers
    
    def load_paper_config(self, paper_name: str) -> Dict[str, Any]:
        """Load paper configuration from YAML file."""
        config_file = self.papers_dir / paper_name / "paper_config.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = ['paper', 'content', 'build', 'latex']
        missing_sections = [s for s in required_sections if s not in config]
        
        if missing_sections:
            raise ValueError(f"Missing required config sections: {missing_sections}")
        
        return config
    
    def validate_paper_environment(self, paper_name: str) -> Dict[str, Any]:
        """Validate paper environment and dependencies."""
        results = {
            "paper_name": paper_name,
            "valid": True,
            "issues": [],
            "warnings": []
        }
        
        paper_dir = self.papers_dir / paper_name
        
        # Check paper directory exists
        if not paper_dir.exists():
            results["issues"].append(f"Paper directory not found: {paper_dir}")
            results["valid"] = False
            return results
        
        # Check configuration
        try:
            config = self.load_paper_config(paper_name)
        except Exception as e:
            results["issues"].append(f"Configuration error: {e}")
            results["valid"] = False
            return results
        
        # Check content files
        content_files = config.get('content', {}).get('files', [])
        missing_files = []
        
        for file_name in content_files:
            file_path = paper_dir / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            results["warnings"].append(f"Missing content files: {missing_files}")
        
        # Check assets directory
        assets_dir = paper_dir / "assets"
        if not assets_dir.exists():
            results["warnings"].append("Assets directory not found - will be created")
        
        # Validate cross-references if core is available
        if CORE_AVAILABLE:
            cross_ref_issues = self._validate_cross_references_for_paper(paper_dir, content_files)
            if cross_ref_issues:
                results["issues"].extend(cross_ref_issues)
                results["valid"] = False
        
        return results
    
    def _validate_cross_references_for_paper(self, paper_dir: Path, content_files: List[str]) -> List[str]:
        """Validate cross-references in all paper content files (paper-wide validation)."""
        issues = []
        
        # Collect all content for paper-wide validation
        all_content = []
        file_boundaries = {}
        
        for file_name in content_files:
            file_path = paper_dir / file_name
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    file_boundaries[file_name] = {
                        'start': len('\n'.join(all_content)),
                        'length': len(content)
                    }
                    
                    all_content.append(f"<!-- FILE: {file_name} -->\n{content}")
                    
                except Exception as e:
                    issues.append(f"Error reading {file_name}: {e}")
        
        # Validate cross-references paper-wide
        if all_content:
            combined_content = '\n\n'.join(all_content)
            validation = validate_cross_references(combined_content, str(paper_dir))
            
            if not validation.get('valid', True):
                # Filter out "unused definition" warnings since they may be used elsewhere
                paper_issues = validation.get('issues', [])
                real_issues = []
                
                for issue in paper_issues:
                    # Only report broken references, not unused definitions
                    if 'Broken reference' in issue:
                        real_issues.append(issue)
                    elif 'Missing figure definition' in issue:
                        real_issues.append(issue)
                    # Skip "unused definition" warnings for paper-wide validation
                
                issues.extend(real_issues)
        
        return issues
    
    def run_analysis_pipeline(self, paper_name: str, config: Dict[str, Any]) -> bool:
        """Run computational analysis pipeline if required."""
        if not config.get('build', {}).get('has_computational_analysis', False):
            return True  # No analysis needed
        
        analysis_config = config.get('analysis', {})
        if not analysis_config:
            print(f"[{paper_name}] No analysis configuration found")
            return True
        
        paper_dir = self.papers_dir / paper_name
        
        # Check for analysis runner module
        runner_module = analysis_config.get('runner_module')
        if runner_module:
            # Try to run analysis using the specified module
            try:
                # For now, use the legacy runner from complexity_energetics
                if paper_name == 'complexity_energetics':
                    manifest_file = paper_dir / analysis_config.get('manifest_file', 'manifest.example.yaml')
                    output_dir = paper_dir / analysis_config.get('output_dir', 'out')
                    
                    if manifest_file.exists():
                        print(f"[{paper_name}] Running analysis pipeline...")
                        
                        # Run the analysis using the original runner
                        runner_script = paper_dir / 'src' / 'ce' / 'runner.py'
                        cmd = [
                            sys.executable, str(runner_script),
                            str(manifest_file), '--out', str(output_dir)
                        ]
                        
                        # Set PYTHONPATH to include the src directory
                        env = os.environ.copy()
                        env['PYTHONPATH'] = str(paper_dir / 'src')

                        result = subprocess.run(
                            cmd,
                            cwd=self.project_root,
                            capture_output=True,
                            text=True,
                            env=env
                        )
                        
                        if result.returncode != 0:
                            print(f"[{paper_name}] Analysis pipeline failed: {result.stderr}")
                            return False
                        else:
                            print(f"[{paper_name}] Analysis pipeline completed successfully")
                            return True
                    else:
                        print(f"[{paper_name}] Manifest file not found: {manifest_file}")
                        return False
                
            except Exception as e:
                print(f"[{paper_name}] Analysis pipeline error: {e}")
                return False
        
        return True
    
    def build_paper(self, paper_name: str, validate_only: bool = False) -> Tuple[bool, Dict[str, Any]]:
        """Build a single paper with comprehensive validation."""
        print(f"\n📄 Building paper: {paper_name}")
        
        # Load configuration
        try:
            config = self.load_paper_config(paper_name)
        except Exception as e:
            return False, {"error": f"Configuration error: {e}"}
        
        paper_dir = self.papers_dir / paper_name
        
        # Environment validation
        validation = self.validate_paper_environment(paper_name)
        if not validation["valid"]:
            print(f"[{paper_name}] Validation failed:")
            for issue in validation.get("issues", []):
                print(f"  ❌ {issue}")
            return False, {"error": "Validation failed", "details": validation}
        
        # Print warnings
        for warning in validation.get("warnings", []):
            print(f"⚠️  {warning}")
        
        if validate_only:
            return True, {"validation": validation}
        
        # Run analysis pipeline if needed
        if not self.run_analysis_pipeline(paper_name, config):
            return False, {"error": "Analysis pipeline failed"}
        
        # Organize assets if core is available
        if CORE_AVAILABLE:
            print(f"[{paper_name}] Organizing figure assets...")
            try:
                organize_figure_assets(paper_dir)
            except Exception as e:
                print(f"[{paper_name}] Warning: Asset organization failed: {e}")
        
        # Build PDF using legacy system (unified pipeline has permission issues)
        success = self._build_pdf_with_legacy_system(paper_name)
        
        build_result = {
            "paper_name": paper_name,
            "success": success,
            "validation": validation,
            "timestamp": datetime.now().isoformat()
        }
        
        self.build_results[paper_name] = build_result
        
        return success, build_result
    
    def _build_pdf_with_legacy_system(self, paper_name: str) -> bool:
        """Build PDF using the existing render_pdf.sh system."""
        render_script = self.tools_dir / "render_pdf.sh"
        
        if not render_script.exists():
            print(f"[{paper_name}] Error: render_pdf.sh not found at {render_script}")
            return False
        
        print(f"[{paper_name}] Building PDF using legacy render system...")
        print(f"[{paper_name}] 📄 Starting PDF generation process...")
        
        try:
            # Set environment variables for the legacy build
            env = os.environ.copy()
            env.update({
                'LINK_COLOR': 'blue',
                'MERMAID_STRATEGY': 'auto',
                'MERMAID_IMG_FORMAT': 'png'
            })
            
            cmd = ['bash', str(render_script), paper_name]
            
            # Use real-time output instead of capturing
            print(f"[{paper_name}] 🔧 Executing: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output in real-time
            output_lines = []
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    line = output.strip()
                    if line:
                        # Filter and format output
                        if line.startswith('[') and ']' in line:
                            print(f"[{paper_name}] {line}")
                        elif 'WARNING' in line or 'ERROR' in line:
                            print(f"[{paper_name}] ⚠️  {line}")
                        elif 'Successfully created PDF' in line:
                            print(f"[{paper_name}] ✅ {line}")
                        elif 'Build validation passed' in line:
                            print(f"[{paper_name}] ✅ {line}")
                        else:
                            print(f"[{paper_name}] 📝 {line}")
                        output_lines.append(line)
            
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code == 0:
                # Check if PDF was actually created
                expected_pdf = self._get_expected_pdf_path(paper_name)
                if expected_pdf.exists():
                    file_size = expected_pdf.stat().st_size
                    print(f"[{paper_name}] ✅ PDF created successfully: {expected_pdf} ({file_size:,} bytes)")
                    return True
                else:
                    print(f"[{paper_name}] ❌ PDF not found at expected location: {expected_pdf}")
                    return False
            else:
                print(f"[{paper_name}] ❌ Build failed with return code: {return_code}")
                # Show last few lines of output for debugging
                if output_lines:
                    print(f"[{paper_name}] Last output lines:")
                    for line in output_lines[-5:]:
                        print(f"[{paper_name}]   {line}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"[{paper_name}] ❌ Build timeout after 5 minutes")
            return False
        except Exception as e:
            print(f"[{paper_name}] ❌ Build error: {e}")
            return False
    
    def _build_pdf_unified(self, paper_name: str, config: Dict[str, Any]) -> bool:
        """Build PDF using unified pipeline (no legacy dependencies)."""
        print(f"[{paper_name}] Building PDF using unified pipeline...")
        print(f"[{paper_name}] 📄 Starting PDF generation process...")
        
        paper_dir = self.papers_dir / paper_name
        
        try:
            # Step 1: Run analysis pipeline if needed
            if not self.run_analysis_pipeline(paper_name, config):
                print(f"[{paper_name}] ❌ Analysis pipeline failed")
                return False
            
            # Step 2: Concatenate markdown files
            temp_md_file = self._concatenate_markdown_files(paper_name, config)
            if not temp_md_file:
                return False
            
            # Step 3: Prerender Mermaid diagrams
            if not self._prerender_mermaid_diagrams(paper_name, temp_md_file, paper_dir):
                print(f"[{paper_name}] ⚠️  Mermaid prerendering had issues, continuing...")
            
            # Step 4: Generate PDF with Pandoc
            success = self._generate_pdf_with_pandoc(paper_name, temp_md_file, config, paper_dir)
            
            # Step 5: Validate build
            if success:
                self._validate_pdf_build(paper_name, paper_dir)
            
            # Cleanup - keep file for debugging for now
            # if not success and temp_md_file and os.path.exists(temp_md_file):
            #     print(f"[{paper_name}] 🧹 Cleaning up temporary file: {temp_md_file}")
            #     os.unlink(temp_md_file)
            
            return success
            
        except Exception as e:
            print(f"[{paper_name}] ❌ Build error: {e}")
            return False
    
    def _concatenate_markdown_files(self, paper_name: str, config: Dict[str, Any]) -> Optional[str]:
        """Concatenate markdown files in the correct order."""
        paper_dir = self.papers_dir / paper_name
        content_files = config.get('content', {}).get('files', [])
        
        if not content_files:
            print(f"[{paper_name}] ❌ No content files specified in config")
            return None
        
        # Create temporary file in project directory with proper permissions
        temp_dir = self.project_root / "temp"
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / f"ant_build_{paper_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        print(f"[{paper_name}] 📄 Concatenating Markdown files into temporary file: {temp_path}")
        
        try:
            # Write to the file using open() with proper permissions
            with open(temp_path, 'w', encoding='utf-8') as temp_file:
                total_files = len(content_files)
                for i, file_name in enumerate(content_files):
                    file_path = paper_dir / file_name
                    file_num = i + 1
                    
                    print(f"[{paper_name}] 📝 Processing file {file_num}/{total_files}: {file_name}")
                    
                    if file_path.exists():
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        temp_file.write(content)
                        
                        # Add page break if not the last file
                        if i < total_files - 1:
                            temp_file.write('\f\n\n')  # Form feed for page break
                    else:
                        print(f"[{paper_name}] ⚠️  Missing file: {file_name}")
            
            # Verify file was created and has content
            if temp_path.exists():
                file_size = temp_path.stat().st_size
                print(f"[{paper_name}] ✅ Concatenated {total_files} files successfully (file size: {file_size} bytes)")
                return str(temp_path)  # Return as string
            else:
                print(f"[{paper_name}] ❌ Temporary file was not created: {temp_path}")
                return None
            
        except Exception as e:
            print(f"[{paper_name}] ❌ Error concatenating files: {e}")
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return None
    
    def _prerender_mermaid_diagrams(self, paper_name: str, temp_md_file: str, paper_dir: Path) -> bool:
        """Prerender Mermaid diagrams to PNG files."""
        print(f"[{paper_name}] Prerendering Mermaid diagrams to local PNGs...")
        
        mermaid_dir = paper_dir / "assets" / "mermaid"
        mermaid_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Use the core Mermaid functionality
            if CORE_AVAILABLE:
                # Check if temp file still exists before reading
                if not os.path.exists(temp_md_file):
                    print(f"[{paper_name}] ❌ Temporary file disappeared before Mermaid processing: {temp_md_file}")
                    return False
                
                with open(temp_md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                print(f"[{paper_name}] 📄 Content length before Mermaid processing: {len(content)} characters")
                
                # Prerender Mermaid diagrams and get processed content
                processed_content = preprocess_mermaid_diagrams(content, str(mermaid_dir))
                
                print(f"[{paper_name}] 📄 Content length after Mermaid processing: {len(processed_content)} characters")
                
                # Write the processed content back to the temp file
                with open(temp_md_file, 'w', encoding='utf-8') as f:
                    f.write(processed_content)
                
                # Verify file still exists after writing
                if os.path.exists(temp_md_file):
                    file_size = os.path.getsize(temp_md_file)
                    print(f"[{paper_name}] ✅ Mermaid diagrams prerendered successfully (final file size: {file_size} bytes)")
                    return True
                else:
                    print(f"[{paper_name}] ❌ Temporary file disappeared after Mermaid processing: {temp_md_file}")
                    return False
            else:
                print(f"[{paper_name}] ⚠️  Core modules not available, skipping Mermaid prerendering")
                return True
                
        except Exception as e:
            print(f"[{paper_name}] ⚠️  Mermaid prerendering failed: {e}")
            return False
    
    def _generate_pdf_with_pandoc(self, paper_name: str, temp_md_file: str, config: Dict[str, Any], paper_dir: Path) -> bool:
        """Generate PDF using Pandoc and XeLaTeX."""
        print(f"[{paper_name}] 🔧 Generating PDF from concatenated file...")
        
        # Check dependencies
        if not shutil.which('pandoc'):
            print(f"[{paper_name}] ❌ pandoc not found")
            return False
        
        if not shutil.which('xelatex'):
            print(f"[{paper_name}] ❌ xelatex not found")
            return False
        
        # Get paper metadata
        paper_meta = config.get('paper', {})
        title = paper_meta.get('title', paper_name)
        author = paper_meta.get('author', 'Daniel Ari Friedman')
        email = paper_meta.get('email', 'daniel@activeinference.institute')
        
        # Create LaTeX header (optional)
        header_file = self._create_latex_header(paper_name)
        if not header_file or not os.path.exists(header_file):
            print(f"[{paper_name}] ⚠️  Header file not available, proceeding without custom LaTeX header")
            header_file = None
        
        # Build Pandoc command
        output_pdf = self._get_expected_pdf_path(paper_name)
        
        # Use relative path from project root
        temp_md_relative = os.path.relpath(temp_md_file, self.project_root)
        pandoc_args = [
            temp_md_relative,
            '-f', 'markdown+tex_math_dollars+raw_tex+autolink_bare_uris',
            '--pdf-engine=xelatex',
            '--toc',
            '--number-sections',
            '--highlight-style=tango',
            '-V', f'geometry:margin=2.5cm',
            '-V', f'title={title}',
            '-V', f'author={author}',
            '-V', f'email={email}',
            '-V', f'date={datetime.now().strftime("%B %d, %Y")}',
            '-V', 'mainfont=Helvetica',
            '--resource-path', f'.:{str(paper_dir)}:{str(paper_dir)}/assets:{str(paper_dir)}/assets/mermaid',
            '-o', str(output_pdf)
        ]
        
        # Add bibliography if specified
        latex_config = config.get('latex', {})
        bibliography = latex_config.get('bibliography')
        if bibliography and (paper_dir / bibliography).exists():
            pandoc_args.extend([
                '--bibliography', str(paper_dir / bibliography),
                '--citeproc'
            ])
        
        # Skip header file and Lua filters for now to avoid permission issues
        # TODO: Debug and re-enable these features
        # if header_file:
        #     pandoc_args.extend(['-H', str(header_file)])
        # 
        # lua_filters = [
        #     'tools/filters/auto_link_code_urls.lua',
        #     'tools/filters/unicode_to_tex.lua'
        # ]
        # 
        # for filter_path in lua_filters:
        #     filter_file = self.project_root / filter_path
        #     if filter_file.exists():
        #         pandoc_args.extend(['--lua-filter', str(filter_file)])
        
        print(f"[{paper_name}] 🔄 Running Pandoc conversion to PDF...")
        print(f"[{paper_name}] 📊 This may take a few minutes for large documents...")
        
        # Check if temp file exists before running Pandoc
        if not os.path.exists(temp_md_file):
            print(f"[{paper_name}] ❌ Temporary file does not exist before Pandoc: {temp_md_file}")
            return False
        
        file_size = os.path.getsize(temp_md_file)
        print(f"[{paper_name}] 📄 Temp file exists, size: {file_size} bytes")
        
        # Debug: print the exact command being run
        print(f"[{paper_name}] 🔧 Pandoc command: pandoc {' '.join(pandoc_args)}")
        
        try:
            result = subprocess.run(
                pandoc_args,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print(f"[{paper_name}] ✅ Pandoc conversion completed")
                return True
            else:
                print(f"[{paper_name}] ❌ Pandoc conversion failed")
                print(f"[{paper_name}] Error output: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"[{paper_name}] ❌ Pandoc conversion timeout after 5 minutes")
            return False
        except Exception as e:
            print(f"[{paper_name}] ❌ Pandoc error: {e}")
            return False
        finally:
            # Cleanup header file
            if header_file and os.path.exists(header_file):
                os.unlink(header_file)
    
    def _create_latex_header(self, paper_name: str) -> Optional[str]:
        """Create LaTeX header file for PDF generation."""
        try:
            # Create temporary file in project directory
            temp_dir = self.project_root / "temp"
            temp_dir.mkdir(exist_ok=True)
            temp_path = temp_dir / f"ant_header_{paper_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex"
            
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(r"""
\usepackage{amsmath}
\usepackage{url}
\usepackage{textcomp}
\usepackage{fontspec}
\usepackage{unicode-math}
\usepackage{hyperref}
\usepackage{cleveref}
\usepackage{newunicodechar}

% Unicode character mappings
\newunicodechar{🔬}{[Lab]}
\newunicodechar{⚙}{[Gear]}
\newunicodechar{🎯}{[Target]}
\newunicodechar{📊}{[Chart]}
\newunicodechar{✅}{[Check]}
\newunicodechar{📖}{[Book]}
\newunicodechar{🔧}{[Tool]}
\newunicodechar{🤖}{[Robot]}
\newunicodechar{🧠}{[Brain]}
\newunicodechar{💭}{[Thought]}
\newunicodechar{⚡}{[Lightning]}

% Math symbols
\newunicodechar{⋅}{\ensuremath{\cdot}}
\newunicodechar{∘}{\ensuremath{\circ}}
\newunicodechar{↑}{\ensuremath{\uparrow}}
\newunicodechar{↓}{\ensuremath{\downarrow}}
\newunicodechar{∝}{\ensuremath{\propto}}
\newunicodechar{∼}{\ensuremath{\sim}}
\newunicodechar{≈}{\ensuremath{\approx}}
\newunicodechar{≤}{\ensuremath{\le}}
\newunicodechar{≥}{\ensuremath{\ge}}
\newunicodechar{≠}{\ensuremath{\ne}}
\newunicodechar{±}{\ensuremath{\pm}}
\newunicodechar{×}{\ensuremath{\times}}
\newunicodechar{÷}{\ensuremath{\div}}
\newunicodechar{−}{-}
\newunicodechar{→}{\ensuremath{\to}}
\newunicodechar{↔}{\ensuremath{\leftrightarrow}}
\newunicodechar{⇒}{\ensuremath{\Rightarrow}}
\newunicodechar{⇔}{\ensuremath{\Leftrightarrow}}
\newunicodechar{∈}{\ensuremath{\in}}
\newunicodechar{∉}{\ensuremath{\notin}}
\newunicodechar{∩}{\ensuremath{\cap}}
\newunicodechar{∪}{\ensuremath{\cup}}
\newunicodechar{⊂}{\ensuremath{\subset}}
\newunicodechar{⊆}{\ensuremath{\subseteq}}
\newunicodechar{⊇}{\ensuremath{\supseteq}}
\newunicodechar{∅}{\ensuremath{\varnothing}}
\newunicodechar{∞}{\ensuremath{\infty}}
\newunicodechar{∇}{\ensuremath{\nabla}}
\newunicodechar{√}{\ensuremath{\sqrt{}}}
\newunicodechar{°}{\ensuremath{^{\circ}}}

% Greek letters
\newunicodechar{α}{\ensuremath{\alpha}}
\newunicodechar{β}{\ensuremath{\beta}}
\newunicodechar{γ}{\ensuremath{\gamma}}
\newunicodechar{δ}{\ensuremath{\delta}}
\newunicodechar{Δ}{\ensuremath{\Delta}}
\newunicodechar{ε}{\ensuremath{\epsilon}}
\newunicodechar{ζ}{\ensuremath{\zeta}}
\newunicodechar{η}{\ensuremath{\eta}}
\newunicodechar{θ}{\ensuremath{\theta}}
\newunicodechar{Θ}{\ensuremath{\Theta}}
\newunicodechar{ι}{\ensuremath{\iota}}
\newunicodechar{κ}{\ensuremath{\kappa}}
\newunicodechar{λ}{\ensuremath{\lambda}}
\newunicodechar{Λ}{\ensuremath{\Lambda}}
\newunicodechar{μ}{\ensuremath{\mu}}
\newunicodechar{µ}{\ensuremath{\mu}}
\newunicodechar{ν}{\ensuremath{\nu}}
\newunicodechar{ξ}{\ensuremath{\xi}}
\newunicodechar{Ξ}{\ensuremath{\Xi}}
\newunicodechar{π}{\ensuremath{\pi}}
\newunicodechar{Π}{\ensuremath{\Pi}}
\newunicodechar{ρ}{\ensuremath{\rho}}
\newunicodechar{σ}{\ensuremath{\sigma}}
\newunicodechar{Σ}{\ensuremath{\Sigma}}
\newunicodechar{τ}{\ensuremath{\tau}}
\newunicodechar{υ}{\ensuremath{\upsilon}}
\newunicodechar{Υ}{\ensuremath{\Upsilon}}
\newunicodechar{φ}{\ensuremath{\phi}}
\newunicodechar{Φ}{\ensuremath{\Phi}}
\newunicodechar{χ}{\ensuremath{\chi}}
\newunicodechar{ψ}{\ensuremath{\psi}}
\newunicodechar{Ψ}{\ensuremath{\Psi}}
\newunicodechar{ω}{\ensuremath{\omega}}
\newunicodechar{Ω}{\ensuremath{\Omega}}

% Mathematical italic variants
\newunicodechar{𝛼}{\ensuremath{\alpha}}
\newunicodechar{𝛽}{\ensuremath{\beta}}
\newunicodechar{𝛾}{\ensuremath{\gamma}}
\newunicodechar{𝛿}{\ensuremath{\delta}}
\newunicodechar{𝛥}{\ensuremath{\Delta}}
\newunicodechar{𝜀}{\ensuremath{\epsilon}}
\newunicodechar{𝜁}{\ensuremath{\zeta}}
\newunicodechar{𝜂}{\ensuremath{\eta}}
\newunicodechar{𝜃}{\ensuremath{\theta}}
\newunicodechar{𝜄}{\ensuremath{\iota}}
\newunicodechar{𝜅}{\ensuremath{\kappa}}
\newunicodechar{𝜆}{\ensuremath{\lambda}}
\newunicodechar{𝜇}{\ensuremath{\mu}}
\newunicodechar{𝜈}{\ensuremath{\nu}}
\newunicodechar{𝜉}{\ensuremath{\xi}}
\newunicodechar{𝜋}{\ensuremath{\pi}}
\newunicodechar{𝜌}{\ensuremath{\rho}}
\newunicodechar{𝜎}{\ensuremath{\sigma}}
\newunicodechar{𝜏}{\ensuremath{\tau}}
\newunicodechar{𝜐}{\ensuremath{\upsilon}}
\newunicodechar{𝜑}{\ensuremath{\phi}}
\newunicodechar{𝜒}{\ensuremath{\chi}}
\newunicodechar{𝜓}{\ensuremath{\psi}}
\newunicodechar{𝜔}{\ensuremath{\omega}}

% Additional math symbols
\newunicodechar{≪}{\ensuremath{\ll}}
\newunicodechar{≫}{\ensuremath{\gg}}
\newunicodechar{≅}{\ensuremath{\cong}}
\newunicodechar{≡}{\ensuremath{\equiv}}
\newunicodechar{⊥}{\ensuremath{\perp}}
\newunicodechar{∥}{\ensuremath{\parallel}}
\newunicodechar{≲}{\ensuremath{\lesssim}}
\newunicodechar{≳}{\ensuremath{\gtrsim}}

% Superscripts
\newunicodechar{⁷}{\ensuremath{^{7}}}
\newunicodechar{⁵}{\ensuremath{^{5}}}
\newunicodechar{⁻}{\ensuremath{^{-1}}}

% URL formatting
\urlstyle{same}

% Commands for consistent formatting
\newcommand{\var}[1]{\textit{#1}}
\newcommand{\code}[1]{\texttt{#1}}

% Configure hyperlinks
\hypersetup{
    colorlinks   = true,
    urlcolor     = blue,
    linkcolor    = blue,
    citecolor    = red,
    breaklinks   = true,
    hidelinks    = false
}

% Set main font
\setmainfont{Helvetica}

% Custom table of contents
\makeatletter
\let\oldtableofcontents\tableofcontents
\renewcommand{\tableofcontents}{%
  \begin{center}\small ORCID: 0000-0001-6232-9096\quad Email: daniel@activeinference.institute\end{center}
  \vspace{0.75em}%
  \oldtableofcontents}
\makeatother
""")
            
            return temp_path
            
        except Exception as e:
            print(f"[{paper_name}] ❌ Error creating LaTeX header: {e}")
            return None
    
    def _validate_pdf_build(self, paper_name: str, paper_dir: Path) -> None:
        """Validate the generated PDF build."""
        print(f"[{paper_name}] Running build validation checks...")
        
        output_pdf = self._get_expected_pdf_path(paper_name)
        validation_issues = 0
        
        # Check PDF exists and has reasonable size
        if output_pdf.exists():
            file_size = output_pdf.stat().st_size
            if file_size < 10000:
                print(f"[{paper_name}] WARNING: PDF file seems too small ({file_size} bytes)")
                validation_issues += 1
            else:
                print(f"[{paper_name}] ✓ PDF created successfully ({file_size:,} bytes)")
        else:
            print(f"[{paper_name}] ERROR: PDF generation failed - output file not found")
            validation_issues += 1
            return
        
        # Check for broken cross-references
        try:
            with open(output_pdf, 'rb') as f:
                pdf_content = f.read()
                if b'Figure~??' in pdf_content:
                    print(f"[{paper_name}] WARNING: Found broken cross-references in PDF")
                    validation_issues += 1
                else:
                    print(f"[{paper_name}] ✓ Cross-references validated")
        except Exception:
            print(f"[{paper_name}] ⚠️  Could not validate cross-references")
        
        # Check Mermaid diagrams
        mermaid_dir = paper_dir / "assets" / "mermaid"
        if mermaid_dir.exists():
            total_diagrams = len(list(mermaid_dir.glob("*.mmd")))
            rendered_diagrams = len(list(mermaid_dir.glob("*.png")))
            
            if total_diagrams > 0:
                if total_diagrams > rendered_diagrams:
                    print(f"[{paper_name}] WARNING: {total_diagrams - rendered_diagrams} Mermaid diagrams failed to render")
                    validation_issues += 1
                else:
                    print(f"[{paper_name}] ✓ All {total_diagrams} Mermaid diagrams rendered successfully")
        
        # Summary
        if validation_issues == 0:
            print(f"[{paper_name}] ✓ Build validation passed - no issues detected")
            print(f"[{paper_name}] Successfully created PDF: {output_pdf}")
        else:
            print(f"[{paper_name}] ⚠ Build validation found {validation_issues} issue(s)")
            print(f"[{paper_name}] PDF created but with potential issues: {output_pdf}")
    
    def _get_expected_pdf_path(self, paper_name: str) -> Path:
        """Get expected PDF output path."""
        if paper_name == 'ant_stack':
            return self.project_root / "1_ant_stack.pdf"
        elif paper_name == 'complexity_energetics':
            return self.project_root / "2_complexity_energetics.pdf"
        elif paper_name == 'cohereAnts':
            return self.project_root / "3_cohereAnts.pdf"
        elif paper_name == 'documentation':
            return self.project_root / "1_ant_stack_documentation.pdf"
        else:
            return self.project_root / f"{paper_name}.pdf"
    
    def build_all_papers(self, validate_only: bool = False, run_tests: bool = False) -> Dict[str, Any]:
        """Build all discovered papers."""
        papers = self.discover_papers()
        
        if not papers:
            return {
                "success": False,
                "error": "No papers found",
                "papers_directory": str(self.papers_dir)
            }
        
        print(f"📚 Found {len(papers)} papers: {', '.join(papers)}")
        
        results = {
            "success": True,
            "papers_built": 0,
            "papers_failed": 0,
            "build_results": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Build each paper
        total_papers = len(papers)
        for i, paper_name in enumerate(papers, 1):
            print(f"\n{'='*60}")
            print(f"📚 Building Paper {i}/{total_papers}: {paper_name}")
            print(f"{'='*60}")
            
            success, build_result = self.build_paper(paper_name, validate_only)
            
            # Ensure build_result has success field for reporting
            build_result["success"] = success
            results["build_results"][paper_name] = build_result
            
            if success:
                results["papers_built"] += 1
                print(f"\n✅ {paper_name} completed successfully ({i}/{total_papers})")
            else:
                results["papers_failed"] += 1
                print(f"\n❌ {paper_name} failed ({i}/{total_papers})")
            
            # Show progress
            print(f"📊 Progress: {results['papers_built']} succeeded, {results['papers_failed']} failed")
        
        # Determine overall success
        results["success"] = results["papers_failed"] == 0
        
        # Run tests if requested and available
        if run_tests and not validate_only:
            print("\n🧪 Running test suite...")
            test_results = self._run_test_suite()
            results["test_results"] = test_results
            
            if not test_results.get("success", False):
                results["success"] = False
        
        # Generate build report
        self._generate_build_report(results)
        
        return results
    
    def _run_test_suite(self) -> Dict[str, Any]:
        """Run the comprehensive test suite."""
        try:
            cmd = [sys.executable, "-m", "pytest", "tests/core_rendering/test_core_refactor.py", "-v"]
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_build_report(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive build report."""
        report_file = self.project_root / "build_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Build Report\n\n")
            f.write(f"**Generated:** {results['timestamp']}\n")
            f.write(f"**Status:** {'✅ SUCCESS' if results['success'] else '❌ FAILED'}\n")
            f.write(f"**Papers Built:** {results['papers_built']}\n")
            f.write(f"**Papers Failed:** {results['papers_failed']}\n\n")
            
            # Paper-specific results
            for paper_name, build_result in results["build_results"].items():
                f.write(f"## {paper_name}\n\n")
                
                success = build_result.get("success", False)
                if success:
                    f.write("✅ **Status:** SUCCESS\n")
                else:
                    f.write("❌ **Status:** FAILED\n")
                    if "error" in build_result:
                        f.write(f"**Error:** {build_result['error']}\n")
                
                # Validation details
                validation = build_result.get("validation", {})
                if validation.get("issues"):
                    f.write("\n**Issues:**\n")
                    for issue in validation["issues"]:
                        f.write(f"- ❌ {issue}\n")
                
                if validation.get("warnings"):
                    f.write("\n**Warnings:**\n")
                    for warning in validation["warnings"]:
                        f.write(f"- ⚠️ {warning}\n")
                
                f.write("\n")
            
            # Test results if available
            if "test_results" in results:
                f.write("## Test Results\n\n")
                test_results = results["test_results"]
                
                if test_results.get("success"):
                    f.write("✅ **Tests:** PASSED\n")
                else:
                    f.write("❌ **Tests:** FAILED\n")
                
                if "stdout" in test_results:
                    f.write("\n**Test Output:**\n")
                    f.write("```\n")
                    f.write(test_results["stdout"])
                    f.write("\n```\n")
        
        print(f"📊 Build report generated: {report_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Modular Scientific Publication Build System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/build_core.py                    # Build all papers
  python3 scripts/build_core.py --paper ant_stack  # Build specific paper
  python3 scripts/build_core.py --validate-only    # Only validate
  python3 scripts/build_core.py --no-tests         # Skip tests
        """
    )
    
    parser.add_argument(
        "--paper",
        help="Build specific paper only"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation, don't build"
    )
    
    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="Skip test execution"
    )
    
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory"
    )
    
    args = parser.parse_args()
    
    # Initialize builder
    builder = ModularPaperBuilder(args.project_root)
    
    if args.paper:
        # Build single paper
        success, result = builder.build_paper(args.paper, args.validate_only)
        
        if success:
            print(f"✅ Paper '{args.paper}' built successfully")
            return 0
        else:
            print(f"❌ Paper '{args.paper}' failed to build")
            if "error" in result:
                print(f"Error: {result['error']}")
            return 1
    else:
        # Build all papers
        results = builder.build_all_papers(
            validate_only=args.validate_only,
            run_tests=not args.no_tests
        )
        
        if results["success"]:
            print(f"✅ All papers built successfully")
            return 0
        else:
            print(f"❌ Build failed")
            return 1


if __name__ == "__main__":
    sys.exit(main())
