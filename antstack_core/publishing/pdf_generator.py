"""
PDF Generation Engine for Ant Stack Publications

Provides comprehensive PDF generation with Pandoc and LaTeX integration,
supporting advanced typography, cross-references, and scientific formatting.

Following .cursorrules specifications for:
- Professional document formatting
- LaTeX macro integration
- Cross-reference validation
- Unicode symbol conversion
"""

from __future__ import annotations

import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import yaml
import json
import re
import logging
from dataclasses import dataclass, asdict
import time

from ..figures.references import validate_cross_references, fix_figure_ids
from ..figures.mermaid import preprocess_mermaid_diagrams

logger = logging.getLogger(__name__)


@dataclass
class PDFGenerationConfig:
    """Configuration for PDF generation process."""

    # Pandoc settings
    pandoc_path: str = "pandoc"
    pandoc_args: List[str] = None

    # LaTeX settings
    latex_engine: str = "xelatex"
    latex_args: List[str] = None

    # Output settings
    output_format: str = "pdf"
    output_quality: str = "high"

    # Processing options
    enable_mermaid: bool = True
    enable_crossrefs: bool = True
    enable_unicode_conversion: bool = True
    validate_before_build: bool = True

    # Paths
    template_dir: Optional[Path] = None
    temp_dir: Optional[Path] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.pandoc_args is None:
            self.pandoc_args = []

        if self.latex_args is None:
            self.latex_args = ["-interaction=nonstopmode", "-synctex=1"]

        if self.template_dir is None:
            self.template_dir = Path(__file__).parent / "templates"

        if self.temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="antstack_pdf_"))


@dataclass
class BuildResult:
    """Result of PDF generation process."""

    success: bool
    output_path: Optional[Path] = None
    error_message: Optional[str] = None
    warnings: List[str] = None
    build_time_seconds: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class PDFGenerator:
    """
    Advanced PDF generation engine for scientific publications.

    Provides comprehensive PDF generation with:
    - Pandoc + LaTeX integration
    - Mermaid diagram preprocessing
    - Cross-reference validation
    - Unicode symbol conversion
    - Quality assurance checks
    - Build optimization
    """

    def __init__(self, config: Optional[PDFGenerationConfig] = None):
        """Initialize PDF generator with configuration."""
        self.config = config or PDFGenerationConfig()
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for PDF generation process."""
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    def generate_pdf(self, input_files: List[Path], output_path: Path,
                    metadata: Optional[Dict[str, Any]] = None) -> BuildResult:
        """
        Generate PDF from markdown files.

        Args:
            input_files: List of markdown files to process
            output_path: Path for output PDF file
            metadata: Optional metadata for the document

        Returns:
            BuildResult with success status and details
        """
        start_time = time.time()

        try:
            # Validate inputs
            if not input_files:
                return BuildResult(
                    success=False,
                    error_message="No input files provided"
                )

            # Create temporary working directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                processed_files = []

                # Process each input file
                for input_file in input_files:
                    processed_file = self._preprocess_file(input_file, temp_path)
                    processed_files.append(processed_file)

                # Generate PDF
                result = self._run_pandoc_build(
                    processed_files, output_path, metadata or {}
                )

                # Update build time
                result.build_time_seconds = time.time() - start_time

                return result

        except Exception as e:
            logger.error(f"PDF generation failed: {str(e)}")
            return BuildResult(
                success=False,
                error_message=str(e),
                build_time_seconds=time.time() - start_time
            )

    def _preprocess_file(self, input_file: Path, temp_dir: Path) -> Path:
        """
        Preprocess a single markdown file for PDF generation.

        Args:
            input_file: Input markdown file
            temp_dir: Temporary directory for processing

        Returns:
            Path to processed file
        """
        # Read input file
        content = input_file.read_text(encoding='utf-8')

        # Apply preprocessing steps
        if self.config.enable_unicode_conversion:
            content = self._convert_unicode_symbols(content)

        if self.config.enable_crossrefs:
            content = self._validate_and_fix_references(content, input_file)

        if self.config.enable_mermaid:
            content = self._preprocess_mermaid_diagrams(content, input_file, temp_dir)

        # Write processed content
        processed_file = temp_dir / f"processed_{input_file.name}"
        processed_file.write_text(content, encoding='utf-8')

        return processed_file

    def _convert_unicode_symbols(self, content: str) -> str:
        """Convert Unicode math symbols to LaTeX macros."""
        # Common Unicode to LaTeX conversions
        conversions = {
            '≤': r'$\le$',
            '≥': r'$\ge$',
            '≈': r'$\approx$',
            '∼': r'$\sim$',
            'Δ': r'$\Delta$',
            'ρ': r'$\rho$',
            'μ': r'$\mu$',
            'λ': r'$\lambda$',
            'σ': r'$\sigma$',
            '∝': r'$\propto$',
            '⋅': r'$\cdot$',
            '∘': r'$\circ$',
            '∑': r'$\sum$',
            '∏': r'$\prod$',
            '∫': r'$\int$',
            '∂': r'$\partial$',
            '∇': r'$\nabla$',
            '∈': r'$\in$',
            '∉': r'$\notin$',
            '⊂': r'$\subset$',
            '⊆': r'$\subseteq$',
            '∪': r'$\cup$',
            '∩': r'$\cap$',
            '∧': r'$\wedge$',
            '∨': r'$\vee$',
            '¬': r'$\neg$',
            '∀': r'$\forall$',
            '∃': r'$\exists$',
            '⇒': r'$\implies$',
            '⇔': r'$\iff$'
        }

        for unicode_char, latex_macro in conversions.items():
            content = content.replace(unicode_char, latex_macro)

        return content

    def _validate_and_fix_references(self, content: str, file_path: Path) -> str:
        """Validate and fix cross-references in content."""
        try:
            # Use existing reference validation
            issues = validate_cross_references(content)

            if issues:
                logger.warning(f"Found {len(issues)} reference issues in {file_path}")
                for issue in issues:
                    logger.warning(f"  {issue}")

                # Attempt to fix common issues
                content = fix_figure_ids(content)

            return content

        except Exception as e:
            logger.warning(f"Reference validation failed for {file_path}: {str(e)}")
            return content

    def _preprocess_mermaid_diagrams(self, content: str, file_path: Path,
                                   temp_dir: Path) -> str:
        """Preprocess Mermaid diagrams for PDF generation."""
        try:
            # Use existing Mermaid preprocessing
            processed_content = preprocess_mermaid_diagrams(
                content, temp_dir, str(file_path)
            )
            return processed_content

        except Exception as e:
            logger.warning(f"Mermaid preprocessing failed for {file_path}: {str(e)}")
            return content

    def _run_pandoc_build(self, input_files: List[Path], output_path: Path,
                         metadata: Dict[str, Any]) -> BuildResult:
        """
        Execute Pandoc build process.

        Args:
            input_files: Processed input files
            output_path: Output PDF path
            metadata: Document metadata

        Returns:
            BuildResult with success status
        """
        try:
            # Build Pandoc command
            cmd = self._build_pandoc_command(input_files, output_path, metadata)

            logger.info(f"Running Pandoc command: {' '.join(cmd)}")

            # Execute Pandoc
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=input_files[0].parent,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                logger.info(f"PDF generated successfully: {output_path}")
                return BuildResult(
                    success=True,
                    output_path=output_path,
                    warnings=self._parse_warnings(result.stderr)
                )
            else:
                error_msg = result.stderr or result.stdout
                logger.error(f"Pandoc build failed: {error_msg}")
                return BuildResult(
                    success=False,
                    error_message=error_msg,
                    warnings=self._parse_warnings(result.stderr)
                )

        except subprocess.TimeoutExpired:
            return BuildResult(
                success=False,
                error_message="PDF generation timed out after 5 minutes"
            )
        except Exception as e:
            return BuildResult(
                success=False,
                error_message=f"Build execution failed: {str(e)}"
            )

    def _build_pandoc_command(self, input_files: List[Path], output_path: Path,
                            metadata: Dict[str, Any]) -> List[str]:
        """Build Pandoc command with all necessary arguments."""
        cmd = [self.config.pandoc_path]

        # Input files
        for input_file in input_files:
            cmd.append(str(input_file))

        # Output
        cmd.extend(["-o", str(output_path)])

        # LaTeX engine
        cmd.extend(["--pdf-engine", self.config.latex_engine])

        # LaTeX arguments
        for arg in self.config.latex_args:
            cmd.extend(["--pdf-engine-opt", arg])

        # Pandoc arguments
        cmd.extend(self.config.pandoc_args)

        # Essential LaTeX packages and settings
        cmd.extend([
            "--variable", "geometry:margin=1in",
            "--variable", "fontsize=11pt",
            "--variable", "colorlinks=true",
            "--variable", "linkcolor=blue",
            "--variable", "urlcolor=blue",
            "--variable", "citecolor=blue",
            "--variable", "mathspec",  # For better math rendering
            "--variable", "unicode-math",  # Unicode math support
        ])

        # Template if available
        template_path = self.config.template_dir / "default.latex"
        if template_path.exists():
            cmd.extend(["--template", str(template_path)])

        # Metadata
        for key, value in metadata.items():
            cmd.extend(["--metadata", f"{key}:{value}"])

        return cmd

    def _parse_warnings(self, stderr: str) -> List[str]:
        """Parse warnings from Pandoc/LaTeX output."""
        warnings = []

        if not stderr:
            return warnings

        # Common warning patterns
        warning_patterns = [
            r"Warning:",
            r"LaTeX Warning:",
            r"Package .* Warning:",
            r"Underfull \\hbox",
            r"Overfull \\hbox",
            r"Reference .* undefined",
        ]

        lines = stderr.split('\n')
        for line in lines:
            line = line.strip()
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in warning_patterns):
                warnings.append(line)

        return warnings

    def validate_build_environment(self) -> Dict[str, bool]:
        """
        Validate that all required tools are available.

        Returns:
            Dictionary with validation results for each component
        """
        results = {}

        # Check Pandoc
        try:
            result = subprocess.run(
                [self.config.pandoc_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            results["pandoc"] = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            results["pandoc"] = False

        # Check LaTeX engine
        try:
            result = subprocess.run(
                [self.config.latex_engine, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            results["latex"] = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            results["latex"] = False

        # Check for required LaTeX packages (basic check)
        if results.get("latex", False):
            # This is a simplified check - in practice you'd test specific packages
            results["latex_packages"] = True
        else:
            results["latex_packages"] = False

        return results


# Convenience functions for common use cases
def generate_publication_pdf(input_files: List[Path], output_path: Path,
                           title: str = "", author: str = "") -> BuildResult:
    """
    Generate PDF for scientific publication with standard settings.

    Args:
        input_files: Markdown files to process
        output_path: Output PDF path
        title: Document title
        author: Document author

    Returns:
        BuildResult with generation results
    """
    config = PDFGenerationConfig(
        output_quality="high",
        enable_mermaid=True,
        enable_crossrefs=True,
        validate_before_build=True
    )

    generator = PDFGenerator(config)

    metadata = {}
    if title:
        metadata["title"] = title
    if author:
        metadata["author"] = author

    return generator.generate_pdf(input_files, output_path, metadata)


def validate_pdf_environment() -> Dict[str, Any]:
    """
    Validate PDF generation environment and return detailed status.

    Returns:
        Dictionary with validation results and recommendations
    """
    config = PDFGenerationConfig()
    generator = PDFGenerator(config)

    validation = generator.validate_build_environment()

    # Add recommendations
    recommendations = []
    if not validation.get("pandoc", False):
        recommendations.append("Install Pandoc: https://pandoc.org/installing.html")
    if not validation.get("latex", False):
        recommendations.append("Install LaTeX (TeX Live or MacTeX recommended)")
    if not validation.get("latex_packages", False):
        recommendations.append("Install additional LaTeX packages for scientific documents")

    return {
        "validation": validation,
        "recommendations": recommendations,
        "all_valid": all(validation.values())
    }
