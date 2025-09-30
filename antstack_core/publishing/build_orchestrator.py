"""
Build Orchestration System for Ant Stack Publications

Comprehensive build management system providing:
- Multi-paper build coordination
- Dependency management between papers
- Incremental build optimization
- Parallel processing capabilities
- Build artifact management
- Continuous integration support

Following .cursorrules specifications for:
- Modular build system architecture
- Comprehensive validation pipelines
- Professional publication workflows
- Automated quality assurance
"""

from __future__ import annotations

import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set, Tuple
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
import json
import logging
from datetime import datetime
import tempfile
import shutil

from .pdf_generator import PDFGenerator, PDFGenerationConfig, validate_pdf_environment
from .quality_validator import QualityValidator, validate_build_readiness
from .reference_manager import ReferenceManager, analyze_document_references

logger = logging.getLogger(__name__)


@dataclass
class BuildTarget:
    """Represents a build target (paper or document)."""

    name: str
    source_files: List[Path]
    output_path: Path
    dependencies: List[str] = field(default_factory=list)
    build_config: Dict[str, Any] = field(default_factory=dict)
    last_build_hash: Optional[str] = None
    build_status: str = "pending"  # pending, building, success, failed

    @property
    def needs_rebuild(self) -> bool:
        """Check if target needs to be rebuilt based on source changes."""
        if self.last_build_hash is None:
            return True

        current_hash = self._calculate_source_hash()
        return current_hash != self.last_build_hash

    def _calculate_source_hash(self) -> str:
        """Calculate hash of all source files."""
        hasher = hashlib.md5()
        for source_file in sorted(self.source_files):
            if source_file.exists():
                hasher.update(source_file.read_bytes())
        return hasher.hexdigest()

    def update_build_hash(self):
        """Update the build hash after successful build."""
        self.last_build_hash = self._calculate_source_hash()


@dataclass
class BuildResult:
    """Result of a build operation."""

    target_name: str
    success: bool
    output_path: Optional[Path] = None
    error_message: Optional[str] = None
    build_time_seconds: float = 0.0
    artifacts: List[Path] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BuildOrchestratorConfig:
    """Configuration for the build orchestrator."""

    build_dir: Path = Path("build")
    temp_dir: Path = Path("temp")
    cache_dir: Path = Path("cache")
    max_workers: int = 4
    enable_parallel: bool = True
    enable_incremental: bool = True
    enable_validation: bool = True
    clean_temp_files: bool = True
    generate_reports: bool = True


class BuildOrchestrator:
    """
    Advanced build orchestration system for scientific publications.

    Features:
    - Multi-target build management
    - Dependency resolution and scheduling
    - Incremental build optimization
    - Parallel processing support
    - Comprehensive build reporting
    - Artifact management and caching
    """

    def __init__(self, config: Optional[BuildOrchestratorConfig] = None):
        """Initialize build orchestrator with configuration."""
        self.config = config or BuildOrchestratorConfig()
        self.targets: Dict[str, BuildTarget] = {}
        self.build_results: Dict[str, BuildResult] = {}
        self._setup_directories()
        self._setup_logging()

        # Initialize build components
        self.pdf_generator = PDFGenerator()
        self.quality_validator = QualityValidator()
        self.reference_manager = ReferenceManager()

    def _setup_directories(self):
        """Create necessary build directories."""
        directories = [
            self.config.build_dir,
            self.config.temp_dir,
            self.config.cache_dir,
            self.config.build_dir / "artifacts",
            self.config.build_dir / "reports",
            self.config.cache_dir / "hashes"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        """Configure logging for build orchestration."""
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    def add_build_target(self, target: BuildTarget):
        """Add a build target to the orchestration system."""
        self.targets[target.name] = target
        logger.info(f"Added build target: {target.name}")

    def create_paper_target(self, paper_name: str, paper_dir: Path,
                          output_dir: Optional[Path] = None) -> BuildTarget:
        """
        Create a build target for a paper from paper directory.

        Args:
            paper_name: Name of the paper
            paper_dir: Directory containing paper files
            output_dir: Output directory for built artifacts

        Returns:
            Configured BuildTarget
        """
        if output_dir is None:
            output_dir = self.config.build_dir / "papers"

        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all markdown files in paper directory
        source_files = []
        for pattern in ["*.md", "*.markdown"]:
            source_files.extend(list(paper_dir.glob(f"**/{pattern}")))

        # Main paper file (usually has same name as directory)
        main_file = paper_dir / f"{paper_name}.md"
        if main_file.exists():
            # Ensure main file is first in the list
            if main_file in source_files:
                source_files.remove(main_file)
            source_files.insert(0, main_file)

        if not source_files:
            raise ValueError(f"No markdown files found in {paper_dir}")

        # Load build configuration if available
        config_file = paper_dir / "paper_config.yaml"
        build_config = {}
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    build_config = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Failed to load config from {config_file}: {str(e)}")

        # Create output path
        output_path = output_dir / f"{paper_name}.pdf"

        target = BuildTarget(
            name=paper_name,
            source_files=source_files,
            output_path=output_path,
            build_config=build_config
        )

        self.add_build_target(target)
        return target

    def build_target(self, target_name: str, force: bool = False) -> BuildResult:
        """
        Build a specific target.

        Args:
            target_name: Name of target to build
            force: Force rebuild even if not needed

        Returns:
            BuildResult for the target
        """
        if target_name not in self.targets:
            return BuildResult(
                target_name=target_name,
                success=False,
                error_message=f"Target '{target_name}' not found"
            )

        target = self.targets[target_name]
        start_time = time.time()

        try:
            logger.info(f"Starting build for target: {target_name}")

            # Check if rebuild is needed
            if not force and not target.needs_rebuild:
                logger.info(f"Target {target_name} is up to date, skipping build")
                return BuildResult(
                    target_name=target_name,
                    success=True,
                    output_path=target.output_path,
                    build_time_seconds=time.time() - start_time
                )

            # Validate build readiness
            if self.config.enable_validation:
                validation_result = self._validate_target(target)
                if not validation_result['passed']:
                    return BuildResult(
                        target_name=target_name,
                        success=False,
                        error_message="Validation failed: " + "; ".join(validation_result['errors']),
                        build_time_seconds=time.time() - start_time
                    )

            # Preprocess files
            processed_files = self._preprocess_target_files(target)

            # Generate PDF
            pdf_result = self._generate_pdf(target, processed_files)

            if pdf_result.success:
                # Update build hash
                target.update_build_hash()

                # Generate build report
                if self.config.generate_reports:
                    self._generate_build_report(target, pdf_result)

                logger.info(f"Successfully built target: {target_name}")

                pdf_result.build_time_seconds = time.time() - start_time
                return pdf_result
            else:
                logger.error(f"Failed to build target {target_name}: {pdf_result.error_message}")
                return BuildResult(
                    target_name=target_name,
                    success=False,
                    error_message=pdf_result.error_message,
                    build_time_seconds=time.time() - start_time
                )

        except Exception as e:
            logger.error(f"Build failed for target {target_name}: {str(e)}")
            return BuildResult(
                target_name=target_name,
                success=False,
                error_message=str(e),
                build_time_seconds=time.time() - start_time
            )

    def build_all(self, force: bool = False, parallel: bool = True) -> Dict[str, BuildResult]:
        """
        Build all registered targets.

        Args:
            force: Force rebuild of all targets
            parallel: Enable parallel building

        Returns:
            Dictionary mapping target names to build results
        """
        logger.info(f"Starting build of {len(self.targets)} targets")

        if parallel and self.config.enable_parallel and len(self.targets) > 1:
            return self._build_parallel(force)
        else:
            return self._build_sequential(force)

    def _build_parallel(self, force: bool) -> Dict[str, BuildResult]:
        """Build targets in parallel."""
        results = {}

        # Resolve dependencies and create build order
        build_order = self._resolve_dependencies()

        # Build targets with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all build tasks
            future_to_target = {
                executor.submit(self.build_target, target_name, force): target_name
                for target_name in build_order
            }

            # Collect results as they complete
            for future in as_completed(future_to_target):
                target_name = future_to_target[future]
                try:
                    result = future.result()
                    results[target_name] = result

                    if result.success:
                        logger.info(f"✅ Completed: {target_name}")
                    else:
                        logger.error(f"❌ Failed: {target_name} - {result.error_message}")

                except Exception as e:
                    logger.error(f"Exception building {target_name}: {str(e)}")
                    results[target_name] = BuildResult(
                        target_name=target_name,
                        success=False,
                        error_message=str(e)
                    )

        return results

    def _build_sequential(self, force: bool) -> Dict[str, BuildResult]:
        """Build targets sequentially."""
        results = {}
        build_order = self._resolve_dependencies()

        for target_name in build_order:
            result = self.build_target(target_name, force)
            results[target_name] = result

            if result.success:
                logger.info(f"✅ Completed: {target_name}")
            else:
                logger.error(f"❌ Failed: {target_name} - {result.error_message}")

        return results

    def _resolve_dependencies(self) -> List[str]:
        """Resolve build dependencies and return build order."""
        # Simple topological sort for dependency resolution
        # This is a simplified implementation - a full implementation would handle cycles

        processed = set()
        result = []

        def process_target(target_name: str):
            if target_name in processed:
                return

            target = self.targets[target_name]

            # Process dependencies first
            for dep in target.dependencies:
                if dep in self.targets:
                    process_target(dep)

            processed.add(target_name)
            result.append(target_name)

        # Process all targets
        for target_name in self.targets:
            process_target(target_name)

        return result

    def _validate_target(self, target: BuildTarget) -> Dict[str, Any]:
        """Validate a build target before building."""
        errors = []
        warnings = []

        # Check source files exist
        for source_file in target.source_files:
            if not source_file.exists():
                errors.append(f"Source file not found: {source_file}")

        if not target.source_files:
            errors.append("No source files specified")

        # Validate output directory is writable
        output_dir = target.output_path.parent
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create output directory: {str(e)}")

        # Run quality validation if files exist
        existing_files = [f for f in target.source_files if f.exists()]
        if existing_files and self.config.enable_validation:
            try:
                validation_report = validate_build_readiness(existing_files)
                if validation_report.errors > 0:
                    errors.append(f"Quality validation failed: {validation_report.errors} errors")

                if validation_report.warnings > 0:
                    warnings.append(f"Quality validation warnings: {validation_report.warnings}")

            except Exception as e:
                warnings.append(f"Quality validation error: {str(e)}")

        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

    def _preprocess_target_files(self, target: BuildTarget) -> List[Path]:
        """Preprocess target files for building."""
        processed_files = []

        for source_file in target.source_files:
            if source_file.exists():
                # For now, just copy files to temp directory
                # In a full implementation, this would include:
                # - Mermaid diagram processing
                # - Cross-reference validation
                # - Unicode symbol conversion
                # - Template processing

                temp_file = self.config.temp_dir / f"processed_{source_file.name}"
                shutil.copy2(source_file, temp_file)
                processed_files.append(temp_file)

        return processed_files

    def _generate_pdf(self, target: BuildTarget, processed_files: List[Path]) -> BuildResult:
        """Generate PDF for a target."""
        try:
            # Extract metadata from build config
            metadata = target.build_config.get('metadata', {})

            # Generate PDF
            result = self.pdf_generator.generate_pdf(
                processed_files,
                target.output_path,
                metadata
            )

            return result

        except Exception as e:
            return BuildResult(
                target_name=target.name,
                success=False,
                error_message=f"PDF generation failed: {str(e)}"
            )

    def _generate_build_report(self, target: BuildTarget, result: BuildResult):
        """Generate build report for a target."""
        report_data = {
            'target_name': target.name,
            'build_time': datetime.now().isoformat(),
            'success': result.success,
            'output_path': str(result.output_path) if result.output_path else None,
            'error_message': result.error_message,
            'build_time_seconds': result.build_time_seconds,
            'source_files': [str(f) for f in target.source_files],
            'warnings': result.warnings,
            'metadata': result.metadata
        }

        report_file = self.config.build_dir / "reports" / f"{target.name}_build_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

    def get_build_status(self) -> Dict[str, Any]:
        """Get overall build status and statistics."""
        total_targets = len(self.targets)
        built_targets = len([t for t in self.targets.values() if t.last_build_hash is not None])
        successful_builds = len([r for r in self.build_results.values() if r.success])

        return {
            'total_targets': total_targets,
            'built_targets': built_targets,
            'successful_builds': successful_builds,
            'failed_builds': len(self.build_results) - successful_builds,
            'build_completion_rate': built_targets / total_targets if total_targets > 0 else 0,
            'success_rate': successful_builds / len(self.build_results) if self.build_results else 0
        }

    def clean_build_artifacts(self, target_name: Optional[str] = None):
        """Clean build artifacts for specified target or all targets."""
        if target_name:
            if target_name in self.targets:
                target = self.targets[target_name]
                if target.output_path.exists():
                    target.output_path.unlink()
                target.last_build_hash = None
                logger.info(f"Cleaned artifacts for target: {target_name}")
        else:
            # Clean all artifacts
            for target in self.targets.values():
                if target.output_path.exists():
                    target.output_path.unlink()
                target.last_build_hash = None

            # Clean temp directory
            if self.config.clean_temp_files:
                shutil.rmtree(self.config.temp_dir, ignore_errors=True)
                self.config.temp_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Cleaned all build artifacts")

    def load_build_state(self, state_file: Path):
        """Load build state from file."""
        if state_file.exists():
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)

                # Restore target build hashes
                for target_name, target_data in state_data.get('targets', {}).items():
                    if target_name in self.targets:
                        self.targets[target_name].last_build_hash = target_data.get('last_build_hash')

                logger.info(f"Loaded build state from {state_file}")

            except Exception as e:
                logger.warning(f"Failed to load build state: {str(e)}")

    def save_build_state(self, state_file: Path):
        """Save build state to file."""
        state_data = {
            'timestamp': datetime.now().isoformat(),
            'targets': {}
        }

        for name, target in self.targets.items():
            state_data['targets'][name] = {
                'last_build_hash': target.last_build_hash,
                'build_status': target.build_status
            }

        state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved build state to {state_file}")


# Convenience functions for common build operations
def create_paper_build_target(paper_name: str, paper_dir: Path,
                            output_dir: Optional[Path] = None) -> BuildTarget:
    """
    Create a build target for a paper.

    Args:
        paper_name: Name of the paper
        paper_dir: Directory containing paper files
        output_dir: Output directory for built artifacts

    Returns:
        Configured BuildTarget
    """
    orchestrator = BuildOrchestrator()
    return orchestrator.create_paper_target(paper_name, paper_dir, output_dir)


def build_paper(paper_name: str, paper_dir: Path,
               output_dir: Optional[Path] = None,
               force: bool = False) -> BuildResult:
    """
    Build a single paper.

    Args:
        paper_name: Name of the paper to build
        paper_dir: Directory containing paper files
        output_dir: Output directory for built artifacts
        force: Force rebuild even if up to date

    Returns:
        BuildResult for the paper
    """
    orchestrator = BuildOrchestrator()
    target = orchestrator.create_paper_target(paper_name, paper_dir, output_dir)
    return orchestrator.build_target(paper_name, force)


def validate_build_environment() -> Dict[str, Any]:
    """
    Validate that the build environment is properly configured.

    Returns:
        Validation results for build environment
    """
    # Check PDF generation environment
    pdf_validation = validate_pdf_environment()

    # Check for required Python packages
    python_packages = ['matplotlib', 'numpy', 'scipy', 'pandas', 'pyyaml', 'jinja2']
    missing_packages = []

    for package in python_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    # Check for build tools
    build_tools = {
        'pandoc': 'pandoc --version',
        'latex': 'xelatex --version',
        'mmdc': 'mmdc --version'  # Mermaid CLI
    }

    tool_status = {}
    for tool, command in build_tools.items():
        try:
            import subprocess
            result = subprocess.run(command.split(), capture_output=True, text=True, timeout=5)
            tool_status[tool] = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            tool_status[tool] = False

    return {
        'pdf_environment': pdf_validation,
        'python_packages': {
            'available': [p for p in python_packages if p not in missing_packages],
            'missing': missing_packages
        },
        'build_tools': tool_status,
        'environment_ready': (
            pdf_validation.get('all_valid', False) and
            not missing_packages and
            all(tool_status.values())
        )
    }
