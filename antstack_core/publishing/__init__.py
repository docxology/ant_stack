"""Scientific publication and PDF generation utilities.

This module provides comprehensive publication workflow management:
- Pandoc-based PDF generation with LaTeX integration
- Template system for consistent document formatting
- Cross-reference validation and figure management
- Quality assurance and validation pipelines
- Multi-paper build orchestration

Following .cursorrules specifications:
- Zero tolerance for broken references
- LaTeX macro conversion for Unicode symbols
- Professional document formatting
- Comprehensive validation reporting
"""

from .pdf_generator import (
    PDFGenerator,
    PDFGenerationConfig,
    BuildResult,
    generate_publication_pdf,
    validate_pdf_environment
)

from .template_engine import (
    TemplateEngine,
    TemplateConfig,
    render_scientific_document,
    create_publication_template
)

from .quality_validator import (
    QualityValidator,
    ValidationReport,
    ValidationIssue,
    validate_markdown_files,
    validate_build_readiness,
    generate_validation_summary
)

from .reference_manager import (
    ReferenceManager,
    ReferenceDefinition,
    ReferenceUsage,
    ReferenceReport,
    analyze_document_references,
    validate_citations,
    generate_reference_health_report
)

from .build_orchestrator import (
    BuildOrchestrator,
    BuildOrchestratorConfig,
    BuildTarget,
    create_paper_build_target,
    build_paper,
    validate_build_environment
)

__version__ = "1.0.0"
__author__ = "Daniel Ari Friedman"
__email__ = "daniel@activeinference.institute"

__all__ = [
    # PDF Generation
    "PDFGenerator",
    "PDFGenerationConfig",
    "BuildResult",
    "generate_publication_pdf",
    "validate_pdf_environment",

    # Template Engine
    "TemplateEngine",
    "TemplateConfig",
    "render_scientific_document",
    "create_publication_template",

    # Quality Validation
    "QualityValidator",
    "ValidationReport",
    "ValidationIssue",
    "validate_markdown_files",
    "validate_build_readiness",
    "generate_validation_summary",

    # Reference Management
    "ReferenceManager",
    "ReferenceDefinition",
    "ReferenceUsage",
    "ReferenceReport",
    "analyze_document_references",
    "validate_citations",
    "generate_reference_health_report",

    # Build Orchestration
    "BuildOrchestrator",
    "BuildOrchestratorConfig",
    "BuildTarget",
    "create_paper_build_target",
    "build_paper",
    "validate_build_environment"
]
