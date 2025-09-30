# Common Pipeline Scripts

Shared infrastructure and build system scripts used across all papers and analyses.

## Purpose

This directory contains the core build system, validation tools, and infrastructure scripts that support the entire Ant Stack research pipeline, including:

- Modular paper building system
- Cross-reference validation and repair
- Citation and formatting tools
- Test suite execution
- Build automation and deployment

## Scripts

### Build System
- `build_core.py` - Modular paper builder with validation and testing
- `build_papers.sh` - Legacy build script for compatibility
- `unified_build.py` - Unified build system for all papers

### Validation and Testing
- `run_validation_suite.py` - Comprehensive test suite execution
- `test_ce_runner.py` - Test runner for complexity-energetics components

### Text Processing and Formatting
- `comprehensive_formatting_fix.py` - Automated document formatting
- Cross-reference validation and repair is handled automatically by the build system
- `diagnose_crossref_issue.py` - Diagnostic tools for cross-reference issues

### Pipeline Utilities
- `trace_pipeline.py` - Pipeline execution tracing and debugging

## Key Features

- **Modular Build System**: Supports multiple papers with shared infrastructure
- **Automated Validation**: Cross-reference checking, citation validation, format compliance
- **Comprehensive Testing**: Unit tests, integration tests, performance benchmarks
- **Build Optimization**: Parallel processing, caching, incremental builds
- **Error Recovery**: Automated repair tools for common issues

## Usage

### Building Papers
```bash
# Build specific paper with validation
python scripts/common_pipeline/build_core.py --paper complexity_energetics

# Build all papers
python scripts/common_pipeline/build_core.py --all

# Validation only mode
python scripts/common_pipeline/build_core.py --validate-only
```

### Validation and Repair
```bash
# Fix cross-references automatically
python3 scripts/common_pipeline/build_core.py --validate-only

# Run comprehensive validation
python scripts/common_pipeline/run_validation_suite.py

# Format documents
python scripts/common_pipeline/comprehensive_formatting_fix.py
```

### Testing
```bash
# Run all tests
python scripts/common_pipeline/run_validation_suite.py --test

# Run specific test suite
python scripts/common_pipeline/test_ce_runner.py
```

## Dependencies

- `antstack_core` package
- pandoc for document processing
- LaTeX distribution (xelatex) for PDF generation
- Standard Python scientific libraries
- pytest for testing framework

## Configuration

Build system uses YAML configuration files in each paper directory:
- `paper_config.yaml` - Paper-specific settings
- `manifest.example.yaml` - Analysis parameters and energy coefficients

## Output

- PDF documents in paper directories
- Validation reports and error logs
- Test results and coverage reports
- Build artifacts and intermediate files
