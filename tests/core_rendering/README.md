# Core Rendering Tests

Tests for core rendering methods and system architecture components.

## Purpose

This directory contains tests for validating the core rendering and system architecture components of the Ant Stack framework, including:

- Figure generation and visualization systems
- Paper configuration and validation
- Core module integration and refactoring
- Build system and infrastructure components

## Test Files

### `test_core_refactor.py`
Core system refactoring and integration tests:
- Core analysis functionality validation
- Figure generation pipeline testing
- Energy estimation and scaling analysis
- Paper configuration loading and validation
- Module integration workflow testing

## Key Test Categories

### Figure Generation Tests
- Bar plot, line plot, and scatter plot generation
- Statistical overlay validation
- Color scheme and styling verification
- Output file format and quality checks

### Configuration Tests
- YAML configuration file parsing
- Paper metadata validation
- Build system configuration verification
- Parameter range and constraint checking

### Integration Tests
- Cross-module functionality validation
- Build pipeline integration testing
- Error handling and recovery mechanisms
- Performance and scalability validation

### Architecture Tests
- Module import and dependency validation
- Class and method interface testing
- Data structure integrity checks
- Memory and resource usage validation

## Running Tests

```bash
# Run all core rendering tests
python -m pytest tests/core_rendering/

# Run with verbose output
python -m pytest -v tests/core_rendering/

# Run with coverage
python -m pytest --cov=antstack_core tests/core_rendering/
```

## Dependencies

- `antstack_core` package (core and analysis modules)
- matplotlib for figure generation testing
- pytest testing framework
- PyYAML for configuration file testing
- Standard Python libraries

## Test Data

Tests use synthetic test data designed to:
- Cover edge cases and boundary conditions
- Validate statistical analysis methods
- Test visualization output quality
- Verify configuration parsing robustness

## Continuous Integration

These tests are designed to run in CI environments and include:
- Headless plotting backend configuration
- Minimal external dependencies
- Fast execution for rapid feedback
- Comprehensive error reporting
