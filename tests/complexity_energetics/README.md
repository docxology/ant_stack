# Complexity & Energetics Tests

Comprehensive tests for computational complexity and energy analysis of embodied AI systems.

## Purpose

This directory contains tests for validating the complexity analysis and energy modeling components of the Ant Stack framework, ensuring:

- Algorithmic complexity characterizations are accurate
- Energy consumption models are validated
- Scaling relationship analysis is robust
- Theoretical limit calculations are correct

## Test Files

### `test_ce.py`
Basic complexity energetics functionality tests:
- Energy estimation and breakdown analysis
- Unit conversions and coefficient validation
- Workload scaling verification
- Cost of transport calculations

### `test_enhanced_ce.py`
Enhanced complexity energetics tests:
- Advanced energy estimation with detailed breakdowns
- Scaling relationship analysis and power law detection
- Realistic workload calculation methods
- Energy efficiency metrics and theoretical limits

### `test_orchestrators.py`
Orchestration script integration tests:
- Contact dynamics analysis orchestrator
- Neural network analysis orchestrator
- Active inference analysis orchestrator
- Output compatibility and parameter validation

### `test_integration_workflows.py`
Complete workflow integration tests:
- End-to-end analysis pipelines
- Comparative analysis across modules
- Theoretical limits validation
- Visualization generation verification

## Key Test Categories

### Energy Modeling Tests
- FLOP, memory, and spike energy calculations
- Hardware coefficient validation
- Energy breakdown accuracy
- Theoretical limit comparisons

### Complexity Analysis Tests
- Algorithmic complexity characterization
- Scaling relationship detection
- Workload modeling validation
- Performance bottleneck identification

### Statistical Validation Tests
- Bootstrap confidence interval calculation
- Power law regression analysis
- Goodness-of-fit metrics
- Uncertainty quantification

### Integration Tests
- Cross-module comparative analysis
- Orchestrator output validation
- Visualization pipeline verification
- End-to-end workflow testing

## Running Tests

```bash
# Run all complexity energetics tests
python -m pytest tests/complexity_energetics/

# Run with verbose output
python -m pytest -v tests/complexity_energetics/

# Run specific test file
python -m pytest tests/complexity_energetics/test_enhanced_ce.py

# Run with coverage
python -m pytest --cov=antstack_core.analysis tests/complexity_energetics/
```

## Dependencies

- `antstack_core.analysis` package
- Scientific Python libraries (numpy, scipy, matplotlib)
- pytest testing framework
- Optional: pandas for data manipulation
