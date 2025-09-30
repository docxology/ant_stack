# AntStack Core Test Suite

Comprehensive test suite for the `antstack_core` package covering all modules and functionality.

## Test Organization

### Core Package Tests
- `test_core_package.py` - Package initialization and dependencies

### Analysis Module Tests
- `test_analysis_energy.py` - Energy estimation and coefficients
- `test_analysis_statistics.py` - Statistical methods and validation
- `test_analysis_workloads.py` - Workload modeling and complexity analysis
- `test_analysis_scaling.py` - Scaling analysis and power law detection
- `test_analysis_complexity.py` - Advanced complexity analysis methods
- `test_analysis_config.py` - Configuration management and experiment setup
- `test_analysis_limits.py` - Theoretical limits and efficiency analysis
- `test_analysis_reporting.py` - Veridical reporting and case studies

### CohereAnts Module Tests
- `test_cohereants_core.py` - Core infrared detection utilities
- `test_cohereants_behavioral.py` - Behavioral response analysis
- `test_cohereants_spectroscopy.py` - Spectral analysis and CHC identification
- `test_cohereants_sensilla.py` - Sensilla morphology and antenna analysis
- `test_cohereants_visualization.py` - Plotting and visualization utilities
- `test_cohereants_case_studies.py` - Specific case study implementations
- `test_cohereants_integrated.py` - Integrated analysis workflows

### Figures Module Tests
- `test_figures_plots.py` - Basic plotting functionality
- `test_figures_publication.py` - Publication-quality figure generation
- `test_figures_advanced.py` - Advanced visualization methods
- `test_figures_mermaid.py` - Mermaid diagram processing
- `test_figures_assets.py` - Asset management and organization
- `test_figures_references.py` - Cross-reference validation

### Publishing Module Tests
- `test_publishing_core.py` - Publishing utilities and templates

## Running Tests

```bash
# Run all antstack_core tests
python -m pytest tests/antstack_core/ -v

# Run specific module tests
python -m pytest tests/antstack_core/test_analysis_energy.py -v

# Run with coverage
python -m pytest tests/antstack_core/ --cov=antstack_core --cov-report=html

# Run through unified test runner
python tests/run_all_tests.py --component antstack_core
```

## Test Principles

Following the .cursorrules specifications:
- ✅ **Real methods only**: No mocks, comprehensive data analysis
- ✅ **Test-driven development**: Statistical validation of scientific methods
- ✅ **Comprehensive coverage**: All classes, methods, and edge cases
- ✅ **Professional documentation**: Detailed docstrings and scientific references
- ✅ **Error handling**: Robust validation and meaningful error messages
- ✅ **Statistical rigor**: Bootstrap confidence intervals, significance testing
- ✅ **Integration testing**: End-to-end workflows and data pipelines

## Coverage Goals

- **Analysis module**: 95%+ coverage across all scientific methods
- **CohereAnts module**: 90%+ coverage for infrared detection methods
- **Figures module**: 85%+ coverage for visualization utilities
- **Publishing module**: 80%+ coverage for publication tools

## Key Test Categories

1. **Unit Tests**: Individual functions and methods
2. **Integration Tests**: Cross-module functionality
3. **Scientific Validation**: Statistical correctness of algorithms
4. **Performance Tests**: Computational efficiency benchmarks
5. **Edge Case Tests**: Boundary conditions and error handling
6. **Regression Tests**: Previously identified bug fixes

## Dependencies

Test suite requires:
- `pytest` - Test framework
- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `matplotlib` - Plotting and visualization
- `pandas` - Data manipulation
- `yaml` - Configuration files
- `antstack_core` - Package under test
