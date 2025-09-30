# Ant Stack Test Suite

Comprehensive test suite for the Ant Stack research framework, organized by component and functionality.

## Directory Structure

### 📁 `ant_stack/`
Tests for Ant Stack framework analysis and development:
- Colony behavior modeling validation
- Multi-agent coordination algorithms
- Stigmergic communication mechanisms
- Swarm intelligence pattern analysis

*Currently empty - add Ant Stack specific tests here*

### 📁 `complexity_energetics/`
Tests for computational complexity and energy analysis:
- Algorithmic complexity characterization
- Energy consumption modeling validation
- Scaling relationship analysis
- Theoretical limit calculations
- Orchestration script integration

**Test Files:**
- `test_ce.py` - Basic complexity energetics functionality
- `test_enhanced_ce.py` - Enhanced energy estimation and scaling analysis
- `test_orchestrators.py` - Orchestration script integration
- `test_integration_workflows.py` - Complete workflow integration

### 📁 `core_rendering/`
Tests for core rendering methods and system architecture:
- Figure generation and visualization systems
- Paper configuration and validation
- Core module integration and refactoring
- Build system and infrastructure components

**Test Files:**
- `test_core_refactor.py` - Core system refactoring and integration

## Test Organization Principles

### By Component
- **Ant Stack**: Colony-level behavior and multi-agent systems
- **Complexity Energetics**: Computational analysis and energy modeling
- **Core Rendering**: System architecture and visualization

### By Scope
- **Unit Tests**: Individual function and method validation
- **Integration Tests**: Cross-module functionality verification
- **Workflow Tests**: End-to-end pipeline validation
- **Orchestration Tests**: Script and tool integration

### By Methodology
- **Real Methods**: No mocks - actual computational methods
- **Statistical Validation**: Bootstrap confidence intervals
- **Performance Testing**: Scaling and efficiency analysis
- **Robustness Testing**: Edge cases and error conditions

## Running Tests

### All Tests
```bash
# Run complete test suite
python -m pytest tests/

# Run with coverage report
python -m pytest --cov=antstack_core --cov-report=html tests/

# Run with parallel execution
python -m pytest -n auto tests/
```

### By Component
```bash
# Complexity Energetics tests
python -m pytest tests/complexity_energetics/

# Core Rendering tests
python -m pytest tests/core_rendering/

# Ant Stack tests (when available)
python -m pytest tests/ant_stack/
```

### Specific Test Categories
```bash
# Energy modeling tests
python -m pytest -k "energy" tests/

# Scaling analysis tests
python -m pytest -k "scaling" tests/

# Integration workflow tests
python -m pytest -k "integration" tests/
```

### Development Mode
```bash
# Run with verbose output and stop on first failure
python -m pytest -v -x tests/

# Run specific test file
python -m pytest tests/complexity_energetics/test_enhanced_ce.py

# Run with debugging
python -m pytest --pdb tests/complexity_energetics/test_ce.py::test_estimator_zero_load
```

## Test Coverage Goals

- **Core Analysis**: 95%+ coverage of energy estimation and scaling analysis
- **Workload Models**: Complete coverage of computational workload calculations
- **Integration**: Full workflow coverage from data generation to visualization
- **Error Handling**: Comprehensive edge case and error condition testing

## Dependencies

### Required
- `antstack_core` package
- `pytest` testing framework
- Scientific Python libraries (numpy, scipy, matplotlib)

### Optional
- `pytest-cov` for coverage reporting
- `pytest-xdist` for parallel execution
- `pytest-html` for HTML reports

## Test Data

Tests use a combination of:
- **Synthetic Data**: Controlled test scenarios with known outcomes
- **Realistic Parameters**: Biologically-inspired parameter ranges
- **Edge Cases**: Boundary conditions and error scenarios
- **Statistical Validation**: Bootstrap sampling and confidence intervals

## Continuous Integration

Tests are optimized for CI environments:
- Headless plotting backends for figure generation tests
- Minimal external dependencies
- Fast execution (< 30 seconds for core tests)
- Comprehensive error reporting and debugging information

## Contributing

When adding new tests:

1. **Choose appropriate directory** based on functionality scope
2. **Follow naming conventions**:
   - `test_*.py` for test files
   - `test_*()` for test functions
   - Descriptive names indicating what is being tested

3. **Include comprehensive documentation**:
   - Clear docstrings explaining test purpose
   - Parameter descriptions and expected outcomes
   - References to requirements or specifications

4. **Test real functionality**:
   - No mocks - test actual computational methods
   - Include statistical validation where appropriate
   - Test edge cases and error conditions

5. **Follow testing best practices**:
   - One assertion per test when possible
   - Independent test execution
   - Fast execution for development workflow
   - Clear failure messages for debugging
