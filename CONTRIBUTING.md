# 🤝 Contributing to Ant Stack

**Guidelines for contributing to the Ant Stack modular scientific publication system.**

---

## 📋 Table of Contents

- [🎯 Mission & Values](#-mission--values)
- [🚀 Development Workflow](#-development-workflow)
- [📝 Code Standards](#-code-standards)
- [🧪 Testing Requirements](#-testing-requirements)
- [📚 Documentation](#-documentation)
- [🔄 Pull Request Process](#-pull-request-process)
- [🐛 Issue Reporting](#-issue-reporting)
- [💬 Communication](#-communication)
- [🏆 Recognition](#-recognition)

---

## 🎯 Mission & Values

### Our Mission

The Ant Stack project advances **reproducible, modular scientific publications** in embodied AI through:

- **🔄 Reproducibility**: Deterministic results with comprehensive validation
- **🧪 Scientific Rigor**: Bootstrap confidence intervals and uncertainty quantification
- **⚡ Performance**: Optimized algorithms with comprehensive benchmarking
- **📚 Modularity**: Reusable components across multiple research domains

### Core Values

- **✅ Professional Excellence**: Code that meets publication standards
- **🔬 Scientific Integrity**: Real data analysis, no mock methods
- **📖 Documentation First**: Comprehensive, clear documentation
- **🧪 Test-Driven Development**: 70%+ test coverage, edge case validation
- **🔧 Maintainability**: Modular, well-documented, clearly reasoned code

---

## 🚀 Development Workflow

### 1. Environment Setup

**Prerequisites:**
```bash
# System requirements
Python 3.8+          # Core language
Node.js 14+          # Mermaid diagram processing
LaTeX distribution   # PDF generation (TeX Live, MacTeX)
Pandoc 2.10+         # Document processing

# Python packages
pip install matplotlib numpy pandas pyyaml pytest scipy

# Node packages
npm install -g mermaid-filter
```

**Repository Setup:**
```bash
# Clone repository
git clone https://github.com/your-repo/ant.git
cd ant

# Install in development mode
pip install -e .

# Verify installation
python -c "import antstack_core; print('✅ Ant Stack installed successfully')"
```

### 2. Development Process

**Feature Development:**
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Implement changes following coding standards
# Add comprehensive tests
# Update documentation

# Run validation
python3 scripts/common_pipeline/build_core.py --validate-only

# Run test suite
python -m pytest tests/ -v

# Build papers to ensure no regressions
python3 scripts/common_pipeline/build_core.py
```

### 3. Pre-Commit Checklist

- [ ] **Code Quality**: Passes all linting and type checks
- [ ] **Tests**: All tests pass with 70%+ coverage
- [ ] **Documentation**: Updated relevant documentation
- [ ] **Validation**: Passes build validation
- [ ] **Cross-References**: No broken references in papers
- [ ] **Style**: Follows repository conventions

---

## 📝 Code Standards

### Python Standards

**Import Organization:**
```python
# Standard library imports
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import yaml

# Local imports
from antstack_core.analysis.energy import EnergyCoefficients
from antstack_core.analysis.statistics import bootstrap_mean_ci
```

**Function Documentation:**
```python
def calculate_energy_scaling(
    workload_data: Dict[str, Any],
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """Calculate energy scaling relationships with uncertainty quantification.

    This function performs comprehensive scaling analysis including:
    - Power-law relationship fitting
    - Bootstrap confidence intervals
    - Cross-validation of scaling exponents
    - Theoretical limit comparisons

    Args:
        workload_data: Dictionary containing computational workload parameters
        confidence_level: Confidence level for uncertainty quantification (0.0 to 1.0)

    Returns:
        Dictionary containing:
        - scaling_exponent: Fitted power-law exponent
        - confidence_interval: Uncertainty bounds
        - goodness_of_fit: R² correlation coefficient
        - theoretical_bounds: Physical constraint validation

    Example:
        >>> workload_data = {'flops': [1e6, 1e7, 1e8], 'energy': [1e-6, 8e-6, 64e-6]}
        >>> result = calculate_energy_scaling(workload_data)
        >>> print(f"Scaling exponent: {result['scaling_exponent']:.3f}")
        Scaling exponent: 0.833

    Note:
        Results validated against physical constraints and theoretical limits.
        See `docs/scientific_validation.md` for validation methodology.
    """
```

**Class Documentation:**
```python
class EnhancedEnergyEstimator:
    """Enhanced energy estimator with comprehensive analysis capabilities.

    Provides detailed energy estimation, scaling analysis, and theoretical
    limit comparisons for all Ant Stack modules (AntBody, AntBrain, AntMind).

    Attributes:
        coefficients (EnergyCoefficients): Energy coefficients for target hardware
        analysis_results (Dict[str, Any]): Cached analysis results
        validation_metrics (Dict[str, float]): Scientific validation metrics

    Methods:
        analyze_body_scaling: Analyze AntBody energy scaling
        analyze_brain_scaling: Analyze AntBrain computational scaling
        analyze_mind_scaling: Analyze AntMind planning complexity
        validate_results: Cross-validate against theoretical limits
    """

    def __init__(self, coefficients: EnergyCoefficients):
        """Initialize estimator with energy coefficients.

        Args:
            coefficients: EnergyCoefficients instance with device parameters
        """
```

### Testing Standards

**Test Structure:**
```python
import pytest
import numpy as np
from antstack_core.analysis.energy import EnergyCoefficients, ComputeLoad

class TestEnergyScaling:
    """Test suite for energy scaling analysis."""

    def setup_method(self):
        """Set up test fixtures."""
        self.coefficients = EnergyCoefficients()
        self.test_data = {
            'x_values': np.array([1, 2, 4, 8, 16, 32]),
            'y_values': np.array([1.0, 2.1, 4.3, 8.7, 17.5, 35.2])
        }

    def test_scaling_relationship(self):
        """Test power-law scaling relationship detection."""
        from antstack_core.analysis.statistics import analyze_scaling_relationship

        result = analyze_scaling_relationship(
            self.test_data['x_values'],
            self.test_data['y_values']
        )

        assert result['scaling_exponent'] > 0
        assert result['r_squared'] > 0.95
        assert 'confidence_interval' in result

    def test_bootstrap_confidence_intervals(self):
        """Test bootstrap confidence interval calculation."""
        from antstack_core.analysis.statistics import bootstrap_mean_ci

        data = [1.2, 1.5, 1.3, 1.8, 1.4, 1.6, 1.7]
        mean, lower, upper = bootstrap_mean_ci(data, n_bootstrap=1000)

        assert lower < mean < upper
        assert upper - lower < 1.0  # Reasonable precision

    @pytest.mark.parametrize("sparsity", [0.01, 0.05, 0.1, 0.2])
    def test_sparsity_parameterization(self, sparsity):
        """Test energy scaling across different sparsity levels."""
        # Test implementation with parameterization
        assert sparsity >= 0
        assert sparsity <= 1
```

### Git Commit Standards

**Commit Message Format:**
```bash
feat: add energy scaling analysis for neural networks

- Add comprehensive scaling analysis function
- Include bootstrap confidence intervals
- Add theoretical limit validation
- Update documentation and examples

Closes #123
```

**Commit Types:**
- `feat:` New feature implementation
- `fix:` Bug fix
- `docs:` Documentation updates
- `test:` Test additions or modifications
- `refactor:` Code restructuring without functionality changes
- `perf:` Performance improvements
- `ci:` Continuous integration changes
- `chore:` Maintenance tasks

---

## 🧪 Testing Requirements

### Test Coverage Requirements

**Minimum Coverage:**
- **Core modules**: 80%+ coverage
- **Analysis methods**: 90%+ coverage
- **Integration tests**: End-to-end validation
- **Performance tests**: Benchmark validation

**Test Categories:**
1. **Unit Tests**: Individual component validation
2. **Integration Tests**: Cross-module functionality
3. **Workflow Tests**: End-to-end pipeline validation
4. **Performance Tests**: Computational benchmarking
5. **Scientific Tests**: Accuracy and reproducibility validation

### Test Implementation

**Scientific Validation Tests:**
```python
def test_reproducibility():
    """Ensure deterministic behavior with fixed seeds."""
    from antstack_core.analysis.statistics import bootstrap_mean_ci

    data = [1.2, 1.5, 1.3, 1.8, 1.4, 1.6, 1.7, 1.9]

    # Run multiple times with same seed
    results = []
    for _ in range(5):
        mean, lower, upper = bootstrap_mean_ci(
            data, n_bootstrap=1000, random_seed=42
        )
        results.append((mean, lower, upper))

    # All results should be identical
    assert all(r == results[0] for r in results[1:])

def test_theoretical_limits():
    """Validate against physical constraints."""
    from antstack_core.analysis.energy import EnergyCoefficients

    # Landauer limit: ~1.4e-21 J/bit at room temperature
    landauer_limit = 1.4e-21

    # Our energy per bit should be above theoretical minimum
    coeffs = EnergyCoefficients()
    energy_per_bit = coeffs.flops_pj / 8  # Rough conversion

    assert energy_per_bit >= landauer_limit
```

---

## 📚 Documentation

### Documentation Standards

**Function Documentation:**
- Complete docstrings with Args, Returns, Examples
- Type hints for all parameters and return values
- Usage examples with realistic data
- References to scientific literature when applicable

**Module Documentation:**
- Overview of module purpose and scope
- Key classes and functions
- Usage patterns and best practices
- Integration with other modules

**Paper Documentation:**
- Clear mathematical notation with LaTeX macros
- Proper figure formatting with captions and IDs
- Descriptive hyperlinks using `\href{URL}{text}` format
- Cross-references using `\ref{type:id}` format

### Documentation Updates

**When to Update Documentation:**
- New feature implementation
- API changes or additions
- Bug fixes with user-visible impact
- Performance improvements
- Configuration changes

---

## 🔄 Pull Request Process

### 1. Pre-Submission

**Code Quality:**
```bash
# Run all tests
python -m pytest tests/ -v

# Check coverage
python -m pytest --cov=antstack_core --cov-report=html

# Validate papers
python3 scripts/common_pipeline/build_core.py --validate-only

# Check formatting
python -m black antstack_core/
python -m isort antstack_core/
python -m flake8 antstack_core/
```

**Documentation:**
- [ ] Updated docstrings for new/changed functions
- [ ] Added examples and usage patterns
- [ ] Updated relevant documentation files
- [ ] Cross-references validated

### 2. Pull Request Template

```markdown
## Description

Brief description of changes and their purpose.

## Changes Made

- **Feature**: Detailed description of new functionality
- **Fix**: Description of bug fixes
- **Documentation**: Documentation updates
- **Tests**: New test cases added

## Validation

- [ ] All tests pass
- [ ] Build validation passes
- [ ] Documentation updated
- [ ] Cross-references validated

## Related Issues

Closes #123, #124

## Testing

- Unit tests added for new functionality
- Integration tests for cross-module features
- Performance benchmarks updated
- Scientific validation tests included
```

### 3. Review Process

**Required Reviews:**
- **Functionality**: Code works as intended
- **Standards**: Follows coding conventions
- **Tests**: Adequate test coverage
- **Documentation**: Complete and accurate
- **Validation**: Passes all validation checks

**Approval Criteria:**
- ✅ All automated checks pass
- ✅ Code review feedback addressed
- ✅ Tests demonstrate functionality
- ✅ Documentation is complete
- ✅ No breaking changes (or documented)

---

## 🐛 Issue Reporting

### Bug Reports

**Required Information:**
```markdown
## Bug Description
Clear description of the bug and expected vs actual behavior.

## Reproduction Steps
1. Step 1: ...
2. Step 2: ...
3. Step 3: ...

## Environment
- OS: [e.g., macOS 12.4]
- Python: [e.g., 3.9.7]
- Dependencies: [e.g., numpy 1.21.2]

## Error Messages
```
Include full error messages and stack traces
```

## Expected Behavior
What should happen instead.

## Additional Context
Any other relevant information.
```

### Feature Requests

**Feature Request Template:**
```markdown
## Feature Description
Detailed description of the proposed feature.

## Use Case
How would this feature be used? Include examples.

## Implementation Ideas
Any suggestions for implementation approach.

## Benefits
Why would this feature be valuable?

## Alternatives Considered
Other approaches and why they were rejected.
```

---

## 💬 Communication

### Communication Channels

**Primary Channels:**
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Technical discussions and Q&A
- **Pull Requests**: Code contributions and reviews

**Community Guidelines:**
- Be respectful and professional
- Provide constructive feedback
- Include evidence for technical claims
- Use appropriate channels for different topics

### Response Times

**Issue Response:**
- **Bug Reports**: Within 48 hours
- **Feature Requests**: Within 1 week
- **Questions**: Within 24 hours

**Pull Request Review:**
- **Initial Review**: Within 48 hours
- **Follow-up Reviews**: Within 24 hours of updates

---

## 🏆 Recognition

### Contributor Recognition

**Recognition Methods:**
- **Contributors List**: Updated in README.md
- **Changelog**: Feature contributions documented
- **Publications**: Contributors acknowledged in papers
- **Presentations**: Credit given in conference talks

**Contribution Types:**
1. **Code Contributions**: Core functionality and bug fixes
2. **Documentation**: Improving guides and examples
3. **Testing**: Comprehensive test coverage
4. **Community Support**: Helping other users
5. **Research Contributions**: Scientific insights and validation

### Hall of Fame

Contributors who have made significant impact:

| Contributor | Contributions | Impact |
|-------------|---------------|---------|
| **Core Team** | Architecture design, core modules | Foundation |
| **Contributors** | Feature implementations, bug fixes | Enhancements |
| **Community** | Documentation, testing, feedback | Improvements |

---

## 📋 Summary

**Contributing to Ant Stack means:**

1. **🔬 Scientific Excellence**: Maintain high standards of scientific rigor
2. **📚 Documentation First**: Comprehensive, clear documentation
3. **🧪 Test-Driven Development**: Thorough testing and validation
4. **🔧 Professional Code**: Well-structured, maintainable implementations
5. **🤝 Collaborative Spirit**: Respectful, constructive engagement

**Thank you for contributing to advance reproducible scientific publications in embodied AI!**

---

*For questions or clarification, please open a GitHub Discussion or Issue.*
