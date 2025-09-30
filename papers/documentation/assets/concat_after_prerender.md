# 🐜 Ant Stack: Modular Scientific Publication System

**A comprehensive framework for reproducible scientific publications in embodied AI, featuring reusable analysis methods, automated validation, and professional presentation standards.**

[![Test Status](https://img.shields.io/badge/tests-passing-brightgreen)](https://github.com/your-repo/ant/actions)
[![Coverage](https://img.shields.io/badge/coverage-70%25-blue)](https://github.com/your-repo/ant)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue)](https://www.python.org/)

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [🏗️ Architecture](#️-architecture)
- [📦 Core Package](#-core-package)
- [📄 Paper Structure](#-paper-structure)
- [🚀 Quick Start](#-quick-start)
- [🔧 Development](#-development)
- [📚 Documentation](#-documentation)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🎯 Overview

### Mission

Ant Stack provides a **modular, reproducible framework** for scientific publications in embodied AI, enabling researchers to:

- ✅ **Reuse validated analysis methods** across papers
- ✅ **Ensure reproducible results** through automated validation
- ✅ **Generate publication-quality figures** with consistent styling
- ✅ **Maintain scientific rigor** with statistical validation
- ✅ **Scale research workflows** with automated build pipelines

### Key Features

| Feature | Description |
|---------|-------------|
| **🔄 Reusability** | Modular analysis methods for energy estimation, statistics, and visualization |
| **📊 Quality Assurance** | Automated validation, cross-reference checking, and statistical verification |
| **🎨 Professional Output** | Publication-ready figures, LaTeX integration, and consistent formatting |
| **⚡ Performance** | Optimized algorithms with comprehensive benchmarking |
| **🔬 Scientific Rigor** | Bootstrap confidence intervals, uncertainty quantification, reproducibility |
| **🧪 Test-Driven** | 70%+ test coverage with comprehensive edge case testing |

### Applications

- **🤖 Embodied AI Research**: Energy analysis for robotic systems
- **🧠 Neuroscience**: Computational complexity of neural networks
- **⚡ Engineering**: Power optimization and scaling analysis
- **📈 Data Science**: Statistical validation and visualization

---

## 🏗️ Architecture

### System Components

![Computational architecture diagram (74d0)](papers/documentation/assets/mermaid/diagram_74d0e6c4.png){ width=70% }

---

## 📦 Core Package (`antstack_core/`)

### Analysis Module (`analysis/`)

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **`energy.py`** | Energy estimation and analysis | Physical modeling, efficiency calculations |
| **`statistics.py`** | Statistical methods and validation | Bootstrap CI, scaling relationships |
| **`workloads.py`** | Computational workload modeling | Body/brain/mind workload patterns |
| **`scaling_analysis.py`** | Scaling relationship analysis | Power laws, regime detection |
| **`enhanced_estimators.py`** | Advanced energy estimation | Multi-scale analysis, theoretical limits |
| **`experiment_config.py`** | Experiment configuration | YAML/JSON management, validation |

### Figures Module (`figures/`)

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **`plots.py`** | Publication-quality plotting | Matplotlib integration, styling |
| **`mermaid.py`** | Diagram preprocessing | Mermaid to PNG conversion |
| **`references.py`** | Cross-reference validation | Figure/table reference checking |
| **`assets.py`** | Asset management | File organization, optimization |

### Publishing Module (`publishing/`)

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **`pdf_generation.py`** | PDF generation utilities | Pandoc integration, LaTeX processing |
| **`templates.py`** | Document templates | Consistent formatting, styling |
| **`validation.py`** | Quality assurance | Automated checking, error detection |

---

## 📄 Paper Structure (`papers/`)

### Ant Stack Framework (`papers/ant_stack/`)

**Focus**: Biological framework for collective intelligence

| Section | File | Purpose |
|---------|------|---------|
| **📖 Introduction** | `Background.md` | Theoretical foundation |
| **🦿 Body Layer** | `AntBody.md` | Locomotion and sensing |
| **🧠 Brain Layer** | `AntBrain.md` | Neural processing and learning |
| **💭 Mind Layer** | `AntMind.md` | Decision making and planning |
| **🔧 Methods** | `Methods.md` | Implementation details |
| **📊 Results** | `Results.md` | Experimental validation |
| **💡 Applications** | `Applications.md` | Real-world use cases |
| **🗣️ Discussion** | `Discussion.md` | Implications and future work |

### Complexity Analysis (`papers/complexity_energetics/`)

**Focus**: Computational complexity and energy scaling

| Section | File | Purpose |
|---------|------|---------|
| **📖 Introduction** | `Background.md` | Problem statement |
| **🔬 Theory** | `Complexity.md` | Complexity analysis framework |
| | `Energetics.md` | Energy modeling approach |
| | `Scaling.md` | Scaling relationship theory |
| **🛠️ Methods** | `Methods.md` | Analysis methodology |
| **📊 Results** | `Generated.md` | Auto-generated analysis results |
| | `Results.md` | Interpretation and validation |
| **🗣️ Discussion** | `Discussion.md` | Scientific implications |

---

## 🚀 Quick Start

### Prerequisites

**System Requirements:**
- Python 3.8+
- Node.js 14+
- LaTeX distribution
- Pandoc 2.10+

### Installation

#### Ubuntu/Debian
```bash
# System dependencies
sudo apt-get update
sudo apt-get install -y pandoc texlive-xetex texlive-fonts-recommended fonts-dejavu nodejs npm

# Enhanced diagram rendering
sudo npm install -g mermaid-filter

# Python dependencies
pip3 install matplotlib numpy pandas pyyaml pytest scipy
```

#### macOS
```bash
# System dependencies
brew install pandoc node python3
brew install --cask mactex-no-gui

# Enhanced diagram rendering
npm install -g mermaid-filter

# Python dependencies
pip3 install matplotlib numpy pandas pyyaml pytest scipy
```

#### Development Setup
```bash
# Clone repository
git clone https://github.com/your-repo/ant.git
cd ant

# Install in development mode
pip install -e .

# Run tests
python -m pytest

# Build documentation
python scripts/build_docs.py
```

### Build Papers

#### Single Paper
```bash
# Ant Stack framework paper
python3 scripts/common_pipeline/build_core.py --paper ant_stack

# Complexity analysis paper
python3 scripts/common_pipeline/build_core.py --paper complexity_energetics
```

#### All Papers
```bash
# Build all papers
python3 scripts/common_pipeline/build_core.py

# With validation only
python3 scripts/common_pipeline/build_core.py --validate-only
```

### Basic Usage

```python
from antstack_core.analysis.energy import EnergyCoefficients, estimate_detailed_energy
from antstack_core.analysis.statistics import bootstrap_mean_ci

# Energy analysis example
coeffs = EnergyCoefficients()
workload = ComputeLoad(flops=1e9, memory_bytes=1e6)
energy = estimate_detailed_energy(workload, coeffs)

# Statistical validation
data = [1.2, 1.5, 1.3, 1.8, 1.4]
mean, ci_lower, ci_upper = bootstrap_mean_ci(data, n_bootstrap=1000)
```

---

## 🔧 Development

### Testing Strategy

**Test Coverage Goals:**
- **Core modules**: 80%+ coverage
- **Analysis methods**: 90%+ coverage
- **Edge cases**: Comprehensive coverage
- **Integration tests**: End-to-end validation

**Running Tests:**
```bash
# All tests
python -m pytest

# With coverage report
python -m pytest --cov=antstack_core --cov-report=html

# Specific module
python -m pytest tests/antstack_core/test_energy.py -v

# Performance benchmarks
python -m pytest tests/ --benchmark-only
```

### Code Quality Standards

**Linting and Formatting:**
```bash
# Run linters
python -m flake8 antstack_core/
python -m black antstack_core/
python -m isort antstack_core/

# Type checking
python -m mypy antstack_core/
```

**Pre-commit Hooks:**
```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Documentation

**Building Docs:**
```bash
# Generate API documentation
python scripts/generate_docs.py

# Build user guide
python scripts/build_user_guide.py

# Deploy to GitHub Pages
python scripts/deploy_docs.py
```

---

## 📚 Documentation

### User Guides

- **[Getting Started](docs/getting_started.md)**: Installation and basic usage
- **[API Reference](docs/api_reference.md)**: Complete method documentation
- **[Best Practices](docs/best_practices.md)**: Development guidelines
- **[Troubleshooting](docs/troubleshooting.md)**: Common issues and solutions

### Scientific Documentation

- **[Theoretical Foundation](docs/theory.md)**: Mathematical underpinnings
- **[Validation Framework](docs/validation.md)**: Quality assurance methods
- **[Benchmarking](docs/benchmarking.md)**: Performance analysis
- **[Reproducibility](docs/reproducibility.md)**: Ensuring scientific validity

### Developer Resources

- **[Contributing Guide](CONTRIBUTING.md)**: Development workflow
- **[Architecture Overview](docs/architecture.md)**: System design
- **[Testing Framework](docs/testing.md)**: Test development guide
- **[CI/CD Pipeline](docs/cicd.md)**: Build and deployment

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/your-feature`
3. **Write tests** for new functionality
4. **Implement** your changes
5. **Run tests**: `python -m pytest`
6. **Update documentation** if needed
7. **Submit** a pull request

### Code Review Process

- All PRs require review
- Tests must pass CI pipeline
- Documentation updates required for API changes
- Maintain backward compatibility

### Issue Reporting

- Use [GitHub Issues](https://github.com/your-repo/ant/issues) for bug reports
- Provide minimal reproducible examples
- Include system information and error traces
- Follow issue templates for consistency

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Scientific Contributors**: Domain experts in embodied AI and computational neuroscience
- **Open Source Community**: Libraries and tools that power this framework
- **Research Institutions**: Partners supporting reproducible science initiatives

---

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/your-repo/ant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/ant/discussions)
- **Email**: research@your-institution.edu

---

*Built with ❤️ for reproducible science in embodied AI*


# PDF Rendering System Guide: Reliable Paper Syntax

## Overview

This guide documents the comprehensive PDF rendering system used across all Ant Stack papers, providing reliable syntax patterns for consistent, professional scientific publications.

## System Architecture

### Core Components

1. **Modular Build System** (`scripts/common_pipeline/build_core.py`)
   - YAML-based paper configuration
   - Automatic paper discovery and validation
   - Cross-reference validation before PDF generation
   - Quality assurance reporting

2. **Legacy Render System** (`tools/render_pdf.sh`)
   - Pandoc + XeLaTeX pipeline
   - Mermaid diagram prerendering
   - Unicode math symbol handling
   - Cross-reference validation

3. **Core Package** (`antstack_core/`)
   - Figure generation and management
   - Cross-reference validation
   - Asset organization
   - Publication-ready formatting

## Paper Configuration System

### YAML Configuration Structure

Each paper requires a `paper_config.yaml` file with this structure:

```yaml
# Paper metadata
paper:
  name: "paper_name"
  title: "Paper Title"
  subtitle: "Subtitle"
  author: "Daniel Ari Friedman"
  email: "daniel@activeinference.institute"
  orcid: "0000-0001-6232-9096"
  output_filename: "N_paper_name.pdf"

# Content organization
content:
  files:
    - "Abstract.md"
    - "Introduction.md"
    # ... other files in build order

# Asset management
assets:
  figures_dir: "assets/figures"
  mermaid_dir: "assets/mermaid"
  tmp_dir: "assets/tmp_images"
  data_dir: "assets/data"

# Build configuration
build:
  has_generated_content: true
  has_computational_analysis: true
  mermaid_preprocessing: true
  cross_reference_validation: true

# LaTeX/Pandoc settings
latex:
  document_class: "article"
  geometry: "margin=2.5cm"
  mainfont: "Latin Modern Roman"
  mathfont: "Latin Modern Math"
  bibliography: "references.bib"

# Quality assurance
validation:
  check_cross_references: true
  check_figure_captions: true
  check_unicode_symbols: true
  require_descriptive_links: true
  validate_analysis_outputs: true
```

## Figure Management System

### Figure Format (CRITICAL)

**ALWAYS use this exact format:**

```markdown
## Figure: Descriptive Title {#fig:identifier}

![Alt text](assets/figures/your_figure_name.png){#fig:identifier}

**Caption:** Detailed description of the figure content, including units and key findings.
```

*Note: This is an example figure reference. Replace `assets/figures/your_figure_name.png` with your actual figure file path.*

**NEVER use:**
- `![caption](path){#fig:id}` (inline figure definitions)
- `\includegraphics` commands in markdown
- Missing figure IDs or captions

### Figure Naming Conventions

- **IDs**: Use descriptive, hierarchical names: `energy_by_workload`, `scaling_brain_K`, `response_time_comparison`
- **Files**: Store in `assets/figures/` with descriptive names
- **References**: Use `\ref{fig:identifier}` for cross-references

### Example Figure Definition

```markdown
## Figure: Energy Scaling Analysis {#fig:energy_scaling}

![Energy scaling with confidence intervals](assets/figures/your_figure_name.png){#fig:energy_scaling}

**Caption:** Energy consumption scaling with system complexity K, showing 95% confidence intervals. The power law relationship E $\propto$ K^$\alpha$ is evident with $\alpha$ $\approx$ 1.2.
```

*Note: This is an example figure reference. Replace `assets/figures/your_figure_name.png` with your actual figure file path.*

## Cross-Reference System

### Reference Types

1. **Figures**: `\ref{fig:identifier}` $\to$ "Figure 1"
2. **Equations**: `\ref{eq:identifier}` $\to$ "Equation (1)"
3. **Sections**: `\ref{sec:identifier}` $\to$ "Section 2.1"
4. **Tables**: `\ref{tab:identifier}` $\to$ "Table 3"

### Section IDs

Use descriptive IDs in section headers:

```markdown
# Introduction {#sec:introduction}

## Methodology {#sec:methodology}

### Data Collection {#sec:data_collection}
```

### Equation Format

```markdown
\begin{equation}
E = \sum_{i=1}^{n} \alpha_i \cdot K_i^{\beta_i}
\label{eq:energy_scaling}
\end{equation}
```

Reference with: `\ref{eq:energy_scaling}`

## Mathematics and Symbols

### LaTeX Math Syntax

**ALWAYS use LaTeX macros, NEVER Unicode symbols:**

```markdown
# Correct
$\mu$, $\lambda$, $\pi$, $\epsilon$, $\Delta$, $\rho$, $\sigma$

# Incorrect  
$\mu$, $\lambda$, $\pi$, $\epsilon$, $\Delta$, $\rho$, $\sigma$
```

### Common Symbol Mappings

| Symbol | LaTeX | Usage |
|--------|-------|-------|
| $\mu$ | `$\mu$` | Micrometers, mean |
| $\lambda$ | `$\lambda$` | Wavelength |
| $\pi$ | `$\pi$` | Pi constant |
| $\epsilon$ | `$\epsilon$` | Epsilon |
| $\Delta$ | `$\Delta$` | Delta, change |
| $\rho$ | `$\rho$` | Density, correlation |
| $\sigma$ | `$\sigma$` | Standard deviation |
| $\pm$ | `$\pm$` | Plus-minus |
| $\le$ | `$\le$` | Less than or equal |
| $\ge$ | `$\ge$` | Greater than or equal |
| $\approx$ | `$\approx$` | Approximately |
| $\propto$ | `$\propto$` | Proportional to |

### Math Environments

```markdown
# Inline math
The energy is $E = mc^2$ joules.

# Display math
$$E = \sum_{i=1}^{n} \alpha_i \cdot K_i^{\beta_i}$$

# Numbered equation
\begin{equation}
E = \sum_{i=1}^{n} \alpha_i \cdot K_i^{\beta_i}
\label{eq:energy_scaling}
\end{equation}
```

## Hyperlinks and References

### External Links

Use descriptive hyperlinks with `\href{URL}{descriptive text}`:

```markdown
# Correct
\href{https://arxiv.org/abs/2505.03764}{arXiv preprint}

# Incorrect
https://arxiv.org/abs/2505.03764
\url{https://arxiv.org/abs/2505.03764}
```

### Internal Cross-References

```markdown
# Section references
See \ref{sec:methodology} for details.

# Figure references  
As shown in \ref{fig:energy_scaling}, the relationship is clear.

# Equation references
From \ref{eq:energy_scaling}, we can derive...
```

## Mermaid Diagrams

### Prerendering System

Mermaid diagrams must be prerendered to local images:

1. **Source**: Store `.mmd` files in `assets/mermaid/`
2. **Rendered**: Convert to `.png` files in same directory
3. **Reference**: Use standard figure format

### Diagram Format

```markdown
## Figure: System Architecture {#fig:system_arch}

![System architecture diagram](assets/mermaid/your_diagram_name.png){#fig:system_arch}

**Caption:** High-level system architecture showing data flow between components.
```

*Note: This is an example figure reference. Replace `assets/mermaid/your_diagram_name.png` with your actual Mermaid diagram output.*

### Mermaid Best Practices

![Computational architecture diagram (40bc)](papers/documentation/assets/mermaid/diagram_40bcc56f.png){ width=70% }

## File Organization

### Directory Structure

```
papers/paper_name/
├── paper_config.yaml          # Paper configuration
├── Abstract.md                # Abstract
├── Introduction.md            # Introduction
├── Methodology.md             # Methods
├── Results.md                 # Results
├── Discussion.md              # Discussion
├── Conclusion.md              # Conclusion
├── References.md              # References
├── Appendices.md              # Appendices
├── assets/
│   ├── figures/              # Generated figures
│   ├── mermaid/              # Mermaid diagrams
│   ├── data/                 # Data files
│   └── tmp_images/           # Temporary images
└── references.bib            # Bibliography
```

### File Naming Conventions

- **Markdown files**: `PascalCase.md` (e.g., `Abstract.md`, `Introduction.md`)
- **Figure files**: `snake_case.png` (e.g., `energy_scaling.png`)
- **Data files**: `descriptive_name.json` (e.g., `analysis_results.json`)

## Build System Usage

### Command Line Interface

```bash
# Build all papers
python3 scripts/common_pipeline/build_core.py

# Build specific paper
python3 scripts/common_pipeline/build_core.py --paper paper_name

# Validate only (no PDF generation)
python3 scripts/common_pipeline/build_core.py --validate-only

# Skip tests
python3 scripts/common_pipeline/build_core.py --no-tests
```

### Legacy System

```bash
# Build all papers
bash tools/render_pdf.sh

# Build specific paper
bash tools/render_pdf.sh paper_name
```

## Quality Assurance

### Pre-Build Validation

The system automatically validates:

1. **Cross-references**: All `\ref{}` commands resolve to valid IDs
2. **Figure captions**: All figures have proper captions
3. **Math symbols**: Unicode symbols converted to LaTeX
4. **File structure**: All referenced files exist
5. **Configuration**: YAML syntax and required fields

### Post-Build Validation

1. **PDF quality**: File size, page count, figure count
2. **Broken references**: Detection of "Figure~??" patterns
3. **Mermaid rendering**: All diagrams successfully converted
4. **Cross-reference consistency**: Definitions match references

### Validation Reports

Build reports are generated in `build_report.md` with:

- Validation results summary
- Error details and fixes
- Quality metrics
- Performance statistics

## Test Suite Integration

### Test Categories

1. **Unit Tests**: Individual component validation
2. **Integration Tests**: Cross-module functionality
3. **Workflow Tests**: End-to-end pipeline validation
4. **Rendering Tests**: PDF generation and quality

### Running Tests

```bash
# All tests
python3 -m pytest tests/

# Specific component
python3 -m pytest tests/core_rendering/

# With coverage
python3 -m pytest --cov=antstack_core tests/
```

## Common Issues and Solutions

### Math Formatting Issues

**Problem**: `\mathrm allowed only in math mode`

**Solution**: Ensure all math is properly wrapped in `$...$` or `$$...$$`

```markdown
# Correct
$\mu\mathrm{m}$

# Incorrect
\(\mu\mathrm{m}\)
```

### Figure Reference Issues

**Problem**: `Figure~??` in PDF

**Solution**: Ensure figure IDs match exactly between definition and reference

```markdown
# Definition
## Figure: Title {#fig:my_figure}

# Reference
\ref{fig:my_figure}
```

### Cross-Reference Issues

**Problem**: Undefined references

**Solution**: Use `\ref{}` instead of `\Cref{}` unless cleveref is properly configured

```markdown
# Correct
\ref{fig:my_figure}

# May cause issues
\Cref{fig:my_figure}
```

## Best Practices Summary

### DO

- ✅ Use consistent figure format with IDs and captions
- ✅ Use LaTeX macros for all math symbols
- ✅ Use descriptive hyperlinks with `\href{}`
- ✅ Validate cross-references before building
- ✅ Use proper file organization
- ✅ Test with real data (no mocks)
- ✅ Generate comprehensive validation reports

### DON'T

- ❌ Use inline figure definitions
- ❌ Use Unicode math symbols in text
- ❌ Use naked URLs
- ❌ Skip cross-reference validation
- ❌ Use mock methods in tests
- ❌ Ignore build warnings

## Troubleshooting

### Build Failures

1. **Check math formatting**: Ensure all math is properly wrapped
2. **Validate cross-references**: Run `--validate-only` first
3. **Check file paths**: Ensure all referenced files exist
4. **Review YAML syntax**: Validate configuration files

### Quality Issues

1. **Broken references**: Check ID matching
2. **Missing figures**: Verify file paths and existence
3. **Math rendering**: Convert Unicode to LaTeX
4. **Cross-references**: Use consistent reference format

This guide ensures reliable, professional PDF generation across all Ant Stack papers with consistent formatting, proper cross-referencing, and comprehensive quality assurance.
