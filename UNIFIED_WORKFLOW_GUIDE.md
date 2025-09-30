# Unified Workflow Guide: Ant Stack Scientific Publication System

## Overview

This guide provides a comprehensive, coherent workflow for operating the Ant Stack scientific publication system across all three papers. The system has been designed with modular architecture, unified style standards, and automated validation to ensure professional, reproducible scientific publications.

## System Architecture

### Core Components

1. **📦 Core Package** (`antstack_core/`): Reusable scientific methods
   - `analysis/`: Energy estimation, statistical methods, workload modeling
   - `figures/`: Publication-quality plotting, cross-reference validation
   - `publishing/`: PDF generation utilities, quality validation

2. **📄 Papers** (`papers/`): Three scientific publications
   - `ant_stack/`: Biological framework (AntBody, AntBrain, AntMind)
   - `complexity_energetics/`: Computational complexity and energy analysis
   - `cohereAnts/`: Infrared vibrational detection in insect olfaction

3. **🔧 Build System** (`scripts/common_pipeline/`): Unified validation and PDF generation
   - `build_core.py`: Primary modular build system
   - `run_validation_suite.py`: Comprehensive test execution

## Quick Start: Complete Workflow

### 1. Environment Setup

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y pandoc texlive-xetex texlive-fonts-recommended fonts-dejavu nodejs npm python3-pip
sudo npm install -g mermaid-filter
pip3 install matplotlib numpy pandas pyyaml pytest

# Install dependencies (macOS)
brew install pandoc node python3
brew install --cask mactex-no-gui
npm install -g mermaid-filter
pip3 install matplotlib numpy pandas pyyaml pytest
```

### 2. Run Complete Test Suite

```bash
# Run all tests with coverage
python3 -m pytest tests/ -v --cov=antstack_core --cov-report=html

# Run specific test categories
python3 -m pytest tests/core_rendering/ -v  # Core system tests
python3 -m pytest tests/complexity_energetics/ -v  # Analysis tests
```

### 3. Generate Analysis and Figures

```bash
# Generate complexity & energetics analysis
python3 -m papers.complexity_energetics.src.ce.runner papers/complexity_energetics/manifest.example.yaml --out papers/complexity_energetics/out

# Generate cohereAnts analysis
cd papers/cohereAnts
python3 scripts/run_all_case_studies.py
cd ../..
```

### 4. Build All Papers

```bash
# Build all papers with validation
python3 scripts/common_pipeline/build_core.py

# Build specific paper
python3 scripts/common_pipeline/build_core.py --paper complexity_energetics

# Validation only (no PDF generation)
python3 scripts/common_pipeline/build_core.py --validate-only
```

### 5. Fix Issues Automatically

```bash
# Fix cross-references
python3 scripts/common_pipeline/build_core.py --validate-only

# Fix math formatting
python3 scripts/common_pipeline/comprehensive_formatting_fix.py
```

## Paper-Specific Workflows

### Ant Stack Paper (Conceptual Framework)

**Purpose**: Biological framework for collective intelligence
**Generated Content**: None (static content)
**Key Files**: `AntBody.md`, `AntBrain.md`, `AntMind.md`

```bash
# Build only
python3 scripts/common_pipeline/build_core.py --paper ant_stack

# Validate cross-references
python3 scripts/common_pipeline/build_core.py --paper ant_stack --validate-only
```

### Complexity & Energetics Paper (Computational Analysis)

**Purpose**: Energy scaling and computational complexity analysis
**Generated Content**: Analysis results, figures, statistical data
**Key Files**: `Generated.md` (auto-generated), `Results.md`

```bash
# 1. Run analysis pipeline
python3 -m papers.complexity_energetics.src.ce.runner papers/complexity_energetics/manifest.example.yaml --out papers/complexity_energetics/out

# 2. Generate manuscript figures
python3 scripts/complexity_energetics/generate_manuscript_figures.py --output papers/complexity_energetics/assets

# 3. Build paper
python3 scripts/common_pipeline/build_core.py --paper complexity_energetics
```

### CohereAnts Paper (Experimental Analysis)

**Purpose**: Infrared vibrational detection in insect olfaction
**Generated Content**: Experimental results, case studies, analysis
**Key Files**: `Experimental_Results.md`, `Empirical_Studies.md`

```bash
# 1. Run case studies
cd papers/cohereAnts
python3 scripts/run_all_case_studies.py
cd ../..

# 2. Generate research figures
cd papers/cohereAnts
python3 scripts/generate_research_figures.py
cd ../..

# 3. Build paper
python3 scripts/common_pipeline/build_core.py --paper cohereAnts
```

## Style Standards and Formatting

### Mathematical Notation

**ALWAYS use LaTeX macros, NEVER Unicode symbols:**

```markdown
# Correct
$\mu$, $\lambda$, $\pi$, $\epsilon$, $\Delta$, $\rho$, $\sigma$

# Incorrect
μ, λ, π, ε, Δ, ρ, σ
```

### Figure Format (CRITICAL)

**ALWAYS use this exact format:**

```markdown
## Figure: Descriptive Title {#fig:identifier}

![Alt text](path/to/figure.png)

**Caption:** Detailed description with units and key findings.
```

**NEVER use:**
- `![caption](path){#fig:id}` (inline figure definitions)
- Missing figure IDs or captions

### Cross-References

```markdown
# Figure references
As shown in \ref{fig:energy_scaling}, the relationship is clear.

# Section references
See \ref{sec:methodology} for details.

# Equation references
From \ref{eq:energy_scaling}, we can derive...
```

### Hyperlinks

```markdown
# External links
\href{https://arxiv.org/abs/2505.03764}{arXiv preprint}

# Internal cross-references
Figure~\ref{fig:energy_scaling}
```

## Quality Assurance

### Pre-Build Validation

The system automatically validates:
- Cross-reference consistency
- Figure format compliance
- Math symbol formatting
- File structure integrity
- YAML configuration syntax

### Post-Build Validation

- PDF quality metrics (size, page count, figure count)
- Broken reference detection
- Mermaid diagram rendering
- Cross-reference consistency

### Automated Repair Tools

```bash
# Fix cross-references
python3 scripts/common_pipeline/build_core.py --validate-only --dry-run  # Preview
python3 scripts/common_pipeline/build_core.py --validate-only            # Apply

# Fix math formatting
python3 scripts/common_pipeline/comprehensive_formatting_fix.py

# Fix citations
# Citations are handled automatically by the build system
```

## Test Suite Integration

### Test Categories

1. **Unit Tests**: Individual component validation
2. **Integration Tests**: Cross-module functionality
3. **Workflow Tests**: End-to-end pipeline validation
4. **Rendering Tests**: PDF generation and quality

### Running Tests

```bash
# All tests
python3 -m pytest tests/ -v

# Specific components
python3 -m pytest tests/core_rendering/ -v
python3 -m pytest tests/complexity_energetics/ -v

# With coverage
python3 -m pytest --cov=antstack_core tests/
```

## Troubleshooting

### Common Issues

1. **Broken References (Figure~??)**
   ```bash
   python3 scripts/common_pipeline/build_core.py --validate-only
   ```

2. **Math Formatting Issues**
   ```bash
   python3 scripts/common_pipeline/comprehensive_formatting_fix.py
   ```

3. **Missing Dependencies**
   ```bash
   pip3 install matplotlib numpy pandas pyyaml pytest
   ```

4. **Build Failures**
   ```bash
   python3 scripts/common_pipeline/build_core.py --validate-only
   ```

### Validation Commands

```bash
# Check environment
python3 scripts/validate_rendering_system.py

# Validate specific paper
python3 scripts/common_pipeline/build_core.py --paper complexity_energetics --validate-only

# Run diagnostics
python3 scripts/common_pipeline/diagnose_crossref_issue.py
```

## File Organization

### Directory Structure

```
ant/
├── antstack_core/           # Core scientific methods
├── papers/                  # Paper content
│   ├── ant_stack/          # Biological framework
│   ├── complexity_energetics/  # Computational analysis
│   └── cohereAnts/         # Experimental analysis
├── scripts/                 # Build and analysis scripts
│   ├── common_pipeline/    # Shared build system
│   └── complexity_energetics/  # Analysis scripts
├── tests/                   # Test suite
└── tools/                   # Legacy build tools
```

### Paper Configuration

Each paper has a `paper_config.yaml` file with:
- Paper metadata and author information
- Content file organization
- Asset management settings
- Build configuration
- LaTeX/Pandoc settings
- Quality assurance settings

## Best Practices

### Development Workflow

1. **Edit Content**: Modify markdown files using proper syntax
2. **Validate**: Run validation before building
3. **Test**: Execute relevant test suites
4. **Build**: Generate PDFs with quality assurance
5. **Review**: Check build reports for issues

### Content Standards

- Use consistent figure format with IDs and captions
- Use LaTeX macros for all math symbols
- Use descriptive hyperlinks with `\href{}`
- Validate cross-references before building
- Test with real data (no mocks)
- Generate comprehensive validation reports

### Maintenance

- Regular validation of cross-references
- Automated testing of build pipelines
- Consistent formatting across papers
- Professional documentation standards
- Zero tolerance for broken references

## Advanced Usage

### Custom Analysis

```bash
# Run specific analysis components
python3 scripts/complexity_energetics/analyze_neural_networks.py --sparsity 0.02
python3 scripts/complexity_energetics/analyze_active_inference.py --horizon 5
```

### Parallel Processing

```bash
# Run tests in parallel
python3 -m pytest -n auto tests/

# Build papers in parallel (if supported)
python3 scripts/common_pipeline/build_core.py --parallel
```

### Debugging

```bash
# Trace pipeline execution
python3 scripts/common_pipeline/trace_pipeline.py

# Diagnose cross-reference issues
python3 scripts/common_pipeline/diagnose_crossref_issue.py
```

This unified workflow ensures consistent, professional, and reproducible scientific publications across all three papers while maintaining the highest quality standards and automated validation.
