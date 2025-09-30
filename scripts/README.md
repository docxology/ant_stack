# Ant Stack Scripts

Organized collection of analysis, build, and utility scripts for the Ant Stack research framework.

## Directory Structure

### 📁 `ant_stack/`
Scripts specific to Ant Stack framework analysis:
- Colony behavior modeling
- Multi-agent coordination
- Stigmergic communication
- Swarm intelligence patterns

*Currently empty - add Ant Stack specific scripts here*

### 📁 `complexity_energetics/`
Scripts for computational complexity and energy analysis:
- Algorithmic complexity characterization
- Energy consumption modeling
- Scaling relationship analysis
- Theoretical limit validation

**Key Scripts:**
- `generate_comprehensive_analysis.py` - Complete analysis pipeline
- `analyze_active_inference.py` - Active inference complexity analysis
- `analyze_neural_networks.py` - Neural network connectivity analysis
- `generate_manuscript_figures.py` - Publication figure generation

### 📁 `common_pipeline/`
Shared infrastructure and build system scripts:
- Modular paper building system
- Cross-reference validation and repair
- Citation and formatting tools
- Comprehensive test suite

**Key Scripts:**
- `build_core.py` - Modular paper builder
- Cross-reference repair is handled automatically by the build system
- `run_validation_suite.py` - Complete test execution
- `unified_build.py` - Unified build system

## Quick Start

### Running Complexity Analysis
```bash
# Complete analysis pipeline
python scripts/complexity_energetics/generate_comprehensive_analysis.py

# Specific component analysis
python scripts/complexity_energetics/analyze_neural_networks.py --sparsity 0.02
```

### Building Papers
```bash
# Build specific paper
python scripts/common_pipeline/build_core.py --paper complexity_energetics

# Build all papers with testing
python scripts/common_pipeline/build_core.py --all --test
```

### Validation and Repair
```bash
# Fix cross-references
python3 scripts/common_pipeline/build_core.py --validate-only

# Run validation suite
python scripts/common_pipeline/run_validation_suite.py
```

## Key Features

- **🔬 Scientific Analysis**: Comprehensive complexity and energy modeling
- **📊 Automated Figure Generation**: Publication-quality plots and tables
- **✅ Rigorous Validation**: Cross-reference checking, citation validation
- **🏗️ Modular Build System**: Support for multiple papers with shared infrastructure
- **🧪 Comprehensive Testing**: Unit tests, integration tests, benchmarks
- **🔧 Automated Repair**: Self-healing tools for common issues

## Dependencies

### Core Requirements
- Python 3.8+
- antstack_core package
- numpy, scipy, matplotlib
- pandas, seaborn

### Build System
- pandoc
- LaTeX distribution (xelatex recommended)
- Graphviz (for mermaid diagrams)

### Optional
- pytest (for testing)
- pynvml (NVIDIA GPU monitoring)
- intel RAPL tools (CPU power monitoring)

## Development Guidelines

### Adding New Scripts

1. **Choose appropriate directory** based on scope:
   - `ant_stack/` - Ant colony specific analysis
   - `complexity_energetics/` - Complexity/energy analysis
   - `common_pipeline/` - Build/validation infrastructure

2. **Follow naming conventions**:
   - `analyze_*.py` - Analysis scripts
   - `generate_*.py` - Data/figure generation
   - `fix_*.py` - Repair/validation tools
   - `build_*.py` - Build system components

3. **Include documentation**:
   - Comprehensive docstrings
   - Usage examples
   - Parameter descriptions

4. **Error handling**:
   - Graceful degradation
   - Informative error messages
   - Logging for debugging

### Script Organization Principles

- **Separation of Concerns**: Each script has a single, well-defined purpose
- **Modular Design**: Reusable components in shared modules
- **Configuration-Driven**: Use YAML/JSON configs for parameters
- **Testable**: Scripts should be testable and validated
- **Documented**: Clear documentation and examples

## Contributing

When adding new scripts:

1. Update this README with script description
2. Add appropriate directory README if creating new categories
3. Include usage examples and parameter documentation
4. Add validation tests where appropriate
5. Follow existing code style and patterns

## Troubleshooting

### Common Issues

**Import Errors**: Ensure `antstack_core` is properly installed
**Build Failures**: Check LaTeX installation and pandoc version
**Figure Generation**: Verify matplotlib backend configuration

### Getting Help

- Check individual script docstrings for usage
- Review error messages for specific guidance
- Use `--help` flag on scripts for parameter information
