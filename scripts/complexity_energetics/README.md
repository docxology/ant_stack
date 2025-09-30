# Complexity & Energetics Analysis Scripts

Scripts for computational complexity analysis and energy modeling of embodied AI systems.

## Purpose

This directory contains specialized scripts for analyzing the computational complexity and energy consumption patterns of the Ant Stack framework, focusing on:

- Algorithmic complexity characterization
- Energy consumption modeling
- Scaling relationship analysis
- Theoretical limit validation
- Performance benchmarking

## Scripts

### Core Analysis Scripts
- `analyze_active_inference.py` - Active inference complexity and bounded rationality analysis
- `analyze_contact_dynamics.py` - Contact dynamics solver performance comparison
- `analyze_neural_networks.py` - Sparse neural network connectivity pattern analysis

### Figure and Results Generation
- `generate_comprehensive_analysis.py` - Complete scaling analysis pipeline
- `generate_manuscript_figures.py` - Publication-quality figure generation
- `generate_multipanel_figures.py` - Multi-panel figure layouts
- `generate_results_figures.py` - Automated results visualization
- `generate_tables_and_numbers.py` - Key numbers and table generation

### Utilities
- `run_ce.sh` - Convenience script for running complexity-energetics analysis

## Key Features

- **Comprehensive Scaling Analysis**: Automated power-law detection and regime classification
- **Energy Modeling**: Device-specific energy coefficients with uncertainty quantification
- **Theoretical Limits**: Comparison against fundamental physical limits (Landauer, thermodynamic)
- **Statistical Rigor**: Bootstrap confidence intervals and hypothesis testing
- **Publication Ready**: Automated figure generation with proper formatting and captions

## Usage

Most scripts can be run directly from the project root:

```bash
# Run comprehensive analysis
python scripts/complexity_energetics/generate_comprehensive_analysis.py

# Analyze specific component
python scripts/complexity_energetics/analyze_neural_networks.py --sparsity 0.02

# Generate figures for paper
python scripts/complexity_energetics/generate_manuscript_figures.py
```

## Dependencies

- `antstack_core.analysis` package
- matplotlib, seaborn for plotting
- numpy, scipy for numerical analysis
- pandas for data manipulation
- pytest for validation

## Output

Scripts generate:
- JSON key numbers files (`key_numbers.json`)
- Publication-quality figures (PNG/SVG)
- Statistical analysis reports
- Validation test results
