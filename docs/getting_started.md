# 🚀 Getting Started with Ant Stack

**Step-by-step guide to install, configure, and use the Ant Stack modular scientific publication system.**

---

## 📋 Prerequisites

### System Requirements

**Operating Systems:**
- ✅ **macOS** 10.15+ (Intel and Apple Silicon)
- ✅ **Ubuntu/Debian** 18.04+
- ✅ **Windows** 10+ (via WSL2)

**Hardware Requirements:**
- **Minimum**: 4GB RAM, 2GB disk space
- **Recommended**: 8GB RAM, 4GB disk space
- **Optimal**: 16GB RAM, SSD storage

### Software Dependencies

**Core Requirements:**
```bash
# Python (3.8+)
python3 --version  # Should show 3.8.0 or higher

# Node.js (14+)
node --version     # Should show v14.0.0 or higher

# LaTeX Distribution
xelatex --version  # Should show XeTeX version

# Pandoc (2.10+)
pandoc --version   # Should show 2.10.0 or higher
```

**Python Packages:**
```bash
pip install matplotlib numpy pandas pyyaml pytest scipy
```

**Node.js Packages:**
```bash
npm install -g mermaid-filter
```

---

## 🛠️ Installation

### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/your-repo/ant.git
cd ant

# Verify repository structure
ls -la
# Should show: README.md, antstack_core/, papers/, scripts/, docs/, etc.
```

### Step 2: Environment Setup

**Option A: Virtual Environment (Recommended)**
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt  # If requirements.txt exists
pip install matplotlib numpy pandas pyyaml pytest scipy

# Install in development mode
pip install -e .
```

**Option B: System-wide Installation**
```bash
# Install system-wide (requires sudo/admin privileges)
sudo pip3 install matplotlib numpy pandas pyyaml pytest scipy

# Install in development mode
sudo pip3 install -e .
```

### Step 3: Verify Installation

```bash
# Test basic import
python3 -c "import antstack_core; print('✅ Ant Stack imported successfully')"

# Run basic functionality test
python3 -c "
from antstack_core.analysis.energy import EnergyCoefficients
coeffs = EnergyCoefficients()
print(f'✅ Energy coefficients loaded: {len([attr for attr in dir(coeffs) if not attr.startswith(\"_\")])} properties')
"

# Check version information
python3 -c "import antstack_core; print(f'✅ Ant Stack version: {antstack_core.__version__ if hasattr(antstack_core, \"__version__\") else \"Development\"}')"
```

---

## ⚙️ Configuration

### Basic Configuration

**Project Structure:**
```
ant/
├── antstack_core/          # Core scientific methods
├── papers/                 # Research papers
├── scripts/               # Build and analysis scripts
├── docs/                  # Documentation
├── tests/                 # Test suite
├── tools/                 # Legacy build tools
└── README.md             # Main documentation
```

**Configuration Files:**
- `paper_config.yaml`: Paper-specific configuration
- `manifest.example.yaml`: Analysis parameters
- `requirements.txt`: Python dependencies
- `package.json`: Node.js dependencies

### Environment Variables

**Optional Environment Configuration:**
```bash
# Build system configuration
export LINK_COLOR=blue                    # PDF link color
export MERMAID_STRATEGY=auto             # Mermaid rendering strategy
export MERMAID_IMG_FORMAT=png            # Image format
export STRICT_MERMAID=0                  # Fail on render errors

# Analysis configuration
export PYTHONPATH="${PYTHONPATH}:$(pwd)" # Add project to Python path
export MPLBACKEND=Agg                    # Matplotlib backend for headless
```

---

## 🔬 Basic Usage

### Step 1: Run Test Suite

**Verify System Functionality:**
```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific test categories
python3 -m pytest tests/core_rendering/ -v    # Core system tests
python3 -m pytest tests/complexity_energetics/ -v  # Analysis tests

# Run with coverage report
python3 -m pytest --cov=antstack_core --cov-report=html tests/

# Open coverage report
open htmlcov/index.html
```

### Step 2: Validate Environment

**Run Build Validation:**
```bash
# Validate all papers
python3 scripts/common_pipeline/build_core.py --validate-only

# Validate specific paper
python3 scripts/common_pipeline/build_core.py --paper complexity_energetics --validate-only

# Check for common issues
python3 scripts/validate_rendering_system.py --verbose
```

### Step 3: Build First Paper

**Build Complexity & Energetics Paper:**
```bash
# Run analysis pipeline
python3 -m papers.complexity_energetics.src.runner papers/complexity_energetics/manifest.example.yaml --out papers/complexity_energetics/out

# Build paper
python3 scripts/common_pipeline/build_core.py --paper complexity_energetics

# Check results
ls -la 2_complexity_energetics.pdf
```

---

## 📊 Analysis Examples

### Energy Analysis

**Basic Energy Estimation:**
```python
from antstack_core.analysis.energy import EnergyCoefficients, ComputeLoad, estimate_detailed_energy

# Define hardware parameters
coeffs = EnergyCoefficients(
    flops_pj=1.5,           # 1.5 pJ per FLOP
    sram_pj_per_byte=0.2,   # 0.2 pJ per SRAM byte
    dram_pj_per_byte=25.0,  # 25 pJ per DRAM byte
    baseline_w=0.8          # 0.8W baseline power
)

# Define computational workload
workload = ComputeLoad(
    flops=1e9,              # 1 billion FLOPs
    sram_bytes=1e6,         # 1 MB SRAM access
    dram_bytes=1e8,         # 100 MB DRAM access
    time_seconds=0.1        # 100ms execution time
)

# Calculate energy consumption
energy = estimate_detailed_energy(workload, coeffs)
print(f"Total energy: {energy.total:.2e} J")
print(f"Energy efficiency: {energy.total / workload.flops:.2e} J/FLOP")
```

### Statistical Analysis

**Bootstrap Confidence Intervals:**
```python
from antstack_core.analysis.statistics import bootstrap_mean_ci
import numpy as np

# Example data
data = np.array([1.2, 1.5, 1.3, 1.8, 1.4, 1.6, 1.7, 1.9])

# Calculate bootstrap confidence interval
mean, lower_ci, upper_ci = bootstrap_mean_ci(
    data,
    n_bootstrap=1000,
    confidence_level=0.95,
    random_seed=42
)

print(f"Mean: {mean:.3f}")
print(f"95% CI: ({lower_ci:.3f}, {upper_ci:.3f})")
print(f"CI Width: {upper_ci - lower_ci:.3f}")
```

### Scaling Analysis

**Power-Law Relationship Analysis:**
```python
from antstack_core.analysis.statistics import analyze_scaling_relationship
import numpy as np

# Example scaling data (energy vs complexity)
complexity = np.array([1, 2, 4, 8, 16, 32])
energy = np.array([1.0, 2.1, 4.3, 8.7, 17.5, 35.2])

# Analyze scaling relationship
result = analyze_scaling_relationship(
    complexity,
    energy,
    log_transform=True,
    confidence_level=0.95
)

print(f"Scaling exponent: {result['scaling_exponent']:.3f}")
print(f"Confidence interval: ±{result['confidence_interval']:.3f}")
print(f"R² correlation: {result['r_squared']:.3f}")
```

---

## 🧪 Advanced Features

### Custom Analysis Pipeline

**Creating Custom Analysis:**
```python
# Example: Custom neural network scaling analysis
from antstack_core.analysis.enhanced_estimators import EnhancedEnergyEstimator
from antstack_core.analysis.energy import EnergyCoefficients

# Initialize estimator
coeffs = EnergyCoefficients()
estimator = EnhancedEnergyEstimator(coeffs)

# Define analysis parameters
brain_params = {
    'C': 1000,              # Channel count
    'sparsity': 0.1,        # Connection sparsity
    'dt': 0.01              # Time step
}

# Run comprehensive analysis
analysis_results = estimator.analyze_brain_scaling(
    c_values=[100, 500, 1000, 2000],
    base_params=brain_params
)

print(f"Scaling analysis complete")
print(f"Results: {analysis_results.keys()}")
```

### Figure Generation

**Publication-Quality Figures:**
```python
from antstack_core.figures.plots import create_scaling_plot
import matplotlib.pyplot as plt

# Generate scaling plot
x_values = [1, 2, 4, 8, 16, 32]
y_values = [1.0, 1.8, 3.2, 5.8, 10.4, 18.8]

fig = create_scaling_plot(
    x_values=x_values,
    y_values=y_values,
    x_label="System Complexity (K)",
    y_label="Energy Consumption (J)",
    title="Energy Scaling Analysis",
    scaling_exponent=0.8,
    confidence_interval=(0.75, 0.85)
)

# Save figure
plt.savefig("energy_scaling.png", dpi=300, bbox_inches='tight')
print("Figure saved: energy_scaling.png")
```

---

## 🔧 Troubleshooting

### Common Issues

**1. Import Errors:**
```bash
# Problem: Module not found
# Solution: Ensure virtual environment is activated
source .venv/bin/activate

# Or install system-wide
sudo pip3 install -e .
```

**2. Build Failures:**
```bash
# Problem: PDF generation fails
# Solution: Check dependencies
python3 scripts/validate_rendering_system.py

# Run validation only
python3 scripts/common_pipeline/build_core.py --validate-only
```

**3. Mermaid Diagram Issues:**
```bash
# Problem: Diagrams not rendering
# Solution: Install Mermaid CLI
npm install -g @mermaid-js/mermaid-cli

# Or use Docker fallback
export MERMAID_STRATEGY=docker
```

**4. LaTeX Errors:**
```bash
# Problem: XeLaTeX compilation fails
# Solution: Reinstall LaTeX distribution
# macOS: brew install --cask mactex-no-gui
# Ubuntu: sudo apt-get install texlive-xetex texlive-fonts-recommended
```

### Getting Help

**Debugging Commands:**
```bash
# Check system dependencies
python3 scripts/validate_rendering_system.py --verbose

# Run comprehensive diagnostics
python3 scripts/common_pipeline/build_core.py --validate-only

# View build reports
cat build_report.md

# Check environment variables
env | grep -E "(PYTHONPATH|PATH|MAMBA|MAMBA_ROOT)" | head -10
```

---

## 📚 Next Steps

### Learning Resources

1. **API Reference**: `docs/api_reference.md` - Complete function documentation
2. **Scientific Validation**: `docs/scientific_validation.md` - Validation methodology
3. **Best Practices**: `docs/best_practices.md` - Development guidelines
4. **Architecture**: `docs/architecture.md` - System design overview

### Advanced Topics

- **Custom Analysis Modules**: Creating new analysis methods
- **Paper Templates**: Developing new paper structures
- **Testing Framework**: Writing comprehensive tests
- **Performance Optimization**: Scaling and optimization techniques

### Community Support

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Technical questions and discussions
- **Documentation**: Contributing to documentation improvements

---

**Congratulations!** You now have a working Ant Stack installation. The system is ready for scientific analysis and publication generation.

For more detailed information about specific features, see the comprehensive documentation in the `docs/` directory.
