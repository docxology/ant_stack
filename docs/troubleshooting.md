# 🔧 Troubleshooting Guide

**Comprehensive solutions for common issues in the Ant Stack scientific publication system.**

---

## 📋 Table of Contents

- [🚨 Build System Issues](#-build-system-issues)
- [📊 Analysis Pipeline Problems](#-analysis-pipeline-problems)
- [🎨 Figure and Visualization Issues](#-figure-and-visualization-issues)
- [📝 PDF Generation Problems](#-pdf-generation-problems)
- [🧪 Testing and Validation Issues](#-testing-and-validation-issues)
- [🔬 Scientific Accuracy Problems](#-scientific-accuracy-problems)
- [⚡ Performance Issues](#-performance-issues)
- [🔧 System Dependencies](#-system-dependencies)
- [📞 Getting Help](#-getting-help)

---

## 🚨 Build System Issues

### 1. Build Command Not Found

**Problem:** `python3 scripts/common_pipeline/build_core.py` returns "command not found"

**Solutions:**
```bash
# Check if file exists
ls -la scripts/common_pipeline/build_core.py

# Check permissions
ls -l scripts/common_pipeline/build_core.py
# Should be executable: -rwxr-xr-x

# Make executable if needed
chmod +x scripts/common_pipeline/build_core.py

# Use python3 explicitly
python3 scripts/common_pipeline/build_core.py --validate-only
```

**Alternative Path:**
```bash
# Full path execution
/Users/4d/Documents/GitHub/ant/scripts/common_pipeline/build_core.py --validate-only
```

### 2. Module Import Errors

**Problem:** `ImportError: No module named 'antstack_core'`

**Solutions:**
```bash
# Install in development mode
pip install -e .

# Check Python path
python3 -c "import sys; print('\\n'.join(sys.path))"

# Verify installation
python3 -c "import antstack_core; print('✅ Successfully imported')"

# Check for conflicts
pip list | grep antstack
```

### 3. Configuration File Not Found

**Problem:** `FileNotFoundError: [Errno 2] No such file or directory: 'paper_config.yaml'`

**Solutions:**
```bash
# Check paper directory structure
ls -la papers/complexity_energetics/

# Verify configuration file exists
ls -la papers/complexity_energetics/paper_config.yaml

# Check file permissions
ls -l papers/complexity_energetics/paper_config.yaml

# Validate YAML syntax
python3 -c "
import yaml
with open('papers/complexity_energetics/paper_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    print('✅ YAML syntax valid')
    print(f'Paper title: {config.get(\"paper\", {}).get(\"title\", \"Not found\")}')
"
```

---

## 📊 Analysis Pipeline Problems

### 1. Analysis Runner Fails

**Problem:** `Analysis pipeline failed: [Errno 2] No such file or directory`

**Solutions:**
```bash
# Check manifest file exists
ls -la papers/complexity_energetics/manifest.example.yaml

# Verify runner script exists
ls -la papers/complexity_energetics/src/

# Check Python path for runner
find . -name "runner.py" -type f

# Run analysis manually
cd papers/complexity_energetics
python3 src/runner.py manifest.example.yaml --out out
cd ../..
```

### 2. Bootstrap Analysis Issues

**Problem:** Bootstrap analysis returns inconsistent results

**Solutions:**
```python
# Check for fixed seed usage
def test_reproducibility():
    from antstack_core.analysis.statistics import bootstrap_mean_ci

    data = [1.2, 1.5, 1.3, 1.8, 1.4, 1.6, 1.7, 1.9]

    # Test multiple runs with same seed
    results = []
    for _ in range(5):
        mean, lower, upper = bootstrap_mean_ci(
            data, n_bootstrap=1000, random_seed=42
        )
        results.append((mean, lower, upper))

    # All should be identical
    assert all(r == results[0] for r in results[1:]), "Non-deterministic results"

# Check data quality
def validate_data_quality(data):
    import numpy as np

    # Check for finite values
    assert np.all(np.isfinite(data)), "Data contains non-finite values"

    # Check for sufficient sample size
    assert len(data) >= 10, "Insufficient sample size"

    # Check for reasonable variance
    std = np.std(data)
    mean = np.mean(data)
    cv = std / mean if mean != 0 else float('inf')
    assert cv < 10, "Coefficient of variation too high"
```

### 3. Scaling Analysis Problems

**Problem:** Scaling analysis returns unexpected exponents

**Solutions:**
```python
# Validate scaling relationship manually
def validate_scaling_manually(x_data, y_data):
    from antstack_core.analysis.statistics import analyze_scaling_relationship
    import numpy as np

    # Log-transform data
    log_x = np.log(x_data)
    log_y = np.log(y_data)

    # Check for linear relationship in log space
    correlation = np.corrcoef(log_x, log_y)[0, 1]
    assert correlation > 0.8, f"Poor correlation: {correlation}"

    # Fit power law: y = a * x^b
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)

    print(f"Scaling exponent: {slope:.3f}")
    print(f"R²: {r_value**2:.3f}")
    print(f"P-value: {p_value:.3f}")

    return slope, r_value**2, p_value

# Check data ranges
def check_data_ranges(x_data, y_data):
    import numpy as np

    print(f"X range: {np.min(x_data):.2e} to {np.max(x_data):.2e}")
    print(f"Y range: {np.min(y_data):.2e} to {np.max(y_data):.2e}")

    # Check for log scaling validity
    x_ratio = np.max(x_data) / np.min(x_data)
    y_ratio = np.max(y_data) / np.min(y_data)

    print(f"X spans {x_ratio:.0f} orders of magnitude")
    print(f"Y spans {y_ratio:.0f} orders of magnitude")

    # Scaling analysis requires at least 2 orders of magnitude
    assert x_ratio > 100, "X data spans insufficient range for scaling analysis"
```

---

## 🎨 Figure and Visualization Issues

### 1. Figure Not Displaying

**Problem:** Generated figures not showing in Jupyter notebooks or displays

**Solutions:**
```python
# Set proper backend
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Or for interactive displays
matplotlib.use('TkAgg')  # For Tkinter
# matplotlib.use('Qt5Agg')  # For Qt

# Check matplotlib configuration
import matplotlib.pyplot as plt
print(f"Backend: {plt.get_backend()}")

# Ensure figures are saved
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
plt.savefig('test_figure.png', dpi=300, bbox_inches='tight')
print("Figure saved: test_figure.png")
```

### 2. Mermaid Diagrams Not Rendering

**Problem:** Mermaid diagrams appear as code blocks instead of images

**Solutions:**
```bash
# Install Mermaid CLI
npm install -g @mermaid-js/mermaid-cli

# Test installation
mmdc --version

# Alternative: Docker-based rendering
docker run --rm -v $(pwd):/data ghcr.io/mermaid-js/mermaid-cli:latest -i /data/input.mmd -o /data/output.png

# Check Kroki service availability
curl -X POST https://kroki.io/mermaid/png -H "Content-Type: text/plain" --data-binary "graph TD; A-->B;" --output test.png
ls -la test.png
```

### 3. LaTeX Math Not Rendering

**Problem:** Mathematical expressions appear as LaTeX source instead of formatted equations

**Solutions:**
```bash
# Check LaTeX installation
xelatex --version

# Test basic LaTeX compilation
echo '\documentclass{article}\begin{document}$E = mc^2$\end{document}' | xelatex

# Check for unicode-math package
kpsewhich unicode-math.sty

# If missing, install LaTeX packages
# Ubuntu/Debian: sudo apt-get install texlive-fonts-recommended texlive-latex-extra
# macOS: brew install --cask mactex-no-gui

# Test math rendering in Python
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.text(0.5, 0.5, r'$\alpha + \beta = \gamma$', fontsize=20)
plt.savefig('math_test.png')
print("Math rendering test saved")
```

---

## 📝 PDF Generation Problems

### 1. Pandoc Not Found

**Problem:** `pandoc: command not found`

**Solutions:**
```bash
# Install Pandoc
# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install pandoc

# macOS:
brew install pandoc

# Windows (Chocolatey):
choco install pandoc

# Verify installation
pandoc --version
```

### 2. XeLaTeX Not Available

**Problem:** `xelatex: command not found`

**Solutions:**
```bash
# Install TeX Live
# Ubuntu/Debian:
sudo apt-get install texlive-xetex texlive-fonts-recommended texlive-latex-extra

# macOS:
brew install --cask mactex-no-gui

# Verify installation
xelatex --version

# Test LaTeX compilation
echo '\documentclass{article}\begin{document}Hello World\end{document}' | xelatex -output-directory /tmp
ls -la /tmp/texput.pdf
```

### 3. Broken Cross-References

**Problem:** PDF contains "Figure~??" or "Section~??"

**Solutions:**
```bash
# Run validation only to identify issues
python3 scripts/common_pipeline/build_core.py --validate-only

# Check cross-reference format in source files
grep -r "ref{" papers/ | head -10

# Validate individual paper
python3 scripts/common_pipeline/build_core.py --paper complexity_energetics --validate-only

# Manual cross-reference check
python3 -c "
import re
with open('papers/complexity_energetics/Results.md', 'r') as f:
    content = f.read()

# Find all figure references
refs = re.findall(r'\\\\ref\{([^}]+)\}', content)
print('References found:', refs)

# Find all figure definitions
defs = re.findall(r'## Figure: [^#]+ {#fig:([^}]+)}', content)
print('Definitions found:', defs)

# Check for mismatches
missing = set(refs) - set(defs)
extra = set(defs) - set(refs)
print('Missing definitions:', missing)
print('Unused definitions:', extra)
"
```

### 4. Unicode Character Issues

**Problem:** PDF contains warnings about undefined Unicode characters

**Solutions:**
```python
# Convert Unicode math symbols to LaTeX macros
def fix_unicode_math(text):
    """Convert Unicode math symbols to LaTeX macros."""
    replacements = {
        'μ': r'$\mu$',
        'λ': r'$\lambda$',
        'π': r'$\pi$',
        'α': r'$\alpha$',
        'β': r'$\beta$',
        'γ': r'$\gamma$',
        'δ': r'$\delta$',
        'Δ': r'$\Delta$',
        'ε': r'$\epsilon$',
        'σ': r'$\sigma$',
        'τ': r'$\tau$',
        'ω': r'$\omega$',
        'Ω': r'$\Omega$',
        '≤': r'$\le$',
        '≥': r'$\ge$',
        '≈': r'$\approx$',
        '≠': r'$\ne$',
        '±': r'$\pm$',
        '×': r'$\times$',
        '÷': r'$\div$',
        '∞': r'$\infty$',
        '∫': r'$\int$',
        '∑': r'$\sum$',
        '∏': r'$\prod$',
        '∂': r'$\partial$',
        '∇': r'$\nabla$',
        '√': r'$\sqrt{}$',
        '∝': r'$\propto$',
        '°': r'$^\circ$',
    }

    for unicode_char, latex_macro in replacements.items():
        text = text.replace(unicode_char, latex_macro)

    return text

# Apply fix to all markdown files
import os
for root, dirs, files in os.walk('papers'):
    for file in files:
        if file.endswith('.md'):
            filepath = os.path.join(root, file)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            fixed_content = fix_unicode_math(content)

            if fixed_content != content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                print(f"Fixed Unicode symbols in: {filepath}")
```

---

## 🧪 Testing and Validation Issues

### 1. Test Failures

**Problem:** `pytest` returns failures or errors

**Solutions:**
```bash
# Run tests with verbose output
python3 -m pytest tests/ -v

# Run specific failing test
python3 -m pytest tests/test_energy.py::TestEnergyAnalysis::test_reproducibility -v

# Check test coverage
python3 -m pytest --cov=antstack_core --cov-report=html tests/
open htmlcov/index.html

# Debug specific function
python3 -c "
from antstack_core.analysis.energy import estimate_detailed_energy
from antstack_core.analysis.energy import ComputeLoad, EnergyCoefficients

# Test with known values
workload = ComputeLoad(flops=1e9, sram_bytes=1e6, time_seconds=0.1)
coeffs = EnergyCoefficients()

try:
    result = estimate_detailed_energy(workload, coeffs)
    print(f'✅ Test passed: {result.total:.2e} J')
except Exception as e:
    print(f'❌ Test failed: {e}')
    import traceback
    traceback.print_exc()
"
```

### 2. Coverage Issues

**Problem:** Test coverage below target thresholds

**Solutions:**
```bash
# Generate coverage report
python3 -m pytest --cov=antstack_core --cov-report=html --cov-report=term-missing tests/

# Identify uncovered lines
python3 -m pytest --cov=antstack_core --cov-report=term-missing --cov-report=html tests/

# Add missing tests
def test_edge_cases():
    """Test edge cases and boundary conditions."""
    # Test with zero values
    # Test with very large values
    # Test with invalid inputs
    # Test error conditions
    pass

def test_error_conditions():
    """Test error handling and validation."""
    from antstack_core.analysis.energy import EnergyCoefficients, ComputeLoad
    import pytest

    coeffs = EnergyCoefficients()

    # Test invalid workload
    with pytest.raises(ValueError):
        invalid_workload = ComputeLoad(flops=-1)  # Negative not allowed
        estimate_detailed_energy(invalid_workload, coeffs)
```

---

## 🔬 Scientific Accuracy Problems

### 1. Unexpected Scaling Exponents

**Problem:** Scaling analysis returns unrealistic exponents

**Solutions:**
```python
# Validate scaling relationship manually
def validate_scaling_analysis():
    import numpy as np
    from antstack_core.analysis.statistics import analyze_scaling_relationship

    # Create known power-law relationship
    x = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    y = 2.5 * x ** 0.8 + np.random.normal(0, 0.1, len(x))  # Known exponent 0.8

    result = analyze_scaling_relationship(x, y)

    print(f"Expected exponent: 0.8")
    print(f"Calculated exponent: {result['scaling_exponent']:.3f}")
    print(f"Confidence interval: ({result['confidence_interval'][0]:.3f}, {result['confidence_interval'][1]:.3f})")
    print(f"R²: {result['r_squared']:.3f}")

    # Check if result is reasonable
    expected_range = (0.7, 0.9)  # Expected range for this test
    actual_exponent = result['scaling_exponent']

    assert expected_range[0] <= actual_exponent <= expected_range[1], \
        f"Exponent {actual_exponent:.3f} outside expected range {expected_range}"

    assert result['r_squared'] > 0.9, f"Poor fit: R² = {result['r_squared']:.3f}"
```

### 2. Energy Estimates Out of Range

**Problem:** Energy estimates seem too high or too low

**Solutions:**
```python
# Compare against theoretical limits
def validate_energy_estimates():
    from antstack_core.analysis.energy import EnergyCoefficients, estimate_detailed_energy
    from antstack_core.analysis.energy import ComputeLoad

    # Test with realistic workload
    workload = ComputeLoad(
        flops=1e9,           # 1 billion FLOPs
        sram_bytes=1e6,      # 1 MB SRAM
        dram_bytes=1e8,      # 100 MB DRAM
        time_seconds=0.1     # 100ms
    )

    coeffs = EnergyCoefficients()

    # Default coefficients
    energy = estimate_detailed_energy(workload, coeffs)

    # Calculate theoretical bounds
    theoretical_min = workload.flops * 1.4e-21  # Landauer limit per bit
    theoretical_max = workload.flops * 1e-12   # Current technology limit

    print(f"Energy estimate: {energy.total:.2e} J")
    print(f"Theoretical range: {theoretical_min:.2e} to {theoretical_max:.2e} J")

    # Check reasonableness
    assert theoretical_min <= energy.total <= theoretical_max, \
        f"Energy {energy.total:.2e} outside theoretical range"

    # Check per-FLOP energy
    energy_per_flop = energy.compute_flops / workload.flops
    print(f"Energy per FLOP: {energy_per_flop:.2e} J/FLOP")

    # Should be in reasonable range for current technology
    assert 1e-15 <= energy_per_flop <= 1e-12, \
        f"Energy per FLOP {energy_per_flop:.2e} seems unreasonable"
```

---

## ⚡ Performance Issues

### 1. Slow Analysis

**Problem:** Analysis takes too long to complete

**Solutions:**
```python
# Profile analysis performance
import time
import cProfile
import pstats

def profile_analysis():
    from antstack_core.analysis.energy import EnergyCoefficients, estimate_detailed_energy
    from antstack_core.analysis.energy import ComputeLoad

    # Create large workload for testing
    large_workload = ComputeLoad(
        flops=1e12,
        sram_bytes=1e9,
        dram_bytes=1e10,
        time_seconds=10.0
    )

    coeffs = EnergyCoefficients()

    # Profile energy estimation
    print("Profiling energy estimation...")
    profiler = cProfile.Profile()
    profiler.enable()

    start_time = time.time()
    energy = estimate_detailed_energy(large_workload, coeffs)
    end_time = time.time()

    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions

    print(f"Analysis time: {end_time - start_time:.3f} seconds")
    print(f"Energy result: {energy.total:.2e} J")

# Run performance analysis
profile_analysis()
```

### 2. Memory Usage Issues

**Problem:** Analysis consumes too much memory

**Solutions:**
```python
# Check memory usage
import psutil
import os

def monitor_memory_usage():
    process = psutil.Process(os.getpid())

    print("Monitoring memory usage...")

    # Get baseline memory
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Baseline memory: {baseline_memory:.1f} MB")

    # Perform memory-intensive operations
    from antstack_core.analysis.statistics import bootstrap_mean_ci
    import numpy as np

    # Create large dataset
    large_data = np.random.normal(0, 1, 1000000)

    # Monitor during bootstrap
    memory_during = process.memory_info().rss / 1024 / 1024
    print(f"Memory during bootstrap setup: {memory_during:.1f} MB")

    # Run bootstrap analysis
    mean, lower, upper = bootstrap_mean_ci(large_data, n_bootstrap=5000)

    memory_after = process.memory_info().rss / 1024 / 1024
    print(f"Memory after bootstrap: {memory_after:.1f} MB")
    print(f"Memory increase: {memory_after - baseline_memory:.1f} MB")

    # Check for memory leaks
    import gc
    gc.collect()
    memory_after_gc = process.memory_info().rss / 1024 / 1024
    print(f"Memory after garbage collection: {memory_after_gc:.1f} MB")

    return memory_after - baseline_memory

# Run memory analysis
memory_increase = monitor_memory_usage()
assert memory_increase < 500, f"Memory usage too high: {memory_increase:.1f} MB"
```

---

## 🔧 System Dependencies

### 1. Python Version Issues

**Problem:** Incompatible Python version

**Solutions:**
```bash
# Check Python version
python3 --version

# Should be 3.8.0 or higher
python3 -c "
import sys
if sys.version_info < (3, 8):
    print('❌ Python version too old')
    sys.exit(1)
else:
    print('✅ Python version compatible')
"

# Install specific Python version
# Ubuntu: sudo apt-get install python3.9 python3.9-venv
# macOS: brew install python@3.9
```

### 2. Package Conflicts

**Problem:** Conflicting package versions

**Solutions:**
```bash
# Check for conflicts
pip check

# List installed packages
pip list

# Create clean virtual environment
python3 -m venv clean_env
source clean_env/bin/activate  # Windows: clean_env\Scripts\activate

# Install only required packages
pip install --upgrade pip
pip install matplotlib numpy pandas pyyaml pytest scipy

# Install Ant Stack
pip install -e .

# Test installation
python3 -c "import antstack_core; print('✅ Clean installation successful')"
```

### 3. Path Issues

**Problem:** Scripts not found in PATH

**Solutions:**
```bash
# Check PATH
echo $PATH

# Add current directory to PATH
export PATH="$PATH:$(pwd)"

# Or use full path
python3 "$(pwd)/scripts/common_pipeline/build_core.py" --validate-only

# Check if file is executable
ls -l scripts/common_pipeline/build_core.py

# Make executable if needed
chmod +x scripts/common_pipeline/build_core.py
```

---

## 📞 Getting Help

### Debug Commands

**Comprehensive System Check:**
```bash
# System information
echo "=== System Information ==="
uname -a
python3 --version
pip --version

# Dependency check
echo -e "\n=== Dependencies ==="
python3 scripts/validate_rendering_system.py --verbose

# Configuration validation
echo -e "\n=== Configuration ==="
python3 scripts/common_pipeline/build_core.py --validate-only

# Test execution
echo -e "\n=== Tests ==="
python3 -m pytest tests/ --tb=short -q

# Build validation
echo -e "\n=== Build ==="
python3 scripts/common_pipeline/build_core.py --paper complexity_energetics --validate-only
```

### Issue Reporting

**When reporting issues, include:**

1. **System Information:**
   ```bash
   # Copy and paste this into issue reports
   echo "=== System Info ==="
   uname -a
   python3 --version
   pip list | grep -E "(numpy|scipy|matplotlib|pandas|pyyaml|pytest)"
   echo "=== End System Info ==="
   ```

2. **Error Messages:** Complete error output, not just summaries

3. **Reproduction Steps:** Exact commands to reproduce the issue

4. **Expected vs Actual:** What you expected vs what happened

5. **Workarounds:** Any temporary solutions you've found

### Community Support

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Technical questions and discussions
- **Documentation**: Contributing improvements to troubleshooting guides

---

This comprehensive troubleshooting guide provides solutions for the most common issues encountered when using the Ant Stack scientific publication system. For additional help, please check the community resources or open a detailed issue report.
