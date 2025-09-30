# Example: Proper Paper Syntax

This document demonstrates the correct syntax patterns for reliable PDF rendering across all Ant Stack papers.

## Paper Structure

### Abstract with Proper Math

```markdown
# Abstract {#sec:abstract}

**Objective:** To demonstrate proper mathematical notation and cross-referencing in scientific papers.

**Methods:** We use LaTeX macros for all mathematical symbols: $\mu$ (micrometers), $\lambda$ (wavelength), and $\pi$ (pi constant). Energy calculations follow the relationship $E = \sum_{i=1}^{n} \alpha_i \cdot K_i^{\beta_i}$ where $\alpha_i$ represents scaling coefficients.

**Results:** Our analysis shows energy scaling with $\beta \approx 1.2$ and confidence intervals of $\pm 0.1\,\mu\mathrm{m}$ resolution.

**Conclusions:** Proper mathematical formatting ensures reliable PDF generation and professional presentation.
```

### Figure Definition (CRITICAL FORMAT)

```markdown
## Figure: Energy Scaling Analysis {#fig:energy_scaling}

![Energy consumption scaling with system complexity](assets/figures/energy_scaling.png)

**Caption:** Energy consumption scaling with system complexity K, showing 95% confidence intervals. The power law relationship $E \propto K^{\beta}$ is evident with $\beta \approx 1.2$. Error bars represent standard deviation across 100 trials.

\href{file:///absolute/path/to/figure.png}{(View absolute file)}
```

### Cross-References

```markdown
# Introduction {#sec:introduction}

As shown in \ref{fig:energy_scaling}, the energy scaling relationship follows a power law. This is consistent with theoretical predictions from \ref{eq:energy_scaling}.

## Methodology {#sec:methodology}

Our approach builds on the framework described in \ref{sec:introduction} and extends it with the energy model from \ref{eq:energy_scaling}.
```

### Equations

```markdown
The energy scaling relationship is given by:

\begin{equation}
E = \sum_{i=1}^{n} \alpha_i \cdot K_i^{\beta_i}
\label{eq:energy_scaling}
\end{equation}

where $E$ is total energy, $\alpha_i$ are scaling coefficients, $K_i$ are complexity parameters, and $\beta_i$ are power law exponents.

From \ref{eq:energy_scaling}, we can derive the efficiency metric:

$$ \eta = \frac{E_{\text{theoretical}}}{E_{\text{measured}}} $$
```

### Tables

```markdown
## Table: Energy Coefficients {#tab:energy_coeffs}

| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| $\alpha_1$ | 0.025 | J | Base energy coefficient |
| $\alpha_2$ | 0.001 | J/K | Scaling coefficient |
| $\beta_1$ | 1.2 | - | Power law exponent |
| $\beta_2$ | 0.8 | - | Efficiency exponent |

**Caption:** Energy scaling parameters derived from experimental data. All values reported with 95% confidence intervals.
```

### Hyperlinks

```markdown
# References

External resources:
- \href{https://arxiv.org/abs/2505.03764}{arXiv preprint on energy scaling}
- \href{https://doi.org/10.1038/s41586-021-03819-2}{Nature article on computational efficiency}
- \href{https://github.com/example/repo}{Source code repository}

Internal references:
- See \ref{sec:methodology} for detailed procedures
- Data available in \ref{tab:energy_coeffs}
- Results shown in \ref{fig:energy_scaling}
```

### Mermaid Diagrams

```markdown
## Figure: System Architecture {#fig:system_arch}

![System architecture diagram](assets/mermaid/system_architecture.png)

**Caption:** High-level system architecture showing data flow between AntBody, AntBrain, and AntMind components. Solid arrows indicate data flow, dashed arrows indicate control signals.

\href{file:///absolute/path/to/diagram.png}{(View absolute file)}
```

### Code Blocks

```markdown
## Implementation

The energy calculation is implemented as follows:

```python
def calculate_energy(K, alpha, beta):
    """Calculate energy using power law scaling.
    
    Args:
        K: Complexity parameter
        alpha: Scaling coefficient
        beta: Power law exponent
        
    Returns:
        Energy in Joules
    """
    return alpha * (K ** beta)

# Example usage
energy = calculate_energy(K=100, alpha=0.025, beta=1.2)
print(f"Energy: {energy:.3f} J")
```

### Appendices

```markdown
# Appendix: Mathematical Derivations {#sec:app_derivations}

## Derivation of Energy Scaling

Starting from the basic energy equation:

$$E = \int_0^T P(t) \, dt$$

where $P(t)$ is power consumption at time $t$.

For constant power $P_0$:

$$E = P_0 \cdot T$$

Substituting the scaling relationship $P_0 = \alpha \cdot K^{\beta}$:

$$E = \alpha \cdot K^{\beta} \cdot T$$

This gives us \ref{eq:energy_scaling} with $T = 1$ second.
```

## Common Mistakes to Avoid

### ❌ Incorrect Math Formatting

```markdown
# WRONG - Unicode symbols
The wavelength is 500 μm with error ±0.1 μm.

# WRONG - Improper LaTeX
The wavelength is 500 \(\mu\mathrm{m}\) with error \(\pm 0.1\,\mu\mathrm{m}\).

# WRONG - Mixed formatting
The wavelength is 500 $\mu$m with error $\pm$ 0.1 $\mu$m.
```

### ✅ Correct Math Formatting

```markdown
# CORRECT - Proper LaTeX macros
The wavelength is 500 $\mu\mathrm{m}$ with error $\pm 0.1\,\mu\mathrm{m}$.
```

### ❌ Incorrect Figure Format

```markdown
# WRONG - Inline figure definition
![Energy scaling](assets/figures/energy.png){#fig:energy}

# WRONG - Missing caption
## Figure: Energy Scaling {#fig:energy}
![Energy scaling](assets/figures/energy.png)
```

### ✅ Correct Figure Format

```markdown
# CORRECT - Proper figure definition
## Figure: Energy Scaling Analysis {#fig:energy_scaling}

![Energy consumption scaling with system complexity](assets/figures/energy_scaling.png)

**Caption:** Energy consumption scaling with system complexity K, showing 95% confidence intervals.
```

### ❌ Incorrect Hyperlinks

```markdown
# WRONG - Naked URLs
See https://arxiv.org/abs/2505.03764 for details.

# WRONG - \url{} commands
See \url{https://arxiv.org/abs/2505.03764} for details.
```

### ✅ Correct Hyperlinks

```markdown
# CORRECT - Descriptive hyperlinks
See \href{https://arxiv.org/abs/2505.03764}{arXiv preprint} for details.
```

## Validation Checklist

Before building your paper, ensure:

- [ ] All figures use the proper format: `## Figure: Title {#fig:id}`
- [ ] All math symbols use LaTeX macros: `$\mu$`, `$\lambda$`, etc.
- [ ] All hyperlinks use `\href{URL}{text}` format
- [ ] All cross-references use `\ref{type:id}` format
- [ ] All figures have descriptive captions
- [ ] All equations are properly numbered with `\label{}`
- [ ] No Unicode math symbols in text mode
- [ ] No inline figure definitions
- [ ] No naked URLs
- [ ] All referenced files exist

## Build Commands

```bash
# Validate paper syntax
python3 scripts/validate_rendering_system.py --paper paper_name --verbose

# Build specific paper
python3 scripts/common_pipeline/build_core.py --paper paper_name

# Build all papers
python3 scripts/common_pipeline/build_core.py

# Validate only (no PDF generation)
python3 scripts/common_pipeline/build_core.py --validate-only
```

This example demonstrates the reliable syntax patterns that ensure consistent, professional PDF generation across all Ant Stack papers.
