# Scientific Paper Syntax: Real Counters, Attached Captions and Literal Contexts

Syntax examples only. Paths must be replaced with caller-owned produced resources; no measurements, fitted exponents, confidence intervals or citations are invented here. Native rendering and inspection remain required: see [PDF rendering](../PDF_RENDERING_GUIDE.md).

## Name the Scientific Section and Its Stable Target

```markdown
# Analytical Workloads and Energy Accounting {#sec:accounting}

These quantities are model estimates under declared assumptions, not measured device energy.
```

## Attach Figure Captions to Native Image Objects

```markdown
![Analytical energy components under the selected coefficients. Units and axis scales are explicit; no empirical calibration is implied.](assets/figures/energy_components.png){#fig:components width=90%}

See \ref{fig:components} for the selected model components.
```

Native inline image IDs are supported, not incorrect syntax. The retained project heading form also binds to the following actual image:

```markdown
## Figure: Declared Model Components {#fig:components}

![Caption describing the actual produced image and its evidence class.](assets/figures/energy_components.png)
```

Use one form per object, with a unique ID. A detached `**Caption:**` paragraph is not an attached image caption. Same-basename images in separate source folders are distinct resources.

## Number Referenced Equations Explicitly

```markdown
\begin{equation}
E_{\mathrm{compute}} = N_F e_F + V_S e_S + V_D e_D
\label{eq:compute}
\end{equation}

All prices in this illustrative equation must first be converted to joules per counted operation or byte. See \ref{eq:compute}.
```

An unnumbered `$$...$$` display cannot supply a numbered equation target merely by containing a label. Preserve SI units and distinguish energy from power and traffic from storage.

## Give Tables Real Captions and Evidence Classes

```markdown
## Table: Declared Inputs, Not Measured Results {#tab:inputs}

| Symbol | Meaning | Unit |
|---|---|---|
| $t$ | Accounting interval | s |
| $f$ | Declared update frequency | Hz |

The table defines inputs, not empirical values. Refer to \ref{tab:inputs}.
```

## Keep Literal Code, URIs and Scientific Math Separate

```python
label = 'α ≤ β'
```

Code literals must stay literal; prose/math normalization must not rewrite code, link destinations or citation keys. Use genuine bibliography keys for citations and verify the primary reference identity independently. A URL query containing `@draft` is not a bibliography citation.

## Render Active Mermaid Instead of Reusing Stale Images

````markdown
## Figure: Analytical Accounting Flow {#fig:accounting_flow}

```mermaid
flowchart LR
  Inputs[Declared inputs] --> Counts[Analytical counts]
  Counts --> Estimate[Estimated energy]
```
````

This conceptual flow is not measured system behavior. Local pinned Mermaid/Puppeteer must genuinely render it with its sandbox enabled; no cached-image or security-disabling fallback qualifies a failed render.

## Resolve References Across the Ordered Document

Footnotes and reference-style links may be defined in another configured source file. Keep configured source order, inspect actual resulting notes/links, and never replace a broken reference by guessing a similarly named target. Compilation alone is not scientific or visual acceptance.
