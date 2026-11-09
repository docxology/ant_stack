# Public API and Error Contracts

## Installed package versus caller-owned project

`__all__` identifies intended package names, not proof that every possible input or hardware platform was exercised. Python/NumPy/Pandas values are preferred at API boundaries; thin CLIs delegate to package-owned implementation. Installed `antstack-build` and `antstack-ce` no longer dynamically import unshipped repository scripts. Paper/config/input files remain caller-owned inputs; the wheel is not a bundle of this repository's manuscripts, Node dependencies, TeX or browser.

| Entry point | Contract |
|---|---|
| `antstack-build [--project-root DIR] [--paper NAME] [--validate-only]` | Read configured ordered sources. Empty/missing/invalid paper sets, colliding effective PDF destinations and failed analysis/render return nonzero before ambiguous publication. Validation writes `build_report.md` but no PDF. No implicit pytest invocation or `--no-tests` switch. |
| `antstack-ce MANIFEST [--validate-only] [--out DIR]` | Validation does not need an output. Generation requires an explicit new/empty artifact root and puts all outputs inside it. A populated root is immutable. |
| `run-all-antstack --config FILE [--out DIR] [--run-id ID] [--tasks LIST] [--validate-only] [--log-level LEVEL]` | Explicit overrides apply before validating the effective configuration and prerequisites. Fresh staged outputs, current-byte verification, owned archived replacements and coherent projections. Validation creates no run layout. |

## Scientific contracts changed in 3.0.0

| Surface | Meaning and failure boundary |
|---|---|
| Workloads and energy | Finite, domain-valid parameters; unsupported solver/topology names fail instead of aliasing another model. Closed-form integer update counts correct one-ULP roundoff at integer boundaries, so a reciprocal 49 Hz window retains its fastest update. Base/enhanced families and explicit joule/traffic/spike/idle boundaries stay distinct. |
| Optional diagnostics | `ComplexityMetrics` fields not computed by the chosen analyzer are `None`, including unrelated graph/thermodynamic/scaling quantities. Real computed or explicitly supplied zero remains zero; uncomputed fields are not scientific estimates. |
| Standalone caps and controls | Contact/neural sample grids and secondary studies obey their supplied maximums and include the endpoint; small grids keep unavailable fits explicit. Planning studies obey the chosen horizon bound and expose a descriptive sampled model-energy cut, not a runtime tractability threshold. Removed label-only contact `--solver`, neural `--pattern`, inverse-input efficiency proxy and `tractable` fields have no shim. |
| `effective_policy_count` | Enhanced leaf-policy counts apply the declared cap at every horizon; `None` selects exact enumeration. It is not a latency or search-quality guarantee. |
| Statistics | Student/Welch degrees of freedom and tails, real confidence intervals, local RNG state, exact constant bootstrap bounds and aligned finite observations. Constant responses retain valid exponents/intercepts with undefined `r_squared=None`; bootstrap R² intervals/statistics remain `None` when unavailable. Missing data cannot become successful validation or invented replication/power. |
| Graph/spectral helpers | Report their defined graph, correlation, baseline/width and complexity quantities; unavailable information is not silently replaced by zero or a fabricated biological regime. |
| Physical lower bounds | Landauer uses `k_B T ln 2` per erased bit. Mechanical and digital terms must be comparable in interval/units; an assumed bit count is not a measured algorithm-erasure count. |
| Power measurement | Null means not measured. Requested but unavailable RAPL/NVML observations raise an explicit error (`PowerMeasurementUnavailable` in `antstack_core.analysis.power_meters`), not zero joules. |
| Plots | Paired finite filtering preserves observation alignment. Invalid categorical data/intervals fail before replacing an output. Supplied intervals retain their actual bounds even when they exclude the displayed mean; no significance is inferred. |
| `interval_plot` | Categorical estimates and actual supplied bounds use an explicit log axis only when every displayed value/bound is positive and the range warrants it. Zero/negative values remain visible on a linear axis; small positive labels are not rounded into zero. Failed image writes preserve the prior image and its mode. |
| Supplemental analysis/figure/table CLIs | All ten generators require explicit fresh absent/empty output roots. Complete families stage before atomic publication, with relative checksummed inventories and source/runtime provenance; manifest-driven analyses retain input snapshots. Populated, linked, package and runtime roots are refused without replacing prior data. Captions attach to real figure IDs; log/symlog comparisons expose small values. These are separate analytical inputs, not canonical run replacements. |
| `MeterConfig` | Only `meter_type`/YAML `type`, NVML `device_index` and RAPL `energy_path` are consumed. Removed calibration/baseline/sampler/channel/logging controls and buffer helpers have no shim; unsupported fields fail before generation. Input `provenance` is notes, not switches disabling mandatory source/version identities. |
| Toy agent model | `complexity_analysis.AgentBasedModel.simulate()` is deterministic with no seed argument and does not reset caller RNGs. `state_pair_similarity` replaces the incorrectly named clustering coefficient; costs are arbitrary units, not joules. The duplicate enhanced-module toy class was removed. |
| Conditional empirical comparison | `TheoreticalLimitsAnalyzer.compare_with_empirical_data` exposes supplied efficiency and an explicitly not-assessed validation status; a supplied efficiency cannot become `validation_score`. Its energy/erasure ratio is conditional on the declared erased-bit assumption, not biological or device validation. |
| Enhanced comprehensive estimate | Optional fits/R² preserve `None` rather than numeric zero. Aggregation requires matched decision windows and describes isolated module sweep means with per-module idle, not a single configuration or whole-device observation. Physical cost of transport is unavailable without locomotion evidence; no hardcoded mass/distance supplies it. |
| Key-number updater CLI | Missing/invalid/unavailable placeholders, missing sections and failed updates report failure and exit nonzero. Exact valid formats retain their actual values; failed substitutions preserve prior source bytes. |

## Publication is failure-honest, not a scientific certification

`PDFGenerator.generate_pdf` returns a structured `BuildResult`. Success requires a newly produced PDF and no fatal undefined-reference/citation/resource/missing-glyph diagnostic; failure preserves the prior PDF. Ordered sources share document-wide footnote/reference definitions while retaining local image-use context. Parsed Markdown image resources render from byte snapshots whose identities are recorded; identical basenames in separate source directories remain distinct. Raw TeX URL/resource arguments are not citation keys. This is not a complete TeX dependency graph or a scientific/visual-readability certification; trusted raw TeX can name additional external inputs.

Figure, table, equation and section IDs must name actual counters. A label inside an unnumbered display is not a numbered equation. Do not fix an unresolved reference by guessing a similar label or citation author/year. Citation-key validation does not verify bibliographic identity; claims/references need independent primary-source review.

`BuildOrchestrator` rejects distinct targets with colliding resolved outputs, honors dependency order and blocks dependants after failed prerequisites. Cache reuse requires current configuration/source/resource/output identities; a missing/tampered output is not a hit. Caller-owned temporary directories and unowned files are not disposable cleanup targets. A source receipt is not deletion permission. Generating-package identity refuses linked package members rather than silently omitting potentially executed source.

Mermaid renders locally through pinned `mmdc`/Puppeteer by default. A remote Kroki endpoint is opt-in, credential-free and caller-trusted; there is no implicit public upload or Docker fallback. Failed rendering never reuses an old diagram as fresh evidence. Do not disable a browser sandbox merely to make a gate pass.

Native Mermaid `accTitle:` values become literal descriptive captions. Markdown/TeX delimiter characters remain text, not injected formatting or mathematics. Fenced code uses `fvextra` line breaking, including highlighted unbroken tokens, without inserting characters into copied literal code; native rendering requires that TeX package.

## Explicit, trusted templates

`TemplateEngine(TemplateConfig(template_dir=Path(...)))` requires a caller-selected directory. `create_publication_template(paper_type, template_dir)` never generates into an installed package and refuses conflicting template files. `render_scientific_document(..., template_dir=...)` uses that explicit directory. Removed ignored base-configuration arguments have no shim.

Missing variables/resources and malformed YAML/JSON fail before atomic document replacement. `validate_template` inspects syntax and reports required variables; `validation_scope: syntax-only` is not a successful data-bound render. Uncertainty formatting rounds the estimate at the uncertainty's significant-digit resolution, rather than treating scientific-notation characters as decimal places.

Jinja, raw TeX, local resources and plotting inputs are trusted author input, **not a security sandbox**. Output path/link checks do not establish isolation from malicious same-UID writers.

## Verify actual consumers

Use the focused behavioral suite plus [native gates](validation.md). `tools/release_smoke.py` builds an isolated consumer environment from a supplied wheel and locked runtime dependencies, exercises real installed generation/rejection/build paths, and retains command/source/artifact receipts. It must not fall back to an editable checkout or site packages. A passing help command alone is not installability proof.
