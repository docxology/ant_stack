# Manuscript Status: Analytical Scope and Source-Bound Render Evidence

Software versioning, scientific validation, document rendering and repository publication are separate claims. This page identifies the reviewed 3.0 manuscript sources and render evidence; live main/tag and GitHub Release observations determine publication state.

## Three Configured Papers Have Separate Source Ownership

| Paper | Authoritative source/configuration | Evidence role |
|---|---|---|
| Ant Stack architecture | [`papers/ant_stack/`](../../papers/ant_stack/) and its `paper_config.yaml` | Conceptual Body/Brain/Mind architecture and proposed evaluations |
| Complexity and energetics | [`papers/complexity_energetics/`](../../papers/complexity_energetics/) and its `paper_config.yaml` | Analytical workload models and energy accounting with illustrative assumptions |
| Documentation/rendering guide | [`papers/documentation/`](../../papers/documentation/) and its `paper_config.yaml` | Software/publication guidance, not an empirical scientific study |

Each configuration's ordered `content.files` determines render inputs. `docs/manuscript/` contains status/navigation guidance, not a parallel manuscript tree and not the documentation paper's configured source. Do not edit generated concatenations or an old PDF as if they were source.

## Native Scientific Evidence Remains Analytical and Synthetic

The [CE example manifest](../../papers/complexity_energetics/manifest.example.yaml) selects deterministic base closed-form counters and a null meter. No independent raw hardware calibration, ant/species validation dataset, measured real-time benchmark or external replication record has been identified for that example. No implemented MuJoCo/FlyBody robot, complete active-inference agent or closed colony loop is established by these counters or the conceptual architecture.

| Claim family | Permitted interpretation |
|---|---|
| Operation, traffic, spike and coefficient-based energy outputs | Analytical model results bound to one completed run, not observed device/animal performance |
| Energy prices, activity fraction, dimensions and update rates | Explicit illustrative assumptions, not calibrated device/species values or biological optima |
| Sobol/Morris indices, posterior calibration, prediction error below 5% | Not established by the native manifest; prospective studies only |
| 100 deterministic repeats / 1000 bootstrap resamples | Computation/resampling effort, not independent sample size, empirical precision or guaranteed power |
| Critical contacts/sparsity/horizon, critical exponents and hysteresis | Not demonstrated by closed-form sweeps or descriptive finite-range fits |
| Biological topology advantage and universal real-time thresholds | Not established; supported counter names are `pgs`/`si`/`ldlt` and `random`/`structured` |
| CoT, metabolism and neural-power comparisons | Require matched primary measurements and boundaries; native CoT is unavailable |
| Mind as zero physical energy | Rejected: conceptual symbolic interpretation does not remove compute energy |
| Inverse-parameter plots | Illustrative proxies, not Pareto frontiers or measured performance trade-offs |
| External literature | Attributed context/results from those sources, not validation of this implementation |

Use [scientific validation](../scientific_validation.md) for claim categories and statistical limitations, [theory](../theory.md) for formulas/accounting, and [external-data requirements](../external_ant_data_integration.md) for the proposed empirical protocol. Schema checks and publication tooling cannot replace independent evidence.

## Base/Enhanced Models and Energy Windows Must Stay Explicit

Selected CE workloads use the **base** Body/Brain/Mind functions. Base Mind sums exact tree nodes through canonical `H_p` decision steps using `B`; enhanced Mind separately counts leaf policies with `max_policies=1000` by default or `None` for exact enumeration. A cap is not demonstrated quality-preserving planning.

Raw rows contain module-isolated compute plus idle over each workload window and must not be summed as shared-controller system energy. Scaling curves are compute-only 0.25 s estimates. The common-window table evaluates the selected models over inverse maximum configured frequency, adds assumed actuation/sensing and counts shared idle once. It is one decision only for modules at that frequency; flooring may omit slower modules when rates differ. CoT requires energy and displacement from the same locomotion interval, which the producer does not supply. A mass of 0.001 kg is 1 g, not 1 mg; do not substitute it for the example's nominal engineering mass.

## Artifact Identity Does Not Establish a Fresh Full Manuscript Render

Direct CE generation confines products to fresh `--out`; governed `outputs/<run_id>/` results carry schema-v2 relative artifact identities, input snapshots and current-invocation inventories. Explicit completed-result projection supplies paper-local compatibility products while preserving source-owned guides. Use the [reproducibility contract](../reproducibility.md), not a second hand-maintained workflow here.

`Generated.md` is numerical evidence only when bound to its selected completed generation and figures/CSV/summary. The three PDFs identified in [manuscript provenance](../../papers/manuscripts.provenance.json) are refreshed, source-bound renders. Other retained PDFs, figures and concatenations remain historical unless a current receipt identifies them. Run-all's copied `historical_paper` products are not new renders; its optional generated-results PDF is not a full manuscript build.

## Reviewed Full Renders, Graphical Assets and Publication Custody

[CI run 42548062547](https://github.com/docxology/ant_stack/actions/runs/42548062547) built all three configured papers from source `422c71e5a402d31e7b2a2676423a2651277509ce`. All 65 resulting pages were rasterized and reviewed for layout, legible graphics, informative headings, mathematics, citations, attached counters and literal code. The [per-manuscript receipt](../../papers/manuscripts.provenance.json) binds actual PDF bytes to authored/generated sources, graphical resources, runtime font and the completed CI run; compilation alone was not accepted as visual proof.

Both conceptual abstracts have vector/raster exports and a source/output receipt in [the graphical family](../../papers/graphical_abstracts/graphical_abstracts.provenance.json). Repeated native exports matched bytes in the recorded environment, and the actual final manuscript schematics were inspected. These schematics contain no measured data. The canonical base-result projection and separately labelled enhanced-model supplements were refreshed from verified completed families, with prior trees preserved.

Local Chromium sandbox and requested-font failures remain recorded limitations of those local attempts, not successful full renders. The Linux proof used the authored configurations and an enabled native browser sandbox. Repository publication requires live refs/checks plus the enabled class, reviewed plan and exact operator phrase; no registry publisher is supplied. See [release notes](../../RELEASE_NOTES.md), [software validation](../validation.md) and [PDF rendering](../../PDF_RENDERING_GUIDE.md).

## Historical DOI Identifies the 2025 v1, Not This Revision

[DataCite's primary metadata for DOI 10.5281/zenodo.17238737](https://api.datacite.org/dois/10.5281/zenodo.17238737) identifies version `v1`, issued 2025-09-30, of *Computational Complexity and Energetics of the Ant Stack*. The deposited abstract retains earlier calibration/critical-behavior/biological-comparison claims that are not established by the revised native analytical workflow. That historical record is not an identifier or validation statement for these revised sources, and repository edits do not correct or republish it. External deposit correction/versioning remains a separate release-owner action.
