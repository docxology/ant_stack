# Publication Pipeline Wrappers: Validation Is Not Repair or Release

These project-owned helpers delegate to package-owned paper, formatting and provenance APIs. Caller-owned manuscript/config files and native renderers remain prerequisites; installed `antstack-build` does not import this scripts directory.

## Build Configured Sources and Run Tests Separately

```bash
uv run antstack-build --validate-only
uv run antstack-build --paper complexity_energetics
uv run antstack-build
uv run pytest -q
uv run ruff check .
```

All-paper validation regenerates ignored `build_report.md`; it does not produce PDFs or certify earlier products. Actual build failure is nonzero. Tests are not implicitly run by paper building. `build_papers.sh` is a thin checkout launcher, not another build engine.

## Preserve Scientific Targets Instead of Guessing Repairs

The reference manager checks declared keys/counters; it cannot infer a missing citation, scientific identity or intended figure. Fix unresolved targets at their authoritative sources. Explicit formatting or key-number tools are author-invoked mutations, not an automatic build repair pass. Key-number updates act on declared placeholders, not arbitrary matching literals.

## Use One Source of Truth for Commands and Evidence

- [PDF rendering](../../PDF_RENDERING_GUIDE.md): supported syntax, toolchain, ordered-source/resource context and actual render proof.
- [Validation](../../docs/validation.md): native gates, installed-wheel smoke and separate remote main/tag/release observations.
- [Reproducibility](../../docs/reproducibility.md): fresh canonical families, current byte identities and owned projections.

Raw TeX/templates are trusted author input, not a sandbox. No browser-sandbox bypass, implicit remote render upload, fabricated result, registry publisher or deployment permission is supplied by these wrappers.
