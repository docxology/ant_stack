# Ant Stack Publication Guide: Configuration, Evidence, and Rendering

This configured documentation paper explains the installed publication path and its evidence limits. It is a software guide, not an empirical embodied-AI study or a certification of publication readiness.

## Validate inputs separately from rendered acceptance

```bash
uv run antstack-build --paper documentation --validate-only
uv run antstack-build --paper documentation
```

The configured source order is this overview followed by `PDF_RENDERING_GUIDE.md`. Examples in the guide are literal code examples; `mermaid_preprocessing: false` keeps them from becoming purported research diagrams. Full rendering still needs the actual Pandoc/XeLaTeX/font toolchain. A compiled file must be inspected for readable mathematics, glyphs, counter references and layout.

## Follow source-owned modular guidance

The root README is the entry point; `docs/public_api_contracts.md`, `docs/reproducibility.md` and `docs/validation.md` own API, evidence and verification contracts. This paper owns publishing syntax and configuration. Update those sources rather than retained concatenation/prerender scratch copies or historical PDFs. Do not infer a current validation result from an old `build_report.md`.
