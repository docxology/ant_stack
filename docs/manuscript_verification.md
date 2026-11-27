# Replay the architecture paper with Template

The shared runner renders the eleven configured sources of *The Ant Stack*.
The [manuscript index](manuscript/README.md) also links the separate
complexity–energetics and documentation papers, which use the
[native paper build](PDF_RENDERING_GUIDE.md).

Use a clean disposable checkout in `projects/working/ant_stack` in
[Template at commit `87704feae0a4490ef7dad9c931d268e8c30273c2`](https://github.com/docxology/template/tree/87704feae0a4490ef7dad9c931d268e8c30273c2).
Provision the Template checkout with `uv sync --locked` and install compatible
Pandoc, pandoc-crossref, TeX and Poppler. From the Template root:

```sh
uv run --no-sync python projects/working/ant_stack/tools/prepare_template_manuscript.py
uv run --no-sync python scripts/pipeline/stage_03_render.py --project working/ant_stack --skip-manuscript-hydration
uv run --no-sync python scripts/pipeline/stage_04_validate.py --project working/ant_stack
```

The staging script declares the generated roster for this architecture-only replay,
matches the native section order and uses the authored
schematic and the current Mermaid diagrams. Their replay PNG filenames bind
them to the SHA-256 prefix of their trimmed Mermaid blocks. The authored
Mermaid remains in the paper and the replay needs no browser. A changed
diagram requires a matching PNG rendered from its updated Mermaid block
with the native sandbox-enabled toolchain.

Read the PDF at `output/pdf/ant_stack_combined.pdf`, the HTML at
`output/web/index.html` and the validation reports under `output/reports/`
in the disposable checkout. The paper reports a proposed architecture,
observational context and finite analytical examples. See the
[scientific status record](manuscript/MANUSCRIPT_STATUS.md) for evidence limits.
