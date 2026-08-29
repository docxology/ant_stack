# Manuscript status — ant_stack

## Repo type
Publication-track reproducible scientific-publication workspace (evidence:
`papers/ant_stack/` with `Abstract.md`, `Background.md`, `AntBody.md`,
`AntBrain.md`, `AntMind.md`, `Discussion.md`, `Appendices.md`, `Glossary.md`,
`paper_config.yaml`, generated PDF; `papers/complexity_energetics/` with its own
sections and `papers/documentation/`).

## Why no top-level `manuscript/` directory was created
The repo already manages its manuscripts under `papers/<name>/` with
per-paper configs (`papers/ant_stack/paper_config.yaml`), section markdown, and
generated PDFs (e.g. `Ant-Stack_v1_DAF_Aug-8-2025.pdf`,
`Complexity-Energetics_AntStack_9-30-2025.pdf` at repo root). Creating a
parallel top-level `manuscript/` with stub sections would duplicate real,
existing manuscript content — not done.

## What would trigger creating one
Migration onto the template's canonical `manuscript/` layout (SECTION files
`00_abstract.md` … `99_references.md` + `config.yaml`) if the `papers/`
structure is retired or unified with the template pipeline.
