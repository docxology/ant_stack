# Tests

The test suite verifies package APIs, complexity energetics workflows, rendering contracts, documentation contracts, and canonical run-all orchestration.

## Layout

| Path | Purpose |
| --- | --- |
| `tests/antstack_core` | Unit and integration tests for public package modules, CLIs, docs contracts, and orchestration. |
| `tests/complexity_energetics` | Complexity energetics runner, generated key-number, and workflow tests. |
| `tests/core_rendering` | Rendering, paper configuration, and core integration tests. |

## Commands

```bash
uv run pytest --collect-only -q
uv run pytest -q
uv run pytest -q tests/antstack_core/test_docs_contract.py
uv run pytest -q tests/antstack_core/test_run_all_antstack.py
```

## Current Contract

- The default suite is expected to pass as `676 passed, 10 subtests passed`.
- Tests should not depend on bytecode, local caches, OS metadata, or external credentials.
- Documentation tests enforce folder signposting, public API signposting, and required Mermaid diagrams in curated guides.
- Output-producing integration tests should use temporary directories unless the requested workflow explicitly updates versioned generated artifacts.

## Adding Tests

- Prefer focused tests that exercise real package methods.
- Keep CLI tests on installed wrapper contracts.
- For docs changes, update `tests/antstack_core/test_docs_contract.py` when the curated guide set changes.
- For generated outputs, assert manifest/provenance content and non-empty artifacts rather than just file existence.
