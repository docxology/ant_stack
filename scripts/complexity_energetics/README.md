# Complexity–Energetics Tools: Analytical Families and Fresh Outputs

Thin wrappers delegate to `antstack_core.analysis`; they do not implement a physics engine, neural circuit, policy controller or hardware calibration.

## Choose the Result Family Before Running

| Surface | Meaning |
|---|---|
| `antstack-ce MANIFEST --out NEW_DIRECTORY` | Canonical base workload family; explicit fresh destination and current-invocation inventory. |
| `run-all-antstack --config FILE` | Governed canonical family, task selection, reports and optional owned paper projection. |
| `generate_tables_and_numbers.py` / `analysis.ce_tables_and_numbers` | Separate enhanced analytical assumptions and descriptive fits; not canonical numerical substitutes. |
| Contact/neural/planning/comprehensive/figure scripts | Supplemental analyses of their declared model inputs. Inspect their options and output boundaries before generation. |

## Use Native Commands from the Checkout Root

```bash
uv run antstack-ce papers/complexity_energetics/manifest.example.yaml --validate-only
# Choose a genuinely new path; a repeated populated destination is an error.
bash scripts/complexity_energetics/run_ce.sh cache/ce-manual-001
uv run python -m antstack_core.analysis.ce_tables_and_numbers --output cache/enhanced-manual-001
```

The shell launcher requires the output path and forwards additional CE options. Never point standalone CE at the retained `papers/complexity_energetics/out` directory. Publication commands may intentionally project a completed owned generation there under the [custody contract](../../docs/reproducibility.md).

## Interpret Generated Values Conservatively

Default energy coefficients are assumptions, not device-specific calibration. Repeated deterministic model values are not empirical trials. Descriptive exponents, conditional erasure comparisons and figure formatting do not establish biological validity, physical efficiency, real-time feasibility or publication readiness. Distinguish module-isolated idle windows from additive shared-system accounting.

Use [scientific validation](../../docs/scientific_validation.md), [API contracts](../../docs/public_api_contracts.md) and [native acceptance checks](../../docs/validation.md). The package/private Node versions are not registry publication receipts.
