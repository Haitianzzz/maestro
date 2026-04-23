# Benchmark task manifest (SWE-bench Lite subset)

Curated via `benchmark/scripts/swebench_adapter.py` from the SWE-bench Lite
test split (Jimenez et al., NeurIPS 2023). Three tasks reproduce reliably
outside the official Docker harness; see the README's "Known limitations"
section for why the rest were excluded.

| status | instance_id | repo | patch Δ | failing_tests | one-line problem |
|---|---|---|---:|---:|---|
| OK | `mwaskom__seaborn-3190` | mwaskom/seaborn | 2 | 1 | Color mapping fails with boolean data |
| OK | `psf__requests-2317` | psf/requests | 4 | 8 | `method = builtin_str(method)` coerces unicode HTTP method to str |
| OK | `pytest-dev__pytest-11143` | pytest-dev/pytest | 1 | 1 | Rewrite fails when first expression of file is a number (mistaken as docstring) |

## Reproduction

To rebuild exactly this set on a clean checkout:

```bash
uv run python benchmark/scripts/swebench_adapter.py \
  --clone-cache benchmark/.clone-cache
```

Each task directory contains:

- `task.json` — `BenchmarkTask` JSON matching `maestro.benchmark.models.BenchmarkTask`
- `before/` — repo state at the SWE-bench `base_commit` (buggy)
- `after/` — `before/` + `patch` + `test_patch` applied (fixed)

Validated invariant: in `before/`, the declared `FAIL_TO_PASS` tests are
RED; in `after/`, they are GREEN. This is what `maestro bench` relies on.
