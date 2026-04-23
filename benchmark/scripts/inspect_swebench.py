"""One-shot introspection of SWE-bench Lite (not intended to be reused)."""

from __future__ import annotations

import sys
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).parent / "swebench_schema.txt"

ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")


def _truncate(value: object, cap: int = 500) -> str:
    s = repr(value)
    if len(s) <= cap:
        return s
    return s[:cap] + f"... ({len(s) - cap} more chars)"


with OUT.open("w", encoding="utf-8") as f:
    print(f"rows: {len(ds)}", file=f)
    print(f"columns: {ds.column_names}", file=f)
    print(file=f)
    print("Per-column feature + Python type (inferred from row 0):", file=f)
    row0 = ds[0]
    for col in ds.column_names:
        feat = ds.features[col]
        py_type = type(row0[col]).__name__
        print(f"  {col}: feature={feat!r}  py_type={py_type}", file=f)
    print(file=f)
    for i in range(3):
        row = ds[i]
        print(f"=== sample row {i} ===", file=f)
        for col in ds.column_names:
            print(f"--- {col} ---", file=f)
            print(_truncate(row[col]), file=f)
        print(file=f)

print(f"wrote {OUT}", file=sys.stderr)
