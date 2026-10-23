#!/usr/bin/env python3
"""Build the ResNet-50-style-only filtered lookup CSV.

Keeps from resnet_block_lookup_jetson_gpu.csv (repo root):
  - stem rows        (block_idx 0, SuperConvK3BNRELU(3,*,2,1))
  - body rows        (block_idx 1-4, SuperResK1K3K1 / K1K5K1 / K1K7K1 only)
  - FC rows          (block_idx 5, SuperConvK1BNRELU proj + Linear(2048,*))
Drops every other body block type (SuperResK3K3 / K5K5 / K7K7).

The beam engine (search_idw_global_constraint.py read_lookup) treats every
block_idx<5 row as a candidate with no hard-coded types, so this filtered CSV
restricts the search space to ResNet-50-style bottleneck blocks without any
engine changes.

Self-checks before writing:
  - row count == 11,941 (5 stem + 11,925 body + 11 FC)
  - per-position counts == {0:5, 1:1800, 2:3240, 3:3645, 4:3240, 5:11}
  - no non-K1K body rows
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # package root
SRC = ROOT / "resnet_block_lookup_jetson_gpu.csv"
OUT = ROOT / "resnet_block_lookup_jetson_gpu_r50style.csv"


def keep(row: dict) -> bool:
    bi = int(row["block_idx"])
    if bi == 0 or bi >= 5:
        return True  # stem + FC
    return row["block_str"].startswith("SuperResK1K")


def main() -> None:
    rows = list(csv.DictReader(open(SRC)))
    print(f"loaded {len(rows)} rows from {SRC}")

    kept = [r for r in rows if keep(r)]
    print(f"kept {len(kept)} rows")

    from collections import Counter
    by_pos = Counter(r["block_idx"] for r in kept)
    expected = {"0": 5, "1": 1800, "2": 3240, "3": 3645, "4": 3240, "5": 11}
    print(f"per-position counts: {dict(sorted(by_pos.items(), key=lambda x: int(x[0])))}")
    assert len(kept) == 11941, f"expected 11941 rows, got {len(kept)}"
    assert dict(by_pos) == expected, f"position counts mismatch: {by_pos}"

    bad = [r["block_str"] for r in kept
           if 0 < int(r["block_idx"]) < 5 and not r["block_str"].startswith("SuperResK1K")]
    assert not bad, f"non-K1K body rows present: {bad[:5]}"
    print("checks passed: 5 stem + 11925 K1K body + 11 FC")

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(kept)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
