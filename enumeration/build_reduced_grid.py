#!/usr/bin/env python3
"""Build a channel-restricted lookup CSV for the #5b reduced-grid check.

Filters idw_block_lookup_jetson_gpu.csv to per-block channel subsets, kernel/
expansion/sub_layers subsets, and asserts connectivity so the beam and exact
searches run on the SAME reduced grid.

Connectivity asserts (hard fail):
  - stem outs   subset of block-1 ins
  - block-i outs subset of block-(i+1) ins (i = 1,2,3)
  - block-4 outs subset of FC proj ins (proj rows must exist for every b4 out)
  - every allowed in-channel of blocks 1..4 has >= 1 row
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
FULL = HERE / "idw_block_lookup_jetson_gpu.csv"
CONSTRAINT = "jetson_gpu_latency_ms_median"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="output reduced CSV path")
    p.add_argument("--stem", required=True, help="comma list of stem out channels")
    p.add_argument("--c1", required=True)
    p.add_argument("--c2", required=True)
    p.add_argument("--c3", required=True)
    p.add_argument("--c4", required=True)
    p.add_argument("--kernels", default="3,5")
    p.add_argument("--exps", default="2,4")
    p.add_argument("--sub", default="1", help="comma list of allowed sub_layers")
    p.add_argument("--num-classes", type=int, default=100)
    return p.parse_args()


def cs(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


def main() -> None:
    args = parse_args()
    sets = {i: cs(v) for i, v in enumerate([args.stem, args.c1, args.c2, args.c3, args.c4], start=0)}
    kernels = cs(args.kernels)
    exps = cs(args.exps)
    subs = cs(args.sub)

    df = pd.read_csv(FULL)
    out_rows = []

    # stem: out in stem set
    stem = df[(df.block_idx == 0) & (df.out_channels.isin(sets[0]))]
    assert len(stem) == len(sets[0]), f"stem: missing rows for outs {sets[0]} (got {stem.out_channels.unique().tolist()})"
    out_rows.append(stem)

    # blocks 1..4: in/out channel sets + kernel/exp/sub filters
    for i in range(1, 5):
        blk = df[
            (df.block_idx == i)
            & (df.in_channels.isin(sets[i - 1]))
            & (df.out_channels.isin(sets[i]))
            & (df.kernel_size.isin(kernels))
            & (df.expansion_ratio.isin(exps))
            & (df.sub_layers.isin(subs))
        ]
        assert len(blk) > 0, f"block {i}: empty after filter"
        for in_ch in sets[i - 1]:
            assert (blk.in_channels == in_ch).any(), f"block {i}: no rows with in_ch={in_ch}"
        out_rows.append(blk)

    # FC rows: proj for every b4 out + Linear(2048, num_classes)
    linear_pat = f"Linear(2048,{args.num_classes})"
    fc = df[
        (df.block_idx >= 5)
        & ((df.in_channels.isin(sets[4])) | (df.block_str.str.contains(linear_pat, regex=False)))
    ]
    for oc in sets[4]:
        assert (fc.in_channels == oc).any(), f"FC: missing proj row for in_ch={oc}"
    assert fc.block_str.str.contains(linear_pat, regex=False).any(), "FC: missing Linear(2048,N) row"
    out_rows.append(fc)

    reduced = pd.concat(out_rows, ignore_index=True)

    # connectivity asserts across blocks
    for i in range(4):
        out_i = set(reduced[reduced.block_idx == i].out_channels)
        in_next = set(reduced[reduced.block_idx == i + 1].in_channels)
        assert out_i <= in_next, f"connectivity broken: block {i} outs {sorted(out_i - in_next)} not in block {i+1} ins"

    print("connectivity asserts passed")
    for i in range(6):
        sub = reduced[reduced.block_idx == i]
        print(f"  block {i}: {len(sub)} rows, outs={sorted(sub.out_channels.unique())}")
    reduced.to_csv(args.out, index=False, float_format="%.17g")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
