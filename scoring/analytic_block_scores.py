#!/usr/bin/env python3
"""Analytic per-block primitive scores: the inner (within-block) aggregation.

These are the `analytic_ssr_exp_proj_dw_{sum,mean,geomean,product,min}` columns
of the block lookup, which the enumeration reads via --metric. They are computed
in closed form from the block string's parameters -- no model is built and no
forward pass is run, which is what makes the search decomposable and
precomputable.

An IDW block decomposes into 6 x sub_layers primitive terms, two units per
sub-layer and three convolutions per unit:

    per sub-layer:
      dw1 = smart_round(bottleneck * E, base=4)
      unit 1:  q_pw(last_channels, dw1)   q_dw(dw1, K)   q_pw(dw1, bottleneck)
      dw2 = smart_round(out * E, base=4)
      unit 2:  q_pw(bottleneck, dw2)      q_dw(dw2, K)   q_pw(dw2, out)
      last_channels = out

    q_pw(cin, cout) = sqrt(cin * cout) / (sqrt(cin) + sqrt(cout))
    q_dw(C, K)      = C * K^2 / (1 + K)^2

The stem (`SuperConvK3BNRELU(3,out,...)`) contributes the single term
q_pw(3, out). The FC head (block_idx 5) scores 0.0: it is architecturally fixed
and the search excludes it from aggregation anyway (--skip-stem-in-agg).

This produces only the score columns. The `jetson_gpu_latency_ms_*` columns
come separately from `scoring.build_block_lookup`, which needs the target
device.

Note this is NOT interchangeable with `scoring/stable_rank.py`. The functions
there walk a *built* module, so the convolution set varies with the residual
projection and expansion ratio and the term count is not 6 x sub_layers. Use
this module to reproduce or regenerate the lookup's score columns.

Usage:
    # score one block
    python -m scoring.analytic_block_scores --block "SuperResIDWE6K3(8,32,2,16,1)"

    # check every row of a lookup against this implementation
    python -m scoring.analytic_block_scores --verify data/block_lookup/idw_block_lookup_jetson_gpu.csv

    # write/refresh the score columns of a lookup
    python -m scoring.analytic_block_scores --input-csv grid.csv --output-csv scored.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys

IDW_PAT = re.compile(r'SuperResIDWE(\d+)K(\d+)\((\d+),(\d+),(\d+),(\d+),(\d+)\)')
STEM_PAT = re.compile(r'SuperConvK3BNRELU\((\d+),(\d+),(\d+),(\d+)\)')

SCORE_COLUMNS = (
    "analytic_ssr_exp_proj_dw_sum",
    "analytic_ssr_exp_proj_dw_mean",
    "analytic_ssr_exp_proj_dw_geomean",
    "analytic_ssr_exp_proj_dw_product",
    "analytic_ssr_exp_proj_dw_min",
)


def smart_round(x: float, base: int = 4) -> int:
    """Round to the nearest multiple of `base`, never below `base`."""
    return max(base, round(x / float(base)) * base)


def q_pw(cin: int, cout: int) -> float:
    """Pointwise term: the asymptotic stable rank of a random cin x cout map."""
    if cin <= 0 or cout <= 0:
        return 0.0
    return math.sqrt(cin * cout) / (math.sqrt(cin) + math.sqrt(cout))


def q_dw(channels: int, kernel: int) -> float:
    """Depthwise term for a kernel x kernel conv: C * K^2 / (1 + K)^2."""
    if channels <= 0 or kernel <= 0:
        return 0.0
    return channels * kernel * kernel / float((1 + kernel) ** 2)


def idw_terms(block_str: str, in_channels: int | None = None) -> list[float]:
    """The 6 x sub_layers primitive terms of one IDW block."""
    m = IDW_PAT.match(block_str)
    if m is None:
        raise ValueError(f"not an IDW block: {block_str!r}")
    exp, kernel, cin, cout, _stride, bottleneck, sub_layers = map(int, m.groups())
    last = cin if in_channels is None else int(in_channels)

    terms: list[float] = []
    for _ in range(sub_layers):
        dw1 = smart_round(bottleneck * exp)
        terms += [q_pw(last, dw1), q_dw(dw1, kernel), q_pw(dw1, bottleneck)]
        dw2 = smart_round(cout * exp)
        terms += [q_pw(bottleneck, dw2), q_dw(dw2, kernel), q_pw(dw2, cout)]
        last = cout
    return terms


def block_terms(block_str: str, in_channels: int | None = None) -> list[float]:
    """Primitive terms for any lookup row: IDW, stem, or FC head."""
    if IDW_PAT.match(block_str):
        return idw_terms(block_str, in_channels)
    m = STEM_PAT.match(block_str)
    if m:
        return [q_pw(int(m.group(1)), int(m.group(2)))]
    # FC head (Linear / SuperConvK1BNRELU projection): fixed, excluded from
    # score aggregation, so it carries no primitive terms.
    return []


def aggregate(terms: list[float]) -> dict[str, float]:
    """The five inner aggregations stored in the lookup."""
    if not terms:
        return {c: 0.0 for c in SCORE_COLUMNS}
    n = len(terms)
    total = math.fsum(terms)
    product = 1.0
    for t in terms:
        product *= t
    return {
        "analytic_ssr_exp_proj_dw_sum": sum(terms),
        "analytic_ssr_exp_proj_dw_mean": sum(terms) / n,
        "analytic_ssr_exp_proj_dw_geomean": product ** (1.0 / n),
        "analytic_ssr_exp_proj_dw_product": product,
        "analytic_ssr_exp_proj_dw_min": min(terms),
    }


def block_scores(block_str: str, in_channels: int | None = None) -> dict[str, float]:
    return aggregate(block_terms(block_str, in_channels))


# ---------------------------------------------------------------------------
def verify(path: str, tol: float = 1e-9) -> int:
    """Compare this implementation against every scored row of a lookup CSV."""
    checked = matched = skipped = 0
    worst = 0.0
    failures = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not all(c in row for c in SCORE_COLUMNS):
                raise SystemExit(f"{path} has no score columns to verify")
            try:
                in_ch = int(row["in_channels"])
            except (KeyError, ValueError):
                in_ch = None
            got = block_scores(row["block_str"], in_ch)
            if not row[SCORE_COLUMNS[0]].strip():
                skipped += 1
                continue
            checked += 1
            rel = 0.0
            for col in SCORE_COLUMNS:
                want = float(row[col])
                rel = max(rel, abs(got[col] - want) / max(abs(want), 1e-12))
            worst = max(worst, rel)
            if rel <= tol:
                matched += 1
            elif len(failures) < 10:
                failures.append((row["block_idx"], row["block_str"], rel))

    print(f"rows checked              : {checked}" + (f"  (skipped {skipped} blank)" if skipped else ""))
    print(f"exact on all 5 aggregates : {matched}/{checked}")
    print(f"worst relative error      : {worst:.3e}")
    for idx, bs, rel in failures:
        print(f"  MISMATCH block_idx={idx} {bs}  rel={rel:.3e}")
    return 0 if matched == checked else 1


def rescore(input_csv: str, output_csv: str) -> None:
    """Write a copy of a lookup with the score columns (re)computed."""
    with open(input_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0].keys()) if rows else []
    for col in SCORE_COLUMNS:
        if col not in fields:
            fields.append(col)
    for row in rows:
        try:
            in_ch = int(row["in_channels"])
        except (KeyError, ValueError):
            in_ch = None
        row.update({k: repr(v) if False else v
                    for k, v in block_scores(row["block_str"], in_ch).items()})
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {output_csv}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analytic per-block primitive scores for the block lookup.")
    parser.add_argument("--block", help="score a single block string")
    parser.add_argument("--in-channels", "--in_channels", dest="in_channels",
                        type=int, default=None,
                        help="override the block's own in_channels")
    parser.add_argument("--verify", metavar="LOOKUP_CSV",
                        help="check every scored row of a lookup against this module")
    parser.add_argument("--tol", type=float, default=1e-9,
                        help="relative tolerance for --verify (default 1e-9)")
    parser.add_argument("--input-csv", "--input_csv", dest="input_csv",
                        help="lookup/grid CSV to (re)score")
    parser.add_argument("--output-csv", "--output_csv", dest="output_csv",
                        help="where to write the rescored CSV")
    args = parser.parse_args()

    if args.block:
        terms = block_terms(args.block, args.in_channels)
        print(f"block      : {args.block}")
        print(f"terms ({len(terms)}) : {[round(t, 6) for t in terms]}")
        for col, val in block_scores(args.block, args.in_channels).items():
            print(f"  {col:34} {val!r}")
        return 0
    if args.verify:
        return verify(args.verify, args.tol)
    if args.input_csv:
        if not args.output_csv:
            parser.error("--input-csv requires --output-csv")
        rescore(args.input_csv, args.output_csv)
        return 0
    parser.error("give --block, --verify, or --input-csv/--output-csv")
    return 2


if __name__ == "__main__":
    sys.exit(main())
