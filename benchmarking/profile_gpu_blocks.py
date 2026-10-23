#!/usr/bin/env python3
"""Profile per-block GPU latency for the IDW block lookup, including the FC head.

For each row in the input lookup CSV, build the PlainNet block, run it on GPU
with random input at the block's input resolution, and record median/min/max
latency over `runs` iterations.

Then (unless --no-fc) synthesise and profile the fixed FC head as block_idx=5
rows, which the enumeration reads via --fc-lookup-from-csv:

  * the projection conv  SuperConvK1BNRELU(<in_ch>,<last_ch>,1,1)  -- one row per
    distinct final-stage out_channels found in the input CSV
  * the classifier       Linear(<last_ch>,<num_classes>)           -- one row per
    --fc-num-classes value

FC rows carry correct flops and params (conv: in*last*res^2 MACs,
in*last + 2*last params including BatchNorm; linear: in*out MACs,
in*out + out params). Their score columns are 0, since the FC head is outside
the aggregation window used by the search.

Output: a CSV with the input rows + jetson_gpu_latency columns, plus the FC rows.
Resume with --skip-existing (keys on block_idx + block_str + in_channels +
out_channels + input_resolution + sub_layers).

Usage:
  python -m benchmarking.profile_gpu_blocks \
      --input-csv  data/block_lookup/idw_block_lookup.csv \
      --output-csv data/block_lookup/idw_block_lookup_jetson_gpu.csv \
      --device cuda:0 --warmup 20 --runs 50 --skip-existing
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

import torch

from utils.paths import LOOKUP_CSV, setup_model_path, setup_repo_path

setup_repo_path()
setup_model_path()
import PlainNet  # noqa: E402


LATENCY_FIELDS = [
    "jetson_gpu_latency_backend",
    "jetson_gpu_latency_ms_median",
    "jetson_gpu_latency_ms_min",
    "jetson_gpu_latency_ms_max",
]


def key_for_row(row: dict) -> tuple:
    return (
        row.get("block_idx", ""),
        row.get("block_str", ""),
        row.get("in_channels", ""),
        row.get("out_channels", ""),
        row.get("input_resolution", ""),
        row.get("sub_layers", ""),
    )


def row_has_latency(row: dict) -> bool:
    val = row.get("jetson_gpu_latency_ms_median", "")
    return bool(val) and val != ""


def build_block_from_row(row: dict):
    """Construct a PlainNet block module from a lookup row."""
    block_str = row["block_str"]
    blocks = PlainNet.create_netblock_list_from_str(block_str, no_create=False)
    if not blocks:
        raise ValueError(f"Failed to parse: {block_str}")
    # Wrap into a small Sequential so we can call it with a standard input
    import torch.nn as nn
    # PlainNet blocks have module_list; we want the actual forward-ready module.
    # Use the block directly since it's a nn.Module.
    block = blocks[0]
    # Make sure in_channels matches the row
    in_ch = int(row["in_channels"])
    block.set_in_channels(in_ch)
    return block


def measure_latency_ms(model, x, warmup=20, runs=50):
    model.eval()
    with torch.no_grad():
        for _ in range(max(1, warmup)):
            _ = model(x)
        torch.cuda.synchronize()

        times = []
        for _ in range(max(1, runs)):
            start = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000.0)
    return statistics.median(times), min(times), max(times)


def profile_row(row: dict, device: torch.device, warmup: int, runs: int) -> dict:
    block = build_block_from_row(row).to(device)
    in_ch = int(row["in_channels"])
    res = int(row["input_resolution"])
    x = torch.randn(1, in_ch, res, res, device=device)

    median_ms, min_ms, max_ms = measure_latency_ms(block, x, warmup=warmup, runs=runs)

    # Free memory aggressively — there are 114K rows
    del block, x
    torch.cuda.empty_cache()

    return {
        "jetson_gpu_latency_backend": str(device),
        "jetson_gpu_latency_ms_median": f"{median_ms:.6f}",
        "jetson_gpu_latency_ms_min": f"{min_ms:.6f}",
        "jetson_gpu_latency_ms_max": f"{max_ms:.6f}",
    }


SCORE_FIELDS = [
    "analytic_ssr_exp_proj_dw_sum",
    "analytic_ssr_exp_proj_dw_mean",
    "analytic_ssr_exp_proj_dw_geomean",
    "analytic_ssr_exp_proj_dw_product",
    "analytic_ssr_exp_proj_dw_min",
]


def _blank_row(fieldnames: list[str]) -> dict:
    row = {k: "" for k in fieldnames}
    for fld in SCORE_FIELDS:
        if fld in row:
            row[fld] = "0.0"
    return row


def final_stage_out_channels(input_rows: list[dict]) -> list[int]:
    """Distinct out_channels of the last searchable block, which feed the FC head."""
    searchable = [r for r in input_rows if r.get("block_idx", "").isdigit()]
    if not searchable:
        return []
    last_idx = max(int(r["block_idx"]) for r in searchable)
    return sorted({int(r["out_channels"]) for r in searchable
                   if int(r["block_idx"]) == last_idx})


def fc_rows(in_channels: list[int], last_channels: int, num_classes: list[int],
            input_res: int, fieldnames: list[str]) -> list[dict]:
    """block_idx=5 rows for the projection conv and the classifier."""
    rows = []
    for in_ch in in_channels:
        row = _blank_row(fieldnames)
        row.update({
            "block_idx": "5",
            "block_str": f"SuperConvK1BNRELU({in_ch},{last_channels},1,1)",
            "in_channels": in_ch, "out_channels": last_channels,
            "bottleneck_channels": 0, "expansion_ratio": 0,
            "kernel_size": 1, "sub_layers": 1, "stride": 1,
            "input_resolution": input_res,
            # 1x1 conv MACs over the spatial map; params include BatchNorm
            "flops": float(in_ch * last_channels * input_res * input_res),
            "params": float(in_ch * last_channels + 2 * last_channels),
        })
        rows.append(row)
    for n_cls in num_classes:
        row = _blank_row(fieldnames)
        row.update({
            "block_idx": "5",
            "block_str": f"Linear({last_channels},{n_cls})",
            "in_channels": last_channels, "out_channels": n_cls,
            "bottleneck_channels": 0, "expansion_ratio": 0,
            "kernel_size": 1, "sub_layers": 1, "stride": 1,
            "input_resolution": 1,
            "flops": float(last_channels * n_cls),
            "params": float(last_channels * n_cls + n_cls),
        })
        rows.append(row)
    return rows


def profile_fc_row(row: dict, device: torch.device, warmup: int, runs: int) -> dict:
    """Time one FC row: the projection conv on a feature map, or the classifier."""
    import torch.nn as nn
    if row["block_str"].startswith("Linear("):
        model = nn.Linear(int(row["in_channels"]), int(row["out_channels"]),
                          bias=True).to(device)
        x = torch.randn(1, int(row["in_channels"]), device=device)
    else:
        model = build_block_from_row(row).to(device)
        res = int(row["input_resolution"])
        x = torch.randn(1, int(row["in_channels"]), res, res, device=device)

    median_ms, min_ms, max_ms = measure_latency_ms(model, x, warmup=warmup, runs=runs)
    del model, x
    torch.cuda.empty_cache()
    return {
        "jetson_gpu_latency_backend": str(device),
        "jetson_gpu_latency_ms_median": f"{median_ms:.6f}",
        "jetson_gpu_latency_ms_min": f"{min_ms:.6f}",
        "jetson_gpu_latency_ms_max": f"{max_ms:.6f}",
    }


def load_existing(output_csv: Path) -> dict:
    if not output_csv.exists():
        return {}
    with output_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {key_for_row(row): row for row in rows}


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    os.makedirs(path.parent, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", "--input_csv", dest="input_csv", required=True,
                        help="IDW lookup CSV (e.g., idw_block_lookup.csv)")
    parser.add_argument("--output-csv", "--output_csv", dest="output_csv", required=True,
                        help="Output CSV with latency columns appended")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--progress-every", "--progress_every", dest="progress_every",
                        type=int, default=50)
    parser.add_argument("--skip-existing", "--skip_existing", dest="skip_existing",
                        action="store_true", help="Skip rows already profiled in output CSV")
    parser.add_argument("--start-row", "--start_row", dest="start_row", type=int, default=1,
                        help="1-indexed start row (inclusive)")
    parser.add_argument("--end-row", "--end_row", dest="end_row", type=int, default=0,
                        help="1-indexed end row (inclusive); 0 means end of file")
    parser.add_argument("--flush-every", "--flush_every", dest="flush_every",
                        type=int, default=200,
                        help="Write output CSV every N rows as a checkpoint")
    parser.add_argument("--no-fc", "--no_fc", dest="no_fc", action="store_true",
                        help="Skip the FC head (block_idx=5) rows")
    parser.add_argument("--fc-last-channels", "--fc_last_channels",
                        dest="fc_last_channels", type=int, default=2048,
                        help="Projection width feeding the classifier (default 2048)")
    parser.add_argument("--fc-num-classes", "--fc_num_classes", dest="fc_num_classes",
                        default="100,1000,1010",
                        help="Comma-separated classifier widths to profile")
    parser.add_argument("--fc-input-res", "--fc_input_res", dest="fc_input_res",
                        type=int, default=7,
                        help="Spatial resolution feeding the projection conv")
    parser.add_argument("--fc-in-channels", "--fc_in_channels", dest="fc_in_channels",
                        default="",
                        help="Comma-separated projection input widths; default is "
                             "every distinct final-stage out_channels in the input CSV")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is not available"
    device = torch.device(args.device)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})", flush=True)

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    with input_csv.open(newline="", encoding="utf-8") as f:
        input_rows = list(csv.DictReader(f))
        input_fieldnames = list(input_rows[0].keys())

    # Output fieldnames = input + latency
    output_fieldnames = list(input_fieldnames)
    for fld in LATENCY_FIELDS:
        if fld not in output_fieldnames:
            output_fieldnames.append(fld)

    # Load existing (resume)
    existing = load_existing(output_csv) if args.skip_existing else {}
    print(f"Input rows: {len(input_rows)}  existing profiled: {len(existing)}", flush=True)

    start = max(0, args.start_row - 1)
    end = args.end_row if args.end_row > 0 else len(input_rows)
    total_range = end - start
    print(f"Profiling rows {start + 1}..{end}  ({total_range} rows)", flush=True)

    # Preserve already-profiled rows from 'existing' for rows we won't touch
    result_rows_by_key: dict = {}
    for k, r in existing.items():
        result_rows_by_key[k] = r

    t0 = time.time()
    profiled = 0
    skipped = 0
    errors = 0

    for idx in range(start, end):
        row = input_rows[idx]
        key = key_for_row(row)

        if args.skip_existing and key in existing and row_has_latency(existing[key]):
            skipped += 1
            # keep existing with all original fields
            merged = {**row, **{fld: existing[key].get(fld, "") for fld in LATENCY_FIELDS}}
            result_rows_by_key[key] = merged
            continue

        try:
            lat = profile_row(row, device, args.warmup, args.runs)
            merged = {**row, **lat}
            result_rows_by_key[key] = merged
            profiled += 1
        except Exception as e:
            errors += 1
            merged = {**row, **{fld: "" for fld in LATENCY_FIELDS}}
            result_rows_by_key[key] = merged
            print(f"[{idx+1}/{end}] ERROR on {row.get('block_str','?')}: {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()

        # Progress
        done = profiled + skipped + errors
        if done % args.progress_every == 0:
            elapsed = time.time() - t0
            rate = profiled / elapsed if elapsed > 0 and profiled > 0 else 0
            remain = (total_range - done) / rate if rate > 0 else 0
            print(
                f"[{idx+1}/{end}] profiled={profiled} skipped={skipped} errors={errors}  "
                f"rate={rate:.1f} rows/s  eta={remain/60:.1f} min",
                flush=True,
            )

        # Checkpoint write
        if done % args.flush_every == 0:
            rows_out = list(result_rows_by_key.values())
            write_rows(output_csv, rows_out, output_fieldnames)

    # FC head (block_idx=5): the enumeration reads these via --fc-lookup-from-csv
    fc_profiled = fc_skipped = fc_errors = 0
    if not args.no_fc:
        if args.fc_in_channels:
            fc_in = [int(v) for v in args.fc_in_channels.split(",") if v.strip()]
        else:
            fc_in = final_stage_out_channels(input_rows)
        fc_classes = [int(v) for v in args.fc_num_classes.split(",") if v.strip()]
        candidates = fc_rows(fc_in, args.fc_last_channels, fc_classes,
                             args.fc_input_res, output_fieldnames)
        print(f"\nFC head: {len(fc_in)} projection widths x last_channels="
              f"{args.fc_last_channels}, classifiers {fc_classes} "
              f"({len(candidates)} rows)", flush=True)
        for row in candidates:
            key = key_for_row({k: str(v) for k, v in row.items()})
            if args.skip_existing and key in existing and row_has_latency(existing[key]):
                fc_skipped += 1
                result_rows_by_key[key] = {**row, **{fld: existing[key].get(fld, "")
                                                     for fld in LATENCY_FIELDS}}
                continue
            try:
                result_rows_by_key[key] = {**row,
                                           **profile_fc_row(row, device,
                                                            args.warmup, args.runs)}
                fc_profiled += 1
            except Exception as e:
                fc_errors += 1
                result_rows_by_key[key] = {**row,
                                           **{fld: "" for fld in LATENCY_FIELDS}}
                print(f"  FC ERROR on {row['block_str']}: {type(e).__name__}: {e}",
                      flush=True)
                torch.cuda.empty_cache()
        print(f"FC head: profiled={fc_profiled} skipped={fc_skipped} "
              f"errors={fc_errors}", flush=True)

    # Final write
    rows_out = list(result_rows_by_key.values())
    write_rows(output_csv, rows_out, output_fieldnames)

    elapsed = time.time() - t0
    print(
        f"\nDone. profiled={profiled} skipped={skipped} errors={errors}  "
        f"total_time={elapsed/60:.1f} min",
        flush=True,
    )
    print(f"Output: {output_csv}")


if __name__ == "__main__":
    main()
