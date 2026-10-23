#!/usr/bin/env python
"""Benchmark every unique IDW block_str in the grid via PyTorch CPU on Jetson.

Matches the protocol used for the earlier `jetson_cpu_latency_ms_median` column:
PyTorch eager mode, float32, single-thread control via --num-threads, warmup
+ timed iterations, per-run median.

Reads `standalone/data/idw_block_lookup.csv`, de-duplicates by (block_str,
input_resolution), builds each block as a PlainNet module, times a
batch=1 forward pass many times, writes median/min/max ms.

Usage (on Jetson):
    python bench_cpu_blocks.py \\
        --in-csv  data/block_lookup/idw_block_lookup.csv \\
        --out-csv data/block_lookup/idw_block_lookup_cpu_torch.csv \\
        --num-threads 8 \\
        [--block-indices 0,1,2,3,4] [--limit 100] [--resume]

Before running, pin Jetson:
    sudo nvpmodel -m 0
    sudo jetson_clocks
"""
from __future__ import annotations
import argparse, csv, os, sys, time
from pathlib import Path

from utils.paths import LOOKUP_CSV, setup_model_path, setup_repo_path

setup_repo_path()
setup_model_path()

import torch
import torch.nn as nn
import PlainNet  # noqa: E402


BLOCK_INPUT_RES = {0: 224, 1: 112, 2: 56, 3: 28, 4: 14}


class BlockModule(nn.Module):
    def __init__(self, block_str: str):
        super().__init__()
        bl = PlainNet.create_netblock_list_from_str(block_str, no_create=False)
        assert len(bl) == 1, f"Expected 1 block, got {len(bl)} from {block_str!r}"
        self.block = bl[0]
    def forward(self, x): return self.block(x)


def time_one_module(module, input_shape, warmup: int, repeats: int) -> dict:
    """Return median/mean/min/max/p90 latency in ms for `repeats` forward passes."""
    module.eval()
    x = torch.randn(*input_shape)
    with torch.no_grad():
        for _ in range(warmup):
            _ = module(x)
    times = []
    with torch.no_grad():
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = module(x)
            times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    n = len(times)
    return {
        'cpu_median_ms': times[n // 2],
        'cpu_mean_ms': sum(times) / n,
        'cpu_min_ms': times[0],
        'cpu_max_ms': times[-1],
        'cpu_p90_ms': times[int(n * 0.9)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-csv', default=str(LOOKUP_CSV))
    ap.add_argument('--out-csv', default=str(LOOKUP_CSV.parent / 'idw_block_lookup_cpu_torch.csv'))
    ap.add_argument('--block-indices', default='0,1,2,3,4')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--num-threads', type=int, default=8,
                    help='torch.set_num_threads(); Jetson Orin NX has 8 Cortex-A78AE cores')
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--repeats', type=int, default=5,
                    help='Timed iterations per block')
    args = ap.parse_args()

    torch.set_num_threads(args.num_threads)
    # Also constrain MKL / OMP parallelism via envs for determinism
    os.environ.setdefault('OMP_NUM_THREADS', str(args.num_threads))
    os.environ.setdefault('MKL_NUM_THREADS', str(args.num_threads))
    print(f"torch threads={torch.get_num_threads()}", flush=True)

    allowed_bi = {int(x) for x in args.block_indices.split(',') if x.strip()}

    with open(args.in_csv) as f:
        rows = list(csv.DictReader(f))
    seen = set(); dedup = []
    for r in rows:
        try: bi = int(r['block_idx'])
        except: continue
        if bi not in allowed_bi: continue
        res = int(r.get('input_resolution') or BLOCK_INPUT_RES.get(bi, 224))
        key = (r['block_str'], res)
        if key in seen: continue
        seen.add(key)
        dedup.append((bi, r['block_str'], int(r['in_channels']),
                      int(r['out_channels']), res))
    print(f"Loaded {len(rows)} rows, {len(dedup)} unique (block_str, res) to benchmark",
          flush=True)
    if args.limit > 0: dedup = dedup[:args.limit]

    done = set()
    if args.resume and os.path.exists(args.out_csv):
        with open(args.out_csv) as f:
            for r in csv.DictReader(f):
                done.add((r['block_str'], int(r.get('input_resolution') or 0)))
        print(f"Resume: skipping {len(done)} rows already benchmarked")

    fieldnames = ['block_idx','block_str','in_channels','out_channels','input_resolution',
                  'cpu_median_ms','cpu_mean_ms','cpu_min_ms','cpu_max_ms','cpu_p90_ms',
                  'num_threads','warmup','repeats','build_ok','bench_ok']
    mode = 'a' if (args.resume and os.path.exists(args.out_csv)) else 'w'
    fh = open(args.out_csv, mode, newline='')
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if mode == 'w':
        writer.writeheader()

    ok = fail = 0
    for bi, block_str, in_ch, out_ch, res in dedup:
        if (block_str, res) in done:
            continue
        build_ok = False
        module = None
        try:
            module = BlockModule(block_str).eval()
            build_ok = True
        except Exception as e:
            print(f"  [build-fail] {block_str}: {e}", flush=True)

        stats = None
        bench_ok = False
        if build_ok:
            try:
                stats = time_one_module(module, (1, in_ch, res, res),
                                        warmup=args.warmup, repeats=args.repeats)
                bench_ok = True
            except Exception as e:
                print(f"  [bench-fail] {block_str}: {e}", flush=True)

        row = {'block_idx': bi, 'block_str': block_str, 'in_channels': in_ch,
               'out_channels': out_ch, 'input_resolution': res,
               'num_threads': args.num_threads, 'warmup': args.warmup,
               'repeats': args.repeats, 'build_ok': int(build_ok),
               'bench_ok': int(bench_ok)}
        if stats: row.update({k: stats[k] for k in
                              ['cpu_median_ms','cpu_mean_ms','cpu_min_ms','cpu_max_ms','cpu_p90_ms']})
        writer.writerow(row); fh.flush()

        if bench_ok:
            ok += 1
            if ok % 100 == 0:
                print(f"  [{ok}/{len(dedup)}] {block_str}  median={stats['cpu_median_ms']:.3f} ms",
                      flush=True)
        else:
            fail += 1

        # drop module to free memory / file descriptors
        del module

    print(f"\nDone: {ok} succeeded, {fail} failed. Output: {args.out_csv}", flush=True)
    fh.close()


if __name__ == '__main__':
    main()
