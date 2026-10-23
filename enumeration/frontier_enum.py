#!/usr/bin/env python3
"""Frontier-based enumeration search on the IDW block lookup table."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict, namedtuple
from pathlib import Path

from scoring.cross_stage_aggregation import (
    ALL_AGGREGATES,
    State,
    aggregate_score,
    append_candidate,
    geomean_score,
    state_rank_key,
)
from scoring.inner_aggregation import SCORE_METRICS, metric_value
from utils.paths import ORIGINAL_SEED_STRUCTURE, setup_model_path, setup_repo_path

setup_repo_path()
setup_model_path()
import PlainNet  # noqa: E402


BlockCandidate = namedtuple(
    "BlockCandidate",
    ["block_idx", "cfg", "input_channels", "output_channels",
     "constraint_value", "params", "flops", "sub_layers", "scores"],
)


def _append(state, candidate, metric, skip_agg=False):
    return append_candidate(state, candidate, metric, metric_value, skip_agg=skip_agg)


def state_key(state, bin_size):
    constraint_bin = int(round(state.constraint / bin_size))
    return (state.next_channels, constraint_bin)


def keep_best(states, agg, target, states_per_key, bin_size):
    buckets = defaultdict(list)
    for s in states:
        buckets[state_key(s, bin_size)].append(s)
    kept = []
    for items in buckets.values():
        items.sort(key=lambda s: state_rank_key(s, agg, target))
        kept.extend(items[:states_per_key])
    return kept


# ---------------------------------------------------------------------------
# Lookup table I/O
# ---------------------------------------------------------------------------

def read_lookup(path, constraint_column, metric, candidates_per_in_ch=200):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # block_idx >= 5 are FC-stage rows (SuperConvK1BNRELU projection / Linear classifier)
            # and are handled by read_fc_lookup + get_fc_block_cost, not the main enumeration loop.
            if int(row.get("block_idx") or 0) >= 5:
                continue
            cv = float(row.get(constraint_column) or 0.0)
            if cv <= 0:
                continue
            scores = {name: float(row.get(name) or 0.0) for name in SCORE_METRICS}
            sl = int(row.get("sub_layers") or 1)
            rows.append(BlockCandidate(
                block_idx=int(row["block_idx"]),
                cfg=row["block_str"],
                input_channels=int(row["in_channels"]),
                output_channels=int(row["out_channels"]),
                constraint_value=cv,
                params=float(row["params"]),
                flops=float(row["flops"]),
                sub_layers=sl,
                scores=scores,
            ))

    # Dedup: keep highest-scoring candidate per (block_idx, cfg, in_channels, out_channels)
    dedup = {}
    for r in rows:
        key = (r.block_idx, r.cfg, r.input_channels, r.output_channels)
        cur = dedup.get(key)
        if cur is None or metric_value(r, metric) > metric_value(cur, metric):
            dedup[key] = r

    grouped = defaultdict(list)
    for r in dedup.values():
        grouped[r.block_idx].append(r)

    # Pre-filter: keep top candidates_per_in_ch per (block_idx, in_channels, out_channels)
    # This preserves diversity across output channel options while reducing btl/sl/type
    # combinations. Filtering by score alone would discard all budget-feasible candidates
    # (since highest-score candidates are the largest).
    for bidx in grouped:
        by_in_out = defaultdict(list)
        for c in grouped[bidx]:
            by_in_out[(c.input_channels, c.output_channels)].append(c)
        filtered = []
        for key, candidates in by_in_out.items():
            candidates.sort(key=lambda c: -metric_value(c, metric))
            filtered.extend(candidates[:candidates_per_in_ch])
        grouped[bidx] = filtered

    for bidx in grouped:
        grouped[bidx].sort(key=lambda c: (c.input_channels, -metric_value(c, metric), c.constraint_value))

    total = sum(len(v) for v in grouped.values())
    print(f"After top-{candidates_per_in_ch} per in_ch filtering: {total} candidates", flush=True)
    return grouped


# ---------------------------------------------------------------------------
# Fixed FC block (block 5)
# ---------------------------------------------------------------------------

def get_fc_block_cost(out_channels: int, input_resolution: int = 7) -> tuple[float, float]:
    """Get FLOPs and params for the fixed SuperConvK1BNRELU(out_channels, 2048, 1, 1)."""
    block_str = f"SuperConvK1BNRELU({out_channels},2048,1,1)"
    blocks = PlainNet.create_netblock_list_from_str(block_str, no_create=True)
    if not blocks:
        return 0.0, 0.0
    block = blocks[0]
    return float(block.get_FLOPs(input_resolution)), float(block.get_model_size())


def read_fc_lookup(lookup_csv: str, constraint_column: str, num_classes: int = 100):
    """Read block_idx=5 entries for CSV-based FC cost (projection + Linear classifier).
    Returns two dicts:
      proj_cost[out_channels] = <constraint_value> for SuperConvK1BNRELU(oc,2048,1,1)
      linear_cost = <constraint_value> for Linear(2048, num_classes)
    Any block not found returns 0.0 (caller may fall back to `get_fc_block_cost`).
    """
    proj_cost, linear_cost = {}, 0.0
    try:
        with open(lookup_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("block_idx") != "5":
                    continue
                bs = row.get("block_str", "")
                try:
                    v = float(row.get(constraint_column) or 0.0)
                except ValueError:
                    continue
                if bs.startswith("SuperConvK1BNRELU"):
                    # e.g. SuperConvK1BNRELU(192,2048,1,1) → key is 192
                    try:
                        oc = int(row.get("in_channels") or 0)
                        if oc > 0:
                            proj_cost[oc] = v
                    except ValueError:
                        pass
                elif bs == f"Linear(2048,{num_classes})":
                    linear_cost = v
    except FileNotFoundError:
        pass
    return proj_cost, linear_cost


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(grouped, metric, agg, target, bin_size, max_factor, states_per_key, max_layers, fc_resolution,
           skip_stem_in_agg=False, fc_cost_fn=None):
    max_constraint = target * max_factor
    num_blocks = max(grouped.keys()) + 1  # 0..4 = 5 blocks

    # Block 0 (stem). If skip_stem_in_agg, its score does not enter aggregation.
    frontier = []
    for c in grouped.get(0, []):
        if c.constraint_value <= max_constraint:
            s = _append(None, c, metric, skip_agg=skip_stem_in_agg)
            if s is not None:
                frontier.append(s)
    frontier = keep_best(frontier, agg, target, states_per_key, bin_size)
    print(f"block 0 frontier {len(frontier)}", flush=True)

    # Blocks 1..4 (IDW blocks, always aggregated).
    for bidx in range(1, num_blocks):
        by_input = defaultdict(list)
        for c in grouped.get(bidx, []):
            by_input[c.input_channels].append(c)

        next_states = []
        for state in frontier:
            for c in by_input.get(state.next_channels, []):
                if state.constraint + c.constraint_value > max_constraint:
                    continue
                if max_layers is not None and state.total_sublayers + c.sub_layers > max_layers:
                    continue
                ns = _append(state, c, metric, skip_agg=False)
                if ns is not None:
                    next_states.append(ns)
        frontier = keep_best(next_states, agg, target, states_per_key, bin_size)
        print(f"block {bidx} frontier {len(frontier)}", flush=True)

    # Append fixed FC block (block 5) costs
    finals = []
    fc_cache = {}
    for state in frontier:
        oc = state.next_channels
        if oc not in fc_cache:
            fc_cache[oc] = get_fc_block_cost(oc, fc_resolution)
        fc_flops, fc_params = fc_cache[oc]
        # Default: FC cost = its FLOPs from PlainNet. With fc_cost_fn (CSV lookup),
        # override with the measured constraint-column value for (proj block + Linear).
        if fc_cost_fn is not None:
            fc_constraint = fc_cost_fn(oc)
        else:
            fc_constraint = fc_flops
        total_constraint = state.constraint + fc_constraint
        if total_constraint > max_constraint:
            continue
        fc_str = f"SuperConvK1BNRELU({oc},2048,1,1)"
        final = State(
            cfgs=state.cfgs + [fc_str],
            inputs=state.inputs + [oc],
            outputs=state.outputs + [2048],
            constraint=total_constraint,
            params=state.params + fc_params,
            flops=state.flops + fc_flops,
            total_sublayers=state.total_sublayers,
            stage_scores=state.stage_scores,
            count_stages=state.count_stages,
            sum_score=state.sum_score,
            min_score=state.min_score,
            log_sum_score=state.log_sum_score,
            harmonic_denom=state.harmonic_denom,
            weighted_sum_score=state.weighted_sum_score,
            weighted_log_sum=state.weighted_log_sum,
            next_channels=2048,
        )
        finals.append(final)

    finals.sort(key=lambda s: state_rank_key(s, agg, target))
    return finals


def choose_best(finals, agg, target, tolerance):
    """Select architectures within strict budget ceiling (constraint <= target + tolerance).
    With tolerance=0, this is a hard cap. Sort by score, not budget proximity."""
    within = [s for s in finals if s.constraint <= target + tolerance]
    if within:
        within.sort(key=lambda s: state_rank_key(s, agg, target))
        return within
    # Fallback: if nothing fits, return closest to target
    finals.sort(key=lambda s: (abs(s.constraint - target),) + state_rank_key(s, agg, target))
    return finals


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_rows(path, rows, args):
    fieldnames = [
        "rank", "structure_str", "block_strs",
        "total_constraint", "total_params", "total_flops",
        "total_sublayers", "stage_metric", "aggregate_mode",
        "total_score", "constraint_error_abs", "stage_scores",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, state in enumerate(rows, start=1):
            writer.writerow({
                "rank": rank,
                "structure_str": "".join(state.cfgs),
                "block_strs": "|".join(state.cfgs),
                "total_constraint": round(state.constraint, 2),
                "total_params": round(state.params, 2),
                "total_flops": round(state.flops, 2),
                "total_sublayers": state.total_sublayers,
                "stage_metric": args.metric,
                "aggregate_mode": args.aggregate,
                "total_score": round(aggregate_score(state, args.aggregate), 6),
                "constraint_error_abs": round(abs(state.constraint - args.target), 2),
                "stage_scores": "|".join(f"{s:.6f}" for s in state.stage_scores),
            })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="IDW frontier-based enumeration search.")
    p.add_argument("--lookup-csv", "--lookup_csv", dest="lookup_csv", required=True)
    p.add_argument("--constraint-column", "--constraint_column", dest="constraint_column", default="flops")
    p.add_argument("--metric", default="analytic_ssr_exp_proj_dw_geomean")
    p.add_argument("--aggregate", choices=ALL_AGGREGATES, default="geomean")
    p.add_argument("--target", type=float, required=True, help="Budget target (FLOPs or latency)")
    p.add_argument("--bin-size", "--bin_size", dest="bin_size", type=float, default=1e6,
                    help="Constraint bin size for frontier bucketing (default 1e6 for FLOPs)")
    p.add_argument("--max-factor", "--max_factor", dest="max_factor", type=float, default=1.0)
    p.add_argument("--states-per-key", "--states_per_key", dest="states_per_key", type=int, default=6)
    p.add_argument("--final-tolerance", "--final_tolerance", dest="final_tolerance", type=float, default=0.0,
                    help="Tolerance for final selection (0 = strict ceiling)")
    p.add_argument("--max-layers", "--max_layers", dest="max_layers", type=int, default=14)
    p.add_argument("--fc-resolution", "--fc_resolution", dest="fc_resolution", type=int, default=7,
                    help="Input resolution for the fixed FC block (7 for 224x224 input)")
    p.add_argument("--candidates-per-in-ch", "--candidates_per_in_ch", dest="candidates_per_in_ch",
                    type=int, default=200,
                    help="Max candidates to keep per (block_idx, in_channels) after sorting by score. Reduces search space.")
    p.add_argument("--top-n", "--top_n", dest="top_n", type=int, default=20)
    p.add_argument("--output-csv", "--output_csv", dest="output_csv", default="")
    p.add_argument("--skip-stem-in-agg", "--skip_stem_in_agg", dest="skip_stem_in_agg",
                   action="store_true", default=False,
                   help="Exclude the stem block (block 0) from score aggregation. Recommended "
                        "because the stem is architecturally fixed and saturates `min`-family "
                        "aggregators. The fixed FC block is already excluded by construction.")
    p.add_argument("--fc-lookup-from-csv", "--fc_lookup_from_csv", dest="fc_lookup_from_csv",
                   action="store_true", default=False,
                   help="Read the FC projection (SuperConvK1BNRELU→2048) and Linear classifier "
                        "cost from block_idx=5 rows of the lookup CSV rather than computing "
                        "FLOPs from PlainNet. Required when --constraint-column is a latency "
                        "column (e.g. jetson_gpu_latency_ms_median).")
    p.add_argument("--fc-num-classes", "--fc_num_classes", dest="fc_num_classes", type=int,
                   default=100, help="num_classes for the Linear(2048,N) classifier row in FC lookup")
    return p.parse_args()


def main():
    args = parse_args()
    if args.final_tolerance is None:
        args.final_tolerance = 0.0  # strict ceiling

    grouped = read_lookup(args.lookup_csv, args.constraint_column, args.metric, args.candidates_per_in_ch)
    print(f"Loaded {sum(len(v) for v in grouped.values())} candidates across {len(grouped)} blocks", flush=True)

    fc_cost_fn = None
    if args.fc_lookup_from_csv:
        proj_cost, linear_cost = read_fc_lookup(args.lookup_csv, args.constraint_column, args.fc_num_classes)
        print(f"FC lookup: {len(proj_cost)} proj entries, linear(2048,{args.fc_num_classes})={linear_cost}", flush=True)
        def fc_cost_fn(oc):
            return float(proj_cost.get(oc, 0.0)) + float(linear_cost)

    finals = search(
        grouped=grouped,
        metric=args.metric,
        agg=args.aggregate,
        target=args.target,
        bin_size=args.bin_size,
        max_factor=args.max_factor,
        states_per_key=args.states_per_key,
        max_layers=args.max_layers,
        fc_resolution=args.fc_resolution,
        skip_stem_in_agg=args.skip_stem_in_agg,
        fc_cost_fn=fc_cost_fn,
    )
    best = choose_best(finals, args.aggregate, args.target, args.final_tolerance)[:args.top_n]

    print(f"\nTop {len(best)} results (target={args.target/1e6:.0f}M, metric={args.metric}, agg={args.aggregate}):")
    for idx, state in enumerate(best[:10], start=1):
        score = aggregate_score(state, args.aggregate)
        print(f"  {idx}. score={score:.4f} flops={state.flops/1e6:.1f}M sl={state.total_sublayers} cfg={''.join(state.cfgs[:2])}...", flush=True)

    if args.output_csv:
        write_rows(args.output_csv, best, args)
        print(f"Written to {args.output_csv}")


if __name__ == "__main__":
    main()
