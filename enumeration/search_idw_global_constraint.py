#!/usr/bin/env python3
"""Frontier-based enumeration search on the IDW block lookup table.

Adapted from search_mbv2_global_latency_only.py for the PlainNet IDW search
space.  Key differences:
  - 5 searchable blocks (0-4) instead of 7 stages; block 5 (fixed FC) appended
  - Constraint column is configurable (flops or latency)
  - Tracks total_sublayers for max_layers enforcement
  - Block 5 cost added to final states
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict, namedtuple
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Package-relative model root (was: standalone.scripts.plainnet_original_space_common)
AZ_ROOT = REPO_ROOT / "model"

ORIGINAL_SEED_STRUCTURE = (
    "SuperConvK3BNRELU(3,8,2,1)"
    "SuperResIDWE6K3(8,32,2,8,1)"
    "SuperResIDWE6K3(32,48,2,32,1)"
    "SuperResIDWE6K3(48,96,2,48,1)"
    "SuperResIDWE6K3(96,128,2,96,1)"
    "SuperConvK1BNRELU(128,2048,1,1)"
)

# PlainNet imports global_utils, which lives in the package's utils/ dir
for _p in (AZ_ROOT, REPO_ROOT / "utils"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import PlainNet  # noqa: E402
from Masternet import MasterNet  # noqa: E402


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

BlockCandidate = namedtuple(
    "BlockCandidate",
    ["block_idx", "cfg", "input_channels", "output_channels",
     "constraint_value", "params", "flops", "sub_layers", "scores"],
)

State = namedtuple(
    "State",
    [
        "cfgs",             # list of block_str
        "inputs",           # list of in_channels
        "outputs",          # list of out_channels
        "constraint",       # cumulative constraint value (flops or latency)
        "params",
        "flops",
        "total_sublayers",  # cumulative sub_layers for max_layers check
        "stage_scores",     # list of per-block score values
        "count_stages",
        "sum_score",
        "min_score",
        "log_sum_score",
        "harmonic_denom",
        "weighted_sum_score",
        "weighted_log_sum",
        "next_channels",
    ],
)


SCORE_METRICS = [
    "analytic_ssr_exp_proj_dw_sum",
    "analytic_ssr_exp_proj_dw_mean",
    "analytic_ssr_exp_proj_dw_geomean",
    "analytic_ssr_exp_proj_dw_product",
    "analytic_ssr_exp_proj_dw_logproduct",
    "analytic_ssr_exp_proj_dw_min",
    "analytic_ssr_exp_proj_dw_bottom2_mean",
]

ALL_AGGREGATES = ["sum", "min", "geomean", "harmonic",
                  "weighted_sum_linear", "weighted_geomean_linear",
                  "sum_plus_min_half", "geomean_plus_min_half",
                  "bottom2_mean", "soft_min_beta2"]


# ---------------------------------------------------------------------------
# Helpers (carried over from search_mbv2_global_latency_only.py)
# ---------------------------------------------------------------------------

def positive_floor(value, eps=1e-12):
    return value if value > eps else eps


def prefix_linear_weight_sum(count_stages):
    return count_stages * (count_stages + 1) / 30.0   # 5 blocks: 5*6/2=15, normalise by 2*15=30


def stage_weight(stage_position):
    return stage_position / 15.0   # 5 blocks: sum(1..5)=15


def mean_score(state):
    if state.count_stages == 0:
        return 0.0
    return state.sum_score / state.count_stages


def geomean_score(state):
    if state.count_stages == 0:
        return 0.0
    return math.exp(state.log_sum_score / state.count_stages)


def aggregate_score(state, agg):
    if state.count_stages == 0:
        # No aggregation-eligible blocks yet (e.g. stem-only frontier with skip-stem-in-agg).
        return 0.0
    if agg == "sum":
        return state.sum_score
    if agg == "min":
        return state.min_score
    if agg == "geomean":
        return geomean_score(state)
    if agg == "harmonic":
        return state.count_stages / state.harmonic_denom
    if agg == "weighted_sum_linear":
        return state.weighted_sum_score / prefix_linear_weight_sum(state.count_stages)
    if agg == "weighted_geomean_linear":
        return math.exp(state.weighted_log_sum / prefix_linear_weight_sum(state.count_stages))
    if agg == "sum_plus_min_half":
        return 0.5 * mean_score(state) + 0.5 * state.min_score
    if agg == "geomean_plus_min_half":
        return 0.5 * geomean_score(state) + 0.5 * state.min_score
    if agg == "bottom2_mean":
        k = min(2, len(state.stage_scores))
        if k == 0:
            return 0.0
        return sum(sorted(state.stage_scores)[:k]) / k
    if agg == "soft_min_beta2":
        # Numerically stable log-sum-exp: soft-min(s) = m - log(sum(exp(-beta*(s-m))))/beta
        # where m = min(s). This avoids underflow for large positive stage_scores.
        beta = 2.0
        m = min(state.stage_scores)
        total = sum(math.exp(-beta * (s - m)) for s in state.stage_scores)
        return m - math.log(total) / beta
    raise ValueError(agg)


def metric_value(candidate, metric):
    return candidate.scores.get(metric, 0.0)


# ---------------------------------------------------------------------------
# State manipulation
# ---------------------------------------------------------------------------

def append_candidate(state, candidate, metric, skip_agg=False):
    """Append a candidate block to the state.

    When skip_agg=True, the candidate's score does NOT enter the aggregation
    counters (used for the stem block 0, which is architecturally fixed
    across candidates and its inclusion in aggregation saturates `min`-family
    aggregators). The candidate's flops/params/channels are still tracked.

    The `logproduct` metric is already in log-space (score = log(product)) and
    may legitimately be negative; accept such values and skip the redundant
    math.log() step when computing geomean-across accumulators.
    """
    score = metric_value(candidate, metric)
    is_log_domain = metric.endswith("logproduct")
    if not is_log_domain and score <= 0.0 and not skip_agg:
        # Non-log-domain metrics require strictly positive per-block scores.
        return None
    if is_log_domain:
        score_log = score  # already log-domain; using `score` directly gives geomean-of-products.
    else:
        score_log = math.log(score) if score > 0.0 else 0.0
    sl = candidate.sub_layers
    if state is None:
        if skip_agg:
            return State(
                cfgs=[candidate.cfg],
                inputs=[candidate.input_channels],
                outputs=[candidate.output_channels],
                constraint=candidate.constraint_value,
                params=candidate.params,
                flops=candidate.flops,
                total_sublayers=sl,
                stage_scores=[],
                count_stages=0,
                sum_score=0.0,
                min_score=float("inf"),
                log_sum_score=0.0,
                harmonic_denom=0.0,
                weighted_sum_score=0.0,
                weighted_log_sum=0.0,
                next_channels=candidate.output_channels,
            )
        # Guard divide-by-zero on harmonic for edge cases (near-zero score)
        inv_score = 1.0 / score if abs(score) > 1e-12 else 0.0
        return State(
            cfgs=[candidate.cfg],
            inputs=[candidate.input_channels],
            outputs=[candidate.output_channels],
            constraint=candidate.constraint_value,
            params=candidate.params,
            flops=candidate.flops,
            total_sublayers=sl,
            stage_scores=[score],
            count_stages=1,
            sum_score=score,
            min_score=score,
            log_sum_score=score_log,
            harmonic_denom=inv_score,
            weighted_sum_score=stage_weight(1) * score,
            weighted_log_sum=stage_weight(1) * score_log,
            next_channels=candidate.output_channels,
        )
    if skip_agg:
        # Preserve aggregation fields unchanged; only record bookkeeping.
        return State(
            cfgs=state.cfgs + [candidate.cfg],
            inputs=state.inputs + [candidate.input_channels],
            outputs=state.outputs + [candidate.output_channels],
            constraint=state.constraint + candidate.constraint_value,
            params=state.params + candidate.params,
            flops=state.flops + candidate.flops,
            total_sublayers=state.total_sublayers + sl,
            stage_scores=state.stage_scores,
            count_stages=state.count_stages,
            sum_score=state.sum_score,
            min_score=state.min_score,
            log_sum_score=state.log_sum_score,
            harmonic_denom=state.harmonic_denom,
            weighted_sum_score=state.weighted_sum_score,
            weighted_log_sum=state.weighted_log_sum,
            next_channels=candidate.output_channels,
        )
    nc = state.count_stages + 1
    nw = stage_weight(nc)
    inv_score = 1.0 / score if abs(score) > 1e-12 else 0.0
    return State(
        cfgs=state.cfgs + [candidate.cfg],
        inputs=state.inputs + [candidate.input_channels],
        outputs=state.outputs + [candidate.output_channels],
        constraint=state.constraint + candidate.constraint_value,
        params=state.params + candidate.params,
        flops=state.flops + candidate.flops,
        total_sublayers=state.total_sublayers + sl,
        stage_scores=state.stage_scores + [score],
        count_stages=nc,
        sum_score=state.sum_score + score,
        min_score=(score if state.count_stages == 0 else min(state.min_score, score)),
        log_sum_score=state.log_sum_score + score_log,
        harmonic_denom=state.harmonic_denom + inv_score,
        weighted_sum_score=state.weighted_sum_score + nw * score,
        weighted_log_sum=state.weighted_log_sum + nw * score_log,
        next_channels=candidate.output_channels,
    )


def state_key(state, bin_size):
    constraint_bin = int(round(state.constraint / bin_size))
    return (state.next_channels, constraint_bin)


def state_rank_key(state, agg, target):
    # Fallback: constraint-only ordering for pre-aggregation frontiers (e.g. stem-only
    # when --skip-stem-in-agg is active).
    if state.count_stages == 0:
        return (0.0, abs(state.constraint - target), state.constraint)
    if agg == "min":
        return (-state.min_score, -geomean_score(state), abs(state.constraint - target), state.constraint)
    return (-aggregate_score(state, agg), abs(state.constraint - target), state.constraint)


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
            s = append_candidate(None, c, metric, skip_agg=skip_stem_in_agg)
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
                ns = append_candidate(state, c, metric, skip_agg=False)
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

    print(f"\nTop {len(best)} results (target={args.target:g} {args.constraint_column}, "
          f"metric={args.metric}, agg={args.aggregate}):")
    for idx, state in enumerate(best[:10], start=1):
        score = aggregate_score(state, args.aggregate)
        print(f"  {idx}. score={score:.4f} flops={state.flops/1e6:.1f}M sl={state.total_sublayers} cfg={''.join(state.cfgs[:2])}...", flush=True)

    if args.output_csv:
        write_rows(args.output_csv, best, args)
        print(f"Written to {args.output_csv}")


if __name__ == "__main__":
    main()
