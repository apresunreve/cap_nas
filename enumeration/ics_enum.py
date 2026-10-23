#!/usr/bin/env python3
"""Additive proxy-score network enumeration.

Direct implementation of `ics_enum_algo.tex`:
  1. Group candidate blocks by stage index.
  2. Sort each stage's candidates by Metric(cfg) descending.
  3. Seed networks from top-k fraction of stage-0 candidates.
  4. For each subsequent stage, expand each partial net by top-k fraction of
     *valid-for-this-net* candidates (in_channels match + latency fits remaining
     budget). After expansion, keep at most max_networks_per_stage by NetScore.
  5. NetScore(net) = sum over blocks of Metric(block)  (strictly additive).
  6. Finally keep top max_final_networks by NetScore.

Differences from `search_idw_global_constraint.py`:
  - Additive NetScore only (no min / geomean / harmonic aggregators)
  - Top-k FRACTION of valid candidates per partial net (not fixed-K global)
  - Branch-and-bound min_score cutoff
  - Beam size applied globally at each stage (no bucket diversification)
  - Handles the IDW FC block (SuperConvK1BNRELU + Linear) as a fixed terminal
    cost looked up from block_idx=5 rows in the CSV.

Usage:
    python ics_enum_algo.py \\
        --lookup-csv data/block_lookup/idw_block_lookup_jetson_gpu.csv \\
        --constraint-column jetson_gpu_latency_ms_median \\
        --score-metric analytic_ssr_exp_proj_dw_mean \\
        --latency-budget 15 \\
        --k-fraction 0.25 \\
        --max-networks-per-stage 5000 \\
        --max-final-networks 10 \\
        --fc-num-classes 100 \\
        --output-csv data/block_lookup/ics_enum/lat15ms_mean.csv
"""
from __future__ import annotations
import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

from scoring.inner_aggregation import SCORE_METRICS
from utils.paths import setup_repo_path

setup_repo_path()


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    stage_idx: int
    cfg: str
    in_channels: int
    out_channels: int
    constraint_value: float   # "latency" in the paper's notation (or flops)
    sub_layers: int
    score: float              # Metric(cfg)
    params: float
    flops: float


@dataclass
class Network:
    blocks: List[Candidate] = field(default_factory=list)
    total_constraint: float = 0.0
    total_score: float = 0.0
    total_flops: float = 0.0
    total_params: float = 0.0
    total_sublayers: int = 0
    next_in_channels: int = 0   # track to match for next stage

    def extend(self, cfg: Candidate) -> "Network":
        return Network(
            blocks=self.blocks + [cfg],
            total_constraint=self.total_constraint + cfg.constraint_value,
            total_score=self.total_score + cfg.score,
            total_flops=self.total_flops + cfg.flops,
            total_params=self.total_params + cfg.params,
            total_sublayers=self.total_sublayers + cfg.sub_layers,
            next_in_channels=cfg.out_channels,
        )


# ---------------------------------------------------------------------------
# Load + group the block lookup CSV
# ---------------------------------------------------------------------------
def load_candidates(path: str, constraint_col: str, score_col: str) -> List[Candidate]:
    rows: List[Candidate] = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            try:
                bi = int(r.get('block_idx') or 0)
                c = Candidate(
                    stage_idx=bi,
                    cfg=r['block_str'].strip(),
                    in_channels=int(r.get('in_channels') or 0),
                    out_channels=int(r.get('out_channels') or 0),
                    constraint_value=float(r.get(constraint_col) or 0.0),
                    sub_layers=int(r.get('sub_layers') or 1),
                    score=float(r.get(score_col) or 0.0),
                    params=float(r.get('params') or 0.0),
                    flops=float(r.get('flops') or 0.0),
                )
            except (ValueError, KeyError):
                continue
            if c.constraint_value <= 0.0:
                continue
            rows.append(c)
    return rows


def load_fc_cost(path: str, constraint_col: str, num_classes: int) -> tuple[dict, float]:
    """Read block_idx==5 rows to get per-in_ch projection cost + classifier cost.

    Returns (proj_cost[in_ch] = float, linear_cost = float).
    """
    proj = {}
    linear = 0.0
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if str(r.get('block_idx')) != '5':
                continue
            bs = (r.get('block_str') or '').strip()
            try:
                v = float(r.get(constraint_col) or 0.0)
            except ValueError:
                continue
            if bs.startswith('SuperConvK1BNRELU'):
                try:
                    ic = int(r.get('in_channels') or 0)
                    if ic > 0:
                        proj[ic] = v
                except ValueError:
                    pass
            elif bs == f'Linear(2048,{num_classes})':
                linear = v
    return proj, linear


# ---------------------------------------------------------------------------
# Metric / NetScore (as in the pseudocode)
# ---------------------------------------------------------------------------
def metric(c: Candidate) -> float:
    return c.score


def net_score(net: Network) -> float:
    return net.total_score  # == sum of Metric(block); kept as incremental total for speed


# ---------------------------------------------------------------------------
# Top-k fraction helper
# ---------------------------------------------------------------------------
def top_k_fraction(sorted_desc: List[Candidate], k: float, floor: int = 0) -> List[Candidate]:
    """Return the first max(floor, ceil(k * len)) elements of a score-desc list."""
    if not sorted_desc:
        return sorted_desc
    keep = max(floor, math.ceil(k * len(sorted_desc)))
    return sorted_desc[:keep]


# ---------------------------------------------------------------------------
# Main enumeration (matches ics_enum_algo.tex)
# ---------------------------------------------------------------------------
def enumerate_networks(
    cfgs_by_stage: dict,
    num_stages: int,
    k: float,
    latency_budget: float,
    max_networks_per_stage: Optional[int],
    min_score: float,
    fc_cost_fn=None,
    fraction_floor: int = 5,
    num_fc_classes: int = 100,
) -> List[Network]:
    # Sort each stage by Metric desc
    for s in cfgs_by_stage:
        cfgs_by_stage[s].sort(key=lambda c: -metric(c))

    # Stage 0 — seed
    stage0_sorted = cfgs_by_stage.get(0, [])
    initial_pool = top_k_fraction(stage0_sorted, k, floor=fraction_floor)
    networks: List[Network] = []
    for c in initial_pool:
        # At stage 0 the FC cost is unknown yet; filter to those that COULD fit.
        # Use min feasible FC cost (i.e. the cheapest proj+linear).
        net = Network(next_in_channels=c.in_channels).extend(c)
        # Always keep stage-0 seeds (feasibility checked at final stage).
        networks.append(net)

    print(f"Stage 0: seeded {len(networks)} networks from top-k fraction", flush=True)

    # Stages 1..num_stages-1 (IDW blocks 1..4 in IDW grid)
    for s in range(1, num_stages):
        cand_stage = cfgs_by_stage.get(s, [])
        # Bucket candidates by in_channels for O(1) chain lookup
        by_in = defaultdict(list)
        for c in cand_stage:
            by_in[c.in_channels].append(c)
        # Already sorted desc because stage_sorted above

        new_networks: List[Network] = []
        for net in networks:
            budget = latency_budget - net.total_constraint
            if budget <= 0:
                continue
            valid = [c for c in by_in.get(net.next_in_channels, []) if c.constraint_value <= budget]
            if not valid:
                continue
            valid = top_k_fraction(valid, k, floor=fraction_floor)
            for c in valid:
                extended = net.extend(c)
                # Hard budget on the chain so far (plus conservative FC min-add later)
                if extended.total_constraint > latency_budget:
                    continue
                if net_score(extended) < min_score:
                    continue
                new_networks.append(extended)

        # Cap beam by NetScore
        if max_networks_per_stage is not None and len(new_networks) > max_networks_per_stage:
            new_networks.sort(key=lambda n: -net_score(n))
            new_networks = new_networks[:max_networks_per_stage]

        networks = new_networks
        print(f"Stage {s}: {len(networks)} partial networks", flush=True)
        if not networks:
            break

    # After last IDW stage — append FC cost and filter by final budget
    finals: List[Network] = []
    for net in networks:
        if fc_cost_fn is not None:
            fc_cost = fc_cost_fn(net.next_in_channels, num_fc_classes)
        else:
            fc_cost = 0.0
        total = net.total_constraint + fc_cost
        if total <= latency_budget:
            # Append FC block as a pseudo-Candidate for output reporting
            fc_block = Candidate(
                stage_idx=5, cfg=f'SuperConvK1BNRELU({net.next_in_channels},2048,1,1)',
                in_channels=net.next_in_channels, out_channels=2048,
                constraint_value=fc_cost, sub_layers=1, score=0.0, params=0.0, flops=0.0)
            net_with_fc = Network(
                blocks=net.blocks + [fc_block],
                total_constraint=total,
                total_score=net.total_score,
                total_flops=net.total_flops,
                total_params=net.total_params,
                total_sublayers=net.total_sublayers,
                next_in_channels=2048,
            )
            finals.append(net_with_fc)
    return finals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lookup-csv', required=True,
                    help='Per-block lookup CSV (e.g. idw_block_lookup_jetson_gpu.csv)')
    ap.add_argument('--constraint-column', default='flops',
                    help='CSV column for the constraint (latency_ms or flops)')
    ap.add_argument('--score-metric', required=True,
                    help='CSV column for Metric(cfg) — e.g. analytic_ssr_exp_proj_dw_mean')
    ap.add_argument('--latency-budget', type=float, required=True,
                    help='Hard budget on the constraint column (same unit as column)')
    ap.add_argument('--k-fraction', type=float, default=0.25,
                    help='Top fraction of valid candidates kept at each stage (pseudocode: k)')
    ap.add_argument('--fraction-floor', type=int, default=5,
                    help='Minimum number of candidates retained after top-k-fraction, per partial net')
    ap.add_argument('--max-networks-per-stage', type=int, default=5000,
                    help='Beam cap after each stage expansion (0 disables)')
    ap.add_argument('--min-score', type=float, default=-float('inf'),
                    help='Branch-and-bound cutoff on partial NetScore')
    ap.add_argument('--max-final-networks', type=int, default=20,
                    help='Top-N output networks by NetScore')
    ap.add_argument('--num-stages', type=int, default=5,
                    help='Number of IDW stages (block_idx 0..num_stages-1). Default 5 = stem + 4 IDW.')
    ap.add_argument('--fc-num-classes', type=int, default=100,
                    help='Num output classes for the Linear classifier in the FC block.')
    ap.add_argument('--include-fc-cost', action='store_true', default=True,
                    help='Add FC stage cost (SuperConvK1BNRELU + Linear) to total_constraint.')
    ap.add_argument('--no-fc-cost', dest='include_fc_cost', action='store_false')
    ap.add_argument('--output-csv', default='',
                    help='Write results to this CSV (one row per returned network)')
    args = ap.parse_args()

    print(f"Loading candidates from {args.lookup_csv} ...", flush=True)
    t0 = time.time()
    all_cands = load_candidates(args.lookup_csv, args.constraint_column, args.score_metric)
    # Group by stage, drop block_idx >= num_stages (FC rows are handled separately)
    cfgs = defaultdict(list)
    for c in all_cands:
        if c.stage_idx < args.num_stages:
            cfgs[c.stage_idx].append(c)
    total = sum(len(v) for v in cfgs.values())
    print(f"Loaded {total} candidates across {len(cfgs)} stages in {time.time()-t0:.1f}s", flush=True)
    for s in sorted(cfgs):
        print(f"  stage {s}: {len(cfgs[s])} candidates", flush=True)

    # FC cost lookup
    fc_cost_fn = None
    if args.include_fc_cost:
        proj, linear = load_fc_cost(args.lookup_csv, args.constraint_column, args.fc_num_classes)
        print(f"FC lookup: {len(proj)} proj entries, linear(2048,{args.fc_num_classes})={linear}",
              flush=True)
        def fc_cost_fn(ic, nc):
            return proj.get(ic, 0.0) + linear

    max_per_stage = args.max_networks_per_stage or None
    if args.max_networks_per_stage == 0:
        max_per_stage = None

    t1 = time.time()
    networks = enumerate_networks(
        cfgs_by_stage=cfgs,
        num_stages=args.num_stages,
        k=args.k_fraction,
        latency_budget=args.latency_budget,
        max_networks_per_stage=max_per_stage,
        min_score=args.min_score,
        fc_cost_fn=fc_cost_fn,
        fraction_floor=args.fraction_floor,
        num_fc_classes=args.fc_num_classes,
    )
    print(f"Enumerated {len(networks)} final networks in {time.time()-t1:.1f}s", flush=True)

    # Sort + top-N
    networks.sort(key=lambda n: -net_score(n))
    networks = networks[: args.max_final_networks]

    # Print top-10 structures briefly
    print(f"\nTop {len(networks)} results (budget={args.latency_budget} {args.constraint_column}):")
    for i, n in enumerate(networks):
        struct = ''.join(b.cfg for b in n.blocks)
        print(f"  {i+1}. score={n.total_score:.4f}  total={n.total_constraint:.4f}  "
              f"flops={n.total_flops/1e6:.1f}M  sl={n.total_sublayers}  "
              f"struct={struct[:70]}...", flush=True)

    # Optional CSV output
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
        with open(args.output_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['rank', 'structure_str', 'block_strs', 'total_constraint',
                        'total_params', 'total_flops', 'total_sublayers',
                        'score_metric', 'net_score', 'constraint_error_abs'])
            for i, n in enumerate(networks):
                struct = ''.join(b.cfg for b in n.blocks)
                blocks = '|'.join(b.cfg for b in n.blocks)
                w.writerow([i+1, struct, blocks, f"{n.total_constraint:.6f}",
                            f"{n.total_params:.1f}", f"{n.total_flops:.1f}",
                            n.total_sublayers, args.score_metric,
                            f"{n.total_score:.6f}",
                            f"{abs(n.total_constraint - args.latency_budget):.6f}"])
        print(f"Wrote {args.output_csv}")


if __name__ == '__main__':
    main()
