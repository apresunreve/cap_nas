"""Cross-stage (across-block) score aggregation for network enumeration."""

from __future__ import annotations

import math
from collections import namedtuple

State = namedtuple(
    "State",
    [
        "cfgs",
        "inputs",
        "outputs",
        "constraint",
        "params",
        "flops",
        "total_sublayers",
        "stage_scores",
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

ALL_AGGREGATES = [
    "sum",
    "min",
    "geomean",
    "harmonic",
    "weighted_sum_linear",
    "weighted_geomean_linear",
    "sum_plus_min_half",
    "geomean_plus_min_half",
    "bottom2_mean",
    "soft_min_beta2",
]


def stage_weight(stage_position: int) -> float:
    return stage_position / 15.0


def prefix_linear_weight_sum(count_stages: int) -> float:
    return count_stages * (count_stages + 1) / 30.0


def mean_score(state: State) -> float:
    if state.count_stages == 0:
        return 0.0
    return state.sum_score / state.count_stages


def geomean_score(state: State) -> float:
    if state.count_stages == 0:
        return 0.0
    return math.exp(state.log_sum_score / state.count_stages)


def aggregate_score(state: State, agg: str) -> float:
    if state.count_stages == 0:
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
        beta = 2.0
        m = min(state.stage_scores)
        total = sum(math.exp(-beta * (s - m)) for s in state.stage_scores)
        return m - math.log(total) / beta
    raise ValueError(agg)


def append_candidate(state, candidate, metric: str, metric_value_fn, skip_agg: bool = False):
    """Append a candidate block to the frontier state."""
    score = metric_value_fn(candidate, metric)
    is_log_domain = metric.endswith("logproduct")
    if not is_log_domain and score <= 0.0 and not skip_agg:
        return None
    if is_log_domain:
        score_log = score
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


def state_rank_key(state: State, agg: str, target: float):
    if state.count_stages == 0:
        return (0.0, abs(state.constraint - target), state.constraint)
    if agg == "min":
        return (-state.min_score, -geomean_score(state), abs(state.constraint - target), state.constraint)
    return (-aggregate_score(state, agg), abs(state.constraint - target), state.constraint)
