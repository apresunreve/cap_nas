"""Within-block (inner) stable-rank score column names and helpers."""

from __future__ import annotations

INNER_METRICS = {
    "sum": "analytic_ssr_exp_proj_dw_sum",
    "mean": "analytic_ssr_exp_proj_dw_mean",
    "geomean": "analytic_ssr_exp_proj_dw_geomean",
    "product": "analytic_ssr_exp_proj_dw_product",
    "logproduct": "analytic_ssr_exp_proj_dw_logproduct",
    "min": "analytic_ssr_exp_proj_dw_min",
    "bottom2_mean": "analytic_ssr_exp_proj_dw_bottom2_mean",
}

SCORE_METRICS = list(INNER_METRICS.values())


def metric_column(name: str) -> str:
    """Resolve a short name (e.g. 'mean') or full column name."""
    if name in INNER_METRICS:
        return INNER_METRICS[name]
    if name in SCORE_METRICS:
        return name
    raise KeyError(f"Unknown inner metric {name!r}; choose from {list(INNER_METRICS)}")


def metric_value(candidate, metric: str) -> float:
    return candidate.scores.get(metric, 0.0)
