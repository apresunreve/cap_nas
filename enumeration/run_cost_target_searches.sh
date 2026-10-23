#!/bin/bash
# Run the nine CAP-NAS cost-target beam searches with production flags:
#   latency 10/15/20 ms, FLOPs 800/1200/1600 M, params 4/6/8 M
#
# Each search is gated on every rank's total_constraint <= target, and the
# top-3 structures are written to top3_structures/capnas_{name}_rank{1..3}.txt
set -e
INVOKE_DIR="$PWD"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

PKG_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
export PYTHONPATH="$PKG_ROOT/model:$PKG_ROOT/utils:$PKG_ROOT:${PYTHONPATH:-}"
PY=${PYTHON:-python3}
LOOKUP="$PKG_ROOT/data/block_lookup/idw_block_lookup_jetson_gpu.csv"
# Results are written outside the package. Override with CAP_NAS_RESULTS_DIR.
RESULTS_DIR="${CAP_NAS_RESULTS_DIR:-$INVOKE_DIR/cap_nas_results}"
OUTDIR="$RESULTS_DIR/cost_targets"
TOPDIR="$RESULTS_DIR/top3_structures"
mkdir -p "$OUTDIR" "$TOPDIR"
echo "Writing results to $RESULTS_DIR"

run_search() {
  local col="$1" target="$2" bin="$3" fc_flag="$4" name="$5"
  local out="$OUTDIR/${name}_capnas.csv"
  local fc_args=()
  [ "$fc_flag" = "yes" ] && fc_args=(--fc-lookup-from-csv)

  if [ -s "$out" ] && [ -s "$TOPDIR/capnas_${name}_rank3.txt" ]; then
    echo "=== cost-target search: $name -- already complete, skipping ==="
    return 0
  fi

  echo "=== cost-target search: $name (${col} <= ${target}) ==="
  "$PY" search_idw_global_constraint.py \
    --lookup-csv "$LOOKUP" --constraint-column "$col" --target "$target" \
    --bin-size "$bin" --metric analytic_ssr_exp_proj_dw_mean --aggregate sum \
    --max-layers 14 --states-per-key 3 --candidates-per-in-ch 200 \
    --top-n 3 --skip-stem-in-agg --fc-num-classes 100 "${fc_args[@]}" \
    --output-csv "$out"

  "$PY" - "$out" "$TOPDIR/capnas_${name}" "$target" <<'EOF'
import csv, sys
out_csv, prefix, target = sys.argv[1], sys.argv[2], float(sys.argv[3])
rows = list(csv.DictReader(open(out_csv)))
assert len(rows) == 3, f"expected 3 rows, got {len(rows)} in {out_csv}"
print(f"{'rank':<6}{'score':>12}{'constraint':>16}")
for r in rows:
    tc = float(r["total_constraint"])
    assert tc <= target, f"rank {r['rank']} over budget: {tc} > {target}"
    print(f"{r['rank']:<6}{float(r['total_score']):>12.4f}{tc:>16.4f}")
    with open(f"{prefix}_rank{r['rank']}.txt", "w") as f:
        f.write(r["structure_str"])
print(f"gate passed: all 3 ranks <= {target:g}")
EOF
}

run_search jetson_gpu_latency_ms_median 10 0.1   yes lat10ms
run_search jetson_gpu_latency_ms_median 15 0.1   yes lat15ms
run_search jetson_gpu_latency_ms_median 20 0.1   yes lat20ms

run_search flops  800000000 1000000 no  flops800M
run_search flops 1200000000 1000000 no  flops1200M
run_search flops 1600000000 1000000 no  flops1600M

run_search params   4000000 100000 yes params4M
run_search params   6000000 100000 yes params6M
run_search params   8000000 100000 yes params8M

echo "Done. CSVs in $OUTDIR/, top-3 structures in $TOPDIR/"
