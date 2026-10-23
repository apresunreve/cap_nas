# CAP-NAS — how to run

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD          # needed for the training commands in section 2
```

Enumeration needs neither `PYTHONPATH` nor a dataset.

Dataset paths have no defaults. Set the ones you need:

| dataset | env var | layout |
|---|---|---|
| `cifar100` | `CAP_NAS_CIFAR100_ROOT` | torchvision CIFAR-100 dir |
| `imagenet100` | `CAP_NAS_IMAGENET100_ROOT` | `train/ val/ test/` of class dirs |
| `inat_resized` | `CAP_NAS_INAT_ROOT` | `train/ val/ test/`, 4-digit class dirs |

An unset variable raises only when that dataset is used, naming what to set.

---

## 1. Enumeration (search)

Produces the top-N architectures under a cost budget. Needs no dataset.

```bash
cd enumeration
python search_idw_global_constraint.py \
  --lookup-csv ../data/block_lookup/idw_block_lookup_jetson_gpu.csv \
  --constraint-column jetson_gpu_latency_ms_median --target 10 \
  --metric analytic_ssr_exp_proj_dw_mean --aggregate sum \
  --max-layers 14 --bin-size 0.1 --states-per-key 3 --candidates-per-in-ch 200 \
  --top-n 3 --skip-stem-in-agg --fc-lookup-from-csv --fc-num-classes 100 \
  --output-csv ~/cap_nas_results/lat10ms.csv
```


**Always pass the flags above.** The CLI defaults (`geomean`/`geomean`, bin
`1e6`) build different networks.

Per-family differences:

All families read the single bundled lookup,
`data/block_lookup/idw_block_lookup_jetson_gpu.csv`, and differ only in:

| family | `--constraint-column` | `--bin-size` | `--fc-lookup-from-csv` |
|---|---|---|---|
| latency 10/15/20 ms | `jetson_gpu_latency_ms_median` | `0.1` | yes |
| FLOPs 800/1200/1600 M | `flops` | `1e6` | **no** |
| params 4/6/8 M | `params` | `1e5` | yes |

All nine families at once:

```bash
cd enumeration
CAP_NAS_RESULTS_DIR=/path/to/results bash run_cost_target_searches.sh
```

Writes one CSV per family to `$CAP_NAS_RESULTS_DIR/cost_targets/` and the top-3
structure strings to `$CAP_NAS_RESULTS_DIR/top3_structures/`. The variable
defaults to `./cap_nas_results` in the directory you run from; results are never
written inside the package. Re-running skips families already complete.

---

## 2. Training

Training takes a file holding one architecture string. The nine-family driver
writes these directly:

```
$CAP_NAS_RESULTS_DIR/top3_structures/capnas_<family>_rank<N>.txt
```

A single-budget search writes a CSV instead, so pull the string out of it:

```bash
python -c "import csv,sys; print(next(csv.DictReader(open(sys.argv[1])))['structure_str'])" \
  ~/cap_nas_results/lat10ms.csv > struct.txt
```

Then:

```bash
export PYTHONPATH=$PWD
export CAP_NAS_IMAGENET100_ROOT=/path/to/imagenet100

python training/train.py \
  --dataset imagenet100 \
  --structure-txt ~/cap_nas_results/top3_structures/capnas_lat10ms_rank1.txt \
  --save-dir ~/runs/lat10ms_rank1 \
  --epochs 200 --batch-size-per-gpu 256 --workers-per-gpu 8 \
  --gpu 0
```

Writes `summary.csv`, `train.done` and `best-params_rank0.pth` to `--save-dir`.
Per-dataset `epochs` / `batch_size_per_gpu` / `num_classes` defaults come from
`config/defaults.yaml` when the flags are omitted.

- `PYTHONPATH` and the dataset env var are both required here (unlike enumeration).
- Multi-GPU: `--dist-mode ddp`, launched under `torchrun`. Default is `single`.
- `--epochs` must exceed `--warmup` (hardcoded 5), so **use >= 6**.
- `--num_classes` and the input size come from `config/defaults.yaml` per dataset.

### Low GPU memory

`training/train_gradckpt.py` takes the same arguments as `train.py` and adds
per-block activation checkpointing, roughly halving activation memory at the
cost of recomputing each block during backward.

---

## Layout

```
scoring/        CAP-NAS score
search_space/   IDW search space
enumeration/    search engine + lookup builders + driver
model/          PlainNet / MasterNet / baselines
dataloader/     the 3 datasets (plain torchvision ImageFolder)
training/       timm_trainer + launchers
benchmarking/   block latency profiling
data/           bundled block lookup CSVs
config/         defaults.yaml
```
