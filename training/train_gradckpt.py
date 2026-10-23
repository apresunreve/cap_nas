#!/usr/bin/env python3
"""Train PlainNet IDW architectures WITH gradient checkpointing.

Identical to train.py (same trainer, same hyperparameters, same completion
files) except the PlainNet model runs with per-block activation
checkpointing: block outputs are stored and intermediate activations are
recomputed during backward. This roughly halves activation GPU memory --
the ~44 GiB peak of the 4-GFLOP structures drops to ~22 GiB -- at ~20-30%
extra compute per step. Gradients are identical up to floating-point noise;
batch size and optimizer are unchanged.

Injection: a generated sitecustomize.py on PYTHONPATH patches
PlainNet.PlainNet.forward at interpreter startup, so no model source file is
modified and no other training path is affected. PlainNet.forward is
out-of-place and sequential (output = block(output) per block), which is
exactly what torch.utils.checkpoint requires.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from utils.config import ensure_dir, get_dataset_defaults, write_text
from utils.paths import MODEL_ROOT, REPO_ROOT, masternet_arch, setup_repo_path

setup_repo_path()

TIMM_TRAINER = Path(__file__).resolve().parent / "timm_trainer.py"

SITECUSTOMIZE = '''
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

import Masternet

_original_forward = Masternet.MasterNet.forward


def _checkpointed_forward(self, x):
    """Per-block activation checkpointing for MasterNet.

    MasterNet overrides PlainNet.forward, so the patch must target MasterNet
    itself. Block outputs are kept; intermediate activations inside each block
    are recomputed during backward. Head (GAP + FC) is left uncheckpointed --
    it is cheap and keeps the graph intact.
    """
    output = x
    for the_block in self.block_list:
        if self.training and torch.is_grad_enabled():
            output = _torch_checkpoint(the_block, output, use_reentrant=False)
        else:
            output = the_block(output)
    output = F.adaptive_avg_pool2d(output, output_size=1)
    output = torch.flatten(output, 1)
    output = self.fc_linear(output)
    return output


Masternet.MasterNet.forward = _checkpointed_forward
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PlainNet IDW models with gradient checkpointing.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--structure-str", "--structure_str", dest="structure_str", default="")
    parser.add_argument("--structure-txt", "--structure_txt", dest="structure_txt", default="")
    parser.add_argument("--save-dir", "--save_dir", dest="save_dir", required=True)
    parser.add_argument("--branch", default="cap_nas")
    parser.add_argument("--method", default="enumeration")
    parser.add_argument("--budget-family", "--budget_family", dest="budget_family", default="custom")
    parser.add_argument("--budget-name", "--budget_name", dest="budget_name", default="custom")
    parser.add_argument("--search-score", "--search_score", dest="search_score", default="nan")
    parser.add_argument("--search-latency-ms", "--search_latency_ms", dest="search_latency_ms", default="nan")
    parser.add_argument("--search-flops", "--search_flops", dest="search_flops", default="nan")
    parser.add_argument("--search-params", "--search_params", dest="search_params", default="nan")
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size-per-gpu", "--batch_size_per_gpu", dest="batch_size_per_gpu", type=int, default=None)
    parser.add_argument("--workers-per-gpu", "--workers_per_gpu", dest="workers_per_gpu", type=int, default=4)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--datapath", default="")
    parser.add_argument("--class-balanced-sampling", "--class_balanced_sampling",
                        dest="class_balanced_sampling", action="store_true",
                        help="sample the train split uniformly over classes "
                             "instead of in proportion to per-class counts")
    parser.add_argument("--balanced-epoch-samples", "--balanced_epoch_samples",
                        dest="balanced_epoch_samples", type=int, default=None,
                        help="draws per epoch when class balancing is on "
                             "(default: train set size)")
    parser.add_argument("--dist-mode", "--dist_mode", dest="dist_mode",
                        default="single",
                        choices=["single", "cpu", "auto", "mpi", "ddp"],
                        help="ddp requires launching under torchrun")
    parser.add_argument("--use-test-set", "--use_test_set", dest="use_test_set", action="store_true")
    return parser.parse_args()


def load_structure(args: argparse.Namespace) -> str:
    if args.structure_str:
        return args.structure_str.strip()
    if args.structure_txt:
        return Path(args.structure_txt).read_text(encoding="utf-8").strip()
    raise ValueError("Provide --structure-str or --structure-txt")


def main() -> None:
    args = parse_args()
    dataset_cfg = get_dataset_defaults(args.dataset)
    save_dir = ensure_dir(Path(args.save_dir))
    structure_str = load_structure(args)
    structure_txt_path = write_text(save_dir / "structure.txt", structure_str + "\n")
    epochs = args.epochs if args.epochs is not None else dataset_cfg["epochs"]
    datapath = args.datapath or dataset_cfg["datapath"]

    command = [
        sys.executable,
        str(TIMM_TRAINER),
        "--model_type", "zen",
        "--arch", masternet_arch(),
        "--plainnet_struct_txt", str(structure_txt_path),
        "--dataset", args.dataset,
        "--num_classes", str(dataset_cfg["num_classes"]),
        "--input_image_size", str(dataset_cfg["input_image_size"]),
        "--data_dir", str(datapath),
        "--img_per_class", "-1",
        "--frac", "1",
        "--epfrac", "1",
        "--epochs", str(epochs),
        "--batch_size", "256",
        "--batch_size_per_gpu", str(args.batch_size_per_gpu or dataset_cfg["batch_size_per_gpu"]),
        "--workers_per_gpu", str(args.workers_per_gpu),
        "--dist_mode", args.dist_mode,
        "--optimizer", "sgd",
        "--lr_per_256", "0.1",
        "--target_lr_per_256", "0.0",
        "--lr_mode", "cosine",
        "--warmup", "5",
        "--bn_momentum", "0.01",
        "--wd", "5e-4",
        "--weight_init", "xavier",
        "--nesterov",
        "--label_smoothing",
        "--random_erase",
        "--mixup",
        "--auto_augment",
        "--gpu", str(args.gpu),
        "--save_dir", str(save_dir),
        "--summary_file", str(save_dir / "summary.csv"),
        "--completion_file", str(save_dir / "train.done"),
        "--expid", str(args.rank),
        "--print_freq", "10000000",
        "--auto_resume",
    ]
    if args.class_balanced_sampling:
        command.append("--class_balanced_sampling")
        if args.balanced_epoch_samples is not None:
            command += ["--balanced_epoch_samples", str(args.balanced_epoch_samples)]
    if args.use_test_set:
        command.append("--use_test_set")

    # Generate the checkpointing hook; sitecustomize is imported automatically
    # at interpreter startup, before the trainer runs.
    hook_dir = tempfile.mkdtemp(prefix="capnas_gradckpt_")
    Path(hook_dir, "sitecustomize.py").write_text(SITECUSTOMIZE)

    env = os.environ.copy()
    repo_root = str(REPO_ROOT)
    pythonpath_parts = [hook_dir, str(MODEL_ROOT), str(REPO_ROOT / "utils"), repo_root, env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = ":".join(p for p in pythonpath_parts if p)

    subprocess.run(command, cwd=str(MODEL_ROOT), check=True, env=env)


if __name__ == "__main__":
    main()
