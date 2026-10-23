#!/usr/bin/env python3
"""Single-GPU PlainNet training via grad-accum timm trainer."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from utils.config import ensure_dir, get_dataset_defaults, write_text
from utils.paths import MODEL_ROOT, REPO_ROOT, masternet_arch, setup_repo_path

setup_repo_path()

TIMM_TRAINER = Path(__file__).resolve().parent / "timm_trainer_grad_accum.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PlainNet IDW models (single GPU, grad accum).")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--structure-str", "--structure_str", dest="structure_str", default="")
    parser.add_argument("--structure-txt", "--structure_txt", dest="structure_txt", default="")
    parser.add_argument("--save-dir", "--save_dir", dest="save_dir", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size-per-gpu", "--batch_size_per_gpu", dest="batch_size_per_gpu", type=int, default=None)
    parser.add_argument("--workers-per-gpu", "--workers_per_gpu", dest="workers_per_gpu", type=int, default=4)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--datapath", default="")
    parser.add_argument("--img-per-class", "--img_per_class", dest="img_per_class", type=int, default=-1)
    parser.add_argument("--frac", type=float, default=1.0)
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
        "--epochs", str(epochs),
        "--frac", str(args.frac),
        "--epfrac", "1",
        "--img_per_class", str(args.img_per_class),
        "--batch_size_per_gpu", str(args.batch_size_per_gpu or dataset_cfg["batch_size_per_gpu"]),
        "--workers_per_gpu", str(args.workers_per_gpu),
        "--optimizer", "sgd",
        "--lr_per_256", "0.1",
        "--lr_mode", "cosine",
        "--wd", "5e-4",
        "--weight_init", "xavier",
        "--nesterov",
        "--label_smoothing",
        "--random_erase",
        "--mixup",
        "--auto_augment",
        "--gpu", str(args.gpu),
        "--save_dir", str(save_dir),
        "--auto_resume",
    ]

    env = os.environ.copy()
    pythonpath_parts = [str(MODEL_ROOT), str(REPO_ROOT), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = ":".join(p for p in pythonpath_parts if p)
    subprocess.run(command, cwd=str(MODEL_ROOT), check=True, env=env)


if __name__ == "__main__":
    main()
