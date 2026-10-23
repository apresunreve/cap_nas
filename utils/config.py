"""Load dataset and pipeline defaults from YAML with env overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from utils.paths import CONFIG_PATH


def _load_yaml() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_dataset_defaults(dataset: str) -> dict[str, Any]:
    cfg = _load_yaml()
    datasets = cfg.get("datasets", {})
    if dataset not in datasets:
        raise KeyError(f"Unknown dataset {dataset!r}; configure it in {CONFIG_PATH}")
    defaults = dict(datasets[dataset])

    # Dataset locations have no built-in defaults. Resolution order:
    #   1. CAP_NAS_<DATASET>_DATAPATH   2. CAP_NAS_<DATASET>_ROOT
    #   3. the `datapath` entry in config/defaults.yaml (null by default)
    env_keys = [
        f"CAP_NAS_{dataset.upper()}_DATAPATH",
        f"CAP_NAS_{dataset.upper()}_ROOT",
    ]
    # a multi-word dataset name also accepts the shared prefix, so e.g.
    # inat_resized is satisfied by CAP_NAS_INAT_ROOT (what the dataloader reads)
    if "_" in dataset:
        prefix = dataset.split("_")[0].upper()
        env_keys += [f"CAP_NAS_{prefix}_DATAPATH", f"CAP_NAS_{prefix}_ROOT"]
    datapath = None
    for env_key in env_keys:
        if os.environ.get(env_key):
            datapath = os.environ[env_key]
            break
    if datapath is None:
        configured = defaults.get("datapath")
        if configured is not None and str(configured).strip():
            datapath = str(configured)
    if datapath is None:
        raise RuntimeError(
            f"Dataset path for {dataset!r} is not configured: set "
            f"{' or '.join(env_keys)}, or fill in `datapath` for "
            f"{dataset!r} in {CONFIG_PATH}."
        )
    defaults["datapath"] = os.path.expanduser(datapath)
    return defaults


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, content: str) -> Path:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")
    return path
