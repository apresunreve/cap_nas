"""Central path configuration for the CAP-NAS pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT
MODEL_ROOT = PROJECT_ROOT / "model"
DATA_DIR = PROJECT_ROOT / "data" / "block_lookup"
CONFIG_PATH = PROJECT_ROOT / "config" / "defaults.yaml"

LOOKUP_GPU_CSV = DATA_DIR / "idw_block_lookup_jetson_gpu.csv"
LOOKUP_CSV = LOOKUP_GPU_CSV

ORIGINAL_SEED_STRUCTURE = (
    "SuperConvK3BNRELU(3,8,2,1)"
    "SuperResIDWE6K3(8,32,2,8,1)"
    "SuperResIDWE6K3(32,48,2,32,1)"
    "SuperResIDWE6K3(48,96,2,48,1)"
    "SuperResIDWE6K3(96,128,2,96,1)"
    "SuperConvK1BNRELU(128,2048,1,1)"
)
MAX_LAYERS = 14

# Dataset roots come from the environment; this package ships no defaults,
# since dataset locations are machine-specific. Unset -> Path("") , which fails
# the .exists() checks callers already perform.


def _dataset_root(env_var: str) -> Path:
    return Path(os.path.expanduser(os.environ.get(env_var, "")))


# iNat 2019 in ImageFolder layout
INAT_ROOT = _dataset_root("CAP_NAS_INAT_ROOT")
IMAGENET100_ROOT = _dataset_root("CAP_NAS_IMAGENET100_ROOT")



def setup_model_path() -> Path:
    """Ensure PlainNet/MasterNet/ModelLoader/DataLoader and global_utils are importable."""
    utils_root = str(PROJECT_ROOT / "utils")
    for root in (str(MODEL_ROOT), utils_root):
        if root not in sys.path:
            sys.path.insert(0, root)
    return MODEL_ROOT


def setup_repo_path() -> Path:
    """Add project root so top-level packages import when running scripts directly."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return PROJECT_ROOT


def masternet_arch() -> str:
    return f"{MODEL_ROOT / 'Masternet.py'}:MasterNet"
