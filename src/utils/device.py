"""
Device selection and reproducible seeding for GPU training.

get_device() picks CUDA when available and falls back to CPU, so the same
code path runs on the GTX 1660 Ti and on a CPU-only machine. set_seed()
seeds Python, NumPy, and torch (CPU + CUDA) for reproducible runs.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the best available compute device.

    Args:
        prefer_cuda: When True (default), use CUDA if available.

    Returns:
        torch.device('cuda') if available and preferred, else torch.device('cpu').
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("Using CUDA device: %s (%.1f GB)", name, total_gb)
        return device

    if prefer_cuda:
        logger.warning(
            "CUDA not available (torch=%s). Falling back to CPU. "
            "If you expected GPU, reinstall the CUDA build: "
            "pip install torch==2.8.0 --index-url "
            "https://download.pytorch.org/whl/cu126",
            torch.__version__,
        )
    return torch.device("cpu")


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed all RNGs for reproducibility.

    Args:
        seed: Random seed.
        deterministic: When True, force deterministic cuDNN kernels. This
            improves reproducibility at some throughput cost; set False to let
            cuDNN autotune for speed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
