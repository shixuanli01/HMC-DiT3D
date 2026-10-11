"""Runtime helpers for reproducibility and device selection."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set all random seeds needed for reproducible experiments.

    Args:
        seed: Random seed value.
        deterministic: Force deterministic kernels and math-only attention.
            When False, seeds are still set but cuDNN autotuning and the
            flash / memory-efficient attention kernels are allowed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if not deterministic:
        LOGGER.info("Global seed set to %d (non-deterministic fast kernels).", seed)
        return
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    LOGGER.info("Global seed set to %d.", seed)


def detect_device(prefer_cuda: bool = True) -> torch.device:
    """Detect the best available torch device and log its details.

    Args:
        prefer_cuda: Whether CUDA should be preferred when available.

    Returns:
        torch.device: Selected device.
    """
    use_cuda = prefer_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        device_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        LOGGER.info(
            "Using device=%s name=%s cuda=%s",
            device,
            device_name,
            cuda_version,
        )
    else:
        LOGGER.info("Using device=%s", device)
    return device
