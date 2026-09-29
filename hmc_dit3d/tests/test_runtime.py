"""Tests for runtime reproducibility helpers."""

from __future__ import annotations

import os

import torch

from hmc_dit3d.utils.runtime import set_global_seed


def test_set_global_seed_sets_reproducibility_flags() -> None:
    """The runtime helper should configure deterministic execution knobs."""
    set_global_seed(123)

    assert os.environ["PYTHONHASHSEED"] == "123"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.are_deterministic_algorithms_enabled()

    if torch.cuda.is_available():
        if hasattr(torch.backends.cuda, "flash_sdp_enabled"):
            assert not torch.backends.cuda.flash_sdp_enabled()
        if hasattr(torch.backends.cuda, "mem_efficient_sdp_enabled"):
            assert not torch.backends.cuda.mem_efficient_sdp_enabled()
        if hasattr(torch.backends.cuda, "math_sdp_enabled"):
            assert torch.backends.cuda.math_sdp_enabled()
