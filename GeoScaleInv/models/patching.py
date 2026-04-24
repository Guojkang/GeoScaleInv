from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class PatchInfo:
    num_global_patches: int
    num_local_patches: int


def compute_num_patches(length: int, patch_len: int, stride: int) -> int:
    if length < patch_len:
        raise ValueError(f"window_length={length} must be >= patch_len={patch_len}")
    return (length - patch_len) // stride + 1


class MultiScalePatching(nn.Module):
    """Create global and local patches from X of shape [B, D, L]."""

    def __init__(self, global_patch_len: int, global_stride: int, local_patch_len: int, local_stride: int):
        super().__init__()
        self.global_patch_len = global_patch_len
        self.global_stride = global_stride
        self.local_patch_len = local_patch_len
        self.local_stride = local_stride

    def get_patch_info(self, window_length: int) -> PatchInfo:
        return PatchInfo(
            num_global_patches=compute_num_patches(window_length, self.global_patch_len, self.global_stride),
            num_local_patches=compute_num_patches(window_length, self.local_patch_len, self.local_stride),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, D, L]
        Returns:
            x_g: [B, D, N_g, P_g]
            x_l: [B, D, N_l, P_l]
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [B, D, L], got {tuple(x.shape)}")
        x_g = x.unfold(dimension=-1, size=self.global_patch_len, step=self.global_stride)
        x_l = x.unfold(dimension=-1, size=self.local_patch_len, step=self.local_stride)
        return x_g.contiguous(), x_l.contiguous()
