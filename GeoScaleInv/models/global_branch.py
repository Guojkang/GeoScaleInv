from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


try:
    from mamba_ssm import Mamba  # type: ignore
    HAS_MAMBA = True
except Exception:
    Mamba = None
    HAS_MAMBA = False


class SequenceMixer(nn.Module):
    """Mamba if explicitly enabled and available; otherwise a GRU fallback."""

    def __init__(self, d_model: int, dropout: float, d_state: int = 16, d_conv: int = 4, expand: int = 2, use_mamba_backend: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_mamba = bool(use_mamba_backend and HAS_MAMBA and torch.cuda.is_available())
        if self.use_mamba:
            self.mixer = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.mixer = nn.GRU(input_size=d_model, hidden_size=d_model, batch_first=True)
            self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        if self.use_mamba:
            x = self.mixer(x)
        else:
            x, _ = self.mixer(x)
            x = self.proj(x)
        x = self.dropout(x)
        return x + residual


class GlobalBranch(nn.Module):
    """
    Input:  x_g [B, D, N_g, P_g]
    Output: z_g [B, D, N_g, d_model]
    """

    def __init__(
        self,
        patch_len: int,
        d_model: int,
        dropout: float,
        bidirectional: bool = True,
        fuse_mode: str = "sum",
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_mamba_backend: bool = False,
        debug_shapes: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.fuse_mode = fuse_mode
        self.debug_shapes = debug_shapes

        self.token_proj = nn.Linear(patch_len, d_model)
        self.forward_mixer = SequenceMixer(d_model, dropout, d_state, d_conv, expand, use_mamba_backend=use_mamba_backend)
        self.out_norm = nn.LayerNorm(d_model)

        if self.bidirectional:
            self.reverse_mixer = SequenceMixer(d_model, dropout, d_state, d_conv, expand, use_mamba_backend=use_mamba_backend)
            if self.fuse_mode == "concat":
                self.fuse_proj = nn.Linear(2 * d_model, d_model)
        elif self.fuse_mode == "concat":
            raise ValueError("fuse_mode='concat' requires bidirectional=True")

    def forward(self, x_g: torch.Tensor) -> torch.Tensor:
        if x_g.ndim != 4:
            raise ValueError(f"Expected x_g with shape [B, D, N_g, P_g], got {tuple(x_g.shape)}")

        b, d, n_g, p_g = x_g.shape
        x = x_g.reshape(b, d * n_g, p_g)             # [B, D*N_g, P_g]
        x = self.token_proj(x)                       # [B, D*N_g, d_model]
        z_f = self.forward_mixer(x)

        if self.bidirectional:
            x_rev = torch.flip(x, dims=[1])
            z_r = self.reverse_mixer(x_rev)
            z_r = torch.flip(z_r, dims=[1])
            if self.fuse_mode == "sum":
                z = z_f + z_r
            else:
                z = self.fuse_proj(torch.cat([z_f, z_r], dim=-1))
        else:
            z = z_f

        z = self.out_norm(z)
        z = z.reshape(b, d, n_g, -1)               # [B, D, N_g, d_model]

        if self.debug_shapes:
            print(f"[GlobalBranch] input={tuple(x_g.shape)} output={tuple(z.shape)}")
        return z
