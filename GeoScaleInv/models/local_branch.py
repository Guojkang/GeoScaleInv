from __future__ import annotations

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.norm1(x)
        attn_out, _ = self.attn(q, q, q, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class LocalBranch(nn.Module):
    """
    Input:  x_l [B, D, N_l, P_l]
    Output: z_l [B, D, N_l, d_model]
    """

    def __init__(self, patch_len: int, d_model: int, num_heads: int, num_layers: int, dropout: float, debug_shapes: bool = False):
        super().__init__()
        self.debug_shapes = debug_shapes
        self.token_proj = nn.Linear(patch_len, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x_l: torch.Tensor) -> torch.Tensor:
        if x_l.ndim != 4:
            raise ValueError(f"Expected x_l with shape [B, D, N_l, P_l], got {tuple(x_l.shape)}")
        b, d, n_l, p_l = x_l.shape
        x = self.token_proj(x_l)                    # [B, D, N_l, d_model]
        x = x.reshape(b * d, n_l, -1)               # variable-wise local modeling
        for layer in self.layers:
            x = layer(x)
        x = self.out_norm(x)
        x = x.reshape(b, d, n_l, -1)
        if self.debug_shapes:
            print(f"[LocalBranch] input={tuple(x_l.shape)} output={tuple(x.shape)}")
        return x
