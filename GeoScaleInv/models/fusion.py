from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    local as query, global as key/value
    local:  [B, D, N_l, d_model]
    global: [B, D, N_g, d_model]
    output: [B, D, N_l, d_model]
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float, use_cross_attention: bool = True, debug_shapes: bool = False):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.debug_shapes = debug_shapes
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor | None) -> torch.Tensor:
        if not self.use_cross_attention or global_feat is None:
            return local_feat

        b, d, n_l, dm = local_feat.shape
        _, _, n_g, _ = global_feat.shape
        q = local_feat.reshape(b * d, n_l, dm)
        kv = global_feat.reshape(b * d, n_g, dm)

        qn = self.norm1(q)
        cross_out, _ = self.cross_attn(qn, kv, kv, need_weights=False)
        x = q + cross_out
        x = x + self.ffn(self.norm2(x))
        x = x.reshape(b, d, n_l, dm)

        if self.debug_shapes:
            print(f"[CrossAttentionFusion] local={tuple(local_feat.shape)} global={tuple(global_feat.shape)} output={tuple(x.shape)}")
        return x
