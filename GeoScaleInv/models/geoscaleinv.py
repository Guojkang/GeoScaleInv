from __future__ import annotations

import torch
import torch.nn as nn

from .fusion import CrossAttentionFusion
from .global_branch import GlobalBranch
from .local_branch import LocalBranch
from .patching import MultiScalePatching


class RegressionHead(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden = max(64, in_dim // 4)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GeoScaleInv(nn.Module):
    """GeoScaleInv template model for TOC regression."""

    def __init__(self, cfg: dict):
        super().__init__()
        mcfg = cfg["model"]
        self.debug_shapes = bool(mcfg.get("debug_shapes", False))
        self.use_global_branch = bool(mcfg.get("use_global_branch", True))
        self.use_local_branch = bool(mcfg.get("use_local_branch", True))
        self.use_cross_attention = bool(mcfg.get("use_cross_attention", True))
        if not self.use_global_branch and not self.use_local_branch:
            raise ValueError("At least one of use_global_branch or use_local_branch must be True.")

        self.window_length = int(mcfg["window_length"])
        self.num_features = len(cfg["data"]["feature_cols"])
        self.d_model = int(mcfg["d_model"])

        self.patching = MultiScalePatching(
            global_patch_len=int(mcfg["global_patch_len"]),
            global_stride=int(mcfg["global_stride"]),
            local_patch_len=int(mcfg["local_patch_len"]),
            local_stride=int(mcfg["local_stride"]),
        )
        info = self.patching.get_patch_info(self.window_length)
        self.num_global_patches = info.num_global_patches
        self.num_local_patches = info.num_local_patches

        if self.use_global_branch:
            self.global_branch = GlobalBranch(
                patch_len=int(mcfg["global_patch_len"]),
                d_model=self.d_model,
                dropout=float(mcfg["dropout"]),
                bidirectional=bool(mcfg.get("bidirectional", True)),
                fuse_mode=str(mcfg.get("fuse_mode", "sum")),
                d_state=int(mcfg.get("d_state", 16)),
                d_conv=int(mcfg.get("d_conv", 4)),
                expand=int(mcfg.get("expand", 2)),
                use_mamba_backend=bool(mcfg.get("use_mamba_backend", False)),
                debug_shapes=self.debug_shapes,
            )
        else:
            self.global_branch = None

        if self.use_local_branch:
            self.local_branch = LocalBranch(
                patch_len=int(mcfg["local_patch_len"]),
                d_model=self.d_model,
                num_heads=int(mcfg["num_heads"]),
                num_layers=int(mcfg["local_layers"]),
                dropout=float(mcfg["dropout"]),
                debug_shapes=self.debug_shapes,
            )
        else:
            self.local_branch = None

        self.fusion = CrossAttentionFusion(
            d_model=self.d_model,
            num_heads=int(mcfg["num_heads"]),
            dropout=float(mcfg["dropout"]),
            use_cross_attention=self.use_cross_attention,
            debug_shapes=self.debug_shapes,
        )

        n_tokens = self.num_local_patches if self.use_local_branch else self.num_global_patches
        head_in_dim = self.num_features * n_tokens * self.d_model
        self.reg_head = RegressionHead(head_in_dim, dropout=float(mcfg["dropout"]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D, L]
        Returns:
            y_hat: [B, 1]
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [B, D, L], got {tuple(x.shape)}")
        if x.shape[1] != self.num_features:
            raise ValueError(f"Expected D={self.num_features}, got {x.shape[1]}")
        if x.shape[2] != self.window_length:
            raise ValueError(f"Expected L={self.window_length}, got {x.shape[2]}")

        x_g, x_l = self.patching(x)

        z_g = self.global_branch(x_g) if self.global_branch is not None else None
        z_l = self.local_branch(x_l) if self.local_branch is not None else None

        if z_l is not None:
            z = self.fusion(z_l, z_g)
        else:
            z = z_g

        if z is None:
            raise RuntimeError("No features available for regression head.")

        b = z.shape[0]
        z = z.reshape(b, -1)
        y_hat = self.reg_head(z)

        if self.debug_shapes:
            print(f"[GeoScaleInv] flattened={tuple(z.shape)} output={tuple(y_hat.shape)}")
        return y_hat
