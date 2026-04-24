from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class WindowedData:
    x: np.ndarray  # [N, D, L]
    y: np.ndarray  # [N, 1]
    meta: Dict[str, np.ndarray]


class TOCWindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def build_windows_from_arrays(
    features: np.ndarray,
    targets: np.ndarray,
    window_length: int,
    drop_boundary: bool = True,
    pad_mode: str = "edge",
    meta: Dict[str, np.ndarray] | None = None,
) -> WindowedData:
    """
    features: [N, D]
    targets:  [N]
    returns x: [M, D, L], y: [M, 1]
    """
    n, d = features.shape
    half = window_length // 2
    xs: List[np.ndarray] = []
    ys: List[float] = []
    metas: Dict[str, List] = {k: [] for k in (meta or {}).keys()}

    for idx in range(n):
        left = idx - half
        right = left + window_length
        if drop_boundary and (left < 0 or right > n):
            continue

        if left < 0 or right > n:
            pad_left = max(0, -left)
            pad_right = max(0, right - n)
            valid_left = max(0, left)
            valid_right = min(n, right)
            window = features[valid_left:valid_right]
            window = np.pad(window, ((pad_left, pad_right), (0, 0)), mode=pad_mode)
        else:
            window = features[left:right]

        xs.append(window.T.astype(np.float32))  # [D, L]
        ys.append(float(targets[idx]))
        for k, arr in (meta or {}).items():
            metas[k].append(arr[idx])

    meta_np = {k: np.asarray(v) for k, v in metas.items()}
    return WindowedData(x=np.stack(xs, axis=0), y=np.asarray(ys, dtype=np.float32).reshape(-1, 1), meta=meta_np)


def build_windows_from_dataframe(df, cfg: dict) -> WindowedData:
    dcfg = cfg["data"]
    feature_cols = list(dcfg["feature_cols"])
    target_col = dcfg["target_col"]
    well_id_col = dcfg.get("well_id_col")
    depth_col = dcfg.get("depth_col")
    window_length = int(cfg["model"]["window_length"])
    drop_boundary = bool(dcfg.get("drop_boundary", True))

    windows: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    meta_store: Dict[str, List] = {"row_index": []}
    if depth_col:
        meta_store[depth_col] = []
    if well_id_col:
        meta_store[well_id_col] = []

    if well_id_col and well_id_col in df.columns:
        groups = df.groupby(well_id_col, sort=False)
    else:
        groups = [("single_well", df)]

    for gid, g in groups:
        g = g.reset_index(drop=True)
        feats = g[feature_cols].to_numpy(dtype=np.float32)
        targs = g[target_col].to_numpy(dtype=np.float32)
        meta = {"row_index": np.arange(len(g))}
        if depth_col and depth_col in g.columns:
            meta[depth_col] = g[depth_col].to_numpy()
        if well_id_col and well_id_col in g.columns:
            meta[well_id_col] = g[well_id_col].to_numpy()

        w = build_windows_from_arrays(feats, targs, window_length=window_length, drop_boundary=drop_boundary, meta=meta)
        windows.append(w.x)
        labels.append(w.y)
        for k, arr in w.meta.items():
            meta_store[k].extend(arr.tolist())

    return WindowedData(
        x=np.concatenate(windows, axis=0),
        y=np.concatenate(labels, axis=0),
        meta={k: np.asarray(v) for k, v in meta_store.items()},
    )


def split_indices_random(n: int, train_ratio: float, val_ratio: float, test_ratio: float, seed: int = 42):
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.")
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def split_indices_by_well(meta: Dict[str, np.ndarray], well_id_col: str, train_ratio: float, val_ratio: float, test_ratio: float, seed: int = 42):
    wells = np.unique(meta[well_id_col])
    rng = np.random.default_rng(seed)
    rng.shuffle(wells)
    n_train = int(len(wells) * train_ratio)
    n_val = int(len(wells) * val_ratio)
    train_wells = set(wells[:n_train])
    val_wells = set(wells[n_train:n_train + n_val])
    test_wells = set(wells[n_train + n_val:])

    all_wells = meta[well_id_col]
    train_idx = np.where(np.isin(all_wells, list(train_wells)))[0]
    val_idx = np.where(np.isin(all_wells, list(val_wells)))[0]
    test_idx = np.where(np.isin(all_wells, list(test_wells)))[0]
    return train_idx, val_idx, test_idx


def subset_windowed_data(w: WindowedData, indices: np.ndarray) -> WindowedData:
    return WindowedData(
        x=w.x[indices],
        y=w.y[indices],
        meta={k: v[indices] for k, v in w.meta.items()},
    )
