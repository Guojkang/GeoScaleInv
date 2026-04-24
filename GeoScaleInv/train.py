from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from models.geoscaleinv import GeoScaleInv
from utils.dataset import (
    TOCWindowDataset,
    build_windows_from_dataframe,
    split_indices_by_well,
    split_indices_random,
    subset_windowed_data,
)
from utils.preprocessing import NumericScaler, preprocess_dataframe, read_table
from utils.trainer import Trainer
from utils.visualization import save_all_plots


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fit_and_apply_scalers(train_w, val_w, test_w, cfg):
    dcfg = cfg["data"]
    feat_scaler = NumericScaler(method=str(dcfg.get("feature_scaler", "standard")))
    targ_scaler = NumericScaler(method=str(dcfg.get("target_scaler", "standard")))

    x_train_flat = train_w.x.transpose(0, 2, 1).reshape(-1, train_w.x.shape[1])
    y_train_flat = train_w.y.reshape(-1, 1)
    feat_scaler.fit(x_train_flat)
    targ_scaler.fit(y_train_flat)

    def apply_x(w):
        x = w.x.transpose(0, 2, 1).reshape(-1, w.x.shape[1])
        x = feat_scaler.transform(x)
        x = x.reshape(w.x.shape[0], w.x.shape[2], w.x.shape[1]).transpose(0, 2, 1)
        w.x = x.astype(np.float32)
        return w

    def apply_y(w):
        w.y = targ_scaler.transform(w.y).astype(np.float32)
        return w

    train_w = apply_x(train_w)
    val_w = apply_x(val_w)
    test_w = apply_x(test_w)

    train_w = apply_y(train_w)
    val_w = apply_y(val_w)
    test_w = apply_y(test_w)
    return train_w, val_w, test_w, feat_scaler, targ_scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--ablation_variant", type=str, default="full", help="full|unidirectional|no_cross_attention|local_only|global_only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_path:
        cfg["data"]["file_path"] = args.data_path
    if args.output_dir:
        cfg["train"]["output_dir"] = args.output_dir

    variant = str(args.ablation_variant).lower()
    if variant == "unidirectional":
        cfg["model"]["bidirectional"] = False
    elif variant == "no_cross_attention":
        cfg["model"]["use_cross_attention"] = False
    elif variant == "local_only":
        cfg["model"]["use_global_branch"] = False
        cfg["model"]["use_cross_attention"] = False
    elif variant == "global_only":
        cfg["model"]["use_local_branch"] = False
        cfg["model"]["use_cross_attention"] = False
    elif variant != "full":
        raise ValueError(f"Unknown ablation_variant: {variant}")

    set_seed(int(cfg.get("seed", 42)))

    file_path = cfg["data"]["file_path"]
    root = Path(__file__).resolve().parent
    if not os.path.isabs(file_path):
        file_path = str(root / file_path)

    output_dir = root / cfg["train"]["output_dir"]
    ckpt_dir = root / cfg["train"]["checkpoint_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / cfg["train"]["save_best_name"]

    print(f"Ablation variant: {variant}")
    print(f"Reading data from: {file_path}")
    df = read_table(file_path=file_path, file_type=str(cfg["data"].get("file_type", "auto")))
    df = preprocess_dataframe(df, cfg)

    windows = build_windows_from_dataframe(df, cfg)
    n_samples = len(windows.x)
    print(f"Constructed {n_samples} window samples. Shape={windows.x.shape}")

    split_mode = str(cfg["data"].get("split_mode", "random"))
    if split_mode == "well":
        well_id_col = cfg["data"].get("well_id_col")
        if not well_id_col:
            raise ValueError("split_mode='well' requires data.well_id_col.")
        train_idx, val_idx, test_idx = split_indices_by_well(
            windows.meta,
            well_id_col=well_id_col,
            train_ratio=float(cfg["data"]["train_ratio"]),
            val_ratio=float(cfg["data"]["val_ratio"]),
            test_ratio=float(cfg["data"]["test_ratio"]),
            seed=int(cfg.get("seed", 42)),
        )
    else:
        train_idx, val_idx, test_idx = split_indices_random(
            n=n_samples,
            train_ratio=float(cfg["data"]["train_ratio"]),
            val_ratio=float(cfg["data"]["val_ratio"]),
            test_ratio=float(cfg["data"]["test_ratio"]),
            seed=int(cfg.get("seed", 42)),
        )

    train_w = subset_windowed_data(windows, train_idx)
    val_w = subset_windowed_data(windows, val_idx)
    test_w = subset_windowed_data(windows, test_idx)
    train_w, val_w, test_w, feat_scaler, targ_scaler = fit_and_apply_scalers(train_w, val_w, test_w, cfg)

    train_ds = TOCWindowDataset(train_w.x, train_w.y)
    val_ds = TOCWindowDataset(val_w.x, val_w.y)
    test_ds = TOCWindowDataset(test_w.x, test_w.y)

    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["data"].get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = GeoScaleInv(cfg)
    trainer = Trainer(model, cfg, target_scaler=targ_scaler)

    extra_state = {
        "feature_scaler": feat_scaler.state_dict(),
        "target_scaler": targ_scaler.state_dict(),
    }
    history = trainer.fit(train_loader, val_loader, checkpoint_path=str(ckpt_path), extra_state=extra_state)

    y_true, y_pred, metrics = trainer.predict(test_loader)
    print("\nTest metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    save_all_plots(str(output_dir), history, y_true, y_pred)

    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(output_dir / "used_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    print(f"\nSaved checkpoint to: {ckpt_path}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
