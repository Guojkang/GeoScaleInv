from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from models.geoscaleinv import GeoScaleInv
from utils.dataset import TOCWindowDataset, build_windows_from_dataframe
from utils.preprocessing import NumericScaler, preprocess_dataframe, read_table
from utils.trainer import Trainer
from utils.visualization import save_all_plots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/eval")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    if args.data_path:
        cfg["data"]["file_path"] = args.data_path

    root = Path(__file__).resolve().parent
    file_path = cfg["data"]["file_path"]
    if not os.path.isabs(file_path):
        file_path = str(root / file_path)
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(file_path=file_path, file_type=str(cfg["data"].get("file_type", "auto")))
    df = preprocess_dataframe(df, cfg)
    windows = build_windows_from_dataframe(df, cfg)

    feat_scaler = NumericScaler.from_state_dict(ckpt["feature_scaler"])
    targ_scaler = NumericScaler.from_state_dict(ckpt["target_scaler"])

    x = windows.x.transpose(0, 2, 1).reshape(-1, windows.x.shape[1])
    x = feat_scaler.transform(x)
    windows.x = x.reshape(windows.x.shape[0], windows.x.shape[2], windows.x.shape[1]).transpose(0, 2, 1)
    windows.y = targ_scaler.transform(windows.y)

    ds = TOCWindowDataset(windows.x, windows.y)
    loader = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)

    model = GeoScaleInv(cfg)
    model.load_state_dict(ckpt["model_state"])
    trainer = Trainer(model, cfg, target_scaler=targ_scaler)
    y_true, y_pred, metrics = trainer.predict(loader)

    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    history = ckpt.get("history", {})
    save_all_plots(str(output_dir), history, y_true, y_pred)
    with open(output_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(output_dir / "used_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    main()
