from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["figure.dpi"] = 130


def plot_loss_curves(history: Dict[str, List[float]], save_path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(history.get("train_loss", []), label="train")
    plt.plot(history.get("val_loss", []), label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_pred_vs_true_scatter(y_true: np.ndarray, y_pred: np.ndarray, save_path: str) -> None:
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.6, s=12)
    xy_min = min(y_true.min(), y_pred.min())
    xy_max = max(y_true.max(), y_pred.max())
    plt.plot([xy_min, xy_max], [xy_min, xy_max], "k--", linewidth=1)
    plt.xlabel("True TOC")
    plt.ylabel("Predicted TOC")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_true_pred_curve(y_true: np.ndarray, y_pred: np.ndarray, save_path: str, max_points: int = 500) -> None:
    n = len(y_true)
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(int)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
    plt.figure(figsize=(8, 4))
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Pred")
    plt.xlabel("Sample Index")
    plt.ylabel("TOC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_residual_hist(y_true: np.ndarray, y_pred: np.ndarray, save_path: str) -> None:
    residual = y_pred - y_true
    plt.figure(figsize=(6, 4))
    plt.hist(residual, bins=30)
    plt.xlabel("Residual (Pred - True)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_all_plots(output_dir: str, history: Dict[str, List[float]], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    os.makedirs(output_dir, exist_ok=True)
    plot_loss_curves(history, os.path.join(output_dir, "loss_curve.png"))
    plot_pred_vs_true_scatter(y_true, y_pred, os.path.join(output_dir, "pred_vs_true_scatter.png"))
    plot_true_pred_curve(y_true, y_pred, os.path.join(output_dir, "true_pred_curve.png"))
    plot_residual_hist(y_true, y_pred, os.path.join(output_dir, "residual_hist.png"))
