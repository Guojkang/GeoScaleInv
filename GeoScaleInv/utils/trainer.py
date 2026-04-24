from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import regression_metrics


class EarlyStopping:
    def __init__(self, patience: int = 10):
        self.patience = patience
        self.best = float("inf")
        self.counter = 0

    def step(self, value: float) -> bool:
        if value < self.best:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class Trainer:
    def __init__(self, model: nn.Module, cfg: dict, target_scaler=None):
        self.model = model
        self.cfg = cfg
        self.target_scaler = target_scaler
        tcfg = cfg["train"]
        device = tcfg.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

        loss_name = str(tcfg.get("loss", "huber")).lower()
        if loss_name == "mse":
            self.criterion = nn.MSELoss()
        elif loss_name == "huber":
            self.criterion = nn.HuberLoss(delta=1.0)
        else:
            raise ValueError(f"Unsupported loss: {loss_name}")

        lr = float(tcfg["lr"])
        wd = float(tcfg.get("weight_decay", 0.0))
        opt_name = str(tcfg.get("optimizer", "adamw")).lower()
        if opt_name == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        sched_name = str(tcfg.get("scheduler", "none")).lower()
        self.scheduler = None
        if sched_name == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=5, factor=0.5)

        self.history = {"train_loss": [], "val_loss": [], "val_RMSE": [], "val_MAE": [], "val_R2": []}

    def _inverse_target(self, x: np.ndarray) -> np.ndarray:
        if self.target_scaler is None:
            return x
        return self.target_scaler.inverse_transform(x)

    def run_epoch(self, loader: DataLoader, train: bool = True) -> Tuple[float, Optional[dict]]:
        self.model.train(train)
        losses = []
        preds, gts = [], []

        for xb, yb in loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            with torch.set_grad_enabled(train):
                out = self.model(xb)
                loss = self.criterion(out, yb)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            losses.append(loss.item())
            preds.append(out.detach().cpu().numpy())
            gts.append(yb.detach().cpu().numpy())

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        if train:
            return mean_loss, None

        y_pred = np.concatenate(preds, axis=0)
        y_true = np.concatenate(gts, axis=0)
        y_pred = self._inverse_target(y_pred)
        y_true = self._inverse_target(y_true)
        metrics = regression_metrics(y_true, y_pred)
        return mean_loss, metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, checkpoint_path: str, extra_state: dict | None = None):
        epochs = int(self.cfg["train"]["epochs"])
        early_stop = EarlyStopping(patience=int(self.cfg["train"].get("early_stopping_patience", 15)))
        best_state = None
        best_val = float("inf")

        for epoch in range(1, epochs + 1):
            train_loss, _ = self.run_epoch(train_loader, train=True)
            val_loss, val_metrics = self.run_epoch(val_loader, train=False)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_RMSE"].append(val_metrics["RMSE"])
            self.history["val_MAE"].append(val_metrics["MAE"])
            self.history["val_R2"].append(val_metrics["R2"])

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"val_RMSE={val_metrics['RMSE']:.4f} | val_MAE={val_metrics['MAE']:.4f} | val_R2={val_metrics['R2']:.4f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                best_state = deepcopy(self.model.state_dict())
                state = {
                    "model_state": best_state,
                    "config": self.cfg,
                    "history": self.history,
                }
                if extra_state:
                    state.update(extra_state)
                torch.save(state, checkpoint_path)

            if early_stop.step(val_loss):
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self.history

    @torch.no_grad()
    def predict(self, loader: DataLoader):
        self.model.eval()
        preds, gts = [], []
        for xb, yb in loader:
            xb = xb.to(self.device)
            out = self.model(xb)
            preds.append(out.detach().cpu().numpy())
            gts.append(yb.detach().cpu().numpy())
        y_pred = np.concatenate(preds, axis=0)
        y_true = np.concatenate(gts, axis=0)
        y_pred_inv = self._inverse_target(y_pred)
        y_true_inv = self._inverse_target(y_true)
        return y_true_inv.reshape(-1), y_pred_inv.reshape(-1), regression_metrics(y_true_inv, y_pred_inv)
