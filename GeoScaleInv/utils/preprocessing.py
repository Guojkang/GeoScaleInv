from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class NumericScaler:
    method: str = "standard"
    eps: float = 1e-8
    center_: Optional[np.ndarray] = None
    scale_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "NumericScaler":
        if self.method == "none":
            self.center_ = np.zeros(x.shape[-1], dtype=np.float32)
            self.scale_ = np.ones(x.shape[-1], dtype=np.float32)
        elif self.method == "standard":
            self.center_ = np.nanmean(x, axis=0).astype(np.float32)
            self.scale_ = (np.nanstd(x, axis=0) + self.eps).astype(np.float32)
        elif self.method == "minmax":
            x_min = np.nanmin(x, axis=0)
            x_max = np.nanmax(x, axis=0)
            self.center_ = x_min.astype(np.float32)
            self.scale_ = (x_max - x_min + self.eps).astype(np.float32)
        else:
            raise ValueError(f"Unsupported scaler method: {self.method}")
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fit before transform.")
        return ((x - self.center_) / self.scale_).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fit before inverse_transform.")
        return (x * self.scale_ + self.center_).astype(np.float32)

    def state_dict(self) -> dict:
        return {"method": self.method, "eps": self.eps, "center_": self.center_, "scale_": self.scale_}

    @classmethod
    def from_state_dict(cls, state: dict) -> "NumericScaler":
        obj = cls(method=state["method"], eps=state.get("eps", 1e-8))
        obj.center_ = state.get("center_")
        obj.scale_ = state.get("scale_")
        return obj


def read_table(file_path: str, file_type: str = "auto") -> pd.DataFrame:
    if file_type == "auto":
        file_type = "xlsx" if file_path.lower().endswith((".xlsx", ".xls")) else "csv"
    if file_type == "csv":
        return pd.read_csv(file_path)
    if file_type == "xlsx":
        return pd.read_excel(file_path)
    raise ValueError(f"Unsupported file_type: {file_type}")


def assert_columns_exist(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c is not None and c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")


def optional_sort_by_depth(df: pd.DataFrame, depth_col: str | None, sort_by_depth: bool = True) -> pd.DataFrame:
    if sort_by_depth and depth_col and depth_col in df.columns:
        return df.sort_values(depth_col).reset_index(drop=True)
    return df.reset_index(drop=True)


def smooth_series(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window=window, min_periods=1, center=True).mean()


def clip_outliers_series(s: pd.Series, method: str = "iqr", z_thresh: float = 3.0) -> pd.Series:
    x = s.astype(float)
    if method == "iqr":
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
    elif method == "zscore":
        mu = x.mean()
        std = x.std() + 1e-8
        lo = mu - z_thresh * std
        hi = mu + z_thresh * std
    else:
        raise ValueError(f"Unsupported outlier method: {method}")
    return x.clip(lower=lo, upper=hi)


def preprocess_dataframe(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    dcfg = cfg["data"]
    feature_cols = list(dcfg["feature_cols"])
    target_col = dcfg["target_col"]
    depth_col = dcfg.get("depth_col")
    well_id_col = dcfg.get("well_id_col")

    required_cols = feature_cols + [target_col]
    if depth_col:
        required_cols.append(depth_col)
    if well_id_col:
        required_cols.append(well_id_col)
    assert_columns_exist(df, required_cols)

    df = optional_sort_by_depth(df, depth_col=depth_col, sort_by_depth=bool(dcfg.get("sort_by_depth", True)))
    df = df.copy()

    for col in feature_cols + [target_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    group_keys = [well_id_col] if well_id_col else None
    if group_keys:
        grouped = []
        for _, g in df.groupby(group_keys[0], sort=False):
            g = g.copy()
            if depth_col and depth_col in g.columns:
                g = g.sort_values(depth_col)
            for col in feature_cols + [target_col]:
                g[col] = g[col].interpolate(method="linear", limit_direction="both")
                g[col] = g[col].ffill().bfill()
                if bool(dcfg.get("enable_outlier_clip", True)):
                    g[col] = clip_outliers_series(g[col], method=str(dcfg.get("outlier_method", "iqr")))
                if bool(dcfg.get("smooth_enabled", False)) and col in feature_cols:
                    g[col] = smooth_series(g[col], int(dcfg.get("smooth_window", 3)))
            grouped.append(g)
        df = pd.concat(grouped, axis=0).reset_index(drop=True)
    else:
        for col in feature_cols + [target_col]:
            df[col] = df[col].interpolate(method="linear", limit_direction="both")
            df[col] = df[col].ffill().bfill()
            if bool(dcfg.get("enable_outlier_clip", True)):
                df[col] = clip_outliers_series(df[col], method=str(dcfg.get("outlier_method", "iqr")))
            if bool(dcfg.get("smooth_enabled", False)) and col in feature_cols:
                df[col] = smooth_series(df[col], int(dcfg.get("smooth_window", 3)))

    return df.reset_index(drop=True)
