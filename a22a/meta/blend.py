"""Blending and stacking helpers for the Phase 14 meta-learner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class StackerResult:
    """Container holding outputs from the stacker training routine."""

    oof: pd.Series
    fitted: pd.Series
    model: object
    info: dict[str, object]


def _as_numpy(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=float)
    return np.asarray(X, dtype=float)


def _forward_splits(n_samples: int, kfold: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_samples <= 1:
        return []
    kfold = max(2, int(kfold))
    boundaries = np.linspace(0, n_samples, kfold + 1, dtype=int)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        train_idx = np.arange(0, start, dtype=int)
        val_idx = np.arange(start, end, dtype=int)
        if train_idx.size == 0 or val_idx.size == 0:
            continue
        folds.append((train_idx, val_idx))
    if not folds:
        cut = max(1, int(round(0.7 * n_samples)))
        train_idx = np.arange(0, cut, dtype=int)
        val_idx = np.arange(cut, n_samples, dtype=int)
        if train_idx.size and val_idx.size:
            folds.append((train_idx, val_idx))
    return folds


def _tune_logit_C(
    X: np.ndarray,
    y: np.ndarray,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    seed: int,
    grid: Sequence[float] = (0.01, 0.1, 0.5, 1.0, 2.0, 5.0),
) -> float:
    best_c = 1.0
    best_score = float("inf")
    if not folds:
        return best_c
    for C in grid:
        scores: list[float] = []
        for train_idx, val_idx in folds:
            if train_idx.size == 0 or val_idx.size == 0:
                continue
            model = make_pipeline(
                StandardScaler(with_mean=True, with_std=True),
                LogisticRegression(max_iter=2000, C=C, solver="lbfgs", random_state=seed),
            )
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict_proba(X[val_idx])[:, 1]
            preds = np.clip(preds, 1e-6, 1 - 1e-6)
            scores.append(log_loss(y[val_idx], preds))
        if scores:
            score = float(np.mean(scores))
            if score < best_score:
                best_score = score
                best_c = float(C)
    return best_c


def _fit_model(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    C: float = 1.0,
) -> object:
    if method == "logit":
        model = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(max_iter=2000, C=C, solver="lbfgs", random_state=seed),
        )
        model.fit(X, y)
        return model
    if method == "gbdt":
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_samples=10,
            max_depth=-1,
            random_state=seed,
        )
        model.fit(X, y)
        return model
    raise ValueError(f"unknown stacker method: {method}")


def _predict(model: object, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raise AttributeError("stacker model must expose predict_proba")


def train_stacker(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    method: str = "logit",
    kfold: int = 5,
    seed: int = 14,
) -> StackerResult:
    """Train a meta-learner using forward-chaining cross-validation."""

    if features.empty:
        raise ValueError("features must contain at least one column for stacking")

    method = method.lower()
    X = _as_numpy(features)
    y = target.to_numpy(dtype=float)
    n_samples = X.shape[0]
    folds = _forward_splits(n_samples, kfold)

    used_folds: list[tuple[np.ndarray, np.ndarray]] = []
    oof = np.full(n_samples, np.nan, dtype=float)

    params: dict[str, float | int] = {}
    if method == "logit":
        params["C"] = _tune_logit_C(X, y, folds, seed=seed)

    for train_idx, val_idx in folds:
        if train_idx.size == 0 or val_idx.size == 0:
            continue
        model = _fit_model(method, X[train_idx], y[train_idx], seed=seed, C=float(params.get("C", 1.0)))
        preds = _predict(model, X[val_idx])
        oof[val_idx] = preds
        used_folds.append((train_idx, val_idx))

    if not used_folds:
        model = _fit_model(method, X, y, seed=seed, C=float(params.get("C", 1.0)))
        preds = _predict(model, X)
        oof[:] = preds
    final_model = _fit_model(method, X, y, seed=seed, C=float(params.get("C", 1.0)))
    fitted = _predict(final_model, X)

    oof_series = pd.Series(oof, index=features.index, name="p_meta")
    fitted_series = pd.Series(fitted, index=features.index, name="p_meta")

    info: dict[str, object] = {
        "method": method,
        "folds": len(used_folds) if used_folds else 1,
        "params": params,
        "n_samples": int(n_samples),
    }
    return StackerResult(oof=oof_series, fitted=fitted_series, model=final_model, info=info)


def stack_logit(p_cols: Iterable[str], df: pd.DataFrame, seed: int = 14) -> pd.Series:
    """Legacy helper retained for backward compatibility in tests."""

    if not p_cols:
        raise ValueError("p_cols must be non-empty for stacking")

    arr = df.loc[:, list(p_cols)].to_numpy(dtype=float)
    arr = np.clip(arr, 1e-6, 1 - 1e-6)
    logits = np.log(arr / (1 - arr))
    mean_logit = logits.mean(axis=1)
    blended = 1 / (1 + np.exp(-mean_logit))
    return pd.Series(blended, index=df.index, name="p_meta").clip(0.0, 1.0)


__all__ = ["StackerResult", "train_stacker", "stack_logit"]
