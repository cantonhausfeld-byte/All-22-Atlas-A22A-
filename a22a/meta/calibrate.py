"""Calibration helpers for meta-learner outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class CalibrationResult:
    calibrated: pd.Series
    info: Dict[str, object]
    transform: Callable[[np.ndarray | pd.Series], np.ndarray]


def _identity_transform(values: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.clip(arr, 0.0, 1.0)


def _platt_scaler(
    probs: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, dict[str, object], Callable[[np.ndarray | pd.Series], np.ndarray]]:
    lr = LogisticRegression(max_iter=2000)
    lr.fit(probs.reshape(-1, 1), targets)

    def transform(values: np.ndarray | pd.Series) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(-1, 1)
        return lr.predict_proba(arr)[:, 1]

    calibrated = transform(probs)
    info = {
        "method": "platt",
        "coef": lr.coef_.ravel().tolist(),
        "intercept": lr.intercept_.ravel().tolist(),
    }
    return calibrated, info, transform


def _isotonic_scaler(
    probs: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, dict[str, object], Callable[[np.ndarray | pd.Series], np.ndarray]]:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs, targets)

    def transform(values: np.ndarray | pd.Series) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return iso.predict(arr)

    calibrated = transform(probs)
    return calibrated, {"method": "isotonic"}, transform


def _venn_abers_scaler(
    probs: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, dict[str, object], Callable[[np.ndarray | pd.Series], np.ndarray]]:
    probs = np.asarray(probs, dtype=float)
    targets = np.asarray(targets, dtype=float)

    def _augmented_fit(value: float, assumed_label: int) -> IsotonicRegression:
        iso = IsotonicRegression(out_of_bounds="clip")
        augmented_probs = np.append(probs, value)
        augmented_labels = np.append(targets, assumed_label)
        iso.fit(augmented_probs, augmented_labels)
        return iso

    cache: dict[tuple[float, int], IsotonicRegression] = {}

    def _predict_single(value: float) -> float:
        key0 = (float(value), 0)
        key1 = (float(value), 1)
        if key0 not in cache:
            cache[key0] = _augmented_fit(value, 0)
        if key1 not in cache:
            cache[key1] = _augmented_fit(value, 1)
        p0 = float(cache[key0].predict([value])[0])
        p1 = float(cache[key1].predict([value])[0])
        return 0.5 * (p0 + p1)

    def transform(values: np.ndarray | pd.Series) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        preds = np.vectorize(_predict_single)(arr)
        return np.clip(preds, 0.0, 1.0)

    calibrated = transform(probs)
    info = {"method": "venn_abers", "note": "simplified"}
    return calibrated, info, transform


def calibrate_probs(
    p: pd.Series,
    y: pd.Series,
    method: str = "isotonic",
) -> CalibrationResult:
    """Calibrate binary win probabilities."""

    method = (method or "isotonic").lower()
    probs_raw = p.astype(float).to_numpy()
    targets_raw = y.astype(float).to_numpy()
    mask = np.isfinite(probs_raw) & np.isfinite(targets_raw)
    probs = np.clip(probs_raw[mask], 1e-6, 1 - 1e-6)
    targets = targets_raw[mask]

    if probs.size < 3 or np.unique(targets).size < 2:
        info = {"method": "identity", "reason": "insufficient data"}
        calibrated = np.clip(p.astype(float).to_numpy(), 0.0, 1.0)
        return CalibrationResult(
            calibrated=pd.Series(calibrated, index=p.index, name="p_calibrated"),
            info=info,
            transform=_identity_transform,
        )

    if method == "platt":
        calibrated, info, transform = _platt_scaler(probs, targets)
    elif method == "venn_abers":
        calibrated, info, transform = _venn_abers_scaler(probs, targets)
    else:
        calibrated, info, transform = _isotonic_scaler(probs, targets)

    calibrated_series = pd.Series(calibrated, index=p.index, name="p_calibrated").clip(0.0, 1.0)
    return CalibrationResult(calibrated=calibrated_series, info=info, transform=transform)


__all__ = ["CalibrationResult", "calibrate_probs"]
