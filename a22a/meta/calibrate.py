"""Calibration helpers for meta-learner outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class CalibrationResult:
    calibrated: pd.Series
    info: Dict[str, object]


def calibrate_probs(
    p: pd.Series,
    y: pd.Series,
    method: str = "isotonic",
) -> Tuple[pd.Series, Dict[str, object]]:
    """Calibrate binary win probabilities.

    Parameters
    ----------
    p:
        Series of uncalibrated probabilities.
    y:
        Series of binary outcomes (0/1).
    method:
        Calibration approach. Supported values: ``"isotonic"``, ``"platt"``,
        ``"venn_abers"``.

    Returns
    -------
    tuple
        Calibrated probabilities and metadata about the procedure.
    """

    method = method.lower()
    probs = p.astype(float).clip(1e-6, 1 - 1e-6)
    targets = y.astype(int)

    if method == "platt":
        lr = LogisticRegression(max_iter=1000)
        lr.fit(probs.to_numpy().reshape(-1, 1), targets.to_numpy())
        calibrated = lr.predict_proba(probs.to_numpy().reshape(-1, 1))[:, 1]
        info = {"method": "platt", "coef": lr.coef_.ravel().tolist(), "intercept": lr.intercept_.tolist()}
    elif method == "venn_abers":
        # Simple Venn-Abers style smoothing: average isotonic fit with original probability.
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(probs.to_numpy(), targets.to_numpy())
        iso_pred = iso.predict(probs.to_numpy())
        calibrated = 0.5 * (iso_pred + probs.to_numpy())
        info = {"method": "venn_abers", "note": "stub isotonic blend"}
    else:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(probs.to_numpy(), targets.to_numpy())
        calibrated = iso.predict(probs.to_numpy())
        info = {"method": "isotonic"}

    calibrated_series = pd.Series(calibrated, index=p.index, name="p_calibrated").clip(0.0, 1.0)
    return calibrated_series, info


__all__ = ["CalibrationResult", "calibrate_probs"]
