"""LightGBM baseline with purged forward chaining and calibration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

FEATURE_DIR = Path("data") / "features"
MODEL_DIR = Path("data") / "model"
REPORT_DIR = Path("reports")
PREDICTION_PATH = MODEL_DIR / "win_probabilities.parquet"
RELIABILITY_PATH = REPORT_DIR / "reliability.png"


@dataclass
class ForwardChainingFold:
    train_weeks: List[int]
    validation_weeks: List[int]


def forward_chaining_schedule(weeks: Iterable[int], purge_gap: int = 1) -> List[ForwardChainingFold]:
    ordered = sorted(set(int(week) for week in weeks))
    folds: List[ForwardChainingFold] = []
    for idx, validation_week in enumerate(ordered[1:], start=1):
        train_cutoff = validation_week - purge_gap
        train_weeks = [week for week in ordered[:idx] if week <= train_cutoff]
        if not train_weeks:
            continue
        folds.append(ForwardChainingFold(train_weeks=train_weeks, validation_weeks=[validation_week]))
    return folds


def _iter_feature_files() -> List[Path]:
    """Return the partitioned feature parquet paths in deterministic order."""

    if not FEATURE_DIR.exists():
        raise FileNotFoundError("Feature directory missing – run `make features` first")

    # Only include partitioned weekly files to avoid double counting the
    # reference snapshot written alongside them.
    paths = sorted(FEATURE_DIR.glob("season=*/week=*.parquet"))
    if not paths:
        raise FileNotFoundError("No feature parquet files found")
    return paths


def _load_feature_table() -> pl.DataFrame:
    frames = [pl.read_parquet(path) for path in _iter_feature_files()]
    return pl.concat(frames, how="vertical")


def _encode_categoricals(frame: pd.DataFrame, categorical_columns: Iterable[str]) -> pd.DataFrame:
    encoded = frame.copy()
    for column in categorical_columns:
        encoded[column] = encoded[column].astype("category").cat.codes
    return encoded


def _prepare_design_matrix(features: pl.DataFrame) -> tuple[pd.DataFrame, pd.Series, List[str]]:
    df = features.to_pandas()
    categorical_columns = ["team_id", "opponent_id"]
    df = _encode_categoricals(df, categorical_columns)
    drop_columns = {"game_id", "game_datetime", "target"}
    feature_columns = [column for column in df.columns if column not in drop_columns]
    X = df[feature_columns]
    y = df["target"].astype(float)
    return X, y, feature_columns


def _calibrate_probabilities(probs: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, str]:
    unique_labels = np.unique(y_true)
    if unique_labels.shape[0] < 2:
        return probs, "identity"
    if len(y_true) < 10:
        calibrator = LogisticRegression(max_iter=1000)
        calibrator.fit(probs.reshape(-1, 1), y_true)
        calibrated = calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
        return calibrated, "platt"
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(probs, y_true)
    calibrated = calibrator.transform(probs)
    return calibrated, "isotonic"


def _expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    """Compute the ECE using equally spaced bins."""

    if len(probs) == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_indices = np.digitize(probs, bin_edges, right=True)
    total = len(probs)
    ece = 0.0
    for bin_id in range(1, bins + 1):
        mask = bin_indices == bin_id
        if not np.any(mask):
            continue
        bin_prob = probs[mask].mean()
        bin_true = y_true[mask].mean()
        weight = mask.sum() / total
        ece += weight * abs(bin_true - bin_prob)
    return float(ece)


def _plot_reliability(y_true: np.ndarray, probs: np.ndarray) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=5, strategy="uniform")
    plt.figure(figsize=(5, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Baseline")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.xlabel("Predicted win probability")
    plt.ylabel("Empirical win rate")
    plt.title("Win probability reliability")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RELIABILITY_PATH, dpi=200)
    plt.close()


def train_baseline(features: pl.DataFrame) -> Dict[str, object]:
    X, y, feature_columns = _prepare_design_matrix(features)
    weeks = features["week"].to_list()
    folds = forward_chaining_schedule(weeks)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    oof_records: List[pd.DataFrame] = []
    models: List[lgb.LGBMClassifier] = []

    for fold_index, fold in enumerate(folds):
        train_mask = X["week"].isin(fold.train_weeks)
        val_mask = X["week"].isin(fold.validation_weeks)
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=200,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model.fit(X[train_mask], y[train_mask])
        val_probs = model.predict_proba(X[val_mask])[:, 1]
        fold_df = pd.DataFrame(
            {
                "game_id": features.filter(pl.col("week").is_in(fold.validation_weeks))["game_id"].to_list(),
                "team_id": features.filter(pl.col("week").is_in(fold.validation_weeks))["team_id"].to_list(),
                "week": X.loc[val_mask, "week"].to_list(),
                "season": X.loc[val_mask, "season"].to_list(),
                "win_prob_raw": val_probs,
                "target": y[val_mask].to_numpy(),
                "fold": fold_index,
            }
        )
        oof_records.append(fold_df)
        models.append(model)

    if not oof_records:
        raise RuntimeError("Not enough data to perform forward-chaining CV")

    oof_df = pd.concat(oof_records, ignore_index=True)
    calibrated_probs, calibration_method = _calibrate_probabilities(
        oof_df["win_prob_raw"].to_numpy(), oof_df["target"].to_numpy()
    )
    oof_df["win_prob_calibrated"] = calibrated_probs
    _plot_reliability(oof_df["target"].to_numpy(), calibrated_probs)

    y_true = oof_df["target"].to_numpy()
    metrics = {
        "brier_score": float(brier_score_loss(y_true, calibrated_probs)),
        "log_loss": float(log_loss(y_true, np.clip(calibrated_probs, 1e-6, 1 - 1e-6))),
        "ece": _expected_calibration_error(y_true, calibrated_probs),
    }

    final_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=21,
    )
    final_model.fit(X, y)
    final_probs = final_model.predict_proba(X)[:, 1]
    final_calibrated, _ = _calibrate_probabilities(final_probs, y.to_numpy())

    prediction_table = pd.DataFrame(
        {
            "game_id": features["game_id"].to_list(),
            "team_id": features["team_id"].to_list(),
            "season": features["season"].to_list(),
            "week": features["week"].to_list(),
            "win_prob_raw": final_probs,
            "win_prob_calibrated": final_calibrated,
        }
    )
    pl.from_pandas(prediction_table).write_parquet(PREDICTION_PATH)

    summary = {
        "folds": len(oof_records),
        "calibration_method": calibration_method,
        "features": feature_columns,
        "predictions_path": PREDICTION_PATH,
        "reliability_plot": RELIABILITY_PATH,
        **metrics,
    }
    return summary


def main() -> None:
    features = _load_feature_table()
    summary = train_baseline(features)
    print(
        "[train] baseline complete | folds={folds} | calibration={calibration_method}"
        .format(**summary)
    )
    print(
        "[train] metrics | brier={brier_score:.4f} | logloss={log_loss:.4f} | ece={ece:.4f}".format(
            **summary
        )
    )
    print(f"[train] predictions saved to {summary['predictions_path']}")
    print(f"[train] reliability plot saved to {summary['reliability_plot']}")


if __name__ == "__main__":
    main()
