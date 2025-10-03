"""Train a gradient boosted baseline with calibration (Phase 4)."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Iterable

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import yaml
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

FEATURE_BLACKLIST = {
    "game_id",
    "team_id",
    "opponent_id",
    "kickoff_datetime",
    "season",
    "week",
    "team_score",
    "opp_score",
    "margin",
    "total_points",
    "win",
}


def _load_config(path: str | Path = "configs/defaults.yaml") -> dict:
    if Path(path).exists():
        return yaml.safe_load(Path(path).read_text())
    return {}


def _list_feature_parts(features_dir: Path) -> list[Path]:
    return sorted(p for p in features_dir.rglob("*.parquet") if "reference" not in p.name)


def _load_feature_frame(features_dir: Path) -> pl.DataFrame:
    files = _list_feature_parts(features_dir)
    if not files:
        raise FileNotFoundError(f"no feature parquet files found in {features_dir}")
    frames = [pl.read_parquet(f) for f in files]
    return pl.concat(frames, how="vertical_relaxed")


def _prepare_matrix(df: pl.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = df.sort(["season", "week", "kickoff_datetime", "team_id"])
    pdf = df.to_pandas()
    feature_cols = [
        col
        for col in pdf.columns
        if col not in FEATURE_BLACKLIST and pd.api.types.is_numeric_dtype(pdf[col])
    ]
    X = pdf[feature_cols].fillna(0.0)
    y = pdf["win"].astype(float)
    meta = pdf[["season", "week"]]
    return X, y, meta


def _forward_folds(df: pd.DataFrame, seasons: Iterable[int]) -> list[tuple[np.ndarray, np.ndarray]]:
    df = df.copy()
    df["season_week"] = list(zip(df["season"], df["week"]))
    ordered = sorted({(int(s), int(w)) for s, w in df["season_week"].unique()})
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for season, week in ordered:
        train_mask = (df["season"] < season) | ((df["season"] == season) & (df["week"] < week))
        val_mask = (df["season"] == season) & (df["week"] == week)
        train_idx = df.index[train_mask].to_numpy()
        val_idx = df.index[val_mask].to_numpy()
        if len(train_idx) < 1 or len(val_idx) == 0:
            continue
        folds.append((train_idx, val_idx))
    return folds


def _train_lightgbm(X: pd.DataFrame, y: np.ndarray, seed: int = 42) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=5,
        random_state=seed,
    )
    model.fit(X, y)
    return model


def _calibrate(probs: np.ndarray, y: np.ndarray) -> tuple[str, object]:
    mask = ~np.isnan(probs)
    probs = probs[mask]
    y = y[mask]
    if len(probs) >= 12 and len(np.unique(probs)) > 3:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(probs, y)
        return "isotonic", iso
    lr = LogisticRegression(max_iter=1000)
    lr.fit(probs.reshape(-1, 1), y)
    return "platt", lr


def _apply_calibrator(method: str, calibrator, preds: np.ndarray) -> np.ndarray:
    if method == "isotonic":
        return calibrator.transform(preds)
    return calibrator.predict_proba(preds.reshape(-1, 1))[:, 1]


def _reliability_table(preds: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    digitized = np.digitize(preds, edges) - 1
    rows = []
    for b in range(bins):
        mask = digitized == b
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin": b,
                "lower": edges[b],
                "upper": edges[b + 1],
                "count": int(mask.sum()),
                "pred_mean": float(preds[mask].mean()),
                "event_rate": float(y[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def train(features_dir: Path, models_dir: Path, reports_dir: Path, seasons: Iterable[int]) -> None:
    features = _load_feature_frame(features_dir)
    X, y, meta = _prepare_matrix(features)
    pdf = meta.copy()
    pdf["season"] = pdf["season"].astype(int)
    pdf["week"] = pdf["week"].astype(int)
    pdf.index = X.index

    folds = _forward_folds(pdf, seasons)
    cv_preds = np.full(len(y), np.nan)
    models: list[lgb.LGBMClassifier] = []
    for train_idx, val_idx in folds:
        model = _train_lightgbm(X.iloc[train_idx], y[train_idx])
        preds = model.predict_proba(X.iloc[val_idx])[:, 1]
        cv_preds[val_idx] = preds
        models.append(model)

    if np.isnan(cv_preds).all():
        model = _train_lightgbm(X, y)
        models = [model]
        cv_preds = model.predict_proba(X)[:, 1]
    else:
        model = _train_lightgbm(X, y)
        models.append(model)

    method, calibrator = _calibrate(cv_preds, y)
    calibrated = _apply_calibrator(method, calibrator, cv_preds)

    metrics = {
        "brier": float(brier_score_loss(y, calibrated)),
        "logloss": float(log_loss(y, np.clip(calibrated, 1e-6, 1 - 1e-6))),
        "roc_auc": float(roc_auc_score(y, calibrated)) if len(np.unique(y)) > 1 else float("nan"),
    }

    final_preds = _apply_calibrator(method, calibrator, model.predict_proba(X)[:, 1])

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "baseline_lgbm.txt"
    calibrator_path = models_dir / "baseline_calibrator.pkl"
    model.booster_.save_model(str(model_path))
    joblib.dump({"method": method, "calibrator": calibrator}, calibrator_path)

    predictions = features.with_columns(pl.Series(name="prob", values=final_preds))
    predictions = predictions.select(
        "game_id", "team_id", "opponent_id", "season", "week", "prob", "win"
    )
    predictions.write_parquet(models_dir / "win_probabilities.parquet")

    reliability = _reliability_table(calibrated, y)
    reliability.to_parquet(reports_dir / "reliability.parquet", index=False)

    metrics_path = models_dir / "baseline_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))


def main() -> None:
    cfg = _load_config()
    paths = cfg.get("paths", {})
    features_dir = Path(paths.get("features", "./data/features"))
    models_dir = Path(paths.get("models", "./data/models"))
    reports_dir = Path(paths.get("reports", "./data/reports"))
    seasons = cfg.get("ingest", {}).get("seasons", [2023])

    start = time.time()
    print("[baseline] training LightGBM baseline")
    train(features_dir, models_dir, reports_dir, seasons)
    print(f"[baseline] completed in {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
