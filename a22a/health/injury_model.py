"""Phase 11 — injury availability and in-game exit risk modelling."""

from __future__ import annotations

import math
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression, LogisticRegression

CONFIG_PATH = pathlib.Path("configs/defaults.yaml")
ARTIFACT_DIR = pathlib.Path("artifacts/health")


@dataclass(frozen=True)
class HealthConfig:
    """Configuration parameters for the health module."""

    half_life_weeks: float = 8.0
    min_events: int = 50
    use_frailty: bool = True
    seed: int = 0

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "HealthConfig":
        raw = dict(config or {})
        seed = int(raw.get("seed", 0))
        health = raw.get("health", {}) or {}
        return cls(
            half_life_weeks=float(health.get("recency_half_life_weeks", 8)),
            min_events=int(health.get("min_events", 50)),
            use_frailty=bool(health.get("frailty", True)),
            seed=seed,
        )


def _load_config(path: pathlib.Path = CONFIG_PATH) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _staged_root(cfg: Mapping[str, Any]) -> pathlib.Path:
    paths = cfg.get("paths", {}) if isinstance(cfg, Mapping) else {}
    return pathlib.Path(paths.get("staged", "./data/staged"))


# ---------------------------------------------------------------------------
# Sample data scaffolding (used when staged data absent)
# ---------------------------------------------------------------------------


def _sample_injury_reports() -> pd.DataFrame:
    """Create a deterministic mini injury log spanning multiple weeks."""

    rows: list[dict[str, Any]] = [
        # Week 1 entries
        {
            "season": 2023,
            "week": 1,
            "team_id": "BUF",
            "player_id": "BUF_QB1",
            "age": 28,
            "position": "QB",
            "practice_status": "full",
            "game_status": "active",
            "historical_snaps": 1100,
            "snaps_played": 64,
            "days_rest": 7,
            "surface": "turf",
            "weather_temp_f": 35,
            "injury_start_week": 1,
            "injury_id": "shoulder-2023",
            "exit_events": 0,
            "exit_exposure": 64,
        },
        {
            "season": 2023,
            "week": 1,
            "team_id": "BUF",
            "player_id": "BUF_WR1",
            "age": 30,
            "position": "WR",
            "practice_status": "limited",
            "game_status": "active",
            "historical_snaps": 780,
            "snaps_played": 52,
            "days_rest": 7,
            "surface": "turf",
            "weather_temp_f": 35,
            "injury_start_week": 1,
            "injury_id": "knee-2023",
            "exit_events": 0,
            "exit_exposure": 52,
        },
        {
            "season": 2023,
            "week": 1,
            "team_id": "BUF",
            "player_id": "BUF_RB1",
            "age": 25,
            "position": "RB",
            "practice_status": "limited",
            "game_status": "active",
            "historical_snaps": 610,
            "snaps_played": 46,
            "days_rest": 7,
            "surface": "turf",
            "weather_temp_f": 35,
            "injury_start_week": 1,
            "injury_id": "ankle-2023",
            "exit_events": 0,
            "exit_exposure": 46,
        },
        {
            "season": 2023,
            "week": 1,
            "team_id": "KC",
            "player_id": "KC_QB1",
            "age": 29,
            "position": "QB",
            "practice_status": "limited",
            "game_status": "active",
            "historical_snaps": 1180,
            "snaps_played": 66,
            "days_rest": 7,
            "surface": "grass",
            "weather_temp_f": 37,
            "injury_start_week": 1,
            "injury_id": "ankle-2023",
            "exit_events": 0,
            "exit_exposure": 66,
        },
        {
            "season": 2023,
            "week": 1,
            "team_id": "KC",
            "player_id": "KC_WR1",
            "age": 31,
            "position": "WR",
            "practice_status": "dnp",
            "game_status": "out",
            "historical_snaps": 800,
            "snaps_played": 0,
            "days_rest": 7,
            "surface": "grass",
            "weather_temp_f": 37,
            "injury_start_week": 1,
            "injury_id": "hamstring-2023",
            "exit_events": 0,
            "exit_exposure": 0,
        },
        {
            "season": 2023,
            "week": 1,
            "team_id": "KC",
            "player_id": "KC_TE1",
            "age": 33,
            "position": "TE",
            "practice_status": "limited",
            "game_status": "active",
            "historical_snaps": 720,
            "snaps_played": 49,
            "days_rest": 7,
            "surface": "grass",
            "weather_temp_f": 37,
            "injury_start_week": 1,
            "injury_id": "knee-2023",
            "exit_events": 0,
            "exit_exposure": 49,
        },
        {
            "season": 2023,
            "week": 1,
            "team_id": "KC",
            "player_id": "KC_RB1",
            "age": 27,
            "position": "RB",
            "practice_status": "full",
            "game_status": "active",
            "historical_snaps": 590,
            "snaps_played": 43,
            "days_rest": 7,
            "surface": "grass",
            "weather_temp_f": 37,
            "injury_start_week": 1,
            "injury_id": "shoulder-2023",
            "exit_events": 0,
            "exit_exposure": 43,
        },
        # Week 2
        {
            "season": 2023,
            "week": 2,
            "team_id": "BUF",
            "player_id": "BUF_QB1",
            "age": 28,
            "position": "QB",
            "practice_status": "limited",
            "game_status": "active",
            "historical_snaps": 1165,
            "snaps_played": 63,
            "days_rest": 4,
            "surface": "turf",
            "weather_temp_f": 38,
            "injury_start_week": 1,
            "injury_id": "shoulder-2023",
            "exit_events": 0,
            "exit_exposure": 63,
        },
        {
            "season": 2023,
            "week": 2,
            "team_id": "BUF",
            "player_id": "BUF_WR1",
            "age": 30,
            "position": "WR",
            "practice_status": "limited",
            "game_status": "questionable",
            "historical_snaps": 832,
            "snaps_played": 47,
            "days_rest": 4,
            "surface": "turf",
            "weather_temp_f": 38,
            "injury_start_week": 1,
            "injury_id": "knee-2023",
            "exit_events": 1,
            "exit_exposure": 47,
        },
        {
            "season": 2023,
            "week": 2,
            "team_id": "BUF",
            "player_id": "BUF_RB1",
            "age": 25,
            "position": "RB",
            "practice_status": "full",
            "game_status": "active",
            "historical_snaps": 668,
            "snaps_played": 44,
            "days_rest": 7,
            "surface": "turf",
            "weather_temp_f": 38,
            "injury_start_week": 1,
            "injury_id": "ankle-2023",
            "exit_events": 0,
            "exit_exposure": 44,
        },
        {
            "season": 2023,
            "week": 2,
            "team_id": "KC",
            "player_id": "KC_QB1",
            "age": 29,
            "position": "QB",
            "practice_status": "limited",
            "game_status": "active",
            "historical_snaps": 1248,
            "snaps_played": 64,
            "days_rest": 4,
            "surface": "grass",
            "weather_temp_f": 41,
            "injury_start_week": 1,
            "injury_id": "ankle-2023",
            "exit_events": 0,
            "exit_exposure": 64,
        },
        {
            "season": 2023,
            "week": 2,
            "team_id": "KC",
            "player_id": "KC_WR1",
            "age": 31,
            "position": "WR",
            "practice_status": "dnp",
            "game_status": "out",
            "historical_snaps": 800,
            "snaps_played": 0,
            "days_rest": 4,
            "surface": "grass",
            "weather_temp_f": 41,
            "injury_start_week": 1,
            "injury_id": "hamstring-2023",
            "exit_events": 0,
            "exit_exposure": 0,
        },
        {
            "season": 2023,
            "week": 2,
            "team_id": "KC",
            "player_id": "KC_TE1",
            "age": 33,
            "position": "TE",
            "practice_status": "limited",
            "game_status": "active",
            "historical_snaps": 768,
            "snaps_played": 45,
            "days_rest": 4,
            "surface": "grass",
            "weather_temp_f": 41,
            "injury_start_week": 1,
            "injury_id": "knee-2023",
            "exit_events": 1,
            "exit_exposure": 45,
        },
        {
            "season": 2023,
            "week": 2,
            "team_id": "KC",
            "player_id": "KC_RB1",
            "age": 27,
            "position": "RB",
            "practice_status": "limited",
            "game_status": "active",
            "historical_snaps": 640,
            "snaps_played": 41,
            "days_rest": 7,
            "surface": "grass",
            "weather_temp_f": 41,
            "injury_start_week": 1,
            "injury_id": "shoulder-2023",
            "exit_events": 0,
            "exit_exposure": 41,
        },
        # Week 3 (holdout for hazard)
        {
            "season": 2023,
            "week": 3,
            "team_id": "BUF",
            "player_id": "BUF_QB1",
            "age": 28,
            "position": "QB",
            "practice_status": "full",
            "game_status": "active",
            "historical_snaps": 1230,
            "snaps_played": 68,
            "days_rest": 7,
            "surface": "turf",
            "weather_temp_f": 45,
            "injury_start_week": 1,
            "injury_id": "shoulder-2023",
            "exit_events": 0,
            "exit_exposure": 68,
        },
        {
            "season": 2023,
            "week": 3,
            "team_id": "BUF",
            "player_id": "BUF_WR1",
            "age": 30,
            "position": "WR",
            "practice_status": "limited",
            "game_status": "questionable",
            "historical_snaps": 884,
            "snaps_played": 44,
            "days_rest": 4,
            "surface": "turf",
            "weather_temp_f": 45,
            "injury_start_week": 2,
            "injury_id": "knee-2023-recur",
            "exit_events": 1,
            "exit_exposure": 44,
        },
        {
            "season": 2023,
            "week": 3,
            "team_id": "BUF",
            "player_id": "BUF_RB1",
            "age": 25,
            "position": "RB",
            "practice_status": "full",
            "game_status": "active",
            "historical_snaps": 714,
            "snaps_played": 45,
            "days_rest": 7,
            "surface": "turf",
            "weather_temp_f": 45,
            "injury_start_week": 1,
            "injury_id": "ankle-2023",
            "exit_events": 0,
            "exit_exposure": 45,
        },
        {
            "season": 2023,
            "week": 3,
            "team_id": "KC",
            "player_id": "KC_QB1",
            "age": 29,
            "position": "QB",
            "practice_status": "limited",
            "game_status": "active",
            "historical_snaps": 1312,
            "snaps_played": 63,
            "days_rest": 4,
            "surface": "grass",
            "weather_temp_f": 48,
            "injury_start_week": 1,
            "injury_id": "ankle-2023",
            "exit_events": 0,
            "exit_exposure": 63,
        },
        {
            "season": 2023,
            "week": 3,
            "team_id": "KC",
            "player_id": "KC_WR1",
            "age": 31,
            "position": "WR",
            "practice_status": "limited",
            "game_status": "active",
            "historical_snaps": 854,
            "snaps_played": 42,
            "days_rest": 4,
            "surface": "grass",
            "weather_temp_f": 48,
            "injury_start_week": 1,
            "injury_id": "hamstring-2023",
            "exit_events": 1,
            "exit_exposure": 42,
        },
        {
            "season": 2023,
            "week": 3,
            "team_id": "KC",
            "player_id": "KC_TE1",
            "age": 33,
            "position": "TE",
            "practice_status": "limited",
            "game_status": "active",
            "historical_snaps": 816,
            "snaps_played": 44,
            "days_rest": 7,
            "surface": "grass",
            "weather_temp_f": 48,
            "injury_start_week": 1,
            "injury_id": "knee-2023",
            "exit_events": 0,
            "exit_exposure": 44,
        },
        {
            "season": 2023,
            "week": 3,
            "team_id": "KC",
            "player_id": "KC_RB1",
            "age": 27,
            "position": "RB",
            "practice_status": "full",
            "game_status": "active",
            "historical_snaps": 686,
            "snaps_played": 40,
            "days_rest": 7,
            "surface": "grass",
            "weather_temp_f": 48,
            "injury_start_week": 1,
            "injury_id": "shoulder-2023",
            "exit_events": 0,
            "exit_exposure": 40,
        },
    ]
    return pd.DataFrame(rows)


def _list_parquet(root: pathlib.Path, folder: str) -> list[pathlib.Path]:
    target = root / folder
    if not target.exists():
        return []
    return sorted(target.rglob("*.parquet"))


def _load_injury_reports(cfg: Mapping[str, Any]) -> pd.DataFrame:
    staged_root = _staged_root(cfg)
    candidates = _list_parquet(staged_root, "injuries")
    frames: list[pd.DataFrame] = []
    for path in candidates:
        try:
            frames.append(pd.read_parquet(path))
        except Exception:
            continue
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        if not combined.empty:
            return combined
    return _sample_injury_reports()


# ---------------------------------------------------------------------------
# Feature engineering and modelling utilities
# ---------------------------------------------------------------------------


def _recency_weight(week: int, max_week: int, half_life: float) -> float:
    decay = (max_week - week) / max(half_life, 1e-6)
    return float(0.5 ** decay)


def _compute_team_frailty(df: pd.DataFrame) -> pd.Series:
    ordered = df.sort_values(["team_id", "season", "week"])  # copy not needed
    def _expanding_mean(series: pd.Series) -> pd.Series:
        return series.shift(1).expanding().mean()

    baseline = ordered.groupby("team_id")["is_active"].apply(_expanding_mean)
    if isinstance(baseline.index, pd.MultiIndex):
        baseline.index = baseline.index.get_level_values(-1)
    baseline = baseline.reindex(df.index)
    global_mean = df["is_active"].mean()
    baseline = baseline.fillna(global_mean if not math.isnan(global_mean) else 0.5)
    odds = np.clip(baseline, 1e-3, 1 - 1e-3) / (1 - np.clip(baseline, 1e-3, 1 - 1e-3))
    return np.log(odds)


def _prepare_dataset(raw: pd.DataFrame, config: HealthConfig) -> pd.DataFrame:
    frame = raw.copy()
    if frame.empty:
        raise ValueError("injury report table is empty")

    frame["practice_status"] = frame["practice_status"].str.lower()
    frame["game_status"] = frame["game_status"].str.lower()
    frame["position"] = frame["position"].str.upper()
    frame["surface"] = frame["surface"].str.lower()

    active_states = {"active", "questionable", "probable"}
    frame["is_active"] = frame["game_status"].isin(active_states).astype(int)

    practice_map = {"full": 0.0, "limited": 1.0, "dnp": 2.0}
    frame["practice_intensity"] = frame["practice_status"].map(practice_map).fillna(1.0)

    frame["days_rest"] = frame["days_rest"].fillna(7).astype(float)
    frame["short_week"] = (frame["days_rest"] < 6).astype(int)

    frame["historical_snaps"] = frame["historical_snaps"].fillna(frame.get("snaps_played", 0)).astype(float)
    frame["snaps_played"] = frame.get("snaps_played", 0).fillna(0).astype(float)
    frame["exit_exposure"] = frame.get("exit_exposure", frame["snaps_played"]).fillna(0).astype(float)
    frame["exit_events"] = frame.get("exit_events", 0).fillna(0).astype(float)

    frame["injury_start_week"] = frame.get("injury_start_week", frame["week"]).fillna(frame["week"]).astype(int)
    frame["time_since_injury_weeks"] = (
        frame["week"].astype(int) - frame["injury_start_week"].astype(int)
    ).clip(lower=0)

    if "injury_id" in frame.columns:
        frame["recurrence_flag"] = frame.groupby("player_id")["injury_id"].transform(
            lambda s: s.duplicated().astype(int)
        )
    else:
        frame["recurrence_flag"] = 0

    max_week = frame["week"].max()
    frame["recency_weight"] = frame["week"].apply(
        lambda w: _recency_weight(int(w), int(max_week), config.half_life_weeks)
    )

    if config.use_frailty:
        frame["team_frailty"] = _compute_team_frailty(frame)
    else:
        frame["team_frailty"] = 0.0

    frame["weather_temp_f"] = frame.get("weather_temp_f", 55).fillna(55).astype(float)
    return frame


def _availability_feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "age",
        "practice_intensity",
        "short_week",
        "time_since_injury_weeks",
        "recurrence_flag",
        "team_frailty",
        "historical_snaps",
        "days_rest",
    ]
    base = frame[numeric_cols].astype(float)
    pos_dummies = pd.get_dummies(frame["position"], prefix="pos")
    surface_dummies = pd.get_dummies(frame["surface"], prefix="surface")
    return pd.concat([base, pos_dummies, surface_dummies], axis=1)


def _calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 8) -> Dict[str, float]:
    brier = float(np.mean((y_true - y_prob) ** 2))
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    total = len(y_prob)
    for idx in range(bins):
        left, right = edges[idx], edges[idx + 1]
        if idx == bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        if not np.any(mask):
            continue
        avg_true = float(np.mean(y_true[mask]))
        avg_prob = float(np.mean(y_prob[mask]))
        ece += (mask.sum() / total) * abs(avg_true - avg_prob)
    return {"brier": brier, "ece": float(ece)}


def _fit_availability(frame: pd.DataFrame, config: HealthConfig) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    features = _availability_feature_matrix(frame)
    y = frame["is_active"].astype(int).to_numpy()
    weights = frame["recency_weight"].to_numpy()

    model = LogisticRegression(max_iter=500, solver="lbfgs", random_state=config.seed)
    model.fit(features, y, sample_weight=weights)

    probs = model.predict_proba(features)[:, 1]
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    frame = frame.copy()
    frame["avail_prob"] = probs

    metrics = _calibration_metrics(y, probs)

    latest_mask = frame.groupby("player_id")["week"].transform("max") == frame["week"]
    availability = frame.loc[
        latest_mask,
        [
            "player_id",
            "team_id",
            "season",
            "week",
            "avail_prob",
            "age",
            "position",
            "practice_status",
            "short_week",
            "time_since_injury_weeks",
            "recurrence_flag",
            "team_frailty",
            "historical_snaps",
            "days_rest",
        ],
    ].reset_index(drop=True)
    availability.sort_values(["team_id", "position", "player_id"], inplace=True)
    return availability, frame, metrics


def _hazard_feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "age",
        "short_week",
        "time_since_injury_weeks",
        "recurrence_flag",
        "team_frailty",
        "historical_snaps",
        "days_rest",
    ]
    base = frame[numeric_cols].astype(float)
    pos_dummies = pd.get_dummies(frame["position"], prefix="pos")
    surface_dummies = pd.get_dummies(frame["surface"], prefix="surface")
    return pd.concat([base, pos_dummies, surface_dummies], axis=1)


def _fit_exit_hazard(frame: pd.DataFrame, config: HealthConfig) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["exit_exposure"] = prepared["exit_exposure"].clip(lower=1.0)
    prepared["exit_events"] = prepared["exit_events"].clip(lower=0.0)

    max_week = int(prepared["week"].max())
    train_mask = prepared["week"] < max_week
    if train_mask.sum() < 3:
        # fall back to using all but the earliest record
        ordered = prepared.sort_values(["season", "week"])
        train_mask = ordered.index != ordered.index.min()
        prepared = ordered

    features = _hazard_feature_matrix(prepared)
    log_rate = np.log((prepared["exit_events"] + 0.25) / (prepared["exit_exposure"] + 1.0))

    model = LinearRegression()
    model.fit(features[train_mask], log_rate[train_mask], sample_weight=prepared.loc[train_mask, "recency_weight"])

    holdout_mask = prepared["week"] == max_week
    holdout_features = features[holdout_mask]
    if holdout_features.empty:
        holdout_features = features.iloc[[-1]]
        holdout_mask = features.index == holdout_features.index[0]

    predicted_log_rate = model.predict(holdout_features)
    hazard_rate = np.exp(predicted_log_rate)
    hazard_rate = np.clip(hazard_rate, 0.0, None)

    hazards = prepared.loc[holdout_mask, [
        "player_id",
        "team_id",
        "season",
        "week",
        "age",
        "position",
        "short_week",
        "time_since_injury_weeks",
        "recurrence_flag",
        "team_frailty",
        "exit_events",
        "exit_exposure",
    ]].copy()
    hazards["exit_hazard_rate"] = hazard_rate
    hazards["exit_hazard_pct"] = hazards["exit_hazard_rate"] * 100.0
    hazards["hazard_rate"] = hazards["exit_hazard_rate"]
    hazards["hazard_pct"] = hazards["exit_hazard_pct"]

    _sanity_check_hazards(hazards)
    hazards.sort_values(["team_id", "position", "player_id"], inplace=True)
    return hazards.reset_index(drop=True)


def _sanity_check_hazards(hazards: pd.DataFrame) -> None:
    if hazards.empty:
        return
    by_age = hazards.sort_values("age")
    if len(by_age) >= 2 and by_age.iloc[-1]["hazard_rate"] < by_age.iloc[0]["hazard_rate"]:
        raise ValueError("older players should not have lower exit hazard than youngest")
    short_mask = hazards["short_week"] == 1
    non_short_mask = hazards["short_week"] == 0
    if short_mask.any() and non_short_mask.any():
        if hazards.loc[short_mask, "hazard_rate"].mean() <= hazards.loc[non_short_mask, "hazard_rate"].mean():
            raise ValueError("short-week hazard should exceed standard-week hazard")


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _write_artifact(df: pd.DataFrame, prefix: str, stamp: str) -> pathlib.Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACT_DIR / f"{prefix}_{stamp}.parquet"
    try:
        df.to_parquet(out_path, index=False)
        return out_path
    except Exception:
        csv_path = out_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    start = time.time()
    config_map = _load_config()
    health_cfg = HealthConfig.from_config(config_map)

    raw_reports = _load_injury_reports(config_map)
    prepared = _prepare_dataset(raw_reports, health_cfg)

    availability, enriched, metrics = _fit_availability(prepared, health_cfg)
    hazards = _fit_exit_hazard(enriched, health_cfg)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    availability_path = _write_artifact(availability, "availability", stamp)
    hazards_path = _write_artifact(hazards, "exit_hazards", stamp)

    duration = time.time() - start
    print(
        "[injuries] wrote %s, %s (brier=%.4f, ece=%.4f, frailty=%s) in %.2fs"
        % (
            availability_path.name,
            hazards_path.name,
            metrics["brier"],
            metrics["ece"],
            health_cfg.use_frailty,
            duration,
        )
    )


if __name__ == "__main__":
    main()
