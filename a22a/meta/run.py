"""Entry-point orchestrating blending, calibration, and conformal control."""

from __future__ import annotations

import json
import math
import pathlib
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from a22a.meta.blend import train_stacker
from a22a.meta.calibrate import calibrate_probs
from a22a.meta.conformal import split_conformal_binary, split_conformal_quantiles
from a22a.metrics.calibration import brier_score, ece, log_loss, reliability_bins
from a22a.units.uer import UER_AXES


def _load_config(path: str = "configs/defaults.yaml") -> dict[str, Any]:
    cfg_path = pathlib.Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _list_parquet(root: pathlib.Path) -> list[pathlib.Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.parquet") if p.is_file())


def _load_games(staged_dir: pathlib.Path) -> pd.DataFrame | None:
    files = _list_parquet(staged_dir / "games")
    if not files:
        return None
    frames = []
    for path in files:
        try:
            frames.append(pd.read_parquet(path))
        except Exception:
            continue
    if not frames:
        return None
    games = pd.concat(frames, ignore_index=True)
    if "kickoff_datetime" in games.columns:
        games["kickoff_datetime"] = pd.to_datetime(games["kickoff_datetime"], errors="coerce")
    return games


def _load_baseline(models_dir: pathlib.Path) -> pd.DataFrame | None:
    path = models_dir / "win_probabilities.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _load_theta(models_dir: pathlib.Path) -> pd.DataFrame | None:
    path = models_dir / "team_strength.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _load_sim_summary(models_dir: pathlib.Path) -> pd.DataFrame | None:
    summary_path = models_dir / "sim" / "summary.json"
    if not summary_path.exists():
        return None
    data = json.loads(summary_path.read_text())
    if not data:
        return None
    df = pd.DataFrame(data)
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _load_sim_draws(models_dir: pathlib.Path) -> dict[str, Any]:
    sim_dir = models_dir / "sim"
    if not sim_dir.exists():
        return {}
    draws: dict[str, Any] = {"per_game": {}}
    margin_samples: list[np.ndarray] = []
    total_samples: list[np.ndarray] = []
    for path in sim_dir.glob("*_samples.parquet"):
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        game_id = path.stem.replace("_samples", "")
        margin = df.get("margin")
        total = df.get("total")
        game_draws: dict[str, np.ndarray] = {}
        if margin is not None:
            arr = margin.to_numpy(dtype=float)
            if arr.size:
                margin_samples.append(arr)
                game_draws["margin"] = arr
        if total is not None:
            arr = total.to_numpy(dtype=float)
            if arr.size:
                total_samples.append(arr)
                game_draws["total"] = arr
        if game_draws:
            draws["per_game"][game_id] = game_draws
    if margin_samples:
        draws["margin"] = np.concatenate(margin_samples)
    if total_samples:
        draws["total"] = np.concatenate(total_samples)
    return draws


def _latest_uer_table(artifact_dir: pathlib.Path = pathlib.Path("artifacts/uer")) -> pd.DataFrame | None:
    if not artifact_dir.exists():
        return None
    files = sorted(artifact_dir.glob("uer_week_*.parquet"))
    if not files:
        return None
    try:
        return pd.read_parquet(files[-1])
    except Exception:
        return None


def _team_uer_features(table: pd.DataFrame | None) -> pd.DataFrame | None:
    if table is None or table.empty or "unit_id" not in table.columns:
        return None
    records: dict[str, dict[str, float | str]] = {}
    unit_ids = table["unit_id"].astype(str)
    for axis in UER_AXES:
        suffix = f"_{axis}"
        mean_col = f"{axis}_mean"
        if mean_col not in table.columns:
            continue
        mask = unit_ids.str.endswith(suffix)
        if not mask.any():
            continue
        subset = table.loc[mask, ["unit_id", mean_col]]
        for _, row in subset.iterrows():
            team = row["unit_id"][: -len(suffix)]
            rec = records.setdefault(team, {"team_id": team})
            rec[mean_col] = float(row[mean_col])
    if not records:
        return None
    return pd.DataFrame(records.values())


def _compute_rest_features(games: pd.DataFrame) -> tuple[pd.Series | None, pd.Series | None]:
    if "kickoff_datetime" not in games.columns:
        return None, None
    schedule_rows: list[dict[str, Any]] = []
    for _, row in games.iterrows():
        kickoff = row.get("kickoff_datetime")
        if pd.isna(kickoff):
            continue
        schedule_rows.append({"team": row.get("home_team"), "game_id": row.get("game_id"), "kickoff": kickoff, "is_home": 1})
        schedule_rows.append({"team": row.get("away_team"), "game_id": row.get("game_id"), "kickoff": kickoff, "is_home": 0})
    schedule = pd.DataFrame(schedule_rows)
    if schedule.empty:
        return None, None
    schedule = schedule.sort_values(["team", "kickoff"])
    schedule["prev_kickoff"] = schedule.groupby("team")["kickoff"].shift(1)
    schedule["rest_hours"] = (schedule["kickoff"] - schedule["prev_kickoff"]).dt.total_seconds() / 3600.0
    schedule["rest_hours"] = schedule["rest_hours"].fillna(7 * 24.0)
    home_rest = schedule.loc[schedule["is_home"] == 1, ["game_id", "rest_hours"]].set_index("game_id")["rest_hours"].astype(float)
    away_rest = schedule.loc[schedule["is_home"] == 0, ["game_id", "rest_hours"]].set_index("game_id")["rest_hours"].astype(float)
    return home_rest, away_rest


def _augment_sim_features(df: pd.DataFrame, draws: dict[str, Any]) -> pd.DataFrame:
    per_game = draws.get("per_game", {}) if isinstance(draws, dict) else {}
    if not per_game:
        return df
    df = df.copy()
    for game_id, game_draws in per_game.items():
        idx = df.index[df["game_id"] == game_id]
        if idx.empty:
            continue
        margin = game_draws.get("margin")
        total = game_draws.get("total")
        if margin is not None and margin.size > 1:
            series = pd.Series(margin)
            df.loc[idx, "sim_margin_skew"] = float(series.skew())
            df.loc[idx, "sim_margin_kurt"] = float(series.kurt())
        if total is not None and total.size > 1:
            series = pd.Series(total)
            df.loc[idx, "sim_total_skew"] = float(series.skew())
            df.loc[idx, "sim_total_kurt"] = float(series.kurt())
    return df


def _sigmoid(x: pd.Series | np.ndarray, scale: float = 12.0) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-arr / max(scale, 1e-6)))


def _synthetic_dataset(seed: int, n: int = 128) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    teams = [f"T{idx:02d}" for idx in range(12)]
    home = rng.choice(teams, size=n)
    away = rng.choice(teams, size=n)
    kickoff = pd.Timestamp("2023-09-01") + pd.to_timedelta(rng.integers(0, 60 * 24, size=n), unit="h")
    theta_home = rng.normal(0, 10, size=n)
    theta_away = rng.normal(0, 10, size=n)
    p_base = np.clip(0.5 + 0.18 * rng.standard_normal(size=n), 1e-3, 1 - 1e-3)
    p_sim = np.clip(p_base + 0.05 * rng.standard_normal(size=n), 1e-3, 1 - 1e-3)
    df = pd.DataFrame(
        {
            "game_id": [f"SYN_{i:04d}" for i in range(n)],
            "season": 2023,
            "week": rng.integers(1, 18, size=n),
            "home_team": home,
            "away_team": away,
            "kickoff_datetime": kickoff,
            "theta_home_mean": theta_home,
            "theta_away_mean": theta_away,
            "p_base": p_base,
            "p_base_away": 1.0 - p_base,
            "p_sim": p_sim,
            "sim_margin_mean": rng.normal(3.0, 7.0, size=n),
            "sim_margin_std": rng.uniform(4.0, 10.0, size=n),
            "sim_total_mean": rng.normal(45.0, 6.0, size=n),
            "sim_total_std": rng.uniform(6.0, 9.0, size=n),
            "home_rest_hours": rng.uniform(120.0, 200.0, size=n),
            "away_rest_hours": rng.uniform(120.0, 200.0, size=n),
        }
    )
    df["p_theta"] = _sigmoid(df["theta_home_mean"] - df["theta_away_mean"])
    df["rest_diff_hours"] = df["home_rest_hours"] - df["away_rest_hours"]
    df["home_short_rest"] = (df["home_rest_hours"] < 96).astype(int)
    df["away_short_rest"] = (df["away_rest_hours"] < 96).astype(int)
    for axis in UER_AXES:
        df[f"home_{axis}_mean"] = rng.normal(0, 0.2, size=n)
        df[f"away_{axis}_mean"] = rng.normal(0, 0.2, size=n)
        df[f"uer_diff_{axis}_mean"] = df[f"home_{axis}_mean"] - df[f"away_{axis}_mean"]
    logits = np.log(df["p_base"] / (1 - df["p_base"])) + 0.4 * (df["theta_home_mean"] - df["theta_away_mean"]) / 10
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logits))).astype(int)
    df["y"] = y
    df["home_score"] = 20 + rng.integers(-10, 20, size=n)
    df["away_score"] = df["home_score"] - rng.integers(-14, 14, size=n)
    df["margin"] = df["home_score"] - df["away_score"]
    df["total_points"] = df["home_score"] + df["away_score"]
    meta = {"source": "synthetic", "rows": int(n)}
    return df, df["y"], meta

def _build_meta_dataset(
    games: pd.DataFrame | None,
    baseline: pd.DataFrame | None,
    theta: pd.DataFrame | None,
    sim_summary: pd.DataFrame | None,
    uer_table: pd.DataFrame | None,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    if games is None or games.empty or baseline is None or baseline.empty:
        return _synthetic_dataset(seed)

    df = games[
        [
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            "kickoff_datetime",
            "home_score",
            "away_score",
        ]
    ].copy()
    df["y"] = (df["home_score"].astype(float) > df["away_score"].astype(float)).astype(int)
    df["margin"] = df["home_score"].astype(float) - df["away_score"].astype(float)
    df["total_points"] = df["home_score"].astype(float) + df["away_score"].astype(float)

    home_probs = baseline.merge(
        df[["game_id", "home_team"]].rename(columns={"home_team": "team_id"}),
        on=["game_id", "team_id"],
        how="inner",
    )
    away_probs = baseline.merge(
        df[["game_id", "away_team"]].rename(columns={"away_team": "team_id"}),
        on=["game_id", "team_id"],
        how="inner",
    )
    df = df.merge(
        home_probs[["game_id", "prob"]].rename(columns={"prob": "p_base"}),
        on="game_id",
        how="left",
    )
    df = df.merge(
        away_probs[["game_id", "prob"]].rename(columns={"prob": "p_base_away"}),
        on="game_id",
        how="left",
    )
    df["p_base"] = df["p_base"].fillna(1.0 - df["p_base_away"].fillna(0.5)).clip(1e-6, 1 - 1e-6)
    df["p_base_away"] = df["p_base_away"].fillna(1.0 - df["p_base"]).clip(1e-6, 1 - 1e-6)
    df["p_base_diff"] = df["p_base"] - df["p_base_away"]

    if theta is not None and not theta.empty:
        theta_cols = theta[["season", "week", "team_id", "theta_mean"]].copy()
        theta_cols = theta_cols.sort_values(["team_id", "season", "week"])
        theta_cols = theta_cols.drop_duplicates(subset=["season", "week", "team_id"], keep="last")
        home_theta = theta_cols.rename(columns={"team_id": "home_team", "theta_mean": "theta_home_mean"})
        away_theta = theta_cols.rename(columns={"team_id": "away_team", "theta_mean": "theta_away_mean"})
        df = df.merge(home_theta, on=["season", "week", "home_team"], how="left")
        df = df.merge(away_theta, on=["season", "week", "away_team"], how="left")
        df["theta_home_mean"] = df["theta_home_mean"].fillna(df["theta_home_mean"].mean())
        df["theta_away_mean"] = df["theta_away_mean"].fillna(df["theta_away_mean"].mean())
        df["theta_diff"] = df["theta_home_mean"] - df["theta_away_mean"]
        df["p_theta"] = _sigmoid(df["theta_diff"].to_numpy())
    else:
        df["theta_home_mean"] = 0.0
        df["theta_away_mean"] = 0.0
        df["theta_diff"] = 0.0
        df["p_theta"] = 0.5

    if sim_summary is not None and not sim_summary.empty:
        summary_cols = sim_summary[
            [
                "game_id",
                "home_win_prob",
                "margin_mean",
                "margin_std",
                "total_mean",
                "total_std",
                "samples",
            ]
        ].copy()
        df = df.merge(summary_cols, on="game_id", how="left")
        df.rename(
            columns={
                "home_win_prob": "p_sim",
                "margin_mean": "sim_margin_mean",
                "margin_std": "sim_margin_std",
                "total_mean": "sim_total_mean",
                "total_std": "sim_total_std",
            },
            inplace=True,
        )
    if "p_sim" not in df.columns:
        df["p_sim"] = df["p_base"]

    uer_features = _team_uer_features(uer_table)
    if uer_features is not None:
        home_uer = uer_features.rename(
            columns={col: f"home_{col}" for col in uer_features.columns if col != "team_id"}
        )
        away_uer = uer_features.rename(
            columns={col: f"away_{col}" for col in uer_features.columns if col != "team_id"}
        )
        df = df.merge(home_uer, left_on="home_team", right_on="team_id", how="left")
        df = df.merge(away_uer, left_on="away_team", right_on="team_id", how="left", suffixes=("", "_away"))
        df.drop(columns=["team_id", "team_id_away"], inplace=True, errors="ignore")
        for axis in UER_AXES:
            h_col = f"home_{axis}_mean"
            a_col = f"away_{axis}_mean"
            if h_col in df.columns and a_col in df.columns:
                df[f"uer_diff_{axis}_mean"] = df[h_col].fillna(0.0) - df[a_col].fillna(0.0)
    else:
        for axis in UER_AXES:
            df[f"uer_diff_{axis}_mean"] = 0.0

    home_rest, away_rest = _compute_rest_features(games)
    if home_rest is not None:
        df["home_rest_hours"] = df["game_id"].map(home_rest)
    else:
        df["home_rest_hours"] = np.nan
    if away_rest is not None:
        df["away_rest_hours"] = df["game_id"].map(away_rest)
    else:
        df["away_rest_hours"] = np.nan
    df["rest_diff_hours"] = df["home_rest_hours"].fillna(0.0) - df["away_rest_hours"].fillna(0.0)
    df["home_short_rest"] = (df["home_rest_hours"].fillna(200.0) < 96.0).astype(int)
    df["away_short_rest"] = (df["away_rest_hours"].fillna(200.0) < 96.0).astype(int)
    df["kickoff_weekday"] = pd.to_datetime(df["kickoff_datetime"], errors="coerce").dt.weekday.fillna(0).astype(int)

    meta = {
        "source": "historical",
        "rows": int(len(df)),
        "games_with_baseline": int(len(home_probs["game_id"].unique())),
    }
    return df, df["y"], meta


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    id_cols = {
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "kickoff_datetime",
        "home_score",
        "away_score",
        "margin",
        "total_points",
        "y",
    }
    feature_cols = [
        col
        for col in df.columns
        if col not in id_cols and pd.api.types.is_numeric_dtype(df[col])
    ]
    features = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    features = features.fillna(features.mean()).fillna(0.0)
    return features, feature_cols


def main() -> None:
    start = time.time()
    cfg = _load_config()
    meta_cfg = cfg.get("meta", {})
    calibrate_cfg = cfg.get("calibrate", {})
    conformal_cfg = cfg.get("conformal", {})
    paths = cfg.get("paths", {})

    staged_dir = pathlib.Path(paths.get("staged", "./data/staged"))
    models_dir = pathlib.Path(paths.get("models", "./data/models"))

    games = _load_games(staged_dir)
    baseline = _load_baseline(models_dir)
    theta = _load_theta(models_dir)
    sim_summary = _load_sim_summary(models_dir)
    sim_draws = _load_sim_draws(models_dir)
    uer_table = _latest_uer_table()

    seed = int(meta_cfg.get("seed", 14))
    df, target, dataset_meta = _build_meta_dataset(
        games,
        baseline,
        theta,
        sim_summary,
        uer_table,
        seed=seed,
    )
    df = _augment_sim_features(df, sim_draws)
    df = df.sort_values(["season", "week", "kickoff_datetime", "game_id"]).reset_index(drop=True)

    features, feature_cols = _prepare_features(df)
    stacker_method = str(meta_cfg.get("stacker", "logit")).lower()
    kfold = int(meta_cfg.get("kfold", 5))
    stack_result = train_stacker(features, target.astype(float), method=stacker_method, kfold=kfold, seed=seed)

    oof = stack_result.oof.dropna()
    y_oof = target.loc[oof.index]
    calib_method = str(calibrate_cfg.get("method", "isotonic"))
    calibration = calibrate_probs(oof, y_oof, method=calib_method)
    calibrated_full = pd.Series(
        calibration.transform(stack_result.fitted.to_numpy()),
        index=df.index,
        name="p_calibrated",
    ).clip(0.0, 1.0)

    bins = int(calibrate_cfg.get("bins", 10))
    calib_target = y_oof.reindex(calibration.calibrated.index)
    ece_val = ece(calibration.calibrated, calib_target, bins=bins)
    brier_val = brier_score(calibration.calibrated, calib_target)
    logloss_val = log_loss(calibration.calibrated, calib_target)

    coverage = float(conformal_cfg.get("coverage", 0.9))
    conformal_info: dict[str, Any] = {"nominal": coverage, "samples": int(len(calibration.calibrated))}
    if len(calibration.calibrated) >= 5:
        conf_binary = split_conformal_binary(calibration.calibrated, calib_target, coverage=coverage)
        residuals = np.abs(calib_target.to_numpy(dtype=float) - calibration.calibrated.to_numpy(dtype=float))
        empirical_cov = float((residuals <= conf_binary["q"]).mean())
        conformal_info.update({"empirical": empirical_cov, **conf_binary})
        if abs(empirical_cov - coverage) > 0.05:
            conformal_info["warning"] = "empirical coverage outside tolerance"
    else:
        conformal_info["empirical"] = None
        conformal_info["message"] = "insufficient calibration samples"

    margin_qs = conformal_cfg.get("margin_quantiles", [0.05, 0.95])
    total_qs = conformal_cfg.get("total_quantiles", [0.05, 0.95])
    margin_draws = sim_draws.get("margin") if isinstance(sim_draws, dict) else None
    total_draws = sim_draws.get("total") if isinstance(sim_draws, dict) else None
    if isinstance(margin_draws, np.ndarray) and margin_draws.size:
        margin_interval = split_conformal_quantiles(margin_draws, q_low=float(margin_qs[0]), q_high=float(margin_qs[1]))
    else:
        margin_interval = (math.nan, math.nan)
    if isinstance(total_draws, np.ndarray) and total_draws.size:
        total_interval = split_conformal_quantiles(total_draws, q_low=float(total_qs[0]), q_high=float(total_qs[1]))
    else:
        total_interval = (math.nan, math.nan)

    outdir = pathlib.Path("artifacts/meta")
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    final_df = df[["game_id"]].copy()
    final_df["p_home"] = calibrated_full
    final_df["p_away"] = (1.0 - final_df["p_home"]).clip(0.0, 1.0)
    final_df["p_calibrated"] = final_df["p_home"]
    final_path = outdir / f"final_probs_{stamp}.parquet"
    final_df.to_parquet(final_path, index=False)

    calib_report = {
        "dataset": {**dataset_meta, "feature_columns": feature_cols},
        "stacker": stack_result.info,
        "calibration_method": calibration.info.get("method"),
        "calibration_info": calibration.info,
        "ece": float(ece_val),
        "brier": float(brier_val),
        "log_loss": float(logloss_val),
        "reliability_bins": reliability_bins(calibration.calibrated, calib_target, bins=bins),
        "conformal": {
            "binary": conformal_info,
            "margin": {
                "low": float(margin_interval[0]) if math.isfinite(margin_interval[0]) else None,
                "high": float(margin_interval[1]) if math.isfinite(margin_interval[1]) else None,
                "quantiles": margin_qs,
                "samples": int(margin_draws.size) if isinstance(margin_draws, np.ndarray) else 0,
            },
            "total": {
                "low": float(total_interval[0]) if math.isfinite(total_interval[0]) else None,
                "high": float(total_interval[1]) if math.isfinite(total_interval[1]) else None,
                "quantiles": total_qs,
                "samples": int(total_draws.size) if isinstance(total_draws, np.ndarray) else 0,
            },
        },
    }
    report_path = outdir / f"calibration_report_{stamp}.json"
    report_path.write_text(json.dumps(calib_report, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x))

    print(
        f"[meta] stacker={stacker_method} folds={stack_result.info.get('folds')} samples={len(df)} features={len(feature_cols)}"
    )
    print(
        f"[meta] calibration method={calibration.info.get('method')} ECE={ece_val:.4f} Brier={brier_val:.4f} LogLoss={logloss_val:.4f}"
    )
    empirical_cov = conformal_info.get("empirical")
    if empirical_cov is not None:
        print(
            f"[meta] conformal coverage nominal={coverage:.2%} empirical={empirical_cov:.2%}"
        )
    else:
        print("[meta] conformal coverage skipped: insufficient calibration samples")
    elapsed = time.time() - start
    print(f"[meta] completed in {elapsed:.2f}s → final_probs: {final_path}")


if __name__ == "__main__":
    main()
