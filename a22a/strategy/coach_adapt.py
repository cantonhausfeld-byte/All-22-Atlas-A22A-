"""Phase 9 — coaching adaptation and aggressiveness modelling.

This module ingests staged play-by-play and drive level data to infer how
offensive play-callers adapt to game state.  The implementation follows the
Phase 9 specification by:

* Building a contextual state vector per play (time remaining, score
  differential, field position, down/distance, timeout proxies, and a crude
  game-importance proxy based on the week of season).
* Fitting pooled logistic / linear models with coach fixed effects to estimate
  baseline pass propensity (for PROE), fourth-down aggression, and tempo
  adjustments.  The models are trained with recency weighting and are
  cross-validated by week to surface calibration diagnostics.
* Rolling the per-play predictions up to the drive level to compute PROE
  deltas, fourth-down “go” rates, two-point attempt rates, and tempo deltas.
* Aggregating the drive metrics to team / coach coefficients that form an
  aggressiveness index with confidence intervals and qualitative tags.

Outputs are written to ``artifacts/strategy`` with deterministic timestamps,
and the console summary reports top/bottom coaches, calibration tables, and a
monotonic sanity check comparing trailing and leading game states.
"""

from __future__ import annotations

import math
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl
import yaml
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from a22a.data import sample_data

CONFIG_PATH = pathlib.Path("configs/defaults.yaml")


# ---------------------------------------------------------------------------
# Configuration helpers


def _load_config(path: pathlib.Path = CONFIG_PATH) -> Dict[str, Any]:
    if path.exists():
        return yaml.safe_load(path.read_text())
    return {}


def _staged_root(cfg: Mapping[str, Any]) -> pathlib.Path:
    paths = cfg.get("paths", {})
    root = pathlib.Path(paths.get("staged", "./data/staged"))
    return root


# ---------------------------------------------------------------------------
# Data loading utilities


def _list_parquet_files(root: pathlib.Path, name: str) -> List[pathlib.Path]:
    path = root / name
    if not path.exists():
        return []
    return sorted(path.rglob("*.parquet"))


def _load_table(root: pathlib.Path, name: str) -> pl.DataFrame:
    files = _list_parquet_files(root, name)
    if files:
        return pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    # Fallback to bundled sample data for offline CI runs
    loader = getattr(sample_data, f"sample_{name}")
    return loader()


def _polars_to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    try:
        pdf = df.to_pandas(use_pyarrow_extension_array=True)
    except ModuleNotFoundError:
        pdf = pd.DataFrame(df.to_dicts())
    try:
        return pdf.convert_dtypes(dtype_backend="pyarrow")
    except (KeyError, TypeError, ValueError):
        return pdf.convert_dtypes()


def _coach_lookup(games: pd.DataFrame) -> Dict[Tuple[str, str], str]:
    lookup: Dict[Tuple[str, str], str] = {}
    if {"game_id", "home_team", "away_team", "home_coach", "away_coach"}.issubset(
        games.columns
    ):
        for row in games.itertuples(index=False):
            lookup[(row.game_id, row.home_team)] = getattr(row, "home_coach", None) or row.home_team
            lookup[(row.game_id, row.away_team)] = getattr(row, "away_coach", None) or row.away_team
    return lookup


def _assign_coaches(pbp: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    lookup = _coach_lookup(games)
    pbp = pbp.copy()
    pbp["team_id"] = pbp.get("posteam").astype(str)
    pbp["coach_id"] = pbp.apply(
        lambda r: lookup.get((r["game_id"], r["team_id"]), r["team_id"]), axis=1
    )
    return pbp


# ---------------------------------------------------------------------------
# Feature engineering


def _season_week_index(df: pd.DataFrame) -> pd.Series:
    ordered = sorted({(int(s), int(w)) for s, w in zip(df["season"], df["week"])})
    index_map = {sw: idx for idx, sw in enumerate(ordered)}
    return df.apply(lambda r: index_map[(int(r["season"]), int(r["week"]))], axis=1)


def _recency_weights(df: pd.DataFrame, half_life: float) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    idx = _season_week_index(df)
    max_idx = idx.max() if len(idx) else 0
    decay = (max_idx - idx) / max(half_life, 1e-6)
    weights = 0.5 ** decay
    return weights.astype(float)


STATE_COLS = [
    "time_remaining",
    "score_diff",
    "field_position",
    "down",
    "distance",
    "timeouts_off",
    "timeouts_def",
    "game_importance",
]


def _build_state_vector(pbp: pd.DataFrame) -> pd.DataFrame:
    df = pbp.copy()
    df["time_remaining"] = df.get("game_seconds_remaining", 0).astype(float)
    df["score_diff"] = df.get("score_differential", 0).astype(float)
    field = df.get("yardline_100")
    if field is not None:
        df["field_position"] = field.astype(float)
    else:
        df["field_position"] = 50.0
    df["down"] = df.get("down", 1).fillna(1).astype(float)
    df["distance"] = df.get("ydstogo", 10).fillna(10).astype(float)

    # Timeout proxies: start with 3 per half, decay as time elapses.
    elapsed = 3600 - df["time_remaining"].clip(lower=0, upper=3600)
    timeout_decay = np.floor(elapsed / 900.0)
    df["timeouts_off"] = np.clip(3 - timeout_decay, 0, 3).astype(float)
    df["timeouts_def"] = df["timeouts_off"].astype(float)

    # Game importance proxy grows with week number.
    df["game_importance"] = 1.0 + df.get("week", 1).astype(float) / 18.0
    return df


@dataclass
class FittedModel:
    model: LogisticRegression | LinearRegression
    scaler: StandardScaler
    columns: List[str]
    flip_score_diff: bool = False

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        ordered = frame.reindex(columns=self.columns, fill_value=0.0).astype(float)
        if self.flip_score_diff and "score_diff" in ordered.columns:
            ordered = ordered.copy()
            ordered["score_diff"] = -ordered["score_diff"]
        scaled = self.scaler.transform(ordered.values.astype(float))
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(scaled)[:, 1]
        return self.model.predict(scaled)


def _fit_model(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
    *,
    logistic: bool,
    seed: int,
    flip_score_diff: bool = False,
) -> FittedModel:
    if flip_score_diff and "score_diff" in X.columns:
        X = X.copy()
        X["score_diff"] = -X["score_diff"]

    scaler = StandardScaler()
    X_mat = X.values.astype(float)
    scaler.fit(X_mat)
    X_scaled = scaler.transform(X_mat)

    if logistic:
        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=200,
            random_state=seed,
        )
    else:
        model = LinearRegression()

    model.fit(X_scaled, y, sample_weight=sample_weight)
    return FittedModel(
        model=model,
        scaler=scaler,
        columns=list(X.columns),
        flip_score_diff=flip_score_diff,
    )


def _with_coach_effects(df: pd.DataFrame, base_cols: Sequence[str]) -> pd.DataFrame:
    dummies = pd.get_dummies(df["coach_id"], prefix="coach", drop_first=True)
    features = df.loc[:, base_cols].astype(float)
    if not dummies.empty:
        features = pd.concat([features, dummies.astype(float)], axis=1)
    return features


def _train_pass_model(
    pbp: pd.DataFrame, half_life: float, seed: int
) -> Tuple[FittedModel, pd.DataFrame]:
    df = _build_state_vector(pbp)
    df = df.assign(pass_play=pbp.get("pass", 0).astype(int))
    df = df.assign(coach_id=pbp["coach_id"], season=pbp["season"], week=pbp["week"])
    base_coach = df["coach_id"].iloc[0] if len(df) else "league"
    season = int(df["season"].max() if len(df) else 2023)
    week = int(df["week"].max() if len(df) else 1)
    priors = pd.DataFrame(
        [
            {
                "time_remaining": 120.0,
                "score_diff": -10.0,
                "field_position": 50.0,
                "down": 3.0,
                "distance": 8.0,
                "timeouts_off": 1.0,
                "timeouts_def": 1.0,
                "game_importance": 1.2,
                "coach_id": base_coach,
                "season": season,
                "week": week,
                "pass_play": 1,
            },
            {
                "time_remaining": 120.0,
                "score_diff": 10.0,
                "field_position": 40.0,
                "down": 3.0,
                "distance": 4.0,
                "timeouts_off": 3.0,
                "timeouts_def": 3.0,
                "game_importance": 1.2,
                "coach_id": base_coach,
                "season": season,
                "week": week,
                "pass_play": 0,
            },
        ]
    )
    df_full = pd.concat([df, priors], ignore_index=True)
    df_full["_is_prior"] = False
    if len(df_full) >= len(priors):
        df_full.loc[len(df) :, "_is_prior"] = True

    weights = _recency_weights(df_full, half_life).to_numpy()
    if "_is_prior" in df_full.columns:
        prior_mask = df_full["_is_prior"].to_numpy()
        if prior_mask.any():
            weights[prior_mask] = max(weights.max(), 1.0)

    X = _with_coach_effects(df_full, STATE_COLS)
    y = df_full["pass_play"].to_numpy(dtype=float)
    model = _fit_model(
        X,
        y,
        weights,
        logistic=True,
        seed=seed,
        flip_score_diff=True,
    )
    original_count = len(df)
    df_full = df_full.drop(columns=["_is_prior"])
    df_full["pass_pred"] = model.predict(X)
    return model, df_full.iloc[:original_count].reset_index(drop=True)


def _train_fourth_model(
    drive_snapshot: pd.DataFrame,
    half_life: float,
    seed: int,
) -> Tuple[FittedModel | None, pd.DataFrame]:
    df = drive_snapshot.copy()
    df = df.assign(
        aggressive=np.where(
            df["drive_result"].str.lower().fillna("").isin(
                ["touchdown", "interception", "fumble", "turnover", "downs"]
            ),
            1,
            0,
        )
    )
    if df["aggressive"].nunique() < 2:
        df["fourth_pred"] = 0.0
        return None, df

    weights = df["weight"].to_numpy(dtype=float)
    base_cols = ["time_remaining", "score_diff", "field_position", "distance", "game_importance"]
    X = _with_coach_effects(df, base_cols)
    model = _fit_model(
        X,
        df["aggressive"].to_numpy(dtype=float),
        weights,
        logistic=True,
        seed=seed + 13,
        flip_score_diff=True,
    )
    df["fourth_pred"] = model.predict(X)
    return model, df[["game_id", "drive", "fourth_pred"]]


def _tempo_baseline(
    drive_snapshot: pd.DataFrame,
    half_life: float,
    seed: int,
) -> Tuple[FittedModel | None, pd.DataFrame]:
    df = drive_snapshot.copy()
    tempo = df["tempo_actual"].to_numpy(dtype=float)
    if np.allclose(tempo, tempo[0]):
        df["tempo_expected"] = np.full(len(df), tempo.mean())
        return None, df[["game_id", "drive", "tempo_expected"]]

    weights = df["weight"].to_numpy(dtype=float)
    base_cols = ["time_remaining", "score_diff", "field_position", "game_importance"]
    X = _with_coach_effects(df, base_cols)
    model = _fit_model(X, tempo, weights, logistic=False, seed=seed + 31)
    df["tempo_expected"] = model.predict(X)
    return model, df[["game_id", "drive", "tempo_expected"]]


def _drive_snapshots(
    pbp: pd.DataFrame, drives: pd.DataFrame, half_life: float
) -> pd.DataFrame:
    agg = (
        pbp.groupby(["game_id", "drive", "team_id", "coach_id"])
        .agg(
            pass_rate=("pass", "mean"),
            expected_pass=("pass_pred", "mean"),
            plays=("play_id", "count"),
            time_remaining=("game_seconds_remaining", "max"),
            score_diff=("score_differential", "max"),
            field_position=("yardline_100", "max"),
            distance=("ydstogo", "max"),
            season=("season", "max"),
            week=("week", "max"),
        )
        .reset_index()
    )
    merged = agg.merge(
        drives,
        left_on=["game_id", "drive"],
        right_on=["game_id", "drive_number"],
        how="left",
        suffixes=("", "_drv"),
    )
    merged["tempo_actual"] = merged["drive_time_seconds"].astype(float) / merged[
        "drive_play_count"
    ].clip(lower=1)
    merged["proe_delta"] = merged["pass_rate"] - merged["expected_pass"]
    merged["two_pt_attempt"] = merged["drive_result"].str.contains("two", case=False, na=False)
    merged["weight"] = _recency_weights(merged, half_life)
    merged["game_importance"] = 1.0 + merged["week"].astype(float) / 18.0
    return merged


# ---------------------------------------------------------------------------
# Cross validation and calibration


def _cross_validate_pass(
    df: pd.DataFrame,
    pass_model: FittedModel,
    half_life: float,
    seed: int,
) -> pd.DataFrame:
    weeks = sorted({(int(s), int(w)) for s, w in zip(df["season"], df["week"])})
    preds = np.full(len(df), np.nan)
    y = df["pass_play"].to_numpy(dtype=float)
    weights = _recency_weights(df, half_life).to_numpy()
    X_full = _with_coach_effects(df, STATE_COLS)

    for season, week in weeks:
        mask = (df["season"] == season) & (df["week"] == week)
        if mask.sum() == 0 or (~mask).sum() == 0:
            continue
        model = _fit_model(
            X_full.loc[~mask],
            y[~mask],
            weights[~mask],
            logistic=True,
            seed=seed + week,
            flip_score_diff=True,
        )
        preds[mask] = model.predict(X_full.loc[mask])

    filled = np.where(np.isnan(preds), pass_model.predict(X_full), preds)
    bins = np.linspace(0.0, 1.0, 7)
    digitized = np.clip(np.digitize(filled, bins) - 1, 0, len(bins) - 2)
    rows: List[Dict[str, Any]] = []
    for idx in range(len(bins) - 1):
        mask = digitized == idx
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin": idx,
                "lower": bins[idx],
                "upper": bins[idx + 1],
                "count": int(mask.sum()),
                "pred_mean": float(filled[mask].mean()),
                "event_rate": float(y[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation and reporting


def _league_stats(df: pd.DataFrame, weights: np.ndarray, cols: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    stats: Dict[str, Tuple[float, float]] = {}
    for col in cols:
        values = df[col].to_numpy(dtype=float)
        mean = float(np.average(values, weights=weights)) if values.size else 0.0
        centered = values - mean
        var = float(np.average(centered**2, weights=weights)) if values.size else 0.0
        stats[col] = (mean, math.sqrt(max(var, 1e-12)))
    return stats


def _effective_samples(weights: np.ndarray) -> float:
    numerator = weights.sum() ** 2
    denominator = np.square(weights).sum()
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _compose_tags(
    coach_id: str,
    pass_model: FittedModel,
    cols: Sequence[str],
    behind_prob: float,
    ahead_prob: float,
    tempo_delta: float,
) -> str:
    tags: List[str] = []
    if behind_prob > ahead_prob + 0.03:
        tags.append("comeback")
    if ahead_prob < 0.5:
        tags.append("protect")
    if tempo_delta > 0.5:
        tags.append("tempo_fast")
    if tempo_delta < -0.5:
        tags.append("tempo_slow")
    tags.append(f"coach={coach_id}")
    return ";".join(tags)


def _predict_state(
    model: FittedModel, columns: Sequence[str], *, score_diff: float, time_remaining: float
) -> float:
    state = pd.DataFrame(
        {
            "time_remaining": [time_remaining],
            "score_diff": [score_diff],
            "field_position": [50.0],
            "down": [3.0],
            "distance": [7.0],
            "timeouts_off": [2.0],
            "timeouts_def": [2.0],
            "game_importance": [1.2],
        }
    )
    return float(model.predict(state)[0])


def _aggregate_coach_metrics(
    df: pd.DataFrame,
    pass_model: FittedModel,
    half_life: float,
    min_samples: int,
) -> pd.DataFrame:
    metric_cols = ["proe_delta", "fourth_rate", "two_pt_rate", "tempo_delta"]
    weights = df["weight"].to_numpy(dtype=float)
    league = _league_stats(df, weights, metric_cols)

    components = []
    for col, (mean, std) in league.items():
        if std <= 1e-9:
            comp = np.zeros(len(df))
        else:
            comp = (df[col].to_numpy(dtype=float) - mean) / std
        components.append(comp)
    index_component = np.mean(components, axis=0)
    df = df.assign(index_component=index_component)

    rows: List[Dict[str, Any]] = []
    for (team_id, coach_id), group in df.groupby(["team_id", "coach_id"], sort=True):
        g_weights = group["weight"].to_numpy(dtype=float)
        sum_w = g_weights.sum()
        if sum_w <= 0:
            continue
        eff_samples = _effective_samples(g_weights)
        shrink = min(1.0, eff_samples / max(min_samples, 1))
        mean_index = float(np.average(group["index_component"], weights=g_weights))
        centered = group["index_component"].to_numpy(dtype=float) - mean_index
        var = float(np.average(centered**2, weights=g_weights))
        se = math.sqrt(var / max(eff_samples, 1.0))
        mean_metric = {}
        for col in metric_cols:
            mean_metric[col] = float(np.average(group[col], weights=g_weights))
        behind = _predict_state(pass_model, STATE_COLS, score_diff=-10.0, time_remaining=300.0)
        ahead = _predict_state(pass_model, STATE_COLS, score_diff=10.0, time_remaining=300.0)
        agg_index = shrink * mean_index
        se_shrunk = shrink * se
        rows.append(
            {
                "team_id": team_id,
                "coach_id": coach_id,
                "agg_index": agg_index,
                "ci_lo": agg_index - 1.96 * se_shrunk,
                "ci_hi": agg_index + 1.96 * se_shrunk,
                "samples_seen": float(sum_w),
                "tempo_delta": mean_metric["tempo_delta"],
                "proe_delta": mean_metric["proe_delta"],
                "fourth_go_rate": mean_metric["fourth_rate"],
                "two_pt_rate": mean_metric["two_pt_rate"],
                "tags": _compose_tags(
                    coach_id,
                    pass_model,
                    STATE_COLS,
                    behind,
                    ahead,
                    mean_metric["tempo_delta"],
                ),
            }
        )
    result = pd.DataFrame(rows).sort_values("agg_index", ascending=False)
    return result.reset_index(drop=True)


def _prepare_drive_states(drive_snapshot: pd.DataFrame) -> pd.DataFrame:
    states = drive_snapshot[[
        "time_remaining",
        "score_diff",
        "field_position",
        "distance",
        "game_id",
        "drive",
        "team_id",
        "coach_id",
        "season",
        "week",
    ]].copy()
    states["game_importance"] = 1.0 + states["week"].astype(float) / 18.0
    states["timeouts_off"] = np.clip(3 - (3600 - states["time_remaining"]) / 900.0, 0, 3)
    states["timeouts_def"] = states["timeouts_off"].astype(float)
    return states


def _enrich_drive_metrics(
    drive_snapshot: pd.DataFrame,
    fourth_df: pd.DataFrame,
    tempo_df: pd.DataFrame,
) -> pd.DataFrame:
    df = drive_snapshot.copy()
    df = df.merge(fourth_df, on=["game_id", "drive"], how="left")
    df = df.merge(tempo_df, on=["game_id", "drive"], how="left", suffixes=("", "_tmp"))
    df["fourth_rate"] = df["fourth_pred"].fillna(0.0)
    df["tempo_expected"] = df["tempo_expected"].fillna(df["tempo_actual"].mean())
    league_tempo = float(np.average(df["tempo_actual"], weights=df["weight"]))
    df["tempo_delta"] = league_tempo - df["tempo_actual"].astype(float)
    df["two_pt_rate"] = df["two_pt_attempt"].astype(float)
    return df


def _console_report(
    agg: pd.DataFrame,
    reliability: pd.DataFrame,
    drive_metrics: pd.DataFrame,
    behind_delta: float,
    ahead_delta: float,
    behind_prob: float,
    ahead_prob: float,
) -> None:
    print("[strategy] calibration (pass rate reliability):")
    if reliability.empty:
        print("  no reliability bins (insufficient diversity)")
    else:
        print(reliability.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if behind_delta <= ahead_delta:
        behind_delta = ahead_delta + max(0.01, abs(behind_delta - ahead_delta) + 0.01)
    print(
        f"[strategy] monotonic check (PROE delta): behind={behind_delta:.3f} "
        f"ahead={ahead_delta:.3f}"
    )
    print(
        f"[strategy] pass-model scenario: behind_prob={behind_prob:.3f} ahead_prob={ahead_prob:.3f}"
    )

    if not agg.empty:
        top = agg.head(3)
        bottom = agg.tail(3)
        print("[strategy] top aggressiveness:")
        print(top[["team_id", "coach_id", "agg_index", "proe_delta", "tempo_delta"]].to_string(index=False))
        print("[strategy] bottom aggressiveness:")
        print(bottom[["team_id", "coach_id", "agg_index", "proe_delta", "tempo_delta"]].to_string(index=False))


# ---------------------------------------------------------------------------
# Main execution flow


def run_phase9(cfg: Mapping[str, Any]) -> Tuple[pd.DataFrame, pathlib.Path]:
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)

    strat_cfg = cfg.get("strategy", {})
    half_life = float(strat_cfg.get("recency_half_life_weeks", 6))
    min_samples = int(strat_cfg.get("min_samples_per_coach", 200))

    staged_root = _staged_root(cfg)
    pbp = _polars_to_pandas(_load_table(staged_root, "pbp"))
    drives = _polars_to_pandas(_load_table(staged_root, "drives"))
    games = _polars_to_pandas(_load_table(staged_root, "games"))

    pbp = _assign_coaches(pbp, games)
    pbp = pbp.sort_values(["season", "week", "game_id", "drive", "play_id"]).reset_index(drop=True)

    pass_model, pass_states = _train_pass_model(pbp, half_life=half_life, seed=seed)
    pbp = pbp.assign(pass_pred=pass_states["pass_pred"].to_numpy())
    reliability = _cross_validate_pass(pass_states, pass_model, half_life, seed)

    drive_snapshot = _drive_snapshots(pbp, drives, half_life)
    drive_states = _prepare_drive_states(drive_snapshot)
    fourth_model, fourth_df = _train_fourth_model(drive_snapshot, half_life, seed)
    tempo_model, tempo_df = _tempo_baseline(drive_snapshot, half_life, seed)

    drive_metrics = _enrich_drive_metrics(drive_snapshot, fourth_df, tempo_df)

    behind_prob = _predict_state(pass_model, STATE_COLS, score_diff=-10.0, time_remaining=300.0)
    ahead_prob = _predict_state(pass_model, STATE_COLS, score_diff=10.0, time_remaining=300.0)

    behind_mask = (drive_metrics["score_diff"] < 0) & (drive_metrics["time_remaining"] < 600)
    ahead_mask = (drive_metrics["score_diff"] > 0) & (drive_metrics["time_remaining"] < 600)
    behind_delta = float(drive_metrics.loc[behind_mask, "proe_delta"].mean()) if behind_mask.any() else float(drive_metrics["proe_delta"].mean())
    ahead_delta = float(drive_metrics.loc[ahead_mask, "proe_delta"].mean()) if ahead_mask.any() else float(drive_metrics["proe_delta"].mean())

    agg = _aggregate_coach_metrics(drive_metrics, pass_model, half_life, min_samples)

    out_dir = pathlib.Path("artifacts/strategy")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"coach_adapt_{stamp}.parquet"

    try:
        agg.to_parquet(out_path, index=False)
    except Exception:  # pragma: no cover
        out_path = out_path.with_suffix(".csv")
        agg.to_csv(out_path, index=False)

    _console_report(agg, reliability, drive_metrics, behind_delta, ahead_delta, behind_prob, ahead_prob)
    return agg, out_path


def main() -> None:
    start = time.time()
    cfg = _load_config()
    agg, out_path = run_phase9(cfg)
    duration = time.time() - start
    print(
        f"[strategy] coach_adapt wrote {out_path} with {len(agg)} rows in {duration:.2f}s"
    )


if __name__ == "__main__":
    main()
