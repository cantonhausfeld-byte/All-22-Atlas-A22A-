"""Construct team-game features (Phase 3)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from a22a.models import team_strength


def _load_config(path: str | Path = "configs/defaults.yaml") -> dict:
    if Path(path).exists():
        return yaml.safe_load(Path(path).read_text())
    return {}


def _load_table(root: Path, name: str) -> pl.LazyFrame:
    target = root / name
    if target.is_dir():
        files = list(target.rglob("*.parquet"))
        if files:
            return pl.scan_parquet(files)
    raise FileNotFoundError(f"missing staged table for {name} in {root}")


def _team_game_meta(games: pl.LazyFrame) -> pl.LazyFrame:
    base = games.select(
        "game_id",
        "season",
        "week",
        "kickoff_datetime",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    )
    home = base.select(
        "game_id",
        "season",
        "week",
        pl.col("kickoff_datetime"),
        pl.col("home_team").alias("team_id"),
        pl.col("away_team").alias("opponent_id"),
        pl.lit(1).alias("is_home"),
        pl.col("home_score").alias("team_score"),
        pl.col("away_score").alias("opp_score"),
    )
    away = base.select(
        "game_id",
        "season",
        "week",
        pl.col("kickoff_datetime"),
        pl.col("away_team").alias("team_id"),
        pl.col("home_team").alias("opponent_id"),
        pl.lit(0).alias("is_home"),
        pl.col("away_score").alias("team_score"),
        pl.col("home_score").alias("opp_score"),
    )
    stacked = pl.concat([home, away])
    stacked = stacked.with_columns(
        pl.col("team_score").cast(pl.Float64),
        pl.col("opp_score").cast(pl.Float64),
        (pl.col("team_score") - pl.col("opp_score")).alias("margin"),
        (pl.col("team_score") + pl.col("opp_score")).alias("total_points"),
        (pl.col("team_score") > pl.col("opp_score")).cast(pl.Int8).alias("win"),
    )
    return stacked.lazy()


def _offensive_features(pbp: pl.LazyFrame) -> pl.LazyFrame:
    offense = pbp.select(
        "game_id",
        "season",
        "week",
        pl.col("posteam").alias("team_id"),
        pl.col("defteam").alias("opponent_id"),
        pl.col("epa").fill_null(0.0).alias("epa"),
        pl.col("success").fill_null(0).alias("success"),
        pl.col("yards_gained").fill_null(0).alias("yards_gained"),
        pl.col("pass").fill_null(0).alias("pass"),
        pl.col("rush").fill_null(0).alias("rush"),
    )
    grouped = offense.group_by(["season", "week", "game_id", "team_id", "opponent_id"]).agg(
        pl.len().alias("plays"),
        pl.mean("epa").alias("epa_mean"),
        pl.sum("epa").alias("epa_sum"),
        pl.mean("success").alias("success_rate"),
        pl.mean("pass").alias("pass_rate"),
        pl.mean("rush").alias("rush_rate"),
        pl.mean("yards_gained").alias("avg_yards"),
    )
    return grouped


def _drive_features(drives: pl.LazyFrame) -> pl.LazyFrame:
    drv = drives.select(
        "game_id",
        "season",
        "week",
        pl.col("posteam").alias("team_id"),
        pl.col("drive_points").alias("drive_points"),
        pl.col("drive_result"),
        pl.col("drive_play_count"),
        pl.col("drive_time_seconds"),
        pl.col("drive_yards"),
    )
    scoring = drv.with_columns(
        pl.when(pl.col("drive_points") >= 7)
        .then(1)
        .otherwise(0)
        .alias("td_drive"),
        pl.when((pl.col("drive_points") >= 3) & (pl.col("drive_points") < 7))
        .then(1)
        .otherwise(0)
        .alias("fg_drive"),
    )
    aggregated = scoring.group_by(["season", "week", "game_id", "team_id"]).agg(
        pl.len().alias("drives"),
        pl.mean("td_drive").alias("drive_td_rate"),
        pl.mean("fg_drive").alias("drive_fg_rate"),
        pl.mean("drive_points").alias("points_per_drive"),
        pl.mean("drive_play_count").alias("plays_per_drive"),
        pl.mean("drive_time_seconds").alias("seconds_per_drive"),
        pl.mean("drive_yards").alias("yards_per_drive"),
    )
    return aggregated


def _rest_features(meta: pl.LazyFrame) -> pl.LazyFrame:
    window = meta.sort(["team_id", "kickoff_datetime"]).with_columns(
        (pl.col("kickoff_datetime").dt.timestamp("us") / 1_000_000.0).alias("kickoff_ts"),
        (pl.col("kickoff_datetime").dt.timestamp("us") / 1_000_000.0)
        .shift(1)
        .over("team_id")
        .alias("prev_kickoff_ts"),
    )
    window = window.with_columns(
        (pl.col("kickoff_ts") - pl.col("prev_kickoff_ts")).alias("rest_seconds"),
        pl.col("kickoff_datetime").dt.weekday().alias("kickoff_weekday"),
    )
    return window.with_columns(
        (pl.col("rest_seconds") / 3600.0).alias("rest_hours"),
        pl.col("rest_seconds").fill_null(7 * 24 * 3600).alias("rest_seconds"),
    )


def _roster_features(roster: pl.LazyFrame) -> pl.LazyFrame:
    grouped = roster.group_by(["season", "team"]).agg(
        pl.len().alias("roster_active"),
        pl.col("position").is_in(["QB", "RB", "WR", "TE"]).cast(pl.Float64).mean().alias(
            "roster_skill_ratio"
        ),
    )
    return grouped.rename({"team": "team_id"})


def _load_theta(models_dir: Path) -> pl.DataFrame | None:
    theta_path = models_dir / "team_strength.parquet"
    if theta_path.exists():
        return pl.read_parquet(theta_path)
    return None


def _theta_prior(theta: pl.DataFrame) -> pl.LazyFrame:
    theta = theta.sort(["team_id", "season", "week"])
    theta = theta.with_columns(
        pl.col("theta_mean").shift(1).over("team_id").alias("theta_mean_prior"),
        pl.col("theta_lo").shift(1).over("team_id").alias("theta_lo_prior"),
        pl.col("theta_hi").shift(1).over("team_id").alias("theta_hi_prior"),
    )
    return theta.lazy().select(
        "season", "week", "team_id", "theta_mean_prior", "theta_lo_prior", "theta_hi_prior"
    )


def _psi(baseline: pl.DataFrame, current: pl.DataFrame, *, bins: int = 10) -> dict[str, float]:
    psi_values: dict[str, float] = {}
    for col in current.columns:
        if not getattr(current[col].dtype, "is_numeric", lambda: False)():
            continue
        cur_series = current[col].to_numpy()
        base_series = baseline[col].to_numpy()
        if len(np.unique(cur_series)) <= 1 or len(np.unique(base_series)) <= 1:
            psi_values[col] = 0.0
            continue
        edges = np.linspace(
            min(np.nanmin(base_series), np.nanmin(cur_series)),
            max(np.nanmax(base_series), np.nanmax(cur_series)),
            bins + 1,
        )
        base_hist, _ = np.histogram(base_series, bins=edges)
        cur_hist, _ = np.histogram(cur_series, bins=edges)
        base_hist = base_hist + 1e-6
        cur_hist = cur_hist + 1e-6
        base_pct = base_hist / base_hist.sum()
        cur_pct = cur_hist / cur_hist.sum()
        psi = ((cur_pct - base_pct) * np.log(cur_pct / base_pct)).sum()
        psi_values[col] = float(psi)
    return psi_values


def _leakage_guard(df: pl.DataFrame) -> None:
    forbidden = [c for c in df.columns if "future" in c or "next_" in c]
    if forbidden:
        raise RuntimeError(f"leakage guard tripped: future-looking columns {forbidden}")
    label_safe = {"team_score", "opp_score", "margin", "total_points", "win"}
    score_cols = [c for c in df.columns if "score" in c]
    unexpected = [c for c in score_cols if c not in label_safe]
    if unexpected:
        raise RuntimeError(f"leakage guard tripped: unexpected score columns {unexpected}")


def build_features(staged_dir: Path, features_dir: Path, models_dir: Path) -> list[Path]:
    start = time.time()
    pbp = _load_table(staged_dir, "pbp")
    drives = _load_table(staged_dir, "drives")
    games = _load_table(staged_dir, "games")
    roster = _load_table(staged_dir, "roster")

    meta = _team_game_meta(games)
    rest = _rest_features(meta).select(
        "game_id",
        "team_id",
        pl.col("rest_hours"),
        pl.col("kickoff_weekday"),
    )

    offense = _offensive_features(pbp)
    drv = _drive_features(drives)
    roster_feats = _roster_features(roster)

    seasons = games.select("season").unique().collect()["season"].to_list()
    theta_df = _load_theta(models_dir)
    if theta_df is None:
        cfg = team_strength.StrengthConfig(seasons=[int(s) for s in seasons])
        team_strength.run_model(staged_dir, models_dir, cfg)
        theta_df = _load_theta(models_dir)
    theta_lazy = _theta_prior(theta_df) if theta_df is not None else None

    feature_lf = (
        meta.join(offense, on=["season", "week", "game_id", "team_id"], how="left")
        .join(drv, on=["season", "week", "game_id", "team_id"], how="left")
        .join(rest, on=["game_id", "team_id"], how="left")
        .join(roster_feats, on=["season", "team_id"], how="left")
    )
    if theta_lazy is not None:
        feature_lf = feature_lf.join(theta_lazy, on=["season", "week", "team_id"], how="left")

    schema = feature_lf.schema
    history_cols = [
        "plays",
        "epa_mean",
        "epa_sum",
        "success_rate",
        "pass_rate",
        "rush_rate",
        "avg_yards",
        "drives",
        "drive_td_rate",
        "drive_fg_rate",
        "points_per_drive",
        "plays_per_drive",
        "seconds_per_drive",
        "yards_per_drive",
    ]
    history_cols = [c for c in history_cols if c in schema]
    for col in history_cols:
        feature_lf = feature_lf.with_columns(pl.col(col).cast(pl.Float64))
        feature_lf = feature_lf.with_columns(
            pl.col(col).shift(1).over("team_id").alias(f"{col}_prior"),
            pl.col(col).shift(1).over("opponent_id").alias(f"opp_{col}_prior"),
        )
    feature_lf = feature_lf.drop(history_cols)

    feature_lf = feature_lf.with_columns(
        pl.col("rest_hours").fill_null(168.0),
        pl.col("kickoff_weekday").fill_null(6),
        pl.col("roster_active").fill_null(53),
        pl.col("roster_skill_ratio").fill_null(0.5),
        pl.col("theta_mean_prior").fill_null(0.0),
        pl.col("theta_lo_prior").fill_null(-3.0),
        pl.col("theta_hi_prior").fill_null(3.0),
    )

    prior_cols = [f"{col}_prior" for col in history_cols] + [f"opp_{col}_prior" for col in history_cols]
    for col in prior_cols:
        if col in feature_lf.schema:
            feature_lf = feature_lf.with_columns(pl.col(col).fill_null(0.0))

    df = feature_lf.collect()
    _leakage_guard(df)

    features_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for season in df.select("season").unique().to_series().to_list():
        sub = df.filter(pl.col("season") == season)
        for week in sorted(sub.select("week").unique().to_series().to_list()):
            part = sub.filter(pl.col("week") == week)
            out = features_dir / f"season={int(season)}" / f"week={int(week):02d}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            part.write_parquet(out)
            outputs.append(out)

    baseline_path = features_dir / "reference.parquet"
    if baseline_path.exists():
        baseline = pl.read_parquet(baseline_path)
        psi = _psi(baseline, df)
        high_drift = {k: v for k, v in psi.items() if v > 0.2}
        if high_drift:
            raise RuntimeError(f"PSI drift check failed: {high_drift}")
    else:
        df.write_parquet(baseline_path)
        psi = {c: 0.0 for c in df.columns if getattr(df[c].dtype, "is_numeric", lambda: False)()}

    (features_dir / "last_run.json").write_text(
        json.dumps(
            {
                "rows": df.height,
                "season_weeks": sorted({(int(r[0]), int(r[1])) for r in df.select(["season", "week"]).unique().rows()}),
                "psi": psi,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "duration_seconds": round(time.time() - start, 3),
            },
            indent=2,
        )
    )
    return outputs


def main() -> None:
    cfg = _load_config()
    staged = Path(cfg.get("paths", {}).get("staged", "./data/staged"))
    features_dir = Path(cfg.get("paths", {}).get("features", "./data/features"))
    models_dir = Path(cfg.get("paths", {}).get("models", "./data/models"))
    print(f"[features] building from {staged}")
    outputs = build_features(staged, features_dir, models_dir)
    for out in outputs:
        print(f"[features] wrote {out}")
    print("[features] complete")


if __name__ == "__main__":
    main()
