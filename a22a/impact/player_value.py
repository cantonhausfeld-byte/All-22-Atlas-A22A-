"""Phase 13 — player impact modelling with depth-aware perturbations."""

from __future__ import annotations

import math
import pathlib
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping

import numpy as np
import pandas as pd
import yaml

from a22a.health import injury_model
from a22a.metrics import summarize_player_metric
from a22a.roster import depth_logic

CONFIG_PATH = pathlib.Path("configs/defaults.yaml")
ARTIFACT_DIR = pathlib.Path("artifacts/impact")
SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True)
class ImpactConfig:
    """Configuration values controlling the player impact simulation."""

    samples_per_player: int = 200
    ci_level: float = 0.9
    max_games_per_batch: int = 8
    seed: int = 13

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object] | None) -> "ImpactConfig":
        data = dict(raw or {})
        return cls(
            samples_per_player=int(data.get("samples_per_player", 200)),
            ci_level=float(data.get("ci_level", 0.9)),
            max_games_per_batch=int(data.get("max_games_per_batch", 8)),
            seed=int(data.get("seed", 13)),
        )


def _load_config(path: pathlib.Path = CONFIG_PATH) -> tuple[dict[str, object], ImpactConfig]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw: dict[str, object] = yaml.safe_load(path.read_text()) or {}
    impact_cfg = ImpactConfig.from_mapping(raw.get("impact", {}))
    return raw, impact_cfg


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    vec_erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + vec_erf(arr / SQRT2))


def _stratified_draws(samples: int, rng: np.random.Generator) -> np.ndarray:
    grid = (np.arange(samples, dtype=float) + 0.5) / float(samples)
    order = rng.permutation(samples)
    return grid[order]


def _player_share(
    samples: int,
    avail_prob: float,
    exit_prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    avail_draws = _stratified_draws(samples, rng)
    exit_draws = _stratified_draws(samples, rng)
    duration_draws = _stratified_draws(samples, rng)

    available = avail_draws < np.clip(avail_prob, 0.0, 1.0)
    exiting = exit_draws < np.clip(exit_prob, 0.0, 1.0)
    partial = 0.35 + 0.5 * duration_draws  # 35–85% share when an exit occurs
    share = np.where(available, np.where(exiting, partial, 1.0), 0.0)
    return share.astype(float)


def _compute_exit_probability(rate: float, exposure: float) -> float:
    rate = max(rate, 0.0)
    exposure = max(exposure, 0.0)
    return float(1.0 - math.exp(-rate * exposure)) if rate > 0 else 0.0


def _load_lineups(cfg_map: Mapping[str, object]) -> pd.DataFrame:
    depth_cfg = depth_logic.DepthConfig.from_config(cfg_map)
    return depth_logic._build_lineups(depth_cfg, cfg_map)


def _load_health(cfg_map: Mapping[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    health_cfg = injury_model.HealthConfig.from_config(cfg_map)
    reports = injury_model._load_injury_reports(cfg_map)
    prepared = injury_model._prepare_dataset(reports, health_cfg)
    availability, enriched, _metrics = injury_model._fit_availability(prepared, health_cfg)
    hazards = injury_model._fit_exit_hazard(enriched, health_cfg)
    return availability, hazards


def _team_ratings(lineups: pd.DataFrame) -> dict[str, float]:
    starters = lineups[lineups["depth_role"] == "starter"].copy()
    grouped = starters.groupby(starters["team_id"].astype(str))["player_effective_uer"].sum()
    if grouped.empty:
        return {}
    league_avg = float(grouped.mean())
    margin_scale = 7.25
    return {team: (float(value) - league_avg) * margin_scale for team, value in grouped.items()}


def _summarise_player(
    player: Mapping[str, object],
    config: ImpactConfig,
    team_mu: float,
    avail_lookup: Mapping[str, float],
    hazard_lookup: Mapping[str, tuple[float, float]],
    rng: np.random.Generator,
) -> Mapping[str, float | str]:
    player_id = str(player["player_id"])
    team_id = str(player.get("team_id", ""))
    position = str(player.get("position", ""))
    depth_role = str(player.get("depth_role", ""))

    player_value = float(player.get("player_effective_uer", 0.0))
    replacement_value = float(player.get("replacement_effective_uer", 0.0))
    delta_rating = player_value - replacement_value

    avail_prob = float(avail_lookup.get(player_id, 0.92))
    rate, exposure = hazard_lookup.get(player_id, (0.0015, 45.0))
    exit_prob = _compute_exit_probability(rate, exposure)

    share = _player_share(config.samples_per_player, avail_prob, exit_prob, rng)
    delta_strength = delta_rating * share

    margin_scale = 7.25
    total_scale = 4.0
    sigma_margin = 9.5

    base_mu = team_mu
    margin_delta_samples = delta_strength * margin_scale
    total_delta_samples = delta_strength * total_scale

    without_prob = _normal_cdf(np.asarray([base_mu / sigma_margin]))[0]
    with_prob = _normal_cdf((base_mu + margin_delta_samples) / sigma_margin)
    win_delta_samples = with_prob - without_prob

    summary = summarize_player_metric(
        win_delta_samples, margin_delta_samples, total_delta_samples, config.ci_level
    )
    return {
        "player_id": player_id,
        "team_id": team_id,
        "position": position,
        "depth_role": depth_role,
        "delta_win_pct": summary["delta_win_pct"],
        "delta_margin": summary["delta_margin"],
        "delta_total": summary["delta_total"],
        "ci_low": summary["delta_win_pct_ci_low"],
        "ci_high": summary["delta_win_pct_ci_high"],
        "samples": config.samples_per_player,
    }


def _prepare_player_table(lineups: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "player_id",
        "team_id",
        "position",
        "depth_role",
        "player_effective_uer",
        "replacement_effective_uer",
    ]
    missing = [c for c in cols if c not in lineups.columns]
    if missing:
        raise ValueError(f"Lineup table missing required columns: {missing}")
    table = lineups[cols].drop_duplicates(subset="player_id").reset_index(drop=True)
    team_labels = table["team_id"].astype(str)
    table["team_key"] = team_labels.where(~team_labels.str.lower().eq("nan"), "__missing__")
    return table


def _write_artifact(df: pd.DataFrame) -> pathlib.Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    nonce = uuid.uuid4().hex[:6]
    out_path = ARTIFACT_DIR / f"player_impact_{timestamp}_{nonce}.parquet"
    try:
        df.to_parquet(out_path, index=False)
    except Exception:
        out_path = out_path.with_suffix(".csv")
        df.to_csv(out_path, index=False)
    return out_path


def _console_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("[impact] no player impact records to summarise")
        return
    top = df.sort_values("delta_win_pct", ascending=False).head(10)
    bottom = df.sort_values("delta_win_pct", ascending=True).head(10)
    print("[impact] top players by Δwin%:")
    for row in top.itertuples(index=False):
        print(
            f"  {row.player_id:>12}  team={row.team_id:<4}  pos={row.position:<3}  Δwin%={row.delta_win_pct:+.4f}"
        )
    print("[impact] bottom players by Δwin%:")
    for row in bottom.itertuples(index=False):
        print(
            f"  {row.player_id:>12}  team={row.team_id:<4}  pos={row.position:<3}  Δwin%={row.delta_win_pct:+.4f}"
        )


def main() -> None:
    start_time = time.time()
    cfg_map, impact_cfg = _load_config()

    lineups = _load_lineups(cfg_map)
    availability, hazards = _load_health(cfg_map)

    players = _prepare_player_table(lineups)

    avail_lookup = availability.set_index("player_id")["avail_prob"].to_dict()
    hazard_lookup = {
        pid: (float(row["exit_hazard_rate"]), float(row.get("exit_exposure", 45.0)))
        for pid, row in hazards.set_index("player_id").iterrows()
    }

    team_mu_map = _team_ratings(lineups)

    seed_seq = np.random.SeedSequence(impact_cfg.seed)
    generators = [np.random.default_rng(s) for s in seed_seq.spawn(len(players))]

    records: list[Mapping[str, float | str]] = []
    teams = sorted(players["team_key"].unique().tolist())
    batch_size = max(impact_cfg.max_games_per_batch, 1)
    gen_iter = iter(generators)
    if not teams:
        teams = ["__missing__"]
    for batch_start in range(0, len(teams), batch_size):
        batch_teams = set(teams[batch_start : batch_start + batch_size])
        batch_players = players[players["team_key"].isin(batch_teams)]
        for player_row in batch_players.to_dict("records"):
            try:
                rng = next(gen_iter)
            except StopIteration:
                rng = np.random.default_rng(seed_seq.spawn(1)[0])
            team_key = str(player_row.get("team_key", "__missing__"))
            mu = team_mu_map.get(team_key if team_key != "__missing__" else "", 0.0)
            records.append(
                _summarise_player(player_row, impact_cfg, mu, avail_lookup, hazard_lookup, rng)
            )

    df = pd.DataFrame.from_records(records)
    artifact_path = _write_artifact(df)
    duration = time.time() - start_time
    print(
        f"[impact] wrote {artifact_path.name} with {len(df)} players in {duration:.2f}s"
    )
    _console_summary(df)


if __name__ == "__main__":
    main()
