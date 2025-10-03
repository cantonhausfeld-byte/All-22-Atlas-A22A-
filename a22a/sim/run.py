"""Drive-level hazard simulator with quasi Monte Carlo sampling (Phase 6)."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy.stats import qmc


RESULT_MAP = {
    "touchdown": "td",
    "rush_td": "td",
    "pass_td": "td",
    "field_goal": "fg",
    "fg": "fg",
    "punt": "punt",
    "end_game": "end",
    "interception": "turnover",
    "fumble": "turnover",
    "fumble_lost": "turnover",
    "turnover_on_downs": "turnover",
    "missed_field_goal": "miss",
}


@dataclass
class Hazard:
    team: str
    td: float
    fg: float
    turnover: float
    miss: float
    other: float
    avg_drives: float
    avg_points: float

    @property
    def td_threshold(self) -> float:
        return self.td

    @property
    def fg_threshold(self) -> float:
        return self.td + self.fg

    @property
    def turnover_threshold(self) -> float:
        return self.td + self.fg + self.turnover

    @property
    def miss_threshold(self) -> float:
        return self.td + self.fg + self.turnover + self.miss


@dataclass
class SimConfig:
    sims: int
    target_margin_ci: float
    seed: int = 42
    batch_power: int = 6  # 64 samples per Sobol block


def _load_config(path: str | Path = "configs/defaults.yaml") -> dict:
    if Path(path).exists():
        return yaml.safe_load(Path(path).read_text())
    return {}


def _load_table(root: Path, name: str) -> pl.DataFrame:
    files = list((root / name).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"missing staged table {name}")
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")


def _categorise_drives(drives: pl.DataFrame) -> pl.DataFrame:
    return drives.with_columns(
        pl.col("drive_result").replace_strict(RESULT_MAP, default="other").alias("result_type"),
    )


def _hazards(drives: pl.DataFrame) -> dict[str, Hazard]:
    cat = _categorise_drives(drives)
    grouped = cat.group_by("posteam").agg(
        pl.len().alias("drives"),
        pl.col("drive_points").mean().alias("avg_points"),
        pl.col("result_type").eq("td").cast(pl.Float64).mean().alias("td"),
        pl.col("result_type").eq("fg").cast(pl.Float64).mean().alias("fg"),
        pl.col("result_type").eq("turnover").cast(pl.Float64).mean().alias("turnover"),
        pl.col("result_type").eq("miss").cast(pl.Float64).mean().alias("miss"),
    )
    hazards = {}
    league = {
        "td": grouped["td"].mean(),
        "fg": grouped["fg"].mean(),
        "turnover": grouped["turnover"].mean(),
        "miss": grouped["miss"].mean(),
        "avg_points": grouped["avg_points"].mean(),
        "avg_drives": grouped["drives"].mean(),
    }
    for row in grouped.iter_rows(named=True):
        other = max(0.0, 1.0 - (row["td"] + row["fg"] + row["turnover"] + row["miss"]))
        hazards[row["posteam"]] = Hazard(
            team=row["posteam"],
            td=float(row["td"]),
            fg=float(row["fg"]),
            turnover=float(row["turnover"]),
            miss=float(row["miss"]),
            other=float(other),
            avg_drives=float(row["drives"]),
            avg_points=float(row["avg_points"]),
        )
    hazards["__league__"] = Hazard(
        team="league",
        td=float(league["td"]),
        fg=float(league["fg"]),
        turnover=float(league["turnover"]),
        miss=float(league["miss"]),
        other=max(0.0, 1.0 - (league["td"] + league["fg"] + league["turnover"] + league["miss"])),
        avg_drives=float(league["avg_drives"]),
        avg_points=float(league["avg_points"]),
    )
    return hazards


def _score_from_draws(draws: np.ndarray, hazard: Hazard) -> float:
    n = max(1, int(round(hazard.avg_drives)))
    score = 0.0
    for i in range(min(n, len(draws))):
        u = draws[i]
        if u < hazard.td_threshold:
            score += 7
        elif u < hazard.fg_threshold:
            score += 3
        elif u < hazard.turnover_threshold:
            score += 0
        elif u < hazard.miss_threshold:
            score += 0
        else:
            # other outcome: approximate residual points from league average
            score += hazard.avg_points * 0.1
    # adjust toward expected points to maintain mean
    expected = hazard.avg_points * n
    score = 0.7 * score + 0.3 * expected
    return score


def _ci_width(samples: list[float]) -> float:
    if len(samples) < 2:
        return float("inf")
    arr = np.asarray(samples)
    std = arr.std(ddof=1)
    return 3.92 * std / math.sqrt(len(arr))


def _simulate_game(hazard_home: Hazard, hazard_away: Hazard, cfg: SimConfig) -> dict[str, object]:
    dim = max(2, int(math.ceil(hazard_home.avg_drives)) + int(math.ceil(hazard_away.avg_drives)))
    engine = qmc.Sobol(d=dim, scramble=True, seed=cfg.seed)
    margin_samples: list[float] = []
    total_samples: list[float] = []
    max_samples = max(cfg.sims * 4, 512)
    batch_size = 2**cfg.batch_power

    while (len(margin_samples) < cfg.sims or _ci_width(margin_samples) > cfg.target_margin_ci) and len(
        margin_samples
    ) < max_samples:
        block = engine.random(batch_size)
        anti = 1.0 - block
        batch = np.vstack([block, anti])
        for row in batch:
            split = int(math.ceil(hazard_home.avg_drives))
            home_draws = row[:split]
            away_draws = row[split: split + int(math.ceil(hazard_away.avg_drives))]
            home_score = _score_from_draws(home_draws, hazard_home)
            away_score = _score_from_draws(away_draws, hazard_away)
            margin = home_score - away_score
            total = home_score + away_score
            margin_samples.append(margin)
            total_samples.append(total)
            if len(margin_samples) >= max_samples:
                break
    margin_arr = np.asarray(margin_samples)
    total_arr = np.asarray(total_samples)
    win_prob = float(np.mean(margin_arr > 0))
    ci = _ci_width(margin_samples)
    return {
        "margin_samples": margin_arr,
        "total_samples": total_arr,
        "win_prob": win_prob,
        "margin_mean": float(margin_arr.mean()),
        "margin_std": float(margin_arr.std(ddof=1)),
        "total_mean": float(total_arr.mean()),
        "total_std": float(total_arr.std(ddof=1)),
        "margin_ci_width": ci,
        "samples": len(margin_arr),
    }


def _ladder(values: np.ndarray, start: int, stop: int, step: int) -> list[dict[str, float]]:
    ladder = []
    for threshold in range(start, stop + step, step):
        ladder.append(
            {
                "threshold": float(threshold),
                "prob_ge": float(np.mean(values >= threshold)),
            }
        )
    return ladder


def run_simulations(staged_dir: Path, models_dir: Path, cfg: SimConfig) -> Path:
    games = _load_table(staged_dir, "games")
    drives = _load_table(staged_dir, "drives")
    hazards = _hazards(drives)
    league = hazards["__league__"]

    sim_dir = models_dir / "sim"
    sim_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for row in games.iter_rows(named=True):
        home = row["home_team"]
        away = row["away_team"]
        hazard_home = hazards.get(home, league)
        hazard_away = hazards.get(away, league)
        result = _simulate_game(hazard_home, hazard_away, cfg)
        margin = result["margin_samples"]
        total = result["total_samples"]
        summary = {
            "game_id": row["game_id"],
            "season": int(row["season"]),
            "week": int(row["week"]),
            "home_team": home,
            "away_team": away,
            "home_win_prob": result["win_prob"],
            "margin_mean": result["margin_mean"],
            "margin_std": result["margin_std"],
            "total_mean": result["total_mean"],
            "total_std": result["total_std"],
            "margin_ci_width": result["margin_ci_width"],
            "samples": result["samples"],
            "margin_ladder": _ladder(margin, -20, 20, 2),
            "total_ladder": _ladder(total, 20, 70, 2),
        }
        summaries.append(summary)
        samples_df = pl.DataFrame(
            {
                "margin": margin,
                "total": total,
            }
        )
        samples_df.write_parquet(sim_dir / f"{row['game_id']}_samples.parquet")
    summary_path = sim_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2))
    return summary_path


def main() -> None:
    cfg_raw = _load_config()
    sim_cfg = SimConfig(
        sims=int(cfg_raw.get("sim", {}).get("sims_per_game", 512)),
        target_margin_ci=float(cfg_raw.get("sim", {}).get("ci_width_target_margin", 0.3)),
    )
    paths = cfg_raw.get("paths", {})
    staged = Path(paths.get("staged", "./data/staged"))
    models_dir = Path(paths.get("models", "./data/models"))
    start = time.time()
    print(f"[sim] running simulations with {sim_cfg.sims} draws per game")
    out = run_simulations(staged, models_dir, sim_cfg)
    print(f"[sim] wrote summary to {out}")
    print(f"[sim] completed in {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
