"""Phase 15 portfolio optimisation: bankroll allocation & correlation guard."""

from __future__ import annotations

import json
import math
import pathlib
import re
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

from a22a.metrics import portfolio as portfolio_metrics


PORTFOLIO_DIR = pathlib.Path("artifacts/portfolio")
META_DIR = pathlib.Path("artifacts/meta")
SIM_DIRS = (
    pathlib.Path("artifacts/sim"),
    pathlib.Path("data/models"),
)


@dataclass
class PCfg:
    """Portfolio configuration parsed from ``configs/defaults.yaml``."""

    bankroll: float
    kelly_fraction: float
    max_stake_pct_per_bet: float
    max_weekly_exposure_pct: float
    max_game_exposure_pct: float
    corr_guard: bool
    corr_threshold: float


def kelly_even(prob: float, fraction: float) -> float:
    """Return fractional Kelly stake for an even-money payoff.

    Parameters
    ----------
    prob:
        Win probability.
    fraction:
        Fraction of full Kelly to deploy.
    """

    edge = 2 * prob - 1
    return max(0.0, edge) * fraction


def load_config() -> PCfg:
    cfg = yaml.safe_load(pathlib.Path("configs/defaults.yaml").read_text())
    return PCfg(**cfg["portfolio"])


def _latest_file(paths: Iterable[pathlib.Path]) -> pathlib.Path | None:
    files = [p for p in paths if p.exists()]
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime)
    return files[-1]


def load_final_probabilities() -> tuple[pd.DataFrame, pathlib.Path | None]:
    """Return the latest calibrated probabilities table.

    Falls back to a deterministic synthetic slate if no Phase 14 artifacts are
    available.  The returned dataframe always contains ``p_home`` and
    ``p_away`` probabilities.
    """

    META_DIR.mkdir(parents=True, exist_ok=True)
    candidates = list(META_DIR.glob("final_probs_*.parquet"))
    latest = _latest_file(candidates)
    if latest is None:
        df = _synth_slate()
        return df, None
    try:
        df = pd.read_parquet(latest)
    except Exception as exc:  # pragma: no cover - fallback rarely triggered
        raise RuntimeError(f"failed to read {latest}: {exc}") from exc

    if "p_home" not in df.columns:
        raise ValueError(f"final probs missing 'p_home' column: {latest}")
    if "p_away" not in df.columns:
        df["p_away"] = 1.0 - df["p_home"]
    return df, latest


def load_sim_summary() -> pd.DataFrame | None:
    """Load simulation summary outputs if they exist."""

    for directory in SIM_DIRS:
        if not directory.exists():
            continue
        for name in ("sim_summary.parquet", "sim_summary.csv"):
            path = directory / name
            if path.exists():
                try:
                    if path.suffix == ".csv":
                        return pd.read_csv(path)
                    return pd.read_parquet(path)
                except Exception:
                    continue
    return None


def _synth_slate(n_games: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(22)
    p_home = np.clip(0.5 + 0.2 * rng.normal(size=n_games), 0.01, 0.99)
    p_home[0] = 0.5
    df = pd.DataFrame({
        "game_id": [f"SYN{i:03d}" for i in range(n_games)],
        "p_home": p_home,
    })
    df["p_away"] = 1.0 - df["p_home"]
    return df


def _parse_team_codes(game_id: str) -> tuple[str | None, str | None]:
    if "@" in game_id:
        away, home = game_id.split("@", 1)
        return away.strip() or None, home.strip() or None
    if "-" in game_id:
        parts = game_id.split("-", 1)
        return parts[0].strip() or None, parts[1].strip() or None
    if "_" in game_id:
        parts = game_id.split("_", 1)
        return parts[0].strip() or None, parts[1].strip() or None
    tokens = re.split(r"\W+", game_id)
    if len(tokens) >= 2:
        return tokens[0] or None, tokens[1] or None
    return None, None


def build_weekly_slate(prob_df: pd.DataFrame, sim_summary: pd.DataFrame | None) -> pd.DataFrame:
    slate = prob_df.copy()
    slate = slate.drop_duplicates(subset=["game_id"]).reset_index(drop=True)
    slate["p_away"] = slate.get("p_away", 1.0 - slate["p_home"])
    slate["confidence"] = slate["p_home"].sub(0.5).abs()
    slate["side"] = np.where(slate["p_home"] >= 0.5, "HOME", "AWAY")
    slate["prob_pick"] = np.where(slate["side"] == "HOME", slate["p_home"], slate["p_away"])
    slate["edge_home"] = 2 * slate["p_home"] - 1
    slate["edge"] = np.where(slate["side"] == "HOME", slate["edge_home"], -slate["edge_home"])
    slate["edge"] = slate["edge"].clip(lower=0.0)
    slate[["team_away", "team_home"]] = slate["game_id"].apply(lambda gid: pd.Series(_parse_team_codes(str(gid))))

    if sim_summary is not None and not sim_summary.empty:
        summary_cols = {col.lower(): col for col in sim_summary.columns}
        margin_mean_col = summary_cols.get("margin_mean")
        margin_std_col = summary_cols.get("margin_std")
        total_mean_col = summary_cols.get("total_mean")
        total_std_col = summary_cols.get("total_std")
        sim_subset = sim_summary.set_index("game_id", drop=False)
        for df_col, sim_col in [
            ("sim_margin_mean", margin_mean_col),
            ("sim_margin_std", margin_std_col),
            ("sim_total_mean", total_mean_col),
            ("sim_total_std", total_std_col),
        ]:
            if sim_col and sim_col in sim_subset.columns:
                slate[df_col] = sim_subset.reindex(slate["game_id"])[sim_col].to_numpy()

    slate = slate.sort_values(["confidence", "edge"], ascending=False).reset_index(drop=True)
    return slate


def _estimate_pairwise_corr(slate: pd.DataFrame) -> pd.DataFrame:
    games = slate["game_id"].tolist()
    matrix = pd.DataFrame(np.eye(len(games)), index=games, columns=games)

    margin_mean = slate.get("sim_margin_mean")
    margin_std = slate.get("sim_margin_std")
    total_mean = slate.get("sim_total_mean")
    total_std = slate.get("sim_total_std")

    for i, gid_i in enumerate(games):
        for j in range(i + 1, len(games)):
            gid_j = games[j]
            corr = 0.0
            # Team overlap heuristic
            teams_i = {slate.at[i, "team_home"], slate.at[i, "team_away"]}
            teams_j = {slate.at[j, "team_home"], slate.at[j, "team_away"]}
            if any(t is not None and t in teams_j for t in teams_i):
                corr = 1.0

            # Simulation proximity heuristic
            sim_corrs = []
            if margin_mean is not None and margin_std is not None:
                m_std = float(margin_std.iloc[i] or 0.0) + float(margin_std.iloc[j] or 0.0) + 1e-6
                m_diff = abs(float(margin_mean.iloc[i] or 0.0) - float(margin_mean.iloc[j] or 0.0))
                sim_corrs.append(math.exp(-m_diff / m_std))
            if total_mean is not None and total_std is not None:
                t_std = float(total_std.iloc[i] or 0.0) + float(total_std.iloc[j] or 0.0) + 1e-6
                t_diff = abs(float(total_mean.iloc[i] or 0.0) - float(total_mean.iloc[j] or 0.0))
                sim_corrs.append(math.exp(-t_diff / t_std))
            if sim_corrs:
                corr = max(corr, float(np.clip(np.mean(sim_corrs), 0.0, 1.0)))

            matrix.iat[i, j] = matrix.iat[j, i] = corr
    return matrix


def apply_correlation_guard(slate: pd.DataFrame, pcfg: PCfg) -> pd.DataFrame:
    if not pcfg.corr_guard or len(slate) <= 1:
        slate = slate.copy()
        slate["corr_flag"] = False
        return slate

    corr_matrix = _estimate_pairwise_corr(slate)
    selected: list[str] = []
    corr_flags: list[bool] = []

    for gid in slate["game_id"]:
        if not selected:
            selected.append(gid)
            corr_flags.append(False)
            continue
        corr_vals = corr_matrix.loc[gid, selected]
        max_corr = float(corr_vals.max()) if not isinstance(corr_vals, float) else float(corr_vals)
        if max_corr > pcfg.corr_threshold:
            corr_flags.append(True)
        else:
            corr_flags.append(False)
            selected.append(gid)

    guarded = slate.copy()
    guarded["corr_flag"] = corr_flags
    return guarded


def size_portfolio(slate: pd.DataFrame, pcfg: PCfg) -> pd.DataFrame:
    sized = slate.copy()
    sized["stake_pct"] = 0.0
    sized["stake_amount"] = 0.0
    sized["exposure_pct"] = 0.0
    sized["exposure_amount"] = 0.0

    total_exposure = 0.0
    per_game_exposure: dict[str, float] = {}

    for idx, row in sized.iterrows():
        gid = str(row["game_id"])
        base_prob = float(row.get("prob_pick", 0.0))
        if row.get("corr_flag", False):
            stake = 0.0
        else:
            stake = kelly_even(base_prob, pcfg.kelly_fraction)
            stake = min(stake, pcfg.max_stake_pct_per_bet)
            used_game = per_game_exposure.get(gid, 0.0)
            remaining_game = pcfg.max_game_exposure_pct - used_game
            stake = max(0.0, min(stake, remaining_game))
            remaining_week = pcfg.max_weekly_exposure_pct - total_exposure
            stake = max(0.0, min(stake, remaining_week))

        if stake > 0:
            per_game_exposure[gid] = per_game_exposure.get(gid, 0.0) + stake
            total_exposure += stake

        sized.at[idx, "stake_pct"] = stake
        sized.at[idx, "stake_amount"] = round(stake * pcfg.bankroll, 2)
        sized.at[idx, "exposure_pct"] = total_exposure
        sized.at[idx, "exposure_amount"] = round(total_exposure * pcfg.bankroll, 2)

    return sized


def write_portfolio(df: pd.DataFrame, outdir: pathlib.Path) -> pathlib.Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    base = outdir / f"picks_week_{stamp}"
    out = base.with_suffix(".parquet")
    try:
        df.to_parquet(out, index=False)
    except Exception:
        out = base.with_suffix(".csv")
        df.to_csv(out, index=False)
    return out


def _herfindahl_index(stakes: pd.Series) -> float:
    total = float(stakes.sum())
    if total <= 0:
        return 0.0
    weights = (stakes / total).to_numpy()
    return float(np.sum(np.square(weights)))


def compute_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    exposure_pct = float(df["stake_pct"].sum())
    active = df.loc[df["stake_pct"] > 0]
    avg_edge = float(active["edge"].mean()) if not active.empty else 0.0
    herfindahl = _herfindahl_index(active.get("stake_pct", pd.Series(dtype=float)))
    num_abstained = int((df["stake_pct"] <= 0).sum())
    dropped_corr = int(df.get("corr_flag", pd.Series(dtype=bool)).sum())
    return {
        "exposure_pct": round(exposure_pct, 6),
        "avg_edge": round(avg_edge, 6),
        "herfindahl": round(herfindahl, 6),
        "number_abstained": num_abstained,
        "dropped_by_corr": dropped_corr,
        "picks_total": int(len(df)),
        "active_picks": int(len(active)),
    }


def write_summary(path: pathlib.Path, metrics: dict) -> pathlib.Path:
    summary_path = path.with_suffix(".json")
    summary_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return summary_path


def main() -> None:
    t0 = time.time()
    pcfg = load_config()
    prob_df, source_path = load_final_probabilities()
    sim_summary = load_sim_summary()
    slate = build_weekly_slate(prob_df, sim_summary)
    guarded = apply_correlation_guard(slate, pcfg)
    sized = size_portfolio(guarded, pcfg)

    out_path = write_portfolio(sized, PORTFOLIO_DIR)
    metrics = compute_metrics(sized)
    write_summary(out_path, metrics)

    exposure_stats = portfolio_metrics.exposure_summary(sized)
    concentration_stats = portfolio_metrics.concentration_summary(sized)

    elapsed = time.time() - t0
    source_msg = source_path.name if source_path else "synthetic"
    print(
        f"[portfolio] source={source_msg} wrote {out_path.name} "
        f"picks={metrics['picks_total']} active={metrics['active_picks']} "
        f"dropped_corr={metrics['dropped_by_corr']} "
        f"exposure={exposure_stats['total_pct']:.3f} "
        f"concentration={concentration_stats['gini']:.3f} "
        f"avg_edge={metrics['avg_edge']:.3f} "
        f"in {elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
