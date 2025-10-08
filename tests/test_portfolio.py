import json
import pathlib
import subprocess
import sys
import time
import uuid
from typing import Iterable

import pandas as pd
import yaml


def _write_final_probs(df: pd.DataFrame) -> pathlib.Path:
    meta_dir = pathlib.Path("artifacts/meta")
    meta_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"_{uuid.uuid4().hex[:6]}"
    path = meta_dir / f"final_probs_{stamp}.parquet"
    df.to_parquet(path, index=False)
    return path


def _data_files(paths: Iterable[pathlib.Path]) -> set[pathlib.Path]:
    return {
        p
        for p in paths
        if p.suffix in {".parquet", ".csv"}
    }


def _run_portfolio():
    outdir = pathlib.Path("artifacts/portfolio")
    before = _data_files(outdir.glob("picks_week_*")) if outdir.exists() else set()
    result = subprocess.run(
        [sys.executable, "-m", "a22a.portfolio.optimize"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "wrote" in result.stdout
    assert outdir.exists(), "portfolio artifacts directory missing"
    after = _data_files(outdir.glob("picks_week_*"))
    new = sorted(after - before, key=lambda p: p.stat().st_mtime)
    if new:
        return new[-1], result.stdout
    outputs = sorted(after, key=lambda p: p.stat().st_mtime)
    assert outputs, "no portfolio outputs produced"
    return outputs[-1], result.stdout


def _read_output(path: pathlib.Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def test_portfolio_caps_and_abstention():
    df = pd.DataFrame(
        {
            "game_id": [
                "NE@BUF",
                "KC@DEN",
                "DAL@PHI",
                "LAR@SEA",
                "PIT@BAL",
                "MIA@NYJ",
                "TB@CAR",
            ],
            "p_home": [0.51, 0.65, 0.6, 0.55, 0.58, 0.52, 0.5],
        }
    )
    _write_final_probs(df)

    path, stdout = _run_portfolio()
    portfolio_df = _read_output(path)
    cfg = yaml.safe_load(pathlib.Path("configs/defaults.yaml").read_text())["portfolio"]

    required_columns = {"game_id", "side", "p_home", "p_away", "edge", "stake_pct", "stake_amount", "corr_flag"}
    assert required_columns.issubset(portfolio_df.columns)
    assert "exposure_pct" in portfolio_df.columns

    assert float(portfolio_df["stake_pct"].max()) <= cfg["max_stake_pct_per_bet"] + 1e-6
    assert float(portfolio_df.groupby("game_id")["stake_pct"].sum().max()) <= cfg["max_game_exposure_pct"] + 1e-6
    assert float(portfolio_df["stake_pct"].sum()) <= cfg["max_weekly_exposure_pct"] + 1e-6
    assert float(portfolio_df["exposure_pct"].max()) <= cfg["max_weekly_exposure_pct"] + 1e-6

    near_coin = portfolio_df[portfolio_df["p_home"].sub(0.5).abs() < 1e-8]
    assert not near_coin.empty
    assert (near_coin["stake_pct"] == 0).all()

    summary_path = path.with_suffix(".json")
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    for key in ["exposure_pct", "avg_edge", "herfindahl", "number_abstained"]:
        assert key in summary
    assert "dropped_corr=" in stdout


def test_portfolio_all_coin_flips_abstain():
    df = pd.DataFrame(
        {
            "game_id": ["GB@CHI", "NO@ATL", "HOU@TEN"],
            "p_home": [0.5, 0.5, 0.5],
        }
    )
    _write_final_probs(df)

    path, _ = _run_portfolio()
    portfolio_df = _read_output(path)

    assert (portfolio_df["stake_pct"] == 0).all()
    summary = json.loads(path.with_suffix(".json").read_text())
    assert summary["exposure_pct"] == 0
    assert summary["active_picks"] == 0


def test_correlation_guard_flags_conflicts():
    df = pd.DataFrame(
        {
            "game_id": ["KC@DEN", "KC@BUF", "NYG@DAL", "SF@SEA"],
            "p_home": [0.66, 0.62, 0.6, 0.59],
        }
    )
    _write_final_probs(df)

    path, stdout = _run_portfolio()
    portfolio_df = _read_output(path)

    kc_games = portfolio_df[portfolio_df["game_id"].str.contains("KC@")]
    assert kc_games["corr_flag"].any(), "one of the KC games should be dropped"
    assert kc_games.loc[kc_games["corr_flag"], "stake_pct"].eq(0).all()
    summary = json.loads(path.with_suffix(".json").read_text())
    assert summary["dropped_by_corr"] >= 1
    assert "dropped_corr=" in stdout and "dropped_corr=0" not in stdout
