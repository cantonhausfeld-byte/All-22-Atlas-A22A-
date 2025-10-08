import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd


def _read_artifact(path: pathlib.Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            path = path.with_suffix(".csv")
    return pd.read_csv(path)


def _run_impact() -> pd.DataFrame:
    outdir = pathlib.Path("artifacts/impact")
    outdir.mkdir(parents=True, exist_ok=True)
    before = set(outdir.glob("player_impact_*"))

    result = subprocess.run(
        [sys.executable, "-m", "a22a.impact.player_value"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "[impact]" in result.stdout

    after = set(outdir.glob("player_impact_*"))
    new_files = sorted(after - before, key=lambda p: p.stat().st_mtime)
    assert new_files, "Expected a new impact artifact to be created"

    artifact = new_files[-1]
    return _read_artifact(artifact)


def test_player_impact_columns_and_ci():
    df = _run_impact()
    required = {"player_id", "delta_win_pct", "delta_margin", "delta_total", "ci_low", "ci_high"}
    assert required.issubset(df.columns)
    assert len(df) > 0
    assert np.isfinite(df["ci_low"]).all()
    assert np.isfinite(df["ci_high"]).all()


def test_qb_outweighs_wr3():
    df = _run_impact()
    qb_mask = df["player_id"].str.contains("QB1")
    wr3_mask = df["player_id"].str.contains("WR3")
    qb_mean = df.loc[qb_mask, "delta_win_pct"].mean()
    wr3_mean = df.loc[wr3_mask, "delta_win_pct"].mean()
    assert qb_mean > wr3_mean


def test_deterministic_seed():
    df1 = _run_impact().set_index("player_id")
    df2 = _run_impact().set_index("player_id")
    cols = ["delta_win_pct", "delta_margin", "delta_total", "ci_low", "ci_high"]
    pd.testing.assert_frame_equal(df1[cols].sort_index(), df2[cols].sort_index())
