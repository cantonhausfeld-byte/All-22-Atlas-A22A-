import pathlib
import subprocess
import sys

from a22a.store.metrics_store import ensure_store


def test_backtest_runs_and_store():
    result = subprocess.run(
        [sys.executable, "-m", "a22a.backtest.run"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[backtest] wrote" in result.stdout

    con = ensure_store()
    con.execute("select 1").fetchall()

    store_path = pathlib.Path("artifacts/store/a22a_metrics.duckdb")
    assert store_path.exists()

    outputs = list(pathlib.Path("artifacts/backtest").glob("summary_*.json"))
    assert outputs, "backtest should write summary artifacts"
