import json
import pathlib
import subprocess
import sys
from typing import Tuple

import duckdb

from a22a.store.metrics_store import ensure_store


def _store_path() -> pathlib.Path:
    return pathlib.Path("artifacts/store/a22a_metrics.duckdb")


def _count_store_rows() -> int:
    path = _store_path()
    if not path.exists():
        return 0
    con = duckdb.connect(str(path))
    try:
        return int(con.execute("SELECT COUNT(*) FROM metrics").fetchone()[0])
    finally:
        con.close()


def _run_backtest() -> Tuple[pathlib.Path, dict[str, object], str]:
    outdir = pathlib.Path("artifacts/backtest")
    before = set(outdir.glob("summary_*.json")) if outdir.exists() else set()
    result = subprocess.run(
        [sys.executable, "-m", "a22a.backtest.run"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    after = set(outdir.glob("summary_*.json"))
    assert after, "backtest should create summary artifacts"
    new = sorted(after - before, key=lambda p: p.stat().st_mtime)
    path = new[-1] if new else sorted(after, key=lambda p: p.stat().st_mtime)[-1]
    payload = json.loads(path.read_text())
    return path, payload, result.stdout


def test_backtest_runs_and_store():
    ensure_store(_store_path())
    before_rows = _count_store_rows()

    first_path, first_payload, first_stdout = _run_backtest()
    after_rows = _count_store_rows()

    assert first_path.name.startswith("summary_")
    assert "[backtest]" in first_stdout
    assert first_payload["results"], "results should not be empty"
    first_result = first_payload["results"][0]
    for key in ["season", "week", "roi", "win_pct", "bankroll"]:
        assert key in first_result

    assert after_rows >= before_rows + len(first_payload["results"])

    second_path, second_payload, _ = _run_backtest()
    assert second_payload["results"] == first_payload["results"]
    assert second_payload["totals"] == first_payload["totals"]

    store_rows = _count_store_rows()
    assert store_rows >= after_rows + len(second_payload["results"])
