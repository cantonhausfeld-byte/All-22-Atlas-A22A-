import json
import pathlib

import duckdb

from a22a.backtest.run import run_backtest


def test_backtest_smoke():
    payload, path = run_backtest()
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["aggregate"] == payload["aggregate"]
    assert payload["seasons"]
    assert payload["seasons"][0]["weeks"]

    store_path = pathlib.Path("artifacts/store/a22a_metrics.duckdb")
    assert store_path.exists()

    conn = duckdb.connect(str(store_path))
    try:
        rows = conn.execute("SELECT COUNT(*) FROM backtest_metrics").fetchone()[0]
    finally:
        conn.close()
    assert rows > 0
