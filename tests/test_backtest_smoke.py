import json
import pathlib

from a22a.backtest.run import run_backtest


def test_backtest_smoke():
    payload, path = run_backtest()
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["aggregate"] == payload["aggregate"]
    assert payload["seasons"]

    store_path = pathlib.Path("artifacts/store/a22a_metrics.duckdb")
    assert store_path.exists()
