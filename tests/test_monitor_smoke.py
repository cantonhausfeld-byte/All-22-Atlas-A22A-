import json
import pathlib

from a22a.monitor.run import run_monitor


def test_monitor_smoke(capsys):
    payload, path = run_monitor()
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["metrics"] == payload["metrics"]

    captured = capsys.readouterr().out
    assert "[monitor] ece=" in captured
    assert "[alerts]" in captured

    latest = sorted(pathlib.Path("artifacts/monitor").glob("health_*.json"))
    assert latest
