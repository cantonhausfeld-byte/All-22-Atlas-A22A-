import json
import pathlib

from a22a.monitor.run import run_monitor


def test_monitor_smoke(capsys):
    payload, path = run_monitor()
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["status"] == payload["status"]
    assert "details" in payload
    assert set(payload["details"]).issuperset({"calibration", "coverage", "clv", "slo"})

    captured = capsys.readouterr().out
    assert "[monitor] summary" in captured
    assert "[monitor] wrote" in captured

    latest = sorted(pathlib.Path("artifacts/monitor").glob("health_*.json"))
    assert latest
