"""
Phase 5: time-varying team strength stub with recency decay and CI fields.
"""
import pandas as pd, numpy as np, yaml, pathlib

def compute_theta_stub(teams=("FAKE","FAK2")):
    rows = []
    for t in teams:
        rows.append({"team_id": t, "theta_mean": 0.0, "theta_lo": -0.2, "theta_hi": 0.2})
    return pd.DataFrame(rows)

def main():
    cfg = yaml.safe_load(open("configs/defaults.yaml"))
    out = pathlib.Path(cfg["paths"]["models"]) / "theta_stub.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = compute_theta_stub()
    try:
        df.to_parquet(out, index=False)
    except Exception:
        df.to_csv(out.with_suffix(".csv"), index=False)
    print(f"[theta] wrote {out}")

if __name__ == "__main__":
    main()
