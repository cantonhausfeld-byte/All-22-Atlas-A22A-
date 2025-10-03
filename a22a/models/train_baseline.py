"""
Phase 4: baseline training stub with purged forward-chaining CV placeholder and calibration scaffold.
"""
import numpy as np, pandas as pd, time, pathlib, yaml
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss

def main():
    t0 = time.time()
    cfg = yaml.safe_load(open("configs/defaults.yaml"))
    models_dir = pathlib.Path(cfg["paths"]["models"])
    models_dir.mkdir(parents=True, exist_ok=True)

    # Dummy training set (replace with real features later)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(500, 5))
    y = (X[:,0] + 0.3*rng.normal(size=500) > 0).astype(int)

    base = LogisticRegression(max_iter=1000)
    base.fit(X, y)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(X, y)
    p = clf.predict_proba(X)[:,1]
    print(f"[baseline] brier={brier_score_loss(y,p):.4f} logloss={log_loss(y,p):.4f}")
    (models_dir / "baseline_stub.pkl").write_bytes(b"stub")
    print(f"[baseline] trained + calibrated in {time.time()-t0:.2f}s (stub)")

if __name__ == "__main__":
    main()
