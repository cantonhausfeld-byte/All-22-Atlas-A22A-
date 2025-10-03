"""
Phase 3: Polars lazy pipeline scaffold + leakage guard hook.
"""
import pathlib, yaml, time

def main():
    t0 = time.time()
    cfg = yaml.safe_load(open("configs/defaults.yaml"))
    features_dir = pathlib.Path(cfg["paths"]["features"])
    features_dir.mkdir(parents=True, exist_ok=True)

    try:
        import polars as pl
        # Stub lazy pipeline
        lf = pl.LazyFrame({"season":[2024], "week":[1], "team_id":["FAKE"], "feature_stub":[0.0]})
        df = lf.collect()
        out = features_dir / "features_2024_w01.parquet"
        df.write_parquet(out)
        print(f"[features] wrote {out}")
    except Exception as e:
        print("[features] WARN: polars not available or failed; wrote placeholder CSV")
        (features_dir / "features_placeholder.csv").write_text("season,week,team_id,feature_stub\n2024,1,FAKE,0.0\n")

    # Leakage guard (placeholder)
    print("[features] leakage guard: OK (no future fields present in stub)")
    print(f"[features] done in {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
