import polars as pl

from a22a.utils.pandas_compat import to_pandas_safe


def test_to_pandas_safe_roundtrip():
    df_pl = pl.DataFrame({"ts": [0, 1, 2], "s": ["a", "b", "c"]})
    df_pd = to_pandas_safe(df_pl)

    assert list(df_pd.columns) == ["ts", "s"]
    assert len(df_pd) == 3
