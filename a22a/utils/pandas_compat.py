from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    import polars as pl


def to_pandas_safe(df_pl: "pl.DataFrame") -> pd.DataFrame:
    """
    Convert a Polars DataFrame to pandas with a robust dtype strategy.
    1) Prefer Arrow extension arrays (fast & safe for timestamps)
    2) Fallback to plain pandas dtypes if Arrow is unavailable
    Returns: pandas.DataFrame
    """
    # Step 1: try Polars' Arrow-backed conversion if available
    try:
        pdf = df_pl.to_pandas(use_pyarrow_extension_array=True)  # polars>=0.20
    except Exception:
        try:
            pdf = df_pl.to_pandas()
        except Exception:
            pdf = pd.DataFrame(df_pl.to_dicts())

    # Step 2: normalize dtypes
    try:
        # If pyarrow + ArrowDtype is present, this will succeed
        return pdf.convert_dtypes(dtype_backend="pyarrow")
    except Exception:
        # Fallback that works everywhere
        return pdf.convert_dtypes(dtype_backend="numpy_nullable")

