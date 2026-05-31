from typing import Optional

import numpy as np
import pandas as pd
import os

from logger import get_logger
logger = get_logger(__name__)

class ParquetOperations:
    """Singleton Read/write Parquet files with optional zstd compression and dtype reduction."""

    compression_level = 4
    dtype_reduction = True

    @staticmethod
    def memory_mb(df: pd.DataFrame) -> float:
        """Return in-memory size of a DataFrame in MB."""
        return df.memory_usage(deep=True).sum() / (1024 * 1024)
    # dtype reduction
    @staticmethod
    def _reduce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Return a dtype-reduced copy (float64→float32, int64→int32, etc.)."""
        if not ParquetOperations.dtype_reduction or df.empty:
            return df

        out = df.copy()

        f32_max = np.finfo(np.float32).max
        i32_info = np.iinfo(np.int32)
        u32_max = np.iinfo(np.uint32).max

        for col in out.columns:
            s = out[col]
            dt = s.dtype

            # floats: 64-bit -> 32-bit (only when finite values fit)
            if dt.kind == "f" and dt.itemsize == 8:
                nullable = isinstance(dt, pd.Float64Dtype)
                arr = s.to_numpy(dtype="float64", na_value=np.nan)
                finite = arr[np.isfinite(arr)]

                if finite.size == 0 or np.nanmax(np.abs(finite)) <= f32_max:
                    out[col] = s.astype("Float32" if nullable else "float32")
                continue

            # integers: 64-bit → 32-bit (only when values fit)
            if dt.kind in ("i", "u") and dt.itemsize == 8:
                unsigned = dt.kind == "u"
                nullable = isinstance(dt, pd.core.arrays.integer.IntegerDtype)

                s_non_na = s.dropna()
                if s_non_na.empty:
                    # All missing — safe to downcast nullable types; numpy int64
                    # columns can't hold NA, so dropna is a no-op for them.
                    if nullable:
                        out[col] = s.astype("UInt32" if unsigned else "Int32")
                    continue

                min_v, max_v = int(s_non_na.min()), int(s_non_na.max())

                if unsigned:
                    fits = 0 <= min_v and max_v <= u32_max
                else:
                    fits = i32_info.min <= min_v and max_v <= i32_info.max

                if fits:
                    if nullable:
                        out[col] = s.astype("UInt32" if unsigned else "Int32")
                    else:
                        out[col] = s.astype("uint32" if unsigned else "int32")

        return out

    # I/O
    @staticmethod
    def save(df: pd.DataFrame, path: str) -> None:
        """Write *df* to a Parquet file, applying dtype reduction and compression."""
        original_mb = ParquetOperations.memory_mb(df)

        df_out = ParquetOperations._reduce_dtypes(df)

        kwargs: dict = {"engine": "pyarrow"}
        if ParquetOperations.compression_level is None:
            kwargs["compression"] = None
        else:
            kwargs["compression"] = "zstd"
            kwargs["compression_level"] = int(ParquetOperations.compression_level)

        df_out.to_parquet(path, **kwargs)

        file_mb = os.path.getsize(path) / (1024 * 1024)
        total_saved_mb = original_mb - file_mb

        logger.info(
            "Total savings: %.2f MB (%.2f MB original -> %.2f MB on disk, %.1f%% reduction) for %s",
            total_saved_mb,
            original_mb,
            file_mb,
            (total_saved_mb / original_mb) * 100 if original_mb > 0 else 0,
            path,
        )

    @staticmethod
    def read(path: str) -> pd.DataFrame:
        """Read a Parquet file back into a DataFrame."""
        return pd.read_parquet(path, engine="pyarrow")