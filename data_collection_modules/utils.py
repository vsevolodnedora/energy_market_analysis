import pandas as pd
import numpy as np

def validate_dataframe(df: pd.DataFrame, text:str = '') -> bool:
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"DataFrame index is not sorted in ascending order | {text}")

    if df.isna().any().any():
        # Find columns with NaN values and their counts
        nan_counts = df.isna().sum()
        nan_columns = nan_counts[nan_counts > 0]

        # Print results
        if nan_columns.empty:
            print(f"No NaN values found in the DataFrame.")
        else:
            print(f"In {text} | Columns with NaN values and their counts:")
            for col, count in nan_columns.items():
                print(f"{col}: {count} NaN values")
        return False

    # Check for infinite values and handle similarly to NaN check
    if np.isinf(df.values).any():
        # Find columns with infinite values and their counts
        inf_counts = np.isinf(df).sum()
        inf_columns = inf_counts[inf_counts > 0]

        if inf_columns.empty:
            print("No infinite values found in the DataFrame.")
        else:
            print(f"In {text} | Columns with infinite values and their counts:")
            for col, count in inf_columns.items():
                print(f"{col}: {count} infinite values")
        return False

    return True


def compare_columns(df1:pd.DataFrame, df2:pd.DataFrame):
    # Get columns from each DataFrame
    df1_cols = set(df1.columns)
    df2_cols = set(df2.columns)

    # Find columns missing in each DataFrame
    missing_in_df2 = df1_cols - df2_cols
    missing_in_df1 = df2_cols - df1_cols

    # Print results
    if not missing_in_df2 and not missing_in_df1:
        print("Both DataFrames have the same columns.")
    else:
        if missing_in_df2:
            print("Columns in first DataFrame but missing in forecast df:", missing_in_df2)
        if missing_in_df1:
            print("Columns in second DataFrame but missing in history df:", missing_in_df1)

