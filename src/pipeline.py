"""
Data pipeline: load, clean, and prepare features/labels.

All features and labels are pre-computed in the CSV.
This module loads and validates them for modeling.
"""

import pandas as pd
from typing import List, Tuple


def load_data(
    filepath: str,
    start_date: str = None,
    end_date: str = None,
    tickers: List[str] = None
) -> pd.DataFrame:
    """
    Load the merged features dataset, optionally filtered by date range and tickers.

    filepath points to merged_features_clean.csv. Rows are returned sorted by date and
    ticker. start_date and end_date filter inclusively, and tickers restricts to a subset.
    """
    df = pd.read_csv(filepath, parse_dates=["Date"])

    if start_date:
        df = df[df["Date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["Date"] <= pd.to_datetime(end_date)]
    if tickers:
        df = df[df["Ticker"].isin(tickers)]

    return df.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def prepare_features_and_labels(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into a feature matrix, target labels, and the Date series.

    feature_cols and target_col name the columns to use. Rows with a missing value in any
    feature or in the target are dropped. Returns (X, y, dates).
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    dates = df["Date"].copy()

    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)
    dates = dates[valid_mask].reset_index(drop=True)

    return X, y, dates
