"""
Derive the label-defining volatility threshold from raw data, verify it reproduces the
stored labels exactly, and persist it into both dataset CSVs.

The NextVolSpike labels were built (in notebook 00) from an expanding 80th-percentile threshold
computed on the full download, including each ticker's warm-up rows. Those rows are dropped from
the clean file, so recomputing from the clean CSV alone would start every expanding window late
and drift from the stored labels. This recomputes from the raw file, asserts it reproduces the
stored labels exactly on both files, and only then writes the vol_threshold column for reuse.

Safe to rerun: any existing vol_threshold column is dropped and rebuilt.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config

# The raw download is not config-owned, only the model-ready file is, so the raw path
# stays a module constant while the clean path follows data.input_path.
RAW_PATH = REPO_ROOT / "data" / "raw" / "merged_features_full.csv"

QUANTILE = 0.8
MIN_PERIODS = 126


def main():
    """Recompute the threshold from raw data, verify it, and write it into both CSVs."""
    clean_path = REPO_ROOT / load_config()["data"]["input_path"]

    # Dates stay as strings so the merge keys and the written files are byte-stable.
    raw = pd.read_csv(RAW_PATH)
    clean = pd.read_csv(clean_path)
    raw = raw.drop(columns=["vol_threshold"], errors="ignore")
    clean = clean.drop(columns=["vol_threshold"], errors="ignore")

    raw = raw.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    raw["vol_threshold"] = raw.groupby("Ticker")["RollingVol"].transform(
        lambda s: s.shift(1).expanding(min_periods=MIN_PERIODS).quantile(QUANTILE)
    )

    # The recomputed threshold must reproduce the stored label exactly, NaNs included,
    # before anything is written. A failure here means the raw file no longer matches
    # the label-generation run and the right fix is a deliberate regeneration, not a
    # silent overwrite.
    next_day_vol = raw.groupby("Ticker")["RollingVol"].shift(-1)
    rederived = np.where(
        next_day_vol.isna() | raw["vol_threshold"].isna(),
        np.nan,
        (next_day_vol > raw["vol_threshold"]).astype(float),
    )
    stored = raw["NextVolSpike"].astype(float).to_numpy()
    both_nan = np.isnan(rederived) & np.isnan(stored)
    equal = rederived == stored
    if not np.all(both_nan | equal):
        bad = int((~(both_nan | equal)).sum())
        raise SystemExit(
            f"Raw-side verification failed: re-derived NextVolSpike differs from the stored "
            f"label on {bad} rows. The raw CSV no longer matches the run that generated the "
            f"labels, regenerate the dataset from notebook 00 rather than writing a "
            f"mismatched threshold."
        )

    merged = clean.merge(
        raw[["Ticker", "Date", "vol_threshold"]],
        on=["Ticker", "Date"],
        how="left",
        validate="one_to_one",
    )
    if len(merged) != len(clean):
        raise SystemExit(f"Merge changed the row count: {len(clean)} -> {len(merged)}")
    if merged["vol_threshold"].isna().any():
        n_missing = int(merged["vol_threshold"].isna().sum())
        raise SystemExit(f"{n_missing} clean rows received no threshold from the raw file.")

    # Clean-side check: within the clean file the stored label must equal the comparison
    # of next-day volatility against the persisted threshold. The only exclusions are each
    # ticker's final row, where the clean file lacks the raw file's trailing day.
    nxt = merged.groupby("Ticker")["RollingVol"].shift(-1)
    comparable = nxt.notna()
    match = (nxt[comparable] > merged.loc[comparable, "vol_threshold"]).astype(int)
    if not (match == merged.loc[comparable, "NextVolSpike"].astype(int)).all():
        bad = int((match != merged.loc[comparable, "NextVolSpike"].astype(int)).sum())
        raise SystemExit(f"Clean-side verification failed on {bad} rows.")

    raw.to_csv(RAW_PATH, index=False)
    merged.to_csv(clean_path, index=False)
    print(f"vol_threshold written to {RAW_PATH.name} ({len(raw)} rows) "
          f"and {clean_path.name} ({len(merged)} rows)")
    print(f"verified: stored NextVolSpike reproduced exactly on both files "
          f"({int(comparable.sum())} comparable clean rows)")


if __name__ == "__main__":
    main()
