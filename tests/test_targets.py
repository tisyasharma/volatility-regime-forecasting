"""
Tests for the grid labeling helper, pinning the (0.80, 10) cell to the shipped labels and
checking that the quantile governs the base rate.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.targets import add_regime_columns, grid_regime_columns

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "merged_features_full.csv"


def _raw():
    """Load the raw download with full warm-up history, date-ordered within each ticker."""
    return pd.read_csv(RAW, parse_dates=["Date"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)


def test_grid_anchor_reproduces_persisted_threshold_and_labels():
    """
    At quantile 0.80 and horizon 10 the recomputed grid threshold matches the shipped
    vol_threshold and the grid regime matches the persisted FwdVolRegime exactly, so the grid
    is anchored to the headline evaluation.
    """
    raw = _raw()
    grid = grid_regime_columns(raw, forward_window=10, quantile=0.8)
    assert np.allclose(grid["grid_threshold"], raw["vol_threshold"], equal_nan=True, atol=1e-9)

    labels = add_regime_columns(raw, forward_window=10)["FwdVolRegime"]
    both = grid["grid_regime"].notna() & labels.notna()
    assert both.any()
    assert (grid.loc[both, "grid_regime"].to_numpy() == labels.loc[both].to_numpy()).all()


def test_grid_quantile_controls_base_rate():
    """A higher regime quantile yields a strictly lower positive rate at a fixed horizon."""
    raw = _raw()
    rates = [
        np.nanmean(grid_regime_columns(raw, forward_window=10, quantile=q)["grid_regime"])
        for q in (0.70, 0.80, 0.90)
    ]
    assert rates[0] > rates[1] > rates[2]
