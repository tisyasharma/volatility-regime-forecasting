"""
Tests for the dependence-aware resampling helpers and the out-of-fold schema they consume.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from src.resampling import _average_precision, paired_edge_stats, date_block_bootstrap_edge

OOF = Path(__file__).resolve().parents[1] / "reports" / "metrics" / "oof_predictions.csv"


def test_average_precision_matches_sklearn_on_continuous_scores():
    """The fast average precision equals sklearn when scores carry no ties."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(100, 500))
        y = (rng.random(n) < 0.3).astype(int)
        if y.min() == y.max():
            continue
        s = rng.random(n)
        assert abs(_average_precision(y, s) - average_precision_score(y, s)) < 1e-9


def test_paired_edge_stats_flags_a_clear_positive_edge():
    """A consistently positive per-fold difference yields a CI above zero and a small p."""
    rng = np.random.default_rng(1)
    diff = 0.05 + rng.normal(0, 0.005, 12)
    fold_ids = np.arange(1, 13)
    s = paired_edge_stats(diff, fold_ids, rng, n_boot=2000)
    assert s["n_folds"] == 12
    assert s["ci_lo"] > 0
    assert s["p"] < 0.05


def test_date_block_point_estimate_equals_fold_mean_edge():
    """The date-block point statistic is exactly the mean over folds of per-fold PR-AUC diffs."""
    rng = np.random.default_rng(2)
    dates = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
    folds = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
    y = np.array([1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0])
    score_model = np.array([0.90, 0.20, 0.10, 0.80, 0.70, 0.30, 0.91, 0.40, 0.21, 0.31, 0.85, 0.25])
    score_naive = np.array([0.50, 0.41, 0.62, 0.53, 0.34, 0.75, 0.52, 0.63, 0.44, 0.66, 0.45, 0.57])

    r = date_block_bootstrap_edge(dates, folds, y, score_model, score_naive, rng, n_boot=200, block=1)
    expected = np.mean([
        average_precision_score(y[folds == f], score_model[folds == f])
        - average_precision_score(y[folds == f], score_naive[folds == f])
        for f in (1, 2)
    ])
    assert abs(r["edge"] - expected) < 1e-9
    assert r["ci_lo"] <= r["edge"] <= r["ci_hi"]


def test_oof_carries_date_and_ticker_when_generated():
    """Once run_evaluation.py has run, every out-of-fold row is labelled with its date and ticker."""
    if not OOF.exists():
        pytest.skip("oof_predictions.csv not generated yet")
    df = pd.read_csv(OOF)
    assert {"Date", "Ticker"}.issubset(df.columns)
    assert df["Date"].notna().all()
    assert df["Ticker"].notna().all()
