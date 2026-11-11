"""
Tests for the dependence-aware resampling helpers.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.metrics import average_precision_score

from src.resampling import _average_precision, paired_edge_stats, date_block_bootstrap_edge

# Labels with both classes, and scores drawn from a few values so most draws contain ties.
labelled_scores = st.integers(min_value=2, max_value=60).flatmap(
    lambda n: st.tuples(
        st.lists(st.integers(0, 1), min_size=n, max_size=n).filter(lambda y: 0 < sum(y) < n),
        st.lists(st.sampled_from([0.1, 0.25, 0.5, 0.75, 0.9]), min_size=n, max_size=n),
    )
)


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


def test_average_precision_groups_tied_scores_like_sklearn():
    """Tied scores share one threshold, for mixed-label ties and for rows repeated by resampling."""
    y = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    s = np.array([0.9, 0.7, 0.7, 0.7, 0.4, 0.4, 0.4, 0.1])
    # Thresholds 0.9, 0.7 and 0.4 give (recall, precision) of (1/4, 1), (3/4, 3/4) and (1, 4/7).
    assert abs(_average_precision(y, s) - 43 / 56) < 1e-12
    assert abs(_average_precision(y, s) - average_precision_score(y, s)) < 1e-12

    rng = np.random.default_rng(3)
    for _ in range(50):
        n = int(rng.integers(50, 300))
        y = (rng.random(n) < 0.3).astype(int)
        s = rng.random(n)
        # Drawing rows with replacement repeats them, as a date drawn twice does in the bootstrap.
        rows = rng.integers(0, n, size=n)
        coarse = np.round(s, 1)
        for yy, ss in [(y[rows], s[rows]), (y, coarse)]:
            if yy.min() == yy.max():
                continue
            assert abs(_average_precision(yy, ss) - average_precision_score(yy, ss)) < 1e-12


# These check correctness, not speed, and a per-example deadline only makes them flaky on a
# loaded CI runner, so none of the three properties sets one.
@settings(deadline=None)
@given(data=labelled_scores, seed=st.integers(0, 2**32 - 1))
def test_average_precision_ignores_row_order(data, seed):
    """Average precision depends only on the scores and labels, not on how the rows are ordered."""
    y, score = np.array(data[0]), np.array(data[1])
    order = np.random.default_rng(seed).permutation(len(y))
    assert abs(_average_precision(y[order], score[order]) - _average_precision(y, score)) < 1e-12


@settings(deadline=None)
@given(data=labelled_scores, copies=st.integers(2, 4))
def test_average_precision_ignores_duplicating_every_row(data, copies):
    """Repeating every row the same number of times leaves precision and recall unchanged."""
    y, score = np.array(data[0]), np.array(data[1])
    repeated = _average_precision(np.repeat(y, copies), np.repeat(score, copies))
    assert abs(repeated - _average_precision(y, score)) < 1e-12


@settings(deadline=None)
@given(data=labelled_scores)
def test_average_precision_matches_sklearn_with_ties(data):
    """With any pattern of tied scores the result equals sklearn's average_precision_score."""
    y, score = np.array(data[0]), np.array(data[1])
    assert abs(_average_precision(y, score) - average_precision_score(y, score)) < 1e-12


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
