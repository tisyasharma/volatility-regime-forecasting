"""
Dependence-aware resampling for the model comparison.

Fold-block resampling captures serial correlation across the time-ordered folds. Date-block
resampling additionally captures cross-sectional co-movement by resampling whole trading dates,
each carrying its full cross-section of tickers, so the 14 co-moving names are not counted as
independent draws.
"""

import warnings

import numpy as np

DEFAULT_N_BOOT = 20000
DEFAULT_BLOCK = 4


def lag1_autocorr(x, fold_ids=None):
    """
    Lag-1 autocorrelation of x, normalized by the total sum of squares.

    When fold_ids is given, only pairs of adjacent folds contribute, so gaps left by skipped
    folds are not treated as consecutive quarters. The numerator is rescaled by
    (n - 1) / n_pairs so that dropping gap-spanning pairs does not shrink the estimate toward
    zero, which would inflate the effective sample size. With no gaps the factor is exactly 1.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom <= 0:
        return 0.0
    if fold_ids is None:
        num = np.sum(x[1:] * x[:-1])
    else:
        adjacent = np.diff(np.asarray(fold_ids)) == 1
        n_pairs = int(adjacent.sum())
        if n_pairs == 0:
            return 0.0
        num = np.sum(x[1:][adjacent] * x[:-1][adjacent]) * (len(x) - 1) / n_pairs
    return float(num / denom)


def moving_block_bootstrap_mean(diff, rng, n_boot=DEFAULT_N_BOOT, block=DEFAULT_BLOCK, fold_ids=None):
    """
    Bootstrap distribution of the mean of diff using moving blocks of consecutive folds.

    When fold_ids is given, a block may only start where the next block-1 fold ids are
    consecutive, preserving the serial structure the block bootstrap relies on. Degenerate
    inputs fall back to smaller blocks instead of crashing.
    """
    # Block resampling follows the moving-block bootstrap of Kuensch (1989).
    n = len(diff)
    block_eff = max(1, min(block, n))
    if fold_ids is None:
        starts_pool = np.arange(0, n - block_eff + 1)
    else:
        ids = np.asarray(fold_ids)
        candidates = np.arange(0, n - block_eff + 1)
        starts_pool = candidates[ids[candidates + block_eff - 1] - ids[candidates] == block_eff - 1]
        if len(starts_pool) == 0:
            warnings.warn(
                "No run of consecutive folds long enough for the block bootstrap, "
                "falling back to single-fold blocks.",
                RuntimeWarning,
            )
            block_eff = 1
            starts_pool = np.arange(0, n)
    n_blocks = int(np.ceil(n / block_eff))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.choice(starts_pool, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, s + block_eff) for s in starts])[:n]
        means[b] = diff[idx].mean()
    return means


def _two_tailed_p(samples):
    """Two-tailed bootstrap p-value for a null of zero, floored at 1/B and capped at 1."""
    tail = min((samples <= 0).mean(), (samples >= 0).mean())
    return min(1.0, max(2 * tail, 1 / len(samples)))


def paired_edge_stats(diff, fold_ids, rng, n_boot=DEFAULT_N_BOOT, block=DEFAULT_BLOCK):
    """
    Fold-block edge statistics for a per-fold paired difference array.

    Returns the mean difference, the AR(1) effective sample size (Bretherton et al. 1999), the
    naive and effective-N t-statistics, and a moving-block bootstrap CI and p-value. The lag-1
    autocorrelation is clipped at zero so the effective-N test can only shrink n and stays at
    least as strict as the naive t.
    """
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    if n < 2:
        return {
            "mean_diff": float(diff.mean()) if n else np.nan, "n_folds": n,
            "fold_autocorr": np.nan, "n_eff": np.nan, "t_naive": np.nan, "t_effN": np.nan,
            "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan,
        }
    mean_d = diff.mean()
    rho = lag1_autocorr(diff, fold_ids=fold_ids)
    rho_pos = max(rho, 0.0)
    n_eff = n * (1 - rho_pos) / (1 + rho_pos)
    sd = diff.std(ddof=1)
    se_naive = sd / np.sqrt(n)
    se_eff = sd / np.sqrt(max(n_eff, 2))
    boot = moving_block_bootstrap_mean(diff, rng, n_boot=n_boot, block=block, fold_ids=fold_ids)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    return {
        "mean_diff": mean_d, "n_folds": n, "fold_autocorr": rho, "n_eff": n_eff,
        "t_naive": mean_d / se_naive if se_naive > 0 else np.nan,
        "t_effN": mean_d / se_eff if se_eff > 0 else np.nan,
        "ci_lo": ci_lo, "ci_hi": ci_hi, "p": _two_tailed_p(boot),
    }


def _average_precision(y, score):
    """
    Average precision from a labels and scores pair, matching sklearn on tie-free scores.

    Recomputed here rather than through sklearn so the date-block bootstrap can call it tens of
    thousands of times without the validation overhead. The model probabilities and the naive
    volatility-to-threshold ratios are continuous, so ties do not arise in this pipeline.
    """
    order = np.argsort(-score, kind="mergesort")
    y = y[order].astype(float)
    tp = np.cumsum(y)
    total_pos = tp[-1]
    if total_pos == 0:
        return 0.0
    precision = tp / np.arange(1, len(y) + 1)
    recall = tp / total_pos
    rec_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - rec_prev) * precision))


def date_block_bootstrap_edge(dates, folds, y_true, score_model, score_naive, rng,
                              n_boot=3000, block=21):
    """
    Cross-sectional-aware bootstrap of the fold-mean PR-AUC edge, model minus matched naive.

    The statistic is the mean over folds of the per-fold PR-AUC difference, the same quantity
    the model comparison reports, so only the interval changes rather than the point estimate.
    Each resample draws moving blocks of consecutive trading dates, every drawn date carrying its
    whole cross-section of tickers, then recomputes each fold's PR-AUC on its resampled dates and
    averages across folds. This keeps both the serial dependence across dates and the co-movement
    of the tickers within a date.
    """
    dates = np.asarray(dates)
    folds = np.asarray(folds)
    y_true = np.asarray(y_true, dtype=int)
    score_model = np.asarray(score_model, dtype=float)
    score_naive = np.asarray(score_naive, dtype=float)

    unique_dates, inverse = np.unique(dates, return_inverse=True)
    d = len(unique_dates)
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse, minlength=d)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rows_by_date = [order[starts[j]:starts[j] + counts[j]] for j in range(d)]

    def fold_mean_edge(row_idx):
        yt, sm, sn, fd = y_true[row_idx], score_model[row_idx], score_naive[row_idx], folds[row_idx]
        diffs = []
        for f in np.unique(fd):
            m = fd == f
            if yt[m].min() == yt[m].max():
                continue
            diffs.append(_average_precision(yt[m], sm[m]) - _average_precision(yt[m], sn[m]))
        return float(np.mean(diffs)) if diffs else np.nan

    point = fold_mean_edge(np.arange(len(y_true)))
    block_eff = max(1, min(block, d))
    start_pool = np.arange(0, d - block_eff + 1)
    n_blocks = int(np.ceil(d / block_eff))
    vals = []
    for _ in range(n_boot):
        picks = rng.choice(start_pool, size=n_blocks, replace=True)
        date_idx = np.concatenate([np.arange(s, s + block_eff) for s in picks])[:d]
        row_idx = np.concatenate([rows_by_date[j] for j in date_idx])
        v = fold_mean_edge(row_idx)
        if not np.isnan(v):
            vals.append(v)
    vals = np.asarray(vals)
    ci_lo, ci_hi = np.percentile(vals, [2.5, 97.5])
    return {
        "edge": point, "ci_lo": ci_lo, "ci_hi": ci_hi, "p": _two_tailed_p(vals),
        "n_dates": d, "n_boot": len(vals),
    }
