"""
Dependence-aware significance for the model comparison.

Per-fold metrics are not independent: volatility clusters in time and the tickers co-move, so
the naive standard error understates uncertainty. This tests whether each model beats the matched
naive baseline on the forward-disjoint target with a moving-block bootstrap over folds and an
AR(1) effective-sample-size adjustment, reported alongside the naive paired t-test so the
inflation is visible.

Both are gap-aware: when a fold is missing for one model, blocks and lag-1 pairs are formed only
across consecutive folds, not across non-adjacent quarters.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
METRICS = REPO_ROOT / "reports" / "metrics"

from src.config import load_config

N_BOOT = 20000
BLOCK = 4          # moving-block length over time-ordered folds
SEED = 42
METRIC = "pr_auc"
BASELINE = "MatchedNaive"


def lag1_autocorr(x, fold_ids=None):
    """
    Lag-1 autocorrelation of x, normalized by the total sum of squares.

    When fold_ids is given, only pairs of adjacent folds contribute, so gaps
    left by skipped folds are not treated as consecutive quarters. The numerator is
    rescaled by (n - 1) / n_pairs so that dropping gap-spanning pairs does not shrink
    the estimate toward zero (which would inflate the effective sample size). With no
    gaps the rescaling factor is exactly 1.
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


def moving_block_bootstrap_mean(diff, rng, n_boot=N_BOOT, block=BLOCK, fold_ids=None):
    """
    Bootstrap distribution of the mean of diff using moving blocks of consecutive folds.

    When fold_ids is given, a block may only start at a position where the next block-1
    fold ids are consecutive, preserving the serial structure the block bootstrap relies on.
    Degenerate inputs fall back to smaller blocks instead of crashing.
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


def main():
    """Run the paired model-vs-baseline tests and write significance.csv."""
    cfg = load_config()
    target = (cfg.get("data") or {}).get("target", "FwdVolRegime")

    per_fold = pd.read_csv(METRICS / "walkforward_per_fold.csv")
    sub = per_fold[per_fold["target"] == target]
    wide = sub.pivot_table(index="fold", columns="model", values=METRIC).sort_index()
    if BASELINE not in wide.columns:
        raise SystemExit(f"{BASELINE} not found for {target}")

    rng = np.random.default_rng(SEED)
    models = [m for m in ["LogisticRegression", "LightGBM", "XGBoost", "HAR"] if m in wide.columns]

    rows = []
    for m in models:
        paired = wide[[m, BASELINE]].dropna()
        diff = (paired[m] - paired[BASELINE]).to_numpy()
        fold_ids = paired.index.to_numpy()
        n = len(diff)

        if n < 2:
            warnings.warn(
                f"Only {n} paired fold(s) for {m} vs {BASELINE}, statistics undefined.",
                RuntimeWarning,
            )
            rows.append({
                "target": target, "metric": METRIC, "model": m, "vs": BASELINE,
                "mean_diff": round(float(diff.mean()), 4) if n else np.nan, "n_folds": n,
                "fold_autocorr": np.nan, "n_eff": np.nan,
                "t_naive": np.nan, "t_effN": np.nan,
                "block_boot_ci_lo": np.nan, "block_boot_ci_hi": np.nan,
                "block_boot_p": np.nan,
            })
            continue

        mean_d = diff.mean()

        rho = lag1_autocorr(diff, fold_ids=fold_ids)
        # AR(1) effective sample size in the form of Bretherton et al. (1999). Negative
        # rho estimates are clipped to zero so the adjustment can only shrink n and the
        # effective-N test stays at least as strict as the naive t.
        rho_pos = max(rho, 0.0)
        n_eff = n * (1 - rho_pos) / (1 + rho_pos)
        se_naive = diff.std(ddof=1) / np.sqrt(n)
        se_eff = diff.std(ddof=1) / np.sqrt(max(n_eff, 2))
        t_naive = mean_d / se_naive if se_naive > 0 else np.nan
        t_eff = mean_d / se_eff if se_eff > 0 else np.nan

        boot = moving_block_bootstrap_mean(diff, rng, fold_ids=fold_ids)
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        tail = min((boot <= 0).mean(), (boot >= 0).mean())
        # Cap at 1 because resampled means tying at exactly zero land in both tails,
        # and floor at 1/B because B resamples cannot resolve a smaller two-tailed p.
        p_boot = min(1.0, max(2 * tail, 1 / len(boot)))

        rows.append({
            "target": target, "metric": METRIC, "model": m, "vs": BASELINE,
            "mean_diff": round(mean_d, 4), "n_folds": n, "fold_autocorr": round(rho, 3),
            "n_eff": round(n_eff, 1),
            "t_naive": round(t_naive, 2), "t_effN": round(t_eff, 2),
            "block_boot_ci_lo": round(ci_lo, 4), "block_boot_ci_hi": round(ci_hi, 4),
            "block_boot_p": round(p_boot, 4),
        })

    out = pd.DataFrame(rows)
    out.to_csv(METRICS / "significance.csv", index=False)
    pd.set_option("display.width", 160)
    print(f"Model minus {BASELINE} on {target} {METRIC} (positive = model wins):\n")
    print(out.to_string(index=False))
    print("\nRead: the naive t treats every fold as independent; effective-N and the moving-block")
    print("bootstrap account for fold serial correlation. A block-bootstrap CI that includes 0")
    print("means the edge over the matched naive baseline is not statistically decisive.")
    return out


if __name__ == "__main__":
    main()
