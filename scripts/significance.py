"""
Dependence-aware significance for the model comparison.

Two resamplings test each model against the matched naive baseline on the forward-disjoint
target. The fold-block bootstrap over the time-ordered folds corrects for serial correlation
only. The date-block bootstrap resamples whole trading dates, each carrying its cross-section
of tickers, so it also corrects for the co-movement of the 14 names. The date-block result is
the primary one, the fold-block result is kept so the inflation from ignoring the cross-section
is visible.
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
from src.resampling import paired_edge_stats, date_block_bootstrap_edge

SEED = 42
METRIC = "pr_auc"
BASELINE = "MatchedNaive"
MODELS = ["LogisticRegression", "LightGBM", "XGBoost", "HAR"]
DATE_BLOCK = 21      # moving-block length in trading days for the cross-sectional bootstrap
N_BOOT_XS = 3000     # date-block resamples, kept modest since average precision is recomputed


def _round(value, digits):
    """Round, passing NaN through untouched."""
    return np.nan if value is None or (isinstance(value, float) and np.isnan(value)) else round(value, digits)


def fold_block_table(target):
    """Fold-block bootstrap of the per-fold PR-AUC edge, the serial-only comparison."""
    per_fold = pd.read_csv(METRICS / "walkforward_per_fold.csv")
    sub = per_fold[per_fold["target"] == target]
    wide = sub.pivot_table(index="fold", columns="model", values=METRIC).sort_index()
    if BASELINE not in wide.columns:
        raise SystemExit(f"{BASELINE} not found for {target}")

    rng = np.random.default_rng(SEED)
    rows = []
    for m in [m for m in MODELS if m in wide.columns]:
        paired = wide[[m, BASELINE]].dropna()
        diff = (paired[m] - paired[BASELINE]).to_numpy()
        fold_ids = paired.index.to_numpy()
        if len(diff) < 2:
            warnings.warn(
                f"Only {len(diff)} paired fold(s) for {m} vs {BASELINE}, statistics undefined.",
                RuntimeWarning,
            )
        s = paired_edge_stats(diff, fold_ids, rng)
        rows.append({
            "target": target, "metric": METRIC, "model": m, "vs": BASELINE,
            "mean_diff": _round(s["mean_diff"], 4), "n_folds": s["n_folds"],
            "fold_autocorr": _round(s["fold_autocorr"], 3), "n_eff": _round(s["n_eff"], 1),
            "t_naive": _round(s["t_naive"], 2), "t_effN": _round(s["t_effN"], 2),
            "block_boot_ci_lo": _round(s["ci_lo"], 4), "block_boot_ci_hi": _round(s["ci_hi"], 4),
            "block_boot_p": _round(s["p"], 4),
        })
    return pd.DataFrame(rows)


def date_block_table(target):
    """Date-block bootstrap of the pooled PR-AUC edge, the cross-sectional comparison."""
    oof = pd.read_csv(METRICS / "oof_predictions.csv", parse_dates=["Date"])
    if not {"Date", "Ticker"}.issubset(oof.columns):
        raise SystemExit(
            "oof_predictions.csv has no Date and Ticker columns. Re-run scripts/run_evaluation.py, "
            "which now carries them so the cross-sectional bootstrap can cluster by date."
        )
    sub = oof[oof["target"] == target]
    base = sub[sub["model"] == BASELINE][["Date", "Ticker", "y_proba"]].rename(
        columns={"y_proba": "score_naive"}
    )
    if base.empty:
        raise SystemExit(f"No {BASELINE} out-of-fold predictions for {target}.")

    rng = np.random.default_rng(SEED)
    rows = []
    for m in [m for m in MODELS if m in sub["model"].unique()]:
        mod = sub[sub["model"] == m][["Date", "Ticker", "fold", "y_true", "y_proba"]].rename(
            columns={"y_proba": "score_model"}
        )
        joined = mod.merge(base, on=["Date", "Ticker"], how="inner")
        if joined.empty or joined["y_true"].nunique() < 2:
            continue
        r = date_block_bootstrap_edge(
            joined["Date"].to_numpy(), joined["fold"].to_numpy(), joined["y_true"].to_numpy(),
            joined["score_model"].to_numpy(), joined["score_naive"].to_numpy(),
            rng, n_boot=N_BOOT_XS, block=DATE_BLOCK,
        )
        rows.append({
            "target": target, "metric": METRIC, "model": m, "vs": BASELINE, "method": "date_block",
            "edge": round(r["edge"], 4), "ci_lo": round(r["ci_lo"], 4), "ci_hi": round(r["ci_hi"], 4),
            "p": round(r["p"], 4), "n_dates": r["n_dates"], "n_rows": len(joined), "n_boot": r["n_boot"],
        })
    return pd.DataFrame(rows)


def main():
    """Write the fold-block and date-block significance tables for the config target."""
    target = (load_config().get("data") or {}).get("target", "FwdVolRegime")

    fold = fold_block_table(target)
    fold.to_csv(METRICS / "significance.csv", index=False)

    date = date_block_table(target)
    date.to_csv(METRICS / "significance_crosssectional.csv", index=False)

    pd.set_option("display.width", 160)
    print(f"Model minus {BASELINE} on {target} {METRIC} (positive = model wins).\n")
    print("Fold-block bootstrap (serial correlation across folds only):")
    print(fold.to_string(index=False))
    print(f"\nDate-block bootstrap (date clusters, block {DATE_BLOCK} trading days, "
          f"{N_BOOT_XS} resamples), the cross-sectional-aware primary:")
    print(date.to_string(index=False))
    print("\nRead: the date-block bootstrap resamples whole trading dates, so the 14 co-moving")
    print("tickers count as one cross-section rather than 14 independent draws. It is the primary")
    print("reading, and its larger p-value than the fold-block test is the honest cost of that.")
    return fold, date


if __name__ == "__main__":
    main()
