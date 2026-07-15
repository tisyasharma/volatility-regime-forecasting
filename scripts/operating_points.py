"""
Alert-budget operating points for the forward volatility-regime forecast.

Using the forecast means picking a cutoff, but the class-weighted models overstate the regime
probability, so a probability threshold would mislead. Instead this flags the top N% of
ticker-days by score and reports the precision and recall that budget buys, holding the matched
naive baseline to the same budget for a like-for-like comparison.

The table lists the tradeoffs at each budget, not a tuned operating point. It is computed from
the pooled walk-forward out-of-fold predictions, so picking the best-looking row after the fact
would be selection on the test set.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
METRICS = REPO_ROOT / "reports" / "metrics"

from src.config import load_config

BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
BASELINE = "MatchedNaive"


def budget_row(df, budget, target):
    """
    Metrics for one model at one alert budget on pooled out-of-fold predictions.

    The cutoff is the (1 - budget) quantile of the model's scores, so roughly the top
    budget share of ticker-days is flagged. The pooled alert rate holds at the budget by
    construction, so the per-window alert rate and the false-alarm count are each reported
    as a min, median, and max across folds, where each fold is one test window of
    split.test_size_days trading days pooled over all tickers, showing how unevenly the
    alerts concentrate.
    """
    score = df["y_proba"].to_numpy()
    y = df["y_true"].to_numpy()
    cutoff = float(np.quantile(score, 1 - budget))
    pred = score >= cutoff

    fp_flags = pred & (y == 0)
    tp = int(np.sum(pred & (y == 1)))
    fp = int(fp_flags.sum())
    fn = int(np.sum(~pred & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else np.nan

    by_fold = df.assign(alert=pred, fp_flag=fp_flags).groupby("fold")
    fp_by_fold = by_fold["fp_flag"].sum()
    alert_rate_by_fold = by_fold["alert"].mean()

    return {
        "target": target,
        "model": df["model"].iloc[0],
        "alert_budget": budget,
        "cutoff": round(cutoff, 4),
        "alert_rate": round(float(pred.mean()), 4),
        "alert_rate_per_fold_min": round(float(alert_rate_by_fold.min()), 4),
        "alert_rate_per_fold_median": round(float(alert_rate_by_fold.median()), 4),
        "alert_rate_per_fold_max": round(float(alert_rate_by_fold.max()), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fp_per_fold_min": int(fp_by_fold.min()),
        "fp_per_fold_median": float(fp_by_fold.median()),
        "fp_per_fold_max": int(fp_by_fold.max()),
    }


def main():
    """Sweep alert budgets for the final model and the matched naive baseline."""
    cfg = load_config()
    target = (cfg.get("data") or {}).get("target", "FwdVolRegime")
    model_name = (cfg.get("evaluation") or {}).get("final_model") or "LogisticRegression"

    oof_path = METRICS / "oof_predictions.csv"
    if not oof_path.exists():
        raise SystemExit(
            f"{oof_path} not found. Run scripts/run_evaluation.py first, "
            "it writes the out-of-fold predictions this script reads."
        )

    oof = pd.read_csv(oof_path)
    sub = oof[oof["target"] == target]

    rows = []
    for model in [model_name, BASELINE]:
        df = sub[sub["model"] == model]
        if df.empty:
            raise SystemExit(f"No out-of-fold predictions for {model} on {target}.")
        for budget in BUDGETS:
            rows.append(budget_row(df, budget, target))

    out = pd.DataFrame(rows)
    out.to_csv(METRICS / "operating_points.csv", index=False)

    base_rate = float(sub[sub["model"] == model_name]["y_true"].mean())
    pd.set_option("display.width", 160)
    print(
        f"{model_name} vs {BASELINE} on {target} by alert budget "
        f"(pooled base rate {base_rate:.3f}):\n"
    )
    print(out.to_string(index=False))
    print("\nRead: cutoffs are score quantiles on the pooled walk-forward predictions, not")
    print(f"calibrated probabilities. {BASELINE}'s score is its volatility-to-threshold")
    print("ratio, thresholded at the same budget, so both columns spend the same alert")
    print("budget and differ only in how they rank the days.")
    return out


if __name__ == "__main__":
    main()
