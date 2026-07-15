"""
Render report figures from the metrics written by run_evaluation.py and the source data.

Reads reports/metrics/*.csv (target-keyed) and data/processed/merged_features_clean.csv,
writes PNGs to reports/figures/. Run run_evaluation.py first.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
METRICS_DIR = REPO_ROOT / "reports" / "metrics"
FIG_DIR = REPO_ROOT / "reports" / "figures"

from src.config import load_config

BLUE = "#2a78d6"
AQUA = "#1baf7a"
ORANGE = "#eb6834"
# The learned models carry the categorical hues. The reference rules (majority, persistence,
# matched naive, HAR) render in neutral grays so they read as benchmarks and recede.
MODEL_ORDER = ["Majority", "Persistence", "MatchedNaive", "HAR",
               "LogisticRegression", "LightGBM", "XGBoost"]
MODEL_COLORS = {
    "Majority": "#b8b6ae",
    "Persistence": "#9c9a92",
    "MatchedNaive": "#52514e",
    "HAR": "#8f8d86",
    "LogisticRegression": BLUE,
    "LightGBM": AQUA,
    "XGBoost": ORANGE,
}
FORWARD = "FwdVolRegime"
NEXTDAY = "NextVolSpike"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 11,
})


def _summary():
    """Read the walk-forward summary metrics."""
    return pd.read_csv(METRICS_DIR / "walkforward_summary.csv")


def _order(models):
    """Sort model names into the canonical display order, unknowns last."""
    present = [m for m in MODEL_ORDER if m in models]
    return present + [m for m in models if m not in present]


def target_reversal():
    """Compare PR-AUC across models. The matched naive baseline scores highest on the next-day target and the learned models score highest on the forward target."""
    s = _summary()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    panels = [(NEXTDAY, "Next-day spike (overlapping-window target)"),
              (FORWARD, "Next-10-day regime (forward-disjoint target)")]
    omitted_all = set()
    for ax, (target, title) in zip(axes, panels):
        # Hard-label baselines carry no ranking score, so their pr_auc is NaN and they
        # are omitted from this panel rather than plotted as fabricated bars.
        sub = s[(s["target"] == target) & s["pr_auc_mean"].notna()]
        omitted_all |= set(s.loc[s["target"] == target, "model"]) - set(sub["model"])
        order = _order(sub["model"].tolist())
        vals = [sub.loc[sub["model"] == m, "pr_auc_mean"].iloc[0] for m in order]
        errs = [sub.loc[sub["model"] == m, "pr_auc_std"].iloc[0] for m in order]
        colors = [MODEL_COLORS.get(m, "#777") for m in order]
        ax.bar(range(len(order)), vals, yerr=errs, capsize=2, color=colors,
               alpha=0.85, edgecolor="darkgrey")
        naive = sub.loc[sub["model"] == "MatchedNaive", "pr_auc_mean"]
        if len(naive):
            ax.axhline(naive.iloc[0], color="#383835", linestyle="--", linewidth=1.2,
                       label="Matched naive baseline")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=9)
        ax.set_xlabel("Model", fontweight="bold")
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.legend(fontsize=9, loc="upper right")
    axes[0].set_ylabel("PR-AUC (mean across folds)", fontweight="bold")
    if omitted_all:
        fig.text(0.125, -0.18,
                 f"{', '.join(sorted(omitted_all))} omitted, no ranking score for hard 0/1 labels",
                 fontsize=8.5, color="#555")
    fig.suptitle("PR-AUC by model on the next-day and forward volatility targets",
                 fontweight="bold", fontsize=12)
    fig.savefig(FIG_DIR / "target_reversal.png")
    plt.close(fig)


def f1_by_fold(target=FORWARD):
    """Plot per-fold F1 over time for the key models, marking the COVID crash."""
    per_fold = pd.read_csv(METRICS_DIR / "walkforward_per_fold.csv")
    per_fold = per_fold[per_fold["target"] == target].copy()
    per_fold["test_start"] = pd.to_datetime(per_fold["test_start"])
    fig, ax = plt.subplots(figsize=(10, 5))
    for model in ["MatchedNaive", "LogisticRegression", "XGBoost"]:
        sub = per_fold[per_fold["model"] == model].sort_values("test_start")
        if sub.empty:
            continue
        ax.plot(sub["test_start"], sub["f1"], marker="o", markersize=3,
                label=model, color=MODEL_COLORS.get(model), linewidth=1.4)
    ax.axvspan(pd.Timestamp("2020-02-15"), pd.Timestamp("2020-06-01"),
               color="#9aa0a6", alpha=0.25, label="COVID crash")
    ax.set_xlabel("Test window start")
    ax.set_ylabel("F1 (per test window)")
    ax.set_title(f"Per-fold F1 over time ({target})", fontweight="bold")
    ax.legend(fontsize=9)
    fig.savefig(FIG_DIR / "f1_by_fold.png")
    plt.close(fig)


def autocorrelation():
    """Plot mean autocorrelation of returns vs absolute returns across tickers."""
    df = pd.read_csv(REPO_ROOT / load_config()["data"]["input_path"], parse_dates=["Date"])
    df["AbsReturn"] = df["Return"].abs()
    lags = range(1, 11)

    def mean_acf(col):
        per_ticker = []
        for _, g in df.groupby("Ticker"):
            s = g.sort_values("Date")[col].dropna()
            per_ticker.append([s.autocorr(lag=k) for k in lags])
        return np.nanmean(per_ticker, axis=0)

    # Absolute daily returns are the standard clustering measure (Cont 2001). Adjacent values
    # do not overlap, so the autocorrelation is not inflated by window smoothing.
    acf_ret = mean_acf("Return")
    acf_absret = mean_acf("AbsReturn")
    x = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.2, acf_ret, 0.4, label="Daily return", color=BLUE,
           alpha=0.85, edgecolor="darkgrey")
    ax.bar(x + 0.2, acf_absret, 0.4, label="|Daily return|", color=ORANGE,
           alpha=0.85, edgecolor="darkgrey")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xlabel("Lag (trading days)")
    ax.set_ylabel("Mean autocorrelation across tickers")
    ax.set_title("Mean autocorrelation of returns and |returns| across 14 tickers")
    ax.legend()
    fig.savefig(FIG_DIR / "autocorrelation.png")
    plt.close(fig)


def main():
    """Render the three metrics-derived figures into reports/figures/.

    The SHAP and calibration figures in the same directory are rendered by
    notebooks/03_explainability.ipynb, not by this script.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    target_reversal()
    f1_by_fold()
    autocorrelation()
    print(f"Figures written to {FIG_DIR}")
    for name in ("target_reversal.png", "f1_by_fold.png", "autocorrelation.png"):
        print(f"  {name}")


if __name__ == "__main__":
    main()
