"""
Threshold and horizon sensitivity grid for the forward volatility-regime forecast.

Sweeps the regime quantile and the forward horizon, recomputing the leakage-safe threshold per
cell so the base rate is governed by the quantile and the cells stay comparable across horizons.
Each cell reports the champion logistic regression edge over the matched naive baseline in mean
walk-forward PR-AUC, with a fold-block bootstrap interval. The (0.80, 10) cell reproduces the
headline result and anchors the grid. Writes reports/metrics/sensitivity_grid.csv and
reports/figures/sensitivity_grid.png.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
METRICS = REPO_ROOT / "reports" / "metrics"
FIG_DIR = REPO_ROOT / "reports" / "figures"

from src.config import load_config
from src.pipeline import load_data
from src.split import generate_expanding_window_splits, get_fold_indices
from src.modeling import train_logistic_regression, predict_with_model, evaluate_model
from src.targets import grid_regime_columns
from src.resampling import paired_edge_stats

RAW_PATH = REPO_ROOT / "data" / "raw" / "merged_features_full.csv"
QUANTILES = [0.70, 0.80, 0.90]
HORIZONS = [5, 10, 21]
SEED = 42

plt.rcParams.update({
    "figure.dpi": 130, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11,
})


def evaluate_cell(clean, raw, splits, features, quantile, horizon):
    """Per-fold logistic-regression and matched-naive PR-AUC for one grid cell."""
    labeled = grid_regime_columns(raw, forward_window=horizon, quantile=quantile)
    keep = ["Ticker", "Date", "grid_regime", "grid_currently_high", "grid_naive_ratio", "grid_label_end"]
    merged = clean.merge(labeled[keep], on=["Ticker", "Date"], how="left", validate="one_to_one")

    dates = merged["Date"]
    label_end = merged["grid_label_end"]
    feat_ok = merged[features].notna().all(axis=1)
    label_ok = merged["grid_regime"].notna()
    naive_ok = merged["grid_currently_high"].notna()

    lr_scores, naive_scores, fold_ids, pos, n = [], [], [], 0, 0
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(splits, 1):
        train_mask, test_mask = get_fold_indices(dates, tr_s, tr_e, te_s, te_e, label_end=label_end)
        train_mask = train_mask & label_ok & feat_ok
        test_mask = test_mask & label_ok & feat_ok & naive_ok
        if train_mask.sum() < 50 or test_mask.sum() == 0:
            continue
        y_test = merged.loc[test_mask, "grid_regime"]
        if y_test.nunique() < 2:
            continue

        model = train_logistic_regression(merged.loc[train_mask, features], merged.loc[train_mask, "grid_regime"])
        y_pred, y_proba = predict_with_model(model, merged.loc[test_mask, features])
        lr = evaluate_model(y_test, y_pred, y_proba)["pr_auc"]
        naive = evaluate_model(
            y_test,
            merged.loc[test_mask, "grid_currently_high"].to_numpy(),
            merged.loc[test_mask, "grid_naive_ratio"].to_numpy(),
        )["pr_auc"]
        lr_scores.append(lr)
        naive_scores.append(naive)
        fold_ids.append(i)
        pos += int(y_test.sum())
        n += int(test_mask.sum())

    return np.array(lr_scores), np.array(naive_scores), np.array(fold_ids), (pos / n if n else np.nan)


def run():
    """Sweep the grid, write the metrics CSV, and render the heatmap."""
    cfg = load_config()
    features = cfg["data"]["features"]
    split_cfg = cfg["split"]

    clean = load_data(REPO_ROOT / cfg["data"]["input_path"])
    raw = pd.read_csv(RAW_PATH, parse_dates=["Date"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    splits = generate_expanding_window_splits(
        dates=clean["Date"],
        train_start=split_cfg["train_start"],
        initial_train_end=split_cfg["initial_train_end"],
        test_size_days=split_cfg["test_size_days"],
        step_size_days=split_cfg["step_size_days"],
        min_train_size_days=split_cfg["min_train_size_days"],
    )
    rng = np.random.default_rng(SEED)

    rows = []
    for horizon in HORIZONS:
        for quantile in QUANTILES:
            lr, naive, fold_ids, base_rate = evaluate_cell(clean, raw, splits, features, quantile, horizon)
            diff = lr - naive
            stats = paired_edge_stats(diff, fold_ids, rng)
            rows.append({
                "target": "FwdVolRegime", "quantile": quantile, "horizon": horizon,
                "base_rate": round(base_rate, 4), "n_folds": stats["n_folds"],
                "pr_auc_lr": round(float(lr.mean()), 4), "pr_auc_naive": round(float(naive.mean()), 4),
                "edge": round(stats["mean_diff"], 4),
                "edge_ci_lo": round(stats["ci_lo"], 4), "edge_ci_hi": round(stats["ci_hi"], 4),
                "edge_p": round(stats["p"], 4),
            })
            print(f"q={quantile:.2f} H={horizon:>2}  base={base_rate:.3f}  "
                  f"edge={stats['mean_diff']:+.4f}  CI[{stats['ci_lo']:+.4f}, {stats['ci_hi']:+.4f}]  p={stats['p']:.4f}")

    out = pd.DataFrame(rows)
    METRICS.mkdir(parents=True, exist_ok=True)
    out.to_csv(METRICS / "sensitivity_grid.csv", index=False)
    _plot(out)
    anchor = out[(out["quantile"] == 0.80) & (out["horizon"] == 10)]["edge"].iloc[0]
    print(f"\nAnchor cell (0.80, 10) edge = {anchor:+.4f} (headline is +0.034)")
    print(f"Wrote {METRICS / 'sensitivity_grid.csv'} and {FIG_DIR / 'sensitivity_grid.png'}")
    return out


def _plot(out):
    """Render the edge heatmap over the quantile by horizon grid."""
    grid = out.pivot(index="horizon", columns="quantile", values="edge")
    sig = out.pivot(index="horizon", columns="quantile", values="edge_ci_lo").gt(0) | \
        out.pivot(index="horizon", columns="quantile", values="edge_ci_hi").lt(0)

    edges = grid.to_numpy()
    # Every cell here is a positive edge, so a sequential scale anchored at zero uses the
    # full color range, where a symmetric diverging map wastes its entire negative half.
    # A zero-centered diverging map is kept as a fallback if a rerun ever turns a cell
    # negative, so the sign is never misread.
    if np.nanmin(edges) < 0:
        span = float(np.nanmax(np.abs(edges)))
        cmap, norm = "RdBu", TwoSlopeNorm(vcenter=0.0, vmin=-span, vmax=span)
    else:
        cmap, norm = "Blues", Normalize(vmin=0.0, vmax=float(np.nanmax(edges)))

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(edges, cmap=cmap, norm=norm, aspect="auto")

    for r, horizon in enumerate(grid.index):
        for c, quantile in enumerate(grid.columns):
            edge = grid.loc[horizon, quantile]
            red, green, blue, _ = im.cmap(im.norm(edge))
            # Flip the label to light on dark cells so every value stays readable.
            text_color = "#111111" if 0.299 * red + 0.587 * green + 0.114 * blue > 0.55 else "#f5f5f5"
            ax.text(c, r, f"{edge:+.3f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=text_color)
            if sig.loc[horizon, quantile]:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                       edgecolor="#e0952b", linewidth=2.8))

    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels([f"{q:.2f}" for q in grid.columns])
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels([str(h) for h in grid.index])
    ax.set_xlabel("Regime quantile", fontweight="bold")
    ax.set_ylabel("Forward horizon (trading days)", fontweight="bold")
    ax.set_title("Logistic regression edge over the matched naive baseline by threshold and horizon",
                 fontweight="bold", fontsize=11)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("PR-AUC edge over matched naive", fontweight="bold")
    fig.text(0.01, -0.04, "Outlined cells: fold-block 95% interval excludes zero",
             fontsize=8.5, color="#555")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "sensitivity_grid.png")
    plt.close(fig)


if __name__ == "__main__":
    run()
