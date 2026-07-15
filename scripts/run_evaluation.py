"""
Walk-forward evaluation driver.

Runs every baseline and model through the same expanding-window walk-forward for each target
in config.yaml and writes per-fold and aggregate metrics to reports/metrics/. Source of truth
for the numbers reported in the README, model card, and docs/METHODOLOGY.md.

The trading-day-exact purge that keeps training labels out of the test period lives in
src/split.py (get_fold_indices with label_end).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.pipeline import load_data
from src.split import generate_expanding_window_splits, get_fold_indices
from src.modeling import (
    MODEL_BUILDERS,
    predict_with_model,
    evaluate_model,
    add_har_features,
    train_har,
    har_forecast,
    threshold_to_prediction,
    HAR_FEATURES,
)
from src.targets import add_regime_columns, matched_naive_predict

# Models that emit hard 0/1 labels only. Average precision needs a graded score to rank
# by, so their pr_auc is recorded as NaN rather than a fabricated number.
HARD_LABEL_MODELS = {"Majority", "Persistence"}

# Continuous volatility each target's spike is defined on (used by the HAR forecaster).
TARGET_CONT_VOL = {"NextVolSpike": "har_next_vol", "FwdVolRegime": "forward_vol"}
# Column giving the date of the last return in each target's forward window (for the purge).
TARGET_LABEL_END = {"NextVolSpike": "label_end_nextday", "FwdVolRegime": "label_end_forward"}


def run():
    """Evaluate all models on all targets and write metrics CSVs."""
    cfg = load_config()
    data_cfg = cfg["data"]
    split_cfg = cfg["split"]
    # `or {}` also covers a section header whose keys are all commented out, which YAML
    # parses as None rather than an empty mapping.
    model_cfg = cfg.get("models") or {}
    eval_cfg = cfg.get("evaluation") or {}
    pos_label = eval_cfg.get("pos_label", 1)
    targets = eval_cfg.get("targets", [data_cfg["target"]])
    feature_cols = data_cfg["features"]

    method = split_cfg.get("method", "expanding")
    if method != "expanding":
        raise NotImplementedError(f"split.method={method!r}, only 'expanding' is implemented")

    df = load_data(
        REPO_ROOT / data_cfg["input_path"],
        start_date=data_cfg.get("start_date"),
        end_date=data_cfg.get("end_date"),
        tickers=data_cfg.get("tickers"),
    )
    df = add_har_features(df)
    df = add_regime_columns(df, forward_window=eval_cfg.get("forward_window", 10))
    dates = df["Date"]

    splits = generate_expanding_window_splits(
        dates=dates,
        train_start=split_cfg["train_start"],
        initial_train_end=split_cfg["initial_train_end"],
        test_size_days=split_cfg["test_size_days"],
        step_size_days=split_cfg["step_size_days"],
        min_train_size_days=split_cfg["min_train_size_days"],
    )
    if not splits:
        raise SystemExit(
            f"Split configuration produced no folds: train_start={split_cfg['train_start']}, "
            f"initial_train_end={split_cfg['initial_train_end']}, "
            f"test_size_days={split_cfg['test_size_days']}, "
            f"step_size_days={split_cfg['step_size_days']}, "
            f"min_train_size_days={split_cfg['min_train_size_days']}, "
            f"data spans {dates.min().date()} to {dates.max().date()}"
        )

    feat_ok = df[feature_cols].notna().all(axis=1)
    thr_ok = df["vol_threshold"].notna()
    har_feat_ok = df[HAR_FEATURES].notna().all(axis=1)
    rows, oof_rows = [], []

    for target_col in targets:
        cont_col = TARGET_CONT_VOL[target_col]
        target_ok = df[target_col].notna()
        cont_ok = df[cont_col].notna()
        label_end = df[TARGET_LABEL_END[target_col]]

        for fold_idx, (tr_start, tr_end, te_start, te_end) in enumerate(splits, 1):
            train_mask, test_mask = get_fold_indices(
                dates, tr_start, tr_end, te_start, te_end, label_end=label_end
            )
            train_mask = train_mask & target_ok & feat_ok & thr_ok
            test_mask = test_mask & target_ok & feat_ok & thr_ok

            if train_mask.sum() < 50 or test_mask.sum() == 0:
                continue
            y_test = df.loc[test_mask, target_col]
            if y_test.nunique() < 2:
                continue

            X_train = df.loc[train_mask, feature_cols]
            y_train = df.loc[train_mask, target_col]
            X_test = df.loc[test_mask, feature_cols]

            def record(name, y_true, y_pred, y_proba, n_test):
                """Score one model on this fold and append its metric and OOF rows."""
                proba_for_metrics = None if name in HARD_LABEL_MODELS else y_proba
                metrics = evaluate_model(y_true, y_pred, proba_for_metrics, pos_label=pos_label)
                rows.append({
                    "target": target_col, "fold": fold_idx, "model": name,
                    "test_start": te_start, "test_end": te_end, "n_test": n_test,
                    "positive_rate": float(np.mean(np.asarray(y_true) == pos_label)),
                    **metrics,
                })
                oof_rows.append(pd.DataFrame({
                    "target": target_col, "fold": fold_idx, "model": name,
                    "test_start": te_start, "y_true": np.asarray(y_true), "y_proba": np.asarray(y_proba),
                }))

            for name, builder in MODEL_BUILDERS.items():
                model = builder(X_train, y_train, model_cfg)
                y_pred, y_proba = predict_with_model(model, X_test)
                record(name, y_test, y_pred, y_proba, int(test_mask.sum()))

            # Matched naive baseline: already above the label's own threshold today.
            naive_pred, naive_score = matched_naive_predict(df.loc[test_mask])
            record("MatchedNaive", y_test, naive_pred, naive_score, int(test_mask.sum()))

            # HAR econometric benchmark: forecast the target volatility, threshold identically.
            har_train_mask = train_mask & cont_ok & har_feat_ok
            har_test_mask = test_mask & har_feat_ok
            if har_train_mask.sum() >= 50 and har_test_mask.sum() > 0:
                har_model = train_har(df.loc[har_train_mask, HAR_FEATURES], df.loc[har_train_mask, cont_col])
                forecast = har_forecast(har_model, df.loc[har_test_mask, HAR_FEATURES])
                thr = df.loc[har_test_mask, "vol_threshold"].to_numpy()
                y_pred, y_proba = threshold_to_prediction(forecast, thr, har_model["scale"])
                y_test_har = df.loc[har_test_mask, target_col]
                if y_test_har.nunique() >= 2:
                    record("HAR", y_test_har, y_pred, y_proba, int(har_test_mask.sum()))

    if not rows:
        raise SystemExit(
            f"No usable folds were produced from {len(splits)} split(s). Check the split "
            f"configuration (train_start={split_cfg['train_start']}, "
            f"initial_train_end={split_cfg['initial_train_end']}, "
            f"test_size_days={split_cfg['test_size_days']}) against the data range "
            f"{dates.min().date()} to {dates.max().date()}."
        )

    per_fold = pd.DataFrame(rows)
    oof = pd.concat(oof_rows, ignore_index=True)

    metric_cols = ["precision", "recall", "f1", "pr_auc"]
    summary = (
        per_fold.groupby(["target", "model"])[metric_cols].agg(["mean", "std"]).round(4)
    )
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    summary = summary.reset_index()

    out_dir = REPO_ROOT / "reports" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(out_dir / "walkforward_per_fold.csv", index=False)
    summary.to_csv(out_dir / "walkforward_summary.csv", index=False)
    oof.to_csv(out_dir / "oof_predictions.csv", index=False)

    for target_col in targets:
        sub = summary[summary["target"] == target_col]
        print(f"\n{target_col} (mean across folds)")
        cols = ["model"] + [f"{m}_mean" for m in metric_cols]
        print(sub[cols].to_string(index=False))
    return per_fold, summary


if __name__ == "__main__":
    run()
