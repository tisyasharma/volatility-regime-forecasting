"""
Train the final model on all available data and persist it for reuse.

Which model gets saved is governed by evaluation.final_model in config.yaml. When that key
is null, the best learned model by walk-forward PR-AUC on the config target is chosen from
reports/metrics/walkforward_summary.csv, so the artifact always reflects the evaluation
rather than a hardcoded pick. The model is trained on the full clean dataset and saved with
the feature list so it can be reloaded and applied to fresh rows via predict_with_model.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.pipeline import load_data, prepare_features_and_labels
from src.modeling import MODEL_BUILDERS
from src.targets import add_regime_columns

LEARNED_MODELS = ["LogisticRegression", "LightGBM", "XGBoost"]


def pick_final_model(cfg):
    """
    Return the model name to persist.

    Uses evaluation.final_model when set. Otherwise picks the learned model with the
    highest mean walk-forward PR-AUC on the config target and prints the choice together
    with its significance row so the selection is reviewable.
    """
    configured = (cfg.get("evaluation") or {}).get("final_model")
    if configured:
        if configured not in MODEL_BUILDERS:
            raise SystemExit(f"evaluation.final_model={configured!r} is not a known model")
        print(f"Final model pinned in config: {configured}")
        return configured

    summary_path = REPO_ROOT / "reports" / "metrics" / "walkforward_summary.csv"
    if not summary_path.exists():
        raise SystemExit("No walkforward_summary.csv found, run `make evaluate` first "
                         "or pin evaluation.final_model in config.yaml.")
    summary = pd.read_csv(summary_path)
    target = cfg["data"]["target"]
    sub = summary[(summary["target"] == target) & summary["model"].isin(LEARNED_MODELS)]
    if sub.empty:
        raise SystemExit(f"No learned-model rows for target {target!r} in the summary.")
    best = sub.sort_values("pr_auc_mean", ascending=False).iloc[0]
    print(f"Final model selected by walk-forward PR-AUC on {target}: "
          f"{best['model']} (pr_auc_mean={best['pr_auc_mean']:.4f})")

    sig_path = REPO_ROOT / "reports" / "metrics" / "significance.csv"
    if sig_path.exists():
        sig = pd.read_csv(sig_path)
        row = sig[sig["model"] == best["model"]]
        if not row.empty:
            print("Significance vs matched naive baseline:")
            print(row.to_string(index=False))
    return best["model"]


def main():
    """Train the selected final model on the full dataset and persist the artifact."""
    cfg = load_config()

    data_cfg = cfg["data"]
    model_cfg = cfg.get("models") or {}
    forward_window = (cfg.get("evaluation") or {}).get("forward_window", 10)
    model_name = pick_final_model(cfg)

    df = load_data(
        REPO_ROOT / data_cfg["input_path"],
        start_date=data_cfg.get("start_date"),
        end_date=data_cfg.get("end_date"),
        tickers=data_cfg.get("tickers"),
    )
    df = add_regime_columns(df, forward_window=forward_window)
    X, y, _ = prepare_features_and_labels(df, data_cfg["features"], data_cfg["target"])

    model = MODEL_BUILDERS[model_name](X, y, model_cfg)

    out_dir = REPO_ROOT / "models" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_name": model_name,
        "model": model,
        "features": data_cfg["features"],
        "target": data_cfg["target"],
    }
    # Filename follows the config target so retargeting the project cannot silently
    # leave a stale artifact name behind.
    out_path = out_dir / f"final_model_{data_cfg['target'].lower()}.joblib"
    joblib.dump(artifact, out_path)
    print(f"Saved {out_path} ({model_name}) trained on {len(X)} rows, "
          f"{len(data_cfg['features'])} features")


if __name__ == "__main__":
    main()
