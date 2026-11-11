"""
Train the final model on all available data and persist it for reuse.

Which model gets saved is set by evaluation.final_model in config.yaml, which pins
LogisticRegression. When that key is null, the learned model with the highest mean
walk-forward PR-AUC on the config target is taken from reports/metrics/walkforward_summary.csv.
The model is trained on every configured clean row with a complete forward label and saved
with its feature list, so it can be reloaded and applied to fresh rows via predict_with_model.
The artifact also records what produced it: the training data's path and SHA-256, the
configured ticker and date filters, the training rows and window, the forward window and the
library versions.
"""

import importlib.metadata
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
from scripts.verify_data_manifest import sha256_of

LEARNED_MODELS = ["LogisticRegression", "LightGBM", "XGBoost"]
# The libraries whose versions can change a fitted model (scipy runs logistic regression's
# lbfgs solver) plus joblib, which writes the file. All are fixed by uv.lock. Python is not
# recorded, because the lock fixes only its minor version and the patch differs by machine.
PROVENANCE_LIBRARIES = ["joblib", "lightgbm", "numpy", "pandas", "scikit-learn", "scipy", "xgboost"]


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
    X, y, dates = prepare_features_and_labels(df, data_cfg["features"], data_cfg["target"])

    model = MODEL_BUILDERS[model_name](X, y, model_cfg)

    out_dir = REPO_ROOT / "models" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_name": model_name,
        "model": model,
        "features": data_cfg["features"],
        "target": data_cfg["target"],
        "provenance": {
            "data_path": data_cfg["input_path"],
            "data_sha256": sha256_of(REPO_ROOT / data_cfg["input_path"]),
            "tickers": data_cfg.get("tickers"),
            "start_date": data_cfg.get("start_date"),
            "end_date": data_cfg.get("end_date"),
            "train_rows": len(X),
            "train_start": dates.min().date().isoformat(),
            "train_end": dates.max().date().isoformat(),
            "forward_window": forward_window,
            "libraries": {
                name: importlib.metadata.version(name) for name in PROVENANCE_LIBRARIES
            },
        },
    }
    # Filename follows the config target so retargeting the project cannot silently
    # leave a stale artifact name behind.
    out_path = out_dir / f"final_model_{data_cfg['target'].lower()}.joblib"
    joblib.dump(artifact, out_path)
    print(f"Saved {out_path} ({model_name}) trained on {len(X)} rows, "
          f"{len(data_cfg['features'])} features")


if __name__ == "__main__":
    main()
