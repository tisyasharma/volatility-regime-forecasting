"""Tests for the committed model artifact written by scripts/save_model.py.

They load the tracked file directly. The golden suite separately checks that retraining
reproduces it.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tomllib
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import InconsistentVersionWarning

from scripts.verify_data_manifest import sha256_of
from src.config import load_config
from src.modeling import MODEL_BUILDERS, predict_with_model
from src.pipeline import load_data, prepare_features_and_labels
from src.targets import add_regime_columns

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = REPO_ROOT / "models" / "artifacts" / "final_model_fwdvolregime.joblib"

# Largest allowed score difference from a fresh fit. Logistic-regression out-of-fold scores were
# measured to drift by at most 3.9e-13 between machines, and the golden suite allows them 1e-12.
SCORE_TOLERANCE = 1e-12

# The libraries that fit, solve (scipy runs the lbfgs solver) and write a logistic-regression
# artifact, so the provenance must record at least these.
REQUIRED_LIBRARIES = {"joblib", "numpy", "scikit-learn", "scipy"}


@pytest.fixture(scope="module")
def artifact() -> dict[str, Any]:
    """The committed artifact, loaded with a scikit-learn version mismatch raised as an error."""
    # Unpickling runs code. This is the repo's own tracked file, whose hash
    # tests/golden/SHA256SUMS records.
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        loaded: dict[str, Any] = joblib.load(ARTIFACT)
    return loaded


@pytest.fixture(scope="module")
def training_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """The features, labels and dates save_model.py trains on, built through the same functions."""
    cfg = load_config()
    data_cfg = cfg["data"]
    df = load_data(
        REPO_ROOT / data_cfg["input_path"],
        start_date=data_cfg.get("start_date"),
        end_date=data_cfg.get("end_date"),
        tickers=data_cfg.get("tickers"),
    )
    df = add_regime_columns(df, forward_window=cfg["evaluation"]["forward_window"])
    X, y, dates = prepare_features_and_labels(df, data_cfg["features"], data_cfg["target"])
    return X, y, dates


def test_artifact_was_built_with_the_locked_libraries(artifact: dict[str, Any]) -> None:
    """The artifact records the uv.lock version of every library that shapes or writes it."""
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
    locked = {package["name"]: package["version"] for package in lock["package"]}
    recorded = artifact["provenance"]["libraries"]
    assert recorded.keys() >= REQUIRED_LIBRARIES
    assert recorded == {name: locked[name] for name in recorded}


def test_artifact_matches_config_and_data(artifact: dict[str, Any]) -> None:
    """The model, features, target and data filters are the configured ones, on committed data."""
    cfg = load_config()
    assert artifact["model_name"] == cfg["evaluation"]["final_model"]
    assert artifact["features"] == cfg["data"]["features"]
    assert artifact["target"] == cfg["data"]["target"]
    assert list(artifact["model"]["scaler"].feature_names_in_) == artifact["features"]
    provenance = artifact["provenance"]
    assert provenance["data_path"] == cfg["data"]["input_path"]
    for key in ["tickers", "start_date", "end_date"]:
        assert provenance[key] == cfg["data"].get(key), key
    assert provenance["forward_window"] == cfg["evaluation"]["forward_window"]
    assert provenance["data_sha256"] == sha256_of(REPO_ROOT / provenance["data_path"])


def test_loaded_artifact_predicts_like_a_fresh_fit(
    artifact: dict[str, Any], training_data: tuple[pd.DataFrame, pd.Series, pd.Series]
) -> None:
    """Loading and predicting gives the labels and scores of a model trained fresh on the data."""
    X, y, dates = training_data
    provenance = artifact["provenance"]
    assert len(X) == provenance["train_rows"]
    assert provenance["train_start"] == dates.min().date().isoformat()
    assert provenance["train_end"] == dates.max().date().isoformat()

    y_pred, y_score = predict_with_model(artifact["model"], X[artifact["features"]])
    fresh = MODEL_BUILDERS[artifact["model_name"]](X, y, load_config().get("models") or {})
    fresh_pred, fresh_score = predict_with_model(fresh, X[artifact["features"]])

    assert np.array_equal(y_pred, fresh_pred)
    assert np.max(np.abs(y_score - fresh_score)) <= SCORE_TOLERANCE
    assert ((y_score >= 0) & (y_score <= 1)).all()


def test_artifact_records_the_configured_filters(tmp_path: Path) -> None:
    """save_model records the ticker and date filters it was run with, not only the null defaults.

    It runs in a copy whose config sets all three, and the training window must fall inside them.
    """
    for relative in ["src", "data/processed"]:
        shutil.copytree(REPO_ROOT / relative, tmp_path / relative)
    (tmp_path / "scripts").mkdir()
    for name in ["save_model.py", "verify_data_manifest.py"]:
        shutil.copyfile(REPO_ROOT / "scripts" / name, tmp_path / "scripts" / name)
    filters = '  start_date: "2018-01-02"\n  end_date: "2024-12-31"\n  tickers: [AAPL, MSFT]\n'
    config = (REPO_ROOT / "config.yaml").read_text()
    assert config.count("  start_date: null\n  end_date: null\n") == 1
    (tmp_path / "config.yaml").write_text(
        config.replace("  start_date: null\n  end_date: null\n", filters)
    )
    result = subprocess.run(
        [sys.executable, "scripts/save_model.py"], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # Unpickling runs code. This file was written a moment ago by the repo's own script.
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        saved: dict[str, Any] = joblib.load(tmp_path / ARTIFACT.relative_to(REPO_ROOT))
    provenance = saved["provenance"]
    assert provenance["tickers"] == ["AAPL", "MSFT"]
    assert (provenance["start_date"], provenance["end_date"]) == ("2018-01-02", "2024-12-31")
    assert "2018-01-02" <= provenance["train_start"] <= provenance["train_end"] <= "2024-12-31"


def test_make_clean_removes_no_tracked_file(tmp_path: Path) -> None:
    """Running `make clean` over a copy of the tracked tree deletes none of its files.

    Every tracked path is recreated as an empty file in a fresh repository, so the test checks
    what the recipe does rather than parsing it.
    """
    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    for relative in filter(None, listed.split("\0")):
        placeholder = tmp_path / relative
        placeholder.parent.mkdir(parents=True, exist_ok=True)
        placeholder.touch()
    shutil.copyfile(REPO_ROOT / "Makefile", tmp_path / "Makefile")

    git = ["git", "-c", "user.name=test", "-c", "user.email=test@example.com"]
    git += ["-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null"]
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "tracked tree"]):
        step = subprocess.run([*git, *args], cwd=tmp_path, capture_output=True, text=True)
        assert step.returncode == 0, step.stderr
    clean = subprocess.run(
        ["make", "--no-print-directory", "clean"], cwd=tmp_path, capture_output=True, text=True
    )
    assert clean.returncode == 0, clean.stderr
    deleted = subprocess.run(
        [*git, "ls-files", "--deleted"], cwd=tmp_path, capture_output=True, text=True, check=True
    ).stdout
    assert deleted == ""
