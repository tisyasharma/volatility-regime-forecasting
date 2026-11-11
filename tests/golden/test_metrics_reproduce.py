"""Golden tests: regenerating the pipeline reproduces the committed metrics.

The six committed reports/metrics/*.csv reproduce byte for byte on macOS arm64 with the locked
packages. That was checked on an M3 Pro, and CI checks it on GitHub's macos-15 runners on every
push. reports/metrics/oof_predictions.csv is untracked. It is compared with the committed
reference run, tests/golden/reference_oof.csv.gz, within per-model tolerances. LightGBM and
XGBoost rows were measured to differ on x86 even for identical inputs, which is why CI runs on
arm64 macOS. The saved model is retrained too. Its metadata must match the committed artifact
exactly and its scores within the logistic-regression tolerance below.
"""

from __future__ import annotations

import platform
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import InconsistentVersionWarning

from tests.golden.support import (
    METRICS,
    MODEL_ARTIFACT,
    OOF,
    REFERENCE_OOF,
    REPO_ROOT,
    describe_csv_difference,
    read_csv_exact,
)

pytestmark = [pytest.mark.golden, pytest.mark.timeout(900)]

OOF_KEYS = ["target", "fold", "model", "test_start", "Date", "Ticker", "y_true"]

# Largest allowed absolute difference in an out-of-fold score, per model. Majority,
# Persistence and MatchedNaive are deterministic arithmetic, and LightGBM reproduced exactly on
# every arm64 run, so all four must match exactly. The largest measured drift was 3.9e-13 for
# LogisticRegression (x86, 2.9e-14 on arm64), so it gets 1e-12, and 2.1e-15 for HAR, so it gets
# 1e-14. XGBoost outputs float32 scores, which moved by up to 1.8e-7 between an M3 Pro and
# GitHub's macos-14 and macos-15 runners in October 2026, so it gets 1e-6.
OOF_SCORE_TOLERANCE = {
    "Majority": 0.0,
    "Persistence": 0.0,
    "MatchedNaive": 0.0,
    "LightGBM": 0.0,
    "LogisticRegression": 1e-12,
    "HAR": 1e-14,
    "XGBoost": 1e-6,
}

X86_MACHINES = {"x86_64", "AMD64"}

# The discrete fitted state of the logistic-regression bundle save_model.py writes. Its float
# arrays are compared through the scores they produce, since BLAS rounding can differ in the last
# digit between machines.
EXACT_FITTED_STATE = {
    "scaler": ["n_samples_seen_", "feature_names_in_"],
    "model": ["classes_", "n_iter_"],
}


def _assert_reproduces(regenerated: Path, name: str) -> None:
    """Fail unless the regenerated metrics file is byte-identical to the committed one."""
    committed = REPO_ROOT / METRICS / name
    produced = regenerated / METRICS / name
    assert produced.is_file(), f"{name} was not written by the pipeline"
    difference = describe_csv_difference(committed, produced)
    machine = platform.machine()
    if difference and machine in X86_MACHINES:
        difference += ". LightGBM and XGBoost rows were measured to differ on x86"
    elif difference and machine != "arm64":
        difference += f". The committed metrics come from macOS arm64, {machine} is unmeasured"
    assert not difference, f"{name} differs from the committed file: {difference}"


def test_walkforward_per_fold_reproduces(regenerated: Path) -> None:
    """Per-fold metrics for every model, target and fold match the committed file."""
    _assert_reproduces(regenerated, "walkforward_per_fold.csv")


def test_walkforward_summary_reproduces(regenerated: Path) -> None:
    """Mean and std of each metric across folds match the committed summary."""
    _assert_reproduces(regenerated, "walkforward_summary.csv")


def test_significance_reproduces(regenerated: Path) -> None:
    """The fold-block bootstrap table matches the committed file."""
    _assert_reproduces(regenerated, "significance.csv")


def test_significance_crosssectional_reproduces(regenerated: Path) -> None:
    """The date-block bootstrap table, the primary inference, matches the committed file."""
    _assert_reproduces(regenerated, "significance_crosssectional.csv")


def test_operating_points_reproduce(regenerated: Path) -> None:
    """The alert-budget operating points match the committed file."""
    _assert_reproduces(regenerated, "operating_points.csv")


def test_sensitivity_grid_reproduces(regenerated: Path) -> None:
    """The quantile by horizon sensitivity grid matches the committed file."""
    _assert_reproduces(regenerated, "sensitivity_grid.csv")


def test_oof_predictions_match_reference_run(regenerated: Path) -> None:
    """Every out-of-fold row and score matches the reference run within its model's tolerance."""
    reference = read_csv_exact(REPO_ROOT / REFERENCE_OOF)
    produced = read_csv_exact(regenerated / METRICS / OOF)
    assert list(produced.columns) == list(reference.columns)
    assert produced[OOF_KEYS].equals(reference[OOF_KEYS]), "OOF rows, keys or labels changed"
    assert set(reference["model"]) == set(OOF_SCORE_TOLERANCE)

    for model, tolerance in OOF_SCORE_TOLERANCE.items():
        rows = reference["model"] == model
        expected = reference.loc[rows, "y_proba"].to_numpy()
        actual = produced.loc[rows, "y_proba"].to_numpy()
        worst = float(np.max(np.abs(actual - expected)))
        assert worst <= tolerance, f"{model} scores moved by up to {worst:.3g} (> {tolerance})"


def test_oof_carries_date_and_ticker(regenerated: Path) -> None:
    """Every OOF row carries its Date, which the date-block bootstrap resamples, and its Ticker,
    which pairs it with the matching naive row."""
    oof = pd.read_csv(regenerated / METRICS / OOF)
    assert {"Date", "Ticker"}.issubset(oof.columns)
    assert oof["Date"].notna().all()
    assert oof["Ticker"].notna().all()


def test_every_metrics_file_was_written_by_this_run(regenerated: Path) -> None:
    """Each regenerated file is newer than the copied config.

    The fixture creates the metrics directory empty, which is what keeps the committed files out.
    This catches a fixture that copied them in with their timestamps, as copy2 and copytree do. A
    copy that sets a new timestamp would pass, so it is a guard, not a proof.
    """
    copied_at = (regenerated / "config.yaml").stat().st_mtime_ns
    produced = sorted((regenerated / METRICS).iterdir())
    assert {p.name for p in produced} >= {OOF}
    for path in produced:
        assert path.stat().st_mtime_ns >= copied_at, f"{path.name} predates this run"


def _load_artifact(path: Path) -> dict[str, Any]:
    """Load a saved model, failing if it was pickled by a different scikit-learn version."""
    # Unpickling runs code. Both files are this repo's own: the committed one, whose hash
    # tests/golden/SHA256SUMS records, and the one this session's pipeline just wrote.
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        artifact: dict[str, Any] = joblib.load(path)
    return artifact


def test_saved_model_reproduces(regenerated: Path) -> None:
    """Retraining the final model reproduces the committed artifact.

    The keys, metadata, provenance, classifier parameters and discrete fitted state must match
    exactly. The scores on every committed clean row must match within the logistic-regression
    out-of-fold tolerance.
    """
    committed = _load_artifact(REPO_ROOT / MODEL_ARTIFACT)
    produced = _load_artifact(regenerated / MODEL_ARTIFACT)
    assert committed["model_name"] == "LogisticRegression", "this comparison covers LR only"
    assert produced.keys() == committed.keys()
    for key in ["model_name", "features", "target", "provenance"]:
        assert produced[key] == committed[key], f"{key} changed"
    for part, attributes in EXACT_FITTED_STATE.items():
        for attribute in attributes:
            expected = getattr(committed["model"][part], attribute)
            actual = getattr(produced["model"][part], attribute)
            assert np.array_equal(actual, expected), f"{part}.{attribute} changed"
    assert produced["model"]["model"].get_params() == committed["model"]["model"].get_params()

    X = pd.read_csv(REPO_ROOT / committed["provenance"]["data_path"])[committed["features"]]
    committed_scores, produced_scores = (
        artifact["model"]["model"].predict_proba(artifact["model"]["scaler"].transform(X))[:, 1]
        for artifact in (committed, produced)
    )
    worst = float(np.max(np.abs(produced_scores - committed_scores)))
    tolerance = OOF_SCORE_TOLERANCE["LogisticRegression"]
    assert worst <= tolerance, f"scores moved by up to {worst:.3g} (> {tolerance})"
