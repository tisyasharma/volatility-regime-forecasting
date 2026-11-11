"""Checks on what the golden tests compare against, fast enough to run without regenerating.

The committed metrics, the reference out-of-fold run and the saved model are pinned by
tests/golden/SHA256SUMS, so overwriting them, for example by running the pipeline in the working
tree, fails here instead of turning the golden tests into a comparison of the output with itself.
"""

from __future__ import annotations

import importlib.metadata
import subprocess
import tomllib
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from scripts.verify_data_manifest import LINE_PATTERN, verify_manifest
from tests.golden.support import (
    COMMITTED_METRICS,
    METRICS,
    MODEL_ARTIFACT,
    OOF,
    REFERENCE_MANIFEST,
    REFERENCE_OOF,
    REPO_ROOT,
    describe_csv_difference,
    read_csv_exact,
)

# The libraries whose arithmetic produces the golden numbers. Any version other than the locked
# one can change them, so a mismatch is reported by name, which a failing golden comparison
# would not do.
MODEL_LIBRARIES = ["lightgbm", "numpy", "pandas", "scikit-learn", "scipy", "xgboost"]

# run_evaluation records no PR-AUC for models that emit only hard labels.
HARD_LABEL_MODELS = {"Majority", "Persistence"}

# run_evaluation predicts the positive class above these scores: MatchedNaive's score is a ratio
# to the threshold, and every other score is 0.5 at the decision boundary.
CUTOFF = {"MatchedNaive": 1.0}
DEFAULT_CUTOFF = 0.5


def _listed(manifest: Path) -> list[str]:
    """The paths a manifest lists, parsed with the checker's own line pattern."""
    lines = manifest.read_text().splitlines()
    return [match["path"] for line in lines if (match := LINE_PATTERN.match(line))]


def test_golden_references_match_manifest() -> None:
    """The committed metrics, reference OOF and saved model hash to what SHA256SUMS records."""
    assert verify_manifest(REPO_ROOT / REFERENCE_MANIFEST, REPO_ROOT, None) == []
    listed = set(_listed(REPO_ROOT / REFERENCE_MANIFEST))
    assert listed == {f"{METRICS}/{name}" for name in COMMITTED_METRICS} | {
        str(REFERENCE_OOF),
        str(MODEL_ARTIFACT),
    }


def test_every_committed_metrics_file_is_compared() -> None:
    """Every tracked metrics CSV is in COMMITTED_METRICS, which the golden tests compare."""
    tracked = subprocess.run(
        ["git", "ls-files", str(METRICS)], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    assert sorted(Path(path).name for path in tracked) == sorted(COMMITTED_METRICS)
    on_disk = {p.name for p in (REPO_ROOT / METRICS).glob("*.csv")} - {OOF}
    assert on_disk == set(COMMITTED_METRICS)


def test_reference_oof_reproduces_committed_per_fold_metrics() -> None:
    """The reference OOF gives every committed per-fold metric: n_test and the base rate, PR-AUC
    to 1e-12, and precision, recall and F1 exactly.

    Precision, recall and F1 come from the hard predictions, so they cover the hard-label models
    too. A different run passes only if every fold's AP and confusion counts are unchanged. Any
    other change to the file is caught by SHA256SUMS.
    """
    reference = read_csv_exact(REPO_ROOT / REFERENCE_OOF)
    per_fold = read_csv_exact(REPO_ROOT / METRICS / "walkforward_per_fold.csv")
    groups = reference.groupby(["target", "fold", "model"], sort=False)
    assert len(groups) == len(per_fold)

    committed: dict[Hashable, dict[Hashable, Any]] = {
        (row["target"], row["fold"], row["model"]): row for row in per_fold.to_dict("records")
    }
    for key, rows in groups:
        row = committed[key]
        assert len(rows) == row["n_test"], key
        assert rows["y_true"].mean() == pytest.approx(row["positive_rate"], abs=1e-12), key
        if row["model"] in HARD_LABEL_MODELS:
            assert np.isnan(row["pr_auc"]), key
        else:
            pr_auc = average_precision_score(rows["y_true"], rows["y_proba"])
            assert pr_auc == pytest.approx(row["pr_auc"], abs=1e-12), key
        predicted = rows["y_proba"] > CUTOFF.get(row["model"], DEFAULT_CUTOFF)
        for name, metric in [
            ("precision", precision_score),
            ("recall", recall_score),
            ("f1", f1_score),
        ]:
            assert metric(rows["y_true"], predicted, zero_division=0) == row[name], (key, name)


def test_installed_libraries_match_the_lock() -> None:
    """The model libraries are the locked versions that produced the committed metrics."""
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
    locked = {package["name"]: package["version"] for package in lock["package"]}
    mismatched = [
        f"{name} {importlib.metadata.version(name)} installed, {locked[name]} locked"
        for name in MODEL_LIBRARIES
        if importlib.metadata.version(name) != locked[name]
    ]
    assert not mismatched, f"{mismatched}. Run `uv sync --locked`."
    assert locked["xgboost"] == "3.2.0", "the committed XGBoost rows were produced by 3.2.0"


def test_hashed_files_are_never_line_ending_converted() -> None:
    """Git checks out every file either manifest hashes unconverted, on every platform."""
    paths = [*_listed(REPO_ROOT / "data/SHA256SUMS"), *_listed(REPO_ROOT / REFERENCE_MANIFEST)]
    result = subprocess.run(
        ["git", "check-attr", "text", *paths],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == [f"{path}: text: unset" for path in paths]


def test_difference_report_names_changed_cell(tmp_path: Path) -> None:
    """A changed value, a format-only change, and changed columns or shape are each reported."""
    expected, actual = tmp_path / "expected.csv", tmp_path / "actual.csv"
    expected.write_text("model,pr_auc\nA,0.3226\nB,0.3604\n")
    actual.write_text("model,pr_auc\nA,0.3226\nB,0.3605\n")
    assert describe_csv_difference(expected, actual) == "row 1 pr_auc: 0.3604 -> 0.3605"

    actual.write_text("model,pr_auc\nA,0.3226\nB,0.36040\n")
    assert "formatting changed" in describe_csv_difference(expected, actual)
    actual.write_text("model,pr_auc,n\nA,0.3226,1\nB,0.3604,1\n")
    assert describe_csv_difference(expected, actual).startswith("columns changed")
    actual.write_text("model,pr_auc\nA,0.3226\n")
    assert describe_csv_difference(expected, actual).startswith("shape changed")
    assert describe_csv_difference(expected, expected) == ""


@given(
    values=st.lists(st.integers(min_value=0, max_value=10**6), min_size=1, max_size=20),
    data=st.data(),
)
def test_any_changed_value_is_reported_at_its_row(
    tmp_path_factory: pytest.TempPathFactory, values: list[int], data: st.DataObject
) -> None:
    """Changing any one value of a column is reported at exactly that row with both values."""
    row = data.draw(st.integers(min_value=0, max_value=len(values) - 1))
    root = tmp_path_factory.mktemp("diff")
    changed = [*values]
    changed[row] += 1
    expected, actual = root / "expected.csv", root / "actual.csv"
    expected.write_text("x\n" + "\n".join(map(str, values)) + "\n")
    actual.write_text("x\n" + "\n".join(map(str, changed)) + "\n")
    assert describe_csv_difference(expected, actual) == (
        f"row {row} x: {values[row]} -> {changed[row]}"
    )
