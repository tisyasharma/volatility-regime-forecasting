"""Paths and comparison helpers shared by the golden tests and their fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS = Path("reports/metrics")
GOLDEN = Path("tests/golden")

COMMITTED_METRICS = [
    "walkforward_per_fold.csv",
    "walkforward_summary.csv",
    "significance.csv",
    "significance_crosssectional.csv",
    "operating_points.csv",
    "sensitivity_grid.csv",
]
OOF = "oof_predictions.csv"
REFERENCE_OOF = GOLDEN / "reference_oof.csv.gz"
MODEL_ARTIFACT = Path("models/artifacts/final_model_fwdvolregime.joblib")
REFERENCE_MANIFEST = GOLDEN / "SHA256SUMS"


def read_csv_exact(path: Path) -> pd.DataFrame:
    """Read a CSV with correctly rounded float parsing, which is the same on every platform."""
    return pd.read_csv(path, float_precision="round_trip")


def describe_csv_difference(expected: Path, actual: Path, max_cells: int = 8) -> str:
    """Describe how two CSV files differ, or return an empty string when they are identical.

    Names the first differing cells with both values, so a failure says what moved rather than
    only that the bytes changed.
    """
    if expected.read_bytes() == actual.read_bytes():
        return ""
    old, new = read_csv_exact(expected), read_csv_exact(actual)
    if list(old.columns) != list(new.columns):
        return f"columns changed: {list(old.columns)} -> {list(new.columns)}"
    if old.shape != new.shape:
        return f"shape changed: {old.shape} -> {new.shape}"

    cells: list[str] = []
    for column in old.columns:
        before, after = old[column], new[column]
        changed = ~((before == after) | (before.isna() & after.isna()))
        for row in np.flatnonzero(changed.to_numpy())[: max_cells - len(cells)]:
            cells.append(f"row {row} {column}: {before.iloc[row]} -> {after.iloc[row]}")
        if len(cells) >= max_cells:
            break
    if not cells:
        return "values are equal but the bytes differ, so the formatting changed"
    return "; ".join(cells)
