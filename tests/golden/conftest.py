"""Regenerate the committed metrics, the OOF predictions and the saved model in a scratch copy."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.golden.support import METRICS, REPO_ROOT

# run_evaluation goes first because significance and operating_points read the
# oof_predictions.csv it writes. sensitivity_grid builds its own folds. save_model runs last,
# because with evaluation.final_model unset it reads the summary run_evaluation writes.
STAGES = ["run_evaluation", "significance", "operating_points", "sensitivity_grid", "save_model"]
OUTPUT_TAIL_CHARS = 4000


@pytest.fixture(scope="session")
def regenerated(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Copy the working tree's code, config and data, run every stage, return the copy's root.

    The scripts resolve their paths from their own location, so they read and write inside the
    copy. The metrics directory is created empty rather than copied, so a stage that silently
    writes nothing leaves its file missing instead of passing on the committed copy.
    """
    root = tmp_path_factory.mktemp("golden")
    for directory in ["src", "scripts", "data"]:
        shutil.copytree(REPO_ROOT / directory, root / directory)
    shutil.copyfile(REPO_ROOT / "config.yaml", root / "config.yaml")
    (root / METRICS).mkdir(parents=True)

    for stage in STAGES:
        result = subprocess.run(
            [sys.executable, f"scripts/{stage}.py"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            pytest.fail(
                f"{stage}.py exited {result.returncode}\n"
                f"stdout:\n{result.stdout[-OUTPUT_TAIL_CHARS:]}\n"
                f"stderr:\n{result.stderr[-OUTPUT_TAIL_CHARS:]}"
            )
    return root
