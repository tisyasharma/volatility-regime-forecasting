"""Tests for scripts/verify_vol_threshold.py.

Most run the script as a subprocess in a copy of the repo, so the real data/ is never at risk.
The tests of failures while writing call main in process, reading the real data but writing only
under tmp_path. A few call check_clean_labels directly on small frames.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import verify_vol_threshold
from scripts.verify_data_manifest import sha256_of

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = "data/raw/merged_features_full.csv"
CLEAN = "data/processed/merged_features_clean.csv"
OUT_DIR = Path("build/vol_threshold")


def _copy_project(root: Path) -> Path:
    """Copy what the script needs into root and return root."""
    shutil.copytree(REPO_ROOT / "src", root / "src")
    (root / "scripts").mkdir()
    shutil.copyfile(
        REPO_ROOT / "scripts/verify_vol_threshold.py", root / "scripts/verify_vol_threshold.py"
    )
    shutil.copyfile(REPO_ROOT / "config.yaml", root / "config.yaml")
    shutil.copytree(REPO_ROOT / "data", root / "data")
    return root


def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the copied script from the copy's root."""
    return subprocess.run(
        [sys.executable, "scripts/verify_vol_threshold.py", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


def _snapshot(root: Path) -> dict[str, str]:
    """SHA-256 of every file under root/data, so any added, removed or changed file shows."""
    data = root / "data"
    return {p.relative_to(data).as_posix(): sha256_of(p) for p in data.rglob("*") if p.is_file()}


def _read(path: Path) -> pd.DataFrame:
    """Read a CSV with every float exactly as written."""
    return pd.read_csv(path, float_precision="round_trip")


def _edit_csv(path: Path, edit: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
    """Rewrite a CSV after applying edit to it, reading it exactly so nothing else changes."""
    edit(_read(path)).to_csv(path, index=False)


def _point_config_at(root: Path, clean: str) -> None:
    """Make the copy's config read the clean dataset from clean instead."""
    config = root / "config.yaml"
    config.write_text(config.read_text().replace(CLEAN, clean))


def _staging_dirs(out_dir: Path) -> list[Path]:
    """Any staging directory a run left next to out_dir."""
    return list(out_dir.parent.glob(f".{out_dir.name}.*"))


def _assert_nothing_written(root: Path, before: dict[str, str]) -> None:
    """data/ is unchanged, and neither the output directory nor a staging directory exists."""
    assert _snapshot(root) == before
    assert not (root / OUT_DIR).exists()
    assert _staging_dirs(root / OUT_DIR) == []


@pytest.fixture(scope="module")
def successful_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, str], str]:
    """One default run in a fresh copy: the root, the data/ snapshot before it, and its output."""
    root = _copy_project(tmp_path_factory.mktemp("threshold"))
    before = _snapshot(root)
    result = _run(root)
    assert result.returncode == 0, result.stdout + result.stderr
    return root, before, result.stdout


def test_writes_new_files_and_leaves_datasets_untouched(
    successful_run: tuple[Path, dict[str, str], str],
) -> None:
    """Both files are written to the output directory, and no file under data/ changes."""
    root, before, stdout = successful_run
    assert _snapshot(root) == before
    for relative in [RAW, CLEAN]:
        assert (root / OUT_DIR / Path(relative).name).is_file()
    assert _staging_dirs(root / OUT_DIR) == []
    assert "verified: stored NextVolSpike reproduced exactly" in stdout


@pytest.mark.golden
def test_output_is_byte_identical_to_committed_datasets(
    successful_run: tuple[Path, dict[str, str], str],
) -> None:
    """Re-deriving the threshold reproduces the committed datasets exactly, and says so.

    This holds on Apple silicon only. On x86, seven derived thresholds differ in the last bit, so
    this is a golden test.
    """
    root, _, stdout = successful_run
    for relative in [RAW, CLEAN]:
        written = root / OUT_DIR / Path(relative).name
        assert written.read_bytes() == (REPO_ROOT / relative).read_bytes(), relative
        assert f"identical to {relative}" in stdout


def test_reports_when_output_differs_from_committed(tmp_path: Path) -> None:
    """A stale threshold in the clean file is replaced by raw's, and the change is reported."""
    root = _copy_project(tmp_path)

    def stale_threshold(frame: pd.DataFrame) -> pd.DataFrame:
        """Double the first stored threshold."""
        thresholds = frame["vol_threshold"].to_numpy(dtype=float, copy=True)
        thresholds[0] *= 2
        frame["vol_threshold"] = thresholds
        return frame

    _edit_csv(root / CLEAN, stale_threshold)
    stale = _read(root / CLEAN).loc[0]
    result = _run(root)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"differs from {CLEAN}" in result.stdout
    raw = _read(root / OUT_DIR / Path(RAW).name).set_index(["Ticker", "Date"])
    written = _read(root / OUT_DIR / Path(CLEAN).name).loc[0]
    derived = raw.loc[(stale["Ticker"], stale["Date"]), "vol_threshold"]
    assert written["vol_threshold"] == derived
    assert written["vol_threshold"] != stale["vol_threshold"]


def test_values_pass_through_unchanged(tmp_path: Path) -> None:
    """A full-precision column the script never recomputes is written back exactly.

    pandas' default float parser misreads most of these values, some by thousands of units in
    the last place, so a run that used it would change them while every label still verified.
    """
    root = _copy_project(tmp_path)

    def full_precision_returns(frame: pd.DataFrame) -> pd.DataFrame:
        """Rescale Return slightly, so most of its values need 16 or 17 significant digits."""
        frame["Return"] = frame["Return"] * (1 + 1e-9)
        return frame

    for relative in [RAW, CLEAN]:
        _edit_csv(root / relative, full_precision_returns)
    result = _run(root)
    assert result.returncode == 0, result.stdout + result.stderr
    for relative in [RAW, CLEAN]:
        edited = _read(root / relative).sort_values(["Ticker", "Date"])["Return"]
        written = _read(root / OUT_DIR / Path(relative).name)["Return"]
        assert np.array_equal(edited.to_numpy(), written.to_numpy(), equal_nan=True), relative


def test_raw_rows_in_any_order_verify(tmp_path: Path) -> None:
    """The raw rows are sorted before the threshold is derived, so their order doesn't matter.

    The expanding threshold runs down each ticker's dates, so deriving it in file order from a
    shuffled file would give wrong thresholds and fail the label check.
    """
    root = _copy_project(tmp_path)
    _edit_csv(root / RAW, lambda frame: frame.sample(frac=1.0, random_state=0))
    result = _run(root)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"differs from {RAW}" in result.stdout
    keys = _read(root / OUT_DIR / Path(RAW).name)[["Ticker", "Date"]]
    assert keys.equals(keys.sort_values(["Ticker", "Date"]).reset_index(drop=True))


def _flip_first_label(frame: pd.DataFrame) -> pd.DataFrame:
    """Flip the first non-missing NextVolSpike label."""
    labels = frame["NextVolSpike"].to_numpy(dtype=float, copy=True)
    first_labelled = int(np.flatnonzero(~np.isnan(labels))[0])
    labels[first_labelled] = 1 - labels[first_labelled]
    frame["NextVolSpike"] = labels
    return frame


def test_label_mismatch_aborts_without_writing(tmp_path: Path) -> None:
    """A raw label the threshold cannot reproduce stops the run before anything is written."""
    root = _copy_project(tmp_path)
    _edit_csv(root / RAW, _flip_first_label)
    before = _snapshot(root)
    result = _run(root)
    assert result.returncode == 1
    assert "Nothing written: raw-side check failed" in result.stdout
    _assert_nothing_written(root, before)


def test_clean_label_mismatch_aborts_without_writing(tmp_path: Path) -> None:
    """A clean label that disagrees with the threshold stops the run before writing."""
    root = _copy_project(tmp_path)
    _edit_csv(root / CLEAN, _flip_first_label)
    before = _snapshot(root)
    result = _run(root)
    assert result.returncode == 1
    assert "Nothing written: clean-side check failed on 1 rows" in result.stdout
    _assert_nothing_written(root, before)


def test_clean_label_on_ticker_last_row_is_checked(tmp_path: Path) -> None:
    """A wrong label on a ticker's last clean row stops the run.

    That row's next day is only in the raw file, so it can't be re-derived from the clean file
    alone. Its label has to match the raw file's.
    """
    root = _copy_project(tmp_path)

    def flip_last_clean_label(frame: pd.DataFrame) -> pd.DataFrame:
        """Flip the label on the first ticker's last row."""
        row = frame.index[frame["Ticker"] == frame["Ticker"].iloc[0]][-1]
        frame.loc[row, "NextVolSpike"] = 1 - frame.loc[row, "NextVolSpike"]
        return frame

    _edit_csv(root / CLEAN, flip_last_clean_label)
    before = _snapshot(root)
    result = _run(root)
    assert result.returncode == 1
    assert "Nothing written: clean-side check failed on 1 rows" in result.stdout
    _assert_nothing_written(root, before)


@pytest.mark.parametrize("position", ["warm_up", "ticker_last"])
def test_label_on_undefined_row_aborts(tmp_path: Path, position: str) -> None:
    """A label where none can be derived is a mismatch rather than a wildcard.

    No label exists before the threshold has its warm-up history, or on a ticker's last day.
    """
    root = _copy_project(tmp_path)

    def invent_label(frame: pd.DataFrame) -> pd.DataFrame:
        """Put a 0 label on the first ticker's first or last raw row, which has none."""
        rows = frame.index[frame["Ticker"] == frame["Ticker"].iloc[0]]
        row = rows[0] if position == "warm_up" else rows[-1]
        assert np.isnan(frame.loc[row, "NextVolSpike"])
        frame.loc[row, "NextVolSpike"] = 0.0
        return frame

    _edit_csv(root / RAW, invent_label)
    before = _snapshot(root)
    result = _run(root)
    assert result.returncode == 1
    assert "Nothing written: raw-side check failed" in result.stdout
    assert "stored label on 1 rows" in result.stdout
    _assert_nothing_written(root, before)


def test_missing_label_where_one_is_defined_aborts(tmp_path: Path) -> None:
    """A blank label on a row whose label can be derived is a mismatch, not a wildcard.

    The row is a ticker's last clean row, blanked in both files. The clean side then sees NaN
    on both sides and has no next day to re-derive from, so only the raw-side check can catch it.
    """
    root = _copy_project(tmp_path)
    clean = _read(root / CLEAN)
    ticker = clean["Ticker"].iloc[0]
    date = clean.loc[clean["Ticker"] == ticker, "Date"].iloc[-1]

    def blank_label(frame: pd.DataFrame) -> pd.DataFrame:
        """Blank the label of that one row."""
        row = (frame["Ticker"] == ticker) & (frame["Date"] == date)
        assert row.sum() == 1
        assert frame.loc[row, "NextVolSpike"].notna().all()
        frame["NextVolSpike"] = frame["NextVolSpike"].astype(float)
        frame.loc[row, "NextVolSpike"] = np.nan
        return frame

    _edit_csv(root / RAW, blank_label)
    _edit_csv(root / CLEAN, blank_label)
    before = _snapshot(root)
    result = _run(root)
    assert result.returncode == 1
    assert "Nothing written: raw-side check failed" in result.stdout
    assert "stored label on 1 rows" in result.stdout
    _assert_nothing_written(root, before)


@pytest.mark.parametrize("relative", [RAW, CLEAN], ids=["raw", "clean"])
def test_repeated_key_aborts(tmp_path: Path, relative: str) -> None:
    """A repeated (Ticker, Date) in either file stops the run with the key named.

    The copy sits right after the original and differs in RSI, so it repeats the key without
    repeating the row. In the raw file it would otherwise shift the ticker's next-day volatility
    and be reported as label drift.
    """
    root = _copy_project(tmp_path)
    ticker, date = _read(root / relative).loc[1000, ["Ticker", "Date"]]

    def repeat_row(frame: pd.DataFrame) -> pd.DataFrame:
        """Insert a copy of row 1000, with a different RSI, right after it."""
        copy = frame.iloc[[1000]].assign(RSI=frame.iloc[1000]["RSI"] + 1)
        return pd.concat([frame.iloc[:1001], copy, frame.iloc[1001:]])

    _edit_csv(root / relative, repeat_row)
    before = _snapshot(root)
    result = _run(root)
    assert result.returncode == 1
    name = Path(relative).name
    expected = f"Nothing written: {name} repeats 1 (Ticker, Date) keys, the first {ticker} {date}"
    assert expected in result.stdout
    _assert_nothing_written(root, before)


@pytest.mark.parametrize("change", ["interior_gap", "extra_final_day", "unknown_date", "swap"])
def test_clean_rows_must_be_the_labelled_raw_rows(tmp_path: Path, change: str) -> None:
    """Each ticker's clean rows must be its raw rows from the first threshold to the day before
    its last, in date order. A dropped, added, unknown or reordered row stops the run.

    This check runs before the label checks and names the problem, which they would otherwise
    miss or report as a label mismatch.
    """
    root = _copy_project(tmp_path)
    clean = _read(root / CLEAN)
    first = clean.loc[0]
    if change == "interior_gap":
        edited = clean.drop(index=50)
        expected = [f"lacks 1 of them, the first {clean.at[50, 'Ticker']} {clean.at[50, 'Date']}"]
    elif change == "extra_final_day":
        raw = _read(root / RAW)
        final = raw[raw["Date"] == raw["Date"].max()]
        edited = pd.concat([clean, final])
        expected = [f"has {len(final)} rows outside them, the first AAPL {raw['Date'].max()}"]
    elif change == "unknown_date":
        edited = clean.copy()
        edited.loc[0, "Date"] = "1999-01-04"
        expected = [
            f"lacks 1 of them, the first {first['Ticker']} {first['Date']}",
            f"has 1 rows outside them, the first {first['Ticker']} 1999-01-04",
        ]
    else:
        order = list(range(len(clean)))
        order[10], order[11] = 11, 10
        edited = clean.iloc[order]
        expected = ["the clean file's dates are out of order within a ticker"]
    edited.to_csv(root / CLEAN, index=False)
    before = _snapshot(root)
    result = _run(root)
    assert result.returncode == 1
    assert "Nothing written: " in result.stdout
    for part in expected:
        assert part in result.stdout
    _assert_nothing_written(root, before)


def test_clean_check_rejects_missing_threshold() -> None:
    """A clean frame with a blank threshold fails rather than comparing against NaN.

    Without the check, NextVolSpike 0 against a NaN threshold would compare as a match.
    """
    merged = pd.DataFrame(
        {
            "Ticker": ["AAPL", "AAPL"],
            "Date": ["2020-01-02", "2020-01-03"],
            "RollingVol": [0.2, 0.3],
            "vol_threshold": [np.nan, 0.25],
            "NextVolSpike": [0, 0],
        }
    )
    raw = merged[["Ticker", "Date", "NextVolSpike"]]
    with pytest.raises(verify_vol_threshold.VerificationError, match="1 clean rows have no"):
        verify_vol_threshold.check_clean_labels(merged, raw)


def _three_clean_rows() -> pd.DataFrame:
    """One ticker's three clean rows, whose labels agree with the next day's vol.

    The last row has no next day here, so its label can only be checked against raw.
    """
    return pd.DataFrame(
        {
            "Ticker": ["AAPL", "AAPL", "AAPL"],
            "Date": ["2020-01-02", "2020-01-03", "2020-01-06"],
            "RollingVol": [0.20, 0.30, 0.10],
            "vol_threshold": [0.25, 0.25, 0.25],
            "NextVolSpike": [1.0, 0.0, 1.0],
        }
    )


@pytest.mark.parametrize("stored", [0.0, np.nan], ids=["wrong_label", "missing_label"])
def test_clean_label_contradicting_its_own_threshold_is_caught(stored: float) -> None:
    """A label the raw file agrees with still fails when the clean file's own data contradicts it.

    The first row's next-day vol, 0.30, is above its threshold of 0.25, so its label must be 1.
    """
    clean = _three_clean_rows()
    assert verify_vol_threshold.check_clean_labels(clean, clean) == 3

    clean.loc[0, "NextVolSpike"] = stored
    raw = clean[["Ticker", "Date", "NextVolSpike"]]
    with pytest.raises(verify_vol_threshold.VerificationError, match="failed on 1 rows"):
        verify_vol_threshold.check_clean_labels(clean, raw)


@pytest.mark.parametrize(
    ("clean_label", "raw_label"),
    [(np.nan, 1.0), (0.0, np.nan)],
    ids=["blank_in_clean", "blank_in_raw"],
)
def test_last_row_label_blank_on_one_side_is_caught(clean_label: float, raw_label: float) -> None:
    """On a ticker's last clean row, a label that is blank in only one file is a mismatch.

    That row has no next day in the clean file, so the comparison with raw is its only check.
    """
    clean = _three_clean_rows()
    assert verify_vol_threshold.check_clean_labels(clean, clean) == 3

    raw = clean[["Ticker", "Date", "NextVolSpike"]].copy()
    clean.loc[2, "NextVolSpike"] = clean_label
    raw.loc[2, "NextVolSpike"] = raw_label
    with pytest.raises(verify_vol_threshold.VerificationError, match="failed on 1 rows"):
        verify_vol_threshold.check_clean_labels(clean, raw)


def test_interior_label_contradicting_raw_is_caught() -> None:
    """A label the clean file's own data supports still fails when raw's label differs (F2).

    The first label and the next day's vol are both changed, so the clean file's re-derivation
    agrees with the wrong label. Only the comparison with raw, which runs on every row and not
    just where the clean file can't re-derive, catches it.
    """
    raw = _three_clean_rows()[["Ticker", "Date", "NextVolSpike"]]
    clean = _three_clean_rows()
    clean.loc[0, "NextVolSpike"] = 0.0
    clean.loc[1, "RollingVol"] = 0.20
    with pytest.raises(verify_vol_threshold.VerificationError, match="failed on 1 rows"):
        verify_vol_threshold.check_clean_labels(clean, raw)


@pytest.mark.parametrize(
    "out_dir",
    ["data/processed", "data/new", "build/../data/new", "linked"],
    ids=["dataset_dir", "new_subdir", "parent_path", "symlink"],
)
def test_refuses_output_inside_data_dir(tmp_path: Path, out_dir: str) -> None:
    """An output directory that resolves under data/ is refused with exit 2 and never created."""
    root = _copy_project(tmp_path)
    (root / "linked").symlink_to(root / "data/processed")
    before = _snapshot(root)
    result = _run(root, "--out-dir", out_dir)
    assert result.returncode == 2
    assert "never writes to" in result.stdout
    assert _snapshot(root) == before
    assert not (root / "data/new").exists()


def test_case_variant_of_the_data_dir_is_refused(tmp_path: Path) -> None:
    """On a disk that ignores case, DATA/raw is data/raw, so it is refused and nothing changes.

    On a case-sensitive disk DATA/raw is a separate new directory, which is allowed. Either way
    data/ is untouched.
    """
    root = _copy_project(tmp_path)
    ignores_case = (root / "DATA").exists()
    before = _snapshot(root)
    result = _run(root, "--out-dir", "DATA/raw")
    assert _snapshot(root) == before
    if ignores_case:
        assert result.returncode == 2
        assert "never writes to" in result.stdout
    else:
        assert result.returncode == 0, result.stdout + result.stderr


def test_clean_input_outside_the_repo_is_reported_by_absolute_path(tmp_path: Path) -> None:
    """A configured input outside the repo is still verified and reported, not a traceback."""
    root = _copy_project(tmp_path / "repo")
    outside = tmp_path / "elsewhere" / Path(CLEAN).name
    outside.parent.mkdir()
    shutil.move(root / CLEAN, outside)
    _point_config_at(root, str(outside))
    result = _run(root)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines()[-1].endswith(f" {outside}")


def test_accepts_sibling_of_data_dir(tmp_path: Path) -> None:
    """A directory whose name merely starts with data is not inside data/."""
    root = _copy_project(tmp_path)
    result = _run(root, "--out-dir", "data_review")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / "data_review" / Path(CLEAN).name).is_file()


@pytest.mark.parametrize("existing", ["empty_dir", "holds_outputs", "file", "symlink"])
def test_existing_out_dir_is_refused(tmp_path: Path, existing: str) -> None:
    """A directory given with --out-dir must be new, so nothing already there can be lost.

    That includes files with the output names, which need not be this script's.
    """
    root = _copy_project(tmp_path)
    target = root / "keep"
    mine = [target / Path(relative).name for relative in [RAW, CLEAN]]
    if existing == "file":
        target.write_text("mine\n")
    elif existing == "symlink":
        (root / "elsewhere").mkdir()
        mine = [root / "elsewhere" / Path(CLEAN).name]
        target.symlink_to(root / "elsewhere")
    else:
        target.mkdir()
    if existing in ("holds_outputs", "symlink"):
        for path in mine:
            path.write_text("mine\n")
    before = _snapshot(root)
    result = _run(root, "--out-dir", "keep")
    assert result.returncode == 2
    assert "already exists, so give a new directory" in result.stdout
    assert _snapshot(root) == before
    assert _staging_dirs(target) == []
    if existing == "file":
        assert target.read_text() == "mine\n"
    elif existing == "empty_dir":
        assert list(target.iterdir()) == []
    else:
        assert all(path.read_text() == "mine\n" for path in mine)


def test_case_variant_of_an_existing_out_dir_is_refused(tmp_path: Path) -> None:
    """On a disk that ignores case, KEEP is the existing keep/, so it is refused and kept.

    On a case-sensitive disk KEEP is a separate new directory, which is allowed.
    """
    root = _copy_project(tmp_path)
    mine = root / "keep" / Path(CLEAN).name
    mine.parent.mkdir()
    mine.write_text("mine\n")
    ignores_case = (root / "KEEP").exists()
    result = _run(root, "--out-dir", "KEEP")
    assert mine.read_text() == "mine\n"
    if ignores_case:
        assert result.returncode == 2
        assert "already exists" in result.stdout
    else:
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("configured", ["direct", "through_symlink", "other_case"])
def test_default_out_dir_holding_an_input_is_not_replaced(tmp_path: Path, configured: str) -> None:
    """When the configured clean file sits inside build/vol_threshold, the run is refused.

    Replacing the directory would delete the input. The paths are compared as files, so an input
    reached through a symlink, or through a case variant on a disk that ignores case, is found
    too. On a case-sensitive disk the case variant is a separate directory, so the run goes ahead.
    """
    root = _copy_project(tmp_path)
    name = Path(CLEAN).name
    moved = root / OUT_DIR / name
    moved.parent.mkdir(parents=True)
    shutil.move(root / CLEAN, moved)
    refused = True
    if configured == "through_symlink":
        (root / "linked").symlink_to(moved.parent)
        _point_config_at(root, f"linked/{name}")
    elif configured == "other_case":
        variant = Path(str(OUT_DIR).upper())
        if not (root / variant).exists():
            refused = False
            (root / variant).mkdir(parents=True)
            moved = Path(shutil.move(moved, root / variant / name))
        _point_config_at(root, f"{variant}/{name}")
    else:
        _point_config_at(root, f"{OUT_DIR}/{name}")
    original = moved.read_bytes()
    result = _run(root)
    assert moved.read_bytes() == original
    if refused:
        assert result.returncode == 2
        assert "holds an input, so it is not replaced" in result.stdout
    else:
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("kind", ["symlink", "file"])
def test_default_out_dir_that_is_not_a_directory_is_kept(tmp_path: Path, kind: str) -> None:
    """A symlink or a file at build/vol_threshold is refused, and a symlink is never followed."""
    root = _copy_project(tmp_path)
    target = root / "elsewhere"
    target.mkdir()
    (target / Path(CLEAN).name).write_text("mine\n")
    (root / OUT_DIR).parent.mkdir(parents=True)
    if kind == "symlink":
        (root / OUT_DIR).symlink_to(target)
    else:
        (root / OUT_DIR).write_text("mine\n")
    result = _run(root)
    assert result.returncode == 2
    assert "is not a plain directory, so it is not replaced" in result.stdout
    assert (root / OUT_DIR).is_symlink() == (kind == "symlink")
    assert (target / Path(CLEAN).name).read_text() == "mine\n"


def test_previous_output_is_replaced_without_following_its_links(tmp_path: Path) -> None:
    """A symlink inside the previous build/vol_threshold goes with it, never written through."""
    root = _copy_project(tmp_path)
    link = root / OUT_DIR / Path(RAW).name
    link.parent.mkdir(parents=True)
    link.symlink_to(root / "data" / "SHA256SUMS")
    before = _snapshot(root)
    result = _run(root)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _snapshot(root) == before
    assert not link.is_symlink()
    assert link.is_file()


def test_failed_run_leaves_no_earlier_output(tmp_path: Path) -> None:
    """The previous build/vol_threshold goes first, so a failed run leaves nothing that looks
    verified."""
    root = _copy_project(tmp_path)
    (root / OUT_DIR).mkdir(parents=True)
    for relative in [RAW, CLEAN]:
        (root / OUT_DIR / Path(relative).name).write_text("from an earlier run")
    _edit_csv(root / RAW, _flip_first_label)
    before = _snapshot(root)
    result = _run(root)
    assert result.returncode == 1
    assert "Nothing written: raw-side check failed" in result.stdout
    _assert_nothing_written(root, before)


def test_undeletable_previous_output_stops_the_run(tmp_path: Path) -> None:
    """A previous build/vol_threshold that can't be removed stops the run with the error named."""
    root = _copy_project(tmp_path)
    out_dir = root / OUT_DIR
    out_dir.mkdir(parents=True)
    stale = out_dir / Path(CLEAN).name
    stale.write_text("from an earlier run")
    before = _snapshot(root)
    out_dir.chmod(0o555)
    try:
        result = _run(root)
    finally:
        out_dir.chmod(0o755)
    assert result.returncode == 1
    assert "Nothing written: could not remove the previous" in result.stdout
    assert "PermissionError" in result.stdout
    assert stale.read_text() == "from an earlier run"
    assert _snapshot(root) == before


@pytest.mark.parametrize(
    "clashing_name", [Path(RAW).name, Path(RAW).name.upper()], ids=["same_name", "other_case"]
)
def test_outputs_with_the_same_name_are_refused(tmp_path: Path, clashing_name: str) -> None:
    """A clean file named like the raw file is refused, also when only the case differs.

    The two outputs would be one file on a disk that ignores case, so case is ignored on every
    disk.
    """
    root = _copy_project(tmp_path)
    clashing = Path(CLEAN).with_name(clashing_name)
    shutil.move(root / CLEAN, root / clashing)
    _point_config_at(root, str(clashing))
    result = _run(root)
    assert result.returncode == 2
    assert "the two outputs would have the same file name" in result.stdout


class _FailOnCall:
    """Wrap a function so that one chosen call raises instead of running."""

    def __init__(self, function: Callable[..., object], failing_call: int, error: BaseException):
        """Raise error on call number failing_call, counting from 1, and pass the others on."""
        self.function = function
        self.failing_call = failing_call
        self.error = error
        self.calls = 0

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Count the call, then raise or call the wrapped function."""
        self.calls += 1
        if self.calls == self.failing_call:
            raise self.error
        return self.function(*args, **kwargs)


def _fail_method_call(
    monkeypatch: pytest.MonkeyPatch, owner: type, name: str, call: int, error: BaseException
) -> None:
    """Make call number call of owner.name raise error."""
    failing = _FailOnCall(getattr(owner, name), call, error)

    def method(self: object, *args: object, **kwargs: object) -> object:
        """A plain function, so it binds as a method where a callable object would not."""
        return failing(self, *args, **kwargs)

    monkeypatch.setattr(owner, name, method)


@pytest.mark.parametrize(
    ("target", "error"),
    [
        pytest.param(
            "verify_written",
            verify_vol_threshold.VerificationError("corrupted on disk"),
            id="verification_fails",
        ),
        pytest.param(
            "verify_written", ValueError("Error tokenizing data"), id="reread_parse_error"
        ),
        pytest.param(
            "verify_written", FileNotFoundError("the written file is gone"), id="reread_os_error"
        ),
        pytest.param("to_csv", OSError("disk full"), id="second_write_fails"),
        pytest.param("rename", OSError("device busy"), id="rename_fails"),
    ],
)
def test_failed_write_or_verification_publishes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    target: str,
    error: Exception,
) -> None:
    """A failure after writing has started returns 1 with its cause, and no output appears.

    Each of these exception types is reported rather than raised, and the staging directory is
    removed. The write and the re-read fail on their second call, after the raw file was done.
    """
    if target == "to_csv":
        _fail_method_call(monkeypatch, pd.DataFrame, "to_csv", 2, error)
    elif target == "rename":
        _fail_method_call(monkeypatch, Path, "rename", 1, error)
    else:
        original = getattr(verify_vol_threshold, target)
        monkeypatch.setattr(verify_vol_threshold, target, _FailOnCall(original, 2, error))
    out_dir = tmp_path / "out"
    assert verify_vol_threshold.main(["--out-dir", str(out_dir)]) == 1
    output = capsys.readouterr().out
    cause = f"({type(error).__name__}: {error})"
    assert f"Writing {out_dir} failed {cause}. Nothing was published." in output
    assert not out_dir.exists()
    assert _staging_dirs(out_dir) == []


def test_output_appears_only_after_both_files_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both files are written and verified in a staging directory beside out_dir, which appears
    only at the end.

    So even a run killed partway, which no cleanup can catch, never leaves out_dir half written,
    and the final rename stays on one file system.
    """
    out_dir = tmp_path / "out"
    original = verify_vol_threshold.verify_written
    checked: list[tuple[Path, bool]] = []

    def verify_written(path: Path, frame: pd.DataFrame) -> None:
        """Note where each file was written and whether out_dir existed yet."""
        checked.append((path, out_dir.exists()))
        original(path, frame)

    monkeypatch.setattr(verify_vol_threshold, "verify_written", verify_written)
    assert verify_vol_threshold.main(["--out-dir", str(out_dir)]) == 0
    assert [exists for _, exists in checked] == [False, False]
    for path, _ in checked:
        assert path.parent.parent == out_dir.parent
        assert path.parent.name.startswith(f".{out_dir.name}.")
    assert sorted(p.name for p in out_dir.iterdir()) == sorted([Path(RAW).name, Path(CLEAN).name])


def test_failed_cleanup_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A staging directory the cleanup can't delete is named, and still nothing is published."""
    original: Callable[..., object] = pd.DataFrame.to_csv
    locked: list[Path] = []

    def to_csv(frame: pd.DataFrame, path: Path, *args: object, **kwargs: object) -> object:
        """Write the raw file, then lock the staging directory and fail the clean file's write."""
        if locked:
            raise OSError("disk full")
        result = original(frame, path, *args, **kwargs)
        locked.append(Path(path).parent)
        locked[0].chmod(0o555)
        return result

    monkeypatch.setattr(pd.DataFrame, "to_csv", to_csv)
    out_dir = tmp_path / "out"
    try:
        assert verify_vol_threshold.main(["--out-dir", str(out_dir)]) == 1
    finally:
        for path in locked:
            path.chmod(0o755)
    output = capsys.readouterr().out
    assert f"Could not remove the staging directory {locked[0]} (PermissionError" in output
    assert "failed (OSError: disk full). Nothing was published." in output
    assert not out_dir.exists()


def _to_csv_with(change: Callable[[pd.DataFrame, str], pd.DataFrame]) -> Callable[..., object]:
    """A DataFrame.to_csv replacement that writes change(frame, file name) instead of frame."""
    original: Callable[..., object] = pd.DataFrame.to_csv

    def to_csv(frame: pd.DataFrame, path: Path, *args: object, **kwargs: object) -> object:
        """Bound replacement for DataFrame.to_csv."""
        return original(change(frame, Path(path).name), path, *args, **kwargs)

    return to_csv


@pytest.mark.parametrize("which", ["raw", "clean"])
def test_written_thresholds_must_equal_the_derived_ones(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    which: str,
) -> None:
    """A threshold changed in its last digits on the way to disk fails re-verification.

    The change is far too small to flip any label, so only the exact comparison of the written
    thresholds with the derived ones can see it.
    """
    target = Path(RAW if which == "raw" else CLEAN).name

    def nudge_threshold(frame: pd.DataFrame, name: str) -> pd.DataFrame:
        """Move the first threshold of the target file by a relative 1e-12."""
        if name != target:
            return frame
        thresholds = frame["vol_threshold"].to_numpy(dtype=float, copy=True)
        first = int(np.flatnonzero(~np.isnan(thresholds))[0])
        thresholds[first] *= 1 + 1e-12
        return frame.assign(vol_threshold=thresholds)

    monkeypatch.setattr(pd.DataFrame, "to_csv", _to_csv_with(nudge_threshold))
    out_dir = tmp_path / "out"
    assert verify_vol_threshold.main(["--out-dir", str(out_dir)]) == 1
    assert f"{target} does not re-read as the verified data" in capsys.readouterr().out
    assert not out_dir.exists()


def test_label_corrupted_on_write_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A label corrupted on its way to disk fails re-verification.

    The corrupted label sits on a ticker's last clean row, which the clean file's own data
    cannot check, so only the comparison with what was verified can see it.
    """
    clean_name = Path(CLEAN).name

    def flip_last_clean_label(frame: pd.DataFrame, name: str) -> pd.DataFrame:
        """Flip one ticker's last label when the clean file is written."""
        if name != clean_name:
            return frame
        frame = frame.copy()
        row = frame.index[frame["Ticker"] == frame["Ticker"].iloc[0]][-1]
        frame.loc[row, "NextVolSpike"] = 1 - frame.loc[row, "NextVolSpike"]
        return frame

    monkeypatch.setattr(pd.DataFrame, "to_csv", _to_csv_with(flip_last_clean_label))
    out_dir = tmp_path / "out"
    assert verify_vol_threshold.main(["--out-dir", str(out_dir)]) == 1
    output = capsys.readouterr().out
    assert f"(VerificationError: {clean_name} does not re-read as the verified data)" in output
    assert not out_dir.exists()


@pytest.mark.parametrize("change", ["rounded_feature", "dropped_column"])
def test_any_column_changed_on_write_is_caught(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    change: str,
) -> None:
    """A write that rounds or drops any column fails re-verification, not only a label or a
    threshold."""
    clean_name = Path(CLEAN).name

    def alter(frame: pd.DataFrame, name: str) -> pd.DataFrame:
        """Round Return to six decimals, or drop VIX, when the clean file is written."""
        if name != clean_name:
            return frame
        if change == "dropped_column":
            return frame.drop(columns="VIX")
        return frame.assign(Return=frame["Return"].round(6))

    monkeypatch.setattr(pd.DataFrame, "to_csv", _to_csv_with(alter))
    out_dir = tmp_path / "out"
    assert verify_vol_threshold.main(["--out-dir", str(out_dir)]) == 1
    assert f"{clean_name} does not re-read as the verified data" in capsys.readouterr().out
    assert not out_dir.exists()


def test_interrupt_while_writing_leaves_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupt during the second write removes the staging directory before the traceback."""
    _fail_method_call(monkeypatch, pd.DataFrame, "to_csv", 2, KeyboardInterrupt())
    out_dir = tmp_path / "out"
    with pytest.raises(KeyboardInterrupt):
        verify_vol_threshold.main(["--out-dir", str(out_dir)])
    assert not out_dir.exists()
    assert _staging_dirs(out_dir) == []
