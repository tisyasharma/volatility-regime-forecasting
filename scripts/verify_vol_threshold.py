"""
Derive the label-defining volatility threshold from raw data, verify it reproduces the
stored labels exactly, and write both datasets with the threshold to a new directory.

The NextVolSpike labels were built (in notebook 00) from an expanding 80th-percentile threshold
computed on the full download, including each ticker's warm-up rows. Those rows are dropped from
the clean file, so recomputing from the clean CSV alone would start every expanding window late
and drift from the stored labels. This recomputes from the raw file and checks that it
reproduces the stored labels exactly on both files, and that the clean file holds exactly the
raw rows those labels are defined on. Only then does it write both files into a staging
directory, re-read them and check they equal the verified data, and rename the staging
directory to the output directory, so the output directory appears complete or not at all. It
reports whether each output is identical to the input it was made from.

Without --out-dir the output goes to build/vol_threshold, which every run replaces. A directory
given with --out-dir must not exist yet. The datasets under data/ are never modified. Replacing
them is a deliberate step taken after reviewing the output, and data/SHA256SUMS has to be
updated with them.

Exit status: 0 when both written files verify, also when they differ from the inputs (each file
is reported as identical or different). 1 when a check fails, when writing, verifying or
renaming the output fails, or when the previous build/vol_threshold can't be removed. 2 when the
output directory is inside data/, when a directory given with --out-dir already exists, when
build/vol_threshold is not a plain directory or holds an input, or when the two outputs would
share a name. An input, config or output path that can't be read or inspected ends in a
traceback before anything is written. Any exception after writing has begun, Ctrl-C included,
removes the staging directory. A run killed outright can leave a hidden .vol_threshold.* staging
directory in build/, which `make clean` removes.
"""

from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config  # noqa: E402

# The raw download is not config-owned, only the model-ready file is, so the raw path
# stays a module constant while the clean path follows data.input_path.
RAW_PATH = REPO_ROOT / "data" / "raw" / "merged_features_full.csv"
DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUT_DIR = REPO_ROOT / "build" / "vol_threshold"
KEYS = ["Ticker", "Date"]

QUANTILE = 0.8
MIN_PERIODS = 126


class VerificationError(Exception):
    """A check on the inputs, or on the files written from them, failed."""


def read_dataset(path: Path) -> pd.DataFrame:
    """Read a dataset exactly, failing if any (Ticker, Date) key appears more than once.

    Floats are parsed with round-trip precision, which is correctly rounded on every platform, so
    a value read and written back is unchanged. Dates stay as strings so the merge keys and the
    written files are byte-stable.
    """
    frame = pd.read_csv(path, float_precision="round_trip")
    repeated = frame.loc[frame.duplicated(KEYS, keep=False), KEYS].drop_duplicates()
    if len(repeated):
        ticker, date = repeated.iloc[0]
        raise VerificationError(
            f"{path.name} repeats {len(repeated)} (Ticker, Date) keys, the first {ticker} {date}"
        )
    return frame


def derive_threshold(raw: pd.DataFrame) -> pd.DataFrame:
    """Return raw sorted by ticker and date with a freshly derived vol_threshold column."""
    raw = raw.drop(columns=["vol_threshold"], errors="ignore")
    raw = raw.sort_values(KEYS).reset_index(drop=True)
    raw["vol_threshold"] = raw.groupby("Ticker")["RollingVol"].transform(
        lambda s: s.shift(1).expanding(min_periods=MIN_PERIODS).quantile(QUANTILE)
    )
    return raw


def check_raw_labels(raw: pd.DataFrame) -> None:
    """Raise unless next-day vol against vol_threshold reproduces NextVolSpike, NaNs included.

    A failure means the raw file no longer matches the label-generation run, and the right fix
    is a deliberate regeneration from notebook 00, not a mismatched threshold.
    """
    next_day_vol = raw.groupby("Ticker")["RollingVol"].shift(-1)
    rederived = np.where(
        next_day_vol.isna() | raw["vol_threshold"].isna(),
        np.nan,
        (next_day_vol > raw["vol_threshold"]).astype(float),
    )
    stored = raw["NextVolSpike"].astype(float).to_numpy()
    agree = (np.isnan(rederived) & np.isnan(stored)) | (rederived == stored)
    if not agree.all():
        raise VerificationError(
            f"raw-side check failed: the re-derived NextVolSpike differs from the stored label "
            f"on {int((~agree).sum())} rows. Regenerate the dataset from notebook 00."
        )


def check_clean_rows(clean: pd.DataFrame, raw: pd.DataFrame) -> None:
    """Raise unless each ticker's clean rows are its raw rows from the first with a threshold to
    the one before its last, in date order.

    The clean-side label check takes the next clean row as the next trading day, so a missing,
    extra or reordered row would pair a label with the wrong day. raw must come from
    derive_threshold, which sorts it.
    """
    has_threshold = raw["vol_threshold"].notna().astype(int).groupby(raw["Ticker"]).cummax()
    before_last = raw.groupby("Ticker").cumcount(ascending=False) > 0
    wanted = set(raw.loc[has_threshold.astype(bool) & before_last, KEYS].itertuples(index=False))
    found = set(clean[KEYS].itertuples(index=False))
    problems = []
    if missing := sorted(wanted - found):
        problems.append(f"lacks {len(missing)} of them, the first {' '.join(missing[0])}")
    if extra := sorted(found - wanted):
        problems.append(f"has {len(extra)} rows outside them, the first {' '.join(extra[0])}")
    if problems:
        raise VerificationError(
            "the clean file should hold each ticker's raw rows from its first threshold to its "
            f"second-to-last date, but it {' and '.join(problems)}"
        )
    if not clean.groupby("Ticker")["Date"].is_monotonic_increasing.all():
        raise VerificationError("the clean file's dates are out of order within a ticker")


def _merge_from_raw(clean: pd.DataFrame, raw: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return clean with raw's column attached by (Ticker, Date).

    read_dataset has already rejected repeated keys, which validate restates.
    """
    return clean.merge(raw[[*KEYS, column]], on=KEYS, how="left", validate="one_to_one")


def merge_threshold(clean: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Return clean with vol_threshold taken from raw.

    check_clean_rows has made sure every clean row is a raw row that has a threshold.
    """
    clean = clean.drop(columns=["vol_threshold"], errors="ignore")
    return _merge_from_raw(clean, raw, "vol_threshold")


def check_clean_labels(clean: pd.DataFrame, raw: pd.DataFrame) -> int:
    """Raise unless every clean label matches raw and the attached threshold, returning the count.

    Every clean row needs a threshold, since the clean file starts after the warm-up. Every
    label must equal the raw file's label for the same row. That covers each ticker's last
    clean row, whose next day is only in the raw file. Where the next day is in the clean file,
    the label is also re-derived from the clean file's volatility and its vol_threshold column,
    which main fills with raw's re-derived threshold through merge_threshold.
    """
    missing = int(clean["vol_threshold"].isna().sum())
    if missing:
        raise VerificationError(f"{missing} clean rows have no threshold")
    raw_label = _merge_from_raw(
        clean[KEYS], raw.rename(columns={"NextVolSpike": "raw_label"}), "raw_label"
    )["raw_label"].to_numpy(dtype=float)
    stored = clean["NextVolSpike"].to_numpy(dtype=float)
    differs_from_raw = ~((stored == raw_label) | (np.isnan(stored) & np.isnan(raw_label)))

    nxt = clean.groupby("Ticker")["RollingVol"].shift(-1)
    comparable = nxt.notna().to_numpy()
    rederived = (nxt > clean["vol_threshold"]).to_numpy(dtype=float)
    differs_from_rederived = comparable & (rederived != stored)

    mismatched = int((differs_from_raw | differs_from_rederived).sum())
    if mismatched:
        raise VerificationError(f"clean-side check failed on {mismatched} rows")
    return len(clean)


def verify_written(path: Path, frame: pd.DataFrame) -> None:
    """Raise unless the written file re-reads as exactly the frame that was verified.

    Floats are parsed with round-trip precision, so every value, the thresholds included, must
    come back bit for bit, and no row or column may go missing.
    """
    if not pd.read_csv(path, float_precision="round_trip").equals(frame):
        raise VerificationError(f"{path.name} does not re-read as the verified data")


def _inside(path: Path, directory: Path) -> bool:
    """Return True when path is directory or below it, comparing files rather than spellings.

    The deepest existing part of the path, and each of its parents, is compared with directory by
    file identity, so a symlink, a ".." or a case variant on a case-insensitive disk can't slip by.
    """
    resolved = path.resolve()
    existing = next(part for part in [resolved, *resolved.parents] if part.exists())
    return any(part.samefile(directory) for part in [existing, *existing.parents])


def _remove_tree(path: Path) -> str:
    """Delete a directory tree, returning "" or the error that stopped it.

    shutil.rmtree removes a symlink inside the tree without following it.
    """
    try:
        shutil.rmtree(path)
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"
    return ""


def _refusal(out_dir: Path, inputs: list[Path], replaces: bool) -> str:
    """Return why the outputs can't be written to out_dir, or "" when they can.

    replaces is True for the default directory, which belongs to this script and may be
    replaced, unless it is a symlink or a file or an input sits inside it. Any other directory has
    to be new.
    """
    if _inside(out_dir, DATA_DIR):
        return f"{out_dir} is inside {DATA_DIR}, which this script never writes to"
    if len({path.name.casefold() for path in inputs}) < len(inputs):
        return "the two outputs would have the same file name"
    if not os.path.lexists(out_dir):
        return ""
    if not replaces:
        return f"--out-dir {out_dir} already exists, so give a new directory"
    if out_dir.is_symlink() or not out_dir.is_dir():
        return f"{out_dir} is not a plain directory, so it is not replaced"
    if any(_inside(path, out_dir) for path in inputs):
        return f"{out_dir} holds an input, so it is not replaced"
    return ""


def _write_verified(frames: dict[str, pd.DataFrame], out_dir: Path) -> str:
    """Write each frame under its name, verify it, and publish them all as out_dir.

    The files go into a staging directory next to out_dir, so the final rename is one step on one
    file system. Returns "" or what failed. On any exception, Ctrl-C included, the staging
    directory is removed, and out_dir never exists half written.
    """
    staging: Path | None = None
    renamed = False
    try:
        try:
            out_dir.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
            for name, frame in frames.items():
                frame.to_csv(staging / name, index=False)
            for name, frame in frames.items():
                verify_written(staging / name, frame)
            staging.rename(out_dir)
            renamed = True
        except (VerificationError, ValueError, OSError) as exc:
            return f"Writing {out_dir} failed ({type(exc).__name__}: {exc}). Nothing was published."
    finally:
        if staging is not None and not renamed:
            left = _remove_tree(staging)
            if left:
                print(f"Could not remove the staging directory {staging} ({left}).")
    return ""


def main(argv: list[str] | None = None) -> int:
    """Derive and verify the threshold, write both files to a new output directory, report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="a new directory for the outputs (default: build/vol_threshold, replaced each run)",
    )
    args = parser.parse_args(argv)
    replaces = args.out_dir is None
    out_dir: Path = DEFAULT_OUT_DIR if replaces else args.out_dir

    clean_path = REPO_ROOT / load_config()["data"]["input_path"]
    refusal = _refusal(out_dir, [RAW_PATH, clean_path], replaces)
    if refusal:
        print(refusal)
        return 2
    # An earlier run's output would still look verified if this run failed, so it goes first.
    if replaces and os.path.lexists(out_dir):
        left = _remove_tree(out_dir)
        if left:
            print(f"Nothing written: could not remove the previous {out_dir} ({left}).")
            return 1

    try:
        raw = derive_threshold(read_dataset(RAW_PATH))
        check_raw_labels(raw)
        clean = read_dataset(clean_path)
        check_clean_rows(clean, raw)
        merged = merge_threshold(clean, raw)
        compared = check_clean_labels(merged, raw)
    except VerificationError as exc:
        print(f"Nothing written: {exc}")
        return 1

    failure = _write_verified({RAW_PATH.name: raw, clean_path.name: merged}, out_dir)
    if failure:
        print(failure)
        return 1

    print(
        f"verified: stored NextVolSpike reproduced exactly on both written files "
        f"({compared} clean rows checked)"
    )
    for source in [RAW_PATH, clean_path]:
        output = out_dir / source.name
        state = "identical to" if filecmp.cmp(source, output, shallow=False) else "differs from"
        shown = source.relative_to(REPO_ROOT) if source.is_relative_to(REPO_ROOT) else source
        print(f"{output}: {state} {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
