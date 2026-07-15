"""
Tests for the walk-forward split logic, the label-end purge, and the label-end
convention the pipeline supplies to it.
"""

import pandas as pd

from src.split import generate_expanding_window_splits, get_fold_indices
from src.targets import add_regime_columns

CAL = pd.bdate_range("2020-01-02", periods=500)
HORIZON = 10


def _dates_and_label_end():
    """
    Build a single-ticker date series plus each row's label-end date, the date of the
    last return entering its forward-looking label. Rows whose label window runs past
    the data get NaT, mirroring how the pipeline leaves incomplete labels missing.
    """
    dates = pd.Series(CAL)
    label_end = dates.shift(-HORIZON)
    return dates, label_end


def _make_splits(**overrides):
    """Generate splits over the synthetic calendar with small, fast defaults."""
    kwargs = dict(
        train_start="2020-01-02",
        initial_train_end=str(CAL[251].date()),
        test_size_days=63,
        step_size_days=63,
        min_train_size_days=100,
    )
    kwargs.update(overrides)
    return generate_expanding_window_splits(pd.Series(CAL), **kwargs)


def test_folds_are_temporally_ordered():
    """Every fold shares the fixed train start and puts train strictly before test."""
    splits = _make_splits()
    assert len(splits) >= 3
    for train_start, train_end, test_start, test_end in splits:
        assert train_start <= train_end < test_start <= test_end
    assert all(s[0] == "2020-01-02" for s in splits)


def test_train_and_test_masks_never_overlap():
    """No row appears in both the train mask and the test mask of any fold."""
    dates = pd.Series(CAL)
    for split in _make_splits():
        train_mask, test_mask = get_fold_indices(dates, *split)
        assert not (train_mask & test_mask).any()
        assert test_mask.any()


def test_purge_keeps_training_labels_out_of_the_test_window():
    """
    With label_end supplied, no surviving training row's label window reaches
    test_start, and the purge actually removes boundary rows rather than being
    a no-op.
    """
    dates, label_end = _dates_and_label_end()
    for split in _make_splits():
        test_start = pd.Timestamp(split[2])
        train_mask, _ = get_fold_indices(dates, *split, label_end=label_end)
        assert (label_end[train_mask] < test_start).all()
        unpurged_train, _ = get_fold_indices(dates, *split)
        assert train_mask.sum() < unpurged_train.sum()
        # no over-purge: every training row whose label closes before test_start survives
        safe = unpurged_train & label_end.notna() & (label_end < test_start)
        assert (train_mask | ~safe).all()


def test_rows_without_a_complete_label_never_train():
    """A row whose label_end is missing is excluded from training by the purge."""
    dates, label_end = _dates_and_label_end()
    label_end.iloc[50] = pd.NaT
    split = _make_splits()[0]
    train_mask, _ = get_fold_indices(dates, *split, label_end=label_end)
    assert not train_mask.iloc[50]


def test_min_train_size_skips_early_folds():
    """Folds whose training window is shorter than the minimum are not emitted."""
    splits = _make_splits(
        initial_train_end=str(CAL[59].date()),
        test_size_days=21,
        step_size_days=21,
        min_train_size_days=252,
    )
    assert splits
    trading = pd.DatetimeIndex(CAL)
    for train_start, train_end, _, _ in splits:
        n_train = trading.searchsorted(
            pd.Timestamp(train_end), side="right"
        ) - trading.searchsorted(pd.Timestamp(train_start), side="left")
        assert n_train >= 252


def test_partial_final_fold_is_kept_only_on_request():
    """The short tail window is evaluated by default and dropped with include_partial=False."""
    with_partial = _make_splits()
    without_partial = _make_splits(include_partial=False)
    assert with_partial[:-1] == without_partial
    trading = pd.DatetimeIndex(CAL)
    _, _, last_test_start, last_test_end = with_partial[-1]
    n_test = trading.searchsorted(
        pd.Timestamp(last_test_end), side="right"
    ) - trading.searchsorted(pd.Timestamp(last_test_start), side="left")
    assert 0 < n_test < 63


def _pipeline_frame(lengths=(40, 35)):
    """Small multi-ticker frame carrying the columns add_regime_columns expects."""
    frames = [
        pd.DataFrame({
            "Ticker": ticker,
            "Date": pd.Series(CAL[:n]),
            "Return": 0.01,
            "RollingVol": 0.02,
            "vol_threshold": 0.05,
        })
        for ticker, n in zip(("AAA", "BBB"), lengths)
    ]
    return pd.concat(frames, ignore_index=True)


def test_pipeline_label_end_columns_follow_the_synthesized_convention():
    """
    add_regime_columns dates each row's label end at t+10 for the forward target and
    t+1 for the next-day target, shifted within each ticker. This pins the convention
    the tests above synthesize to what the pipeline actually produces.
    """
    df = add_regime_columns(_pipeline_frame(), forward_window=HORIZON)
    for _, group in df.groupby("Ticker"):
        assert group["label_end_forward"].equals(group["Date"].shift(-HORIZON))
        assert group["label_end_nextday"].equals(group["Date"].shift(-1))


def test_pipeline_label_end_pins_the_purge_boundary_exactly():
    """
    Feeding the pipeline's own label_end_forward through get_fold_indices leaves the
    last surviving training row exactly HORIZON + 1 trading days before test_start, so
    its label window closes on the final day before the test period. A one-day drift
    in either the label-end convention or the purge comparison moves this boundary.
    """
    frame = pd.DataFrame({
        "Ticker": "AAA",
        "Date": pd.Series(CAL),
        "Return": 0.01,
        "RollingVol": 0.02,
        "vol_threshold": 0.05,
    })
    df = add_regime_columns(frame, forward_window=HORIZON)
    trading = pd.DatetimeIndex(CAL)
    for split in _make_splits():
        train_mask, _ = get_fold_indices(
            df["Date"], *split, label_end=df["label_end_forward"]
        )
        test_start_pos = trading.searchsorted(pd.Timestamp(split[2]), side="left")
        assert df.loc[train_mask, "Date"].max() == trading[test_start_pos - (HORIZON + 1)]
