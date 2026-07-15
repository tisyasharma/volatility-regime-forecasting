"""
Time-based train/test splits for walk-forward validation.

Window sizes are counted in trading days observed in the data, not calendar days, so a
63-day test window is one full quarter of market sessions regardless of weekends and
holidays. All splits maintain temporal ordering, and get_fold_indices can additionally
purge training rows whose forward-looking label reaches into the test period.
"""

import warnings
from typing import List, Optional, Tuple

import pandas as pd


def generate_expanding_window_splits(
    dates: pd.Series,
    train_start: str,
    initial_train_end: str,
    test_size_days: int,
    step_size_days: int,
    min_train_size_days: int = 252,
    include_partial: bool = True,
) -> List[Tuple[str, str, str, str]]:
    """
    Generate expanding-window walk-forward splits over the observed trading calendar.

    Training starts at train_start and grows; each test window spans test_size_days
    consecutive trading dates starting the day after the training window and advancing by
    step_size_days per fold. All window sizes count unique trading dates present in the data,
    and folds with fewer than min_train_size_days training dates are skipped.

    A final window shorter than test_size_days is kept by default (flagged with a RuntimeWarning)
    so the newest data is still evaluated; pass include_partial=False to drop it.

    Returns (train_start, train_end, test_start, test_end) date strings per fold, with train_end
    the last trading date before test_start so the inclusive ranges are disjoint by construction.
    """
    trading = pd.DatetimeIndex(pd.Series(dates).dropna().unique()).sort_values()
    train_start_pos = trading.searchsorted(pd.Timestamp(train_start), side="left")
    test_start_pos = trading.searchsorted(pd.Timestamp(initial_train_end), side="right")
    if test_start_pos == 0:
        raise ValueError("initial_train_end precedes the first trading date in the data")

    splits = []
    while test_start_pos < len(trading):
        test_dates = trading[test_start_pos : test_start_pos + test_size_days]
        is_partial = len(test_dates) < test_size_days
        if is_partial and not include_partial:
            break

        n_train = test_start_pos - train_start_pos
        if n_train >= min_train_size_days:
            splits.append((
                train_start,
                trading[test_start_pos - 1].strftime("%Y-%m-%d"),
                test_dates[0].strftime("%Y-%m-%d"),
                test_dates[-1].strftime("%Y-%m-%d"),
            ))
            if is_partial:
                warnings.warn(
                    f"Final fold is partial, {len(test_dates)} of "
                    f"{test_size_days} test days",
                    RuntimeWarning,
                )

        # A partial window means the data is exhausted. Stepping further (possible when
        # step_size_days < test_size_days) would only emit nested sub-windows of the
        # same tail, so stop here.
        if is_partial:
            break
        test_start_pos += step_size_days

    return splits


def get_fold_indices(
    dates: pd.Series,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    label_end: Optional[pd.Series] = None,
) -> Tuple[pd.Series, pd.Series]:
    """
    Return boolean (train_mask, test_mask) for one fold, selecting rows of dates inside the
    inclusive [train_start, train_end] and [test_start, test_end] ranges.

    When label_end is given (aligned with dates, the date of the last return entering each row's
    forward-looking label), the train mask additionally keeps only rows whose label window closes
    strictly before test_start. This trading-day-exact purge keeps no training label from peeking
    into the test period, matched to each target's horizon and robust to holidays (Lopez de
    Prado, 2018).
    """
    train_mask = (dates >= train_start) & (dates <= train_end)
    test_mask = (dates >= test_start) & (dates <= test_end)
    if label_end is not None:
        train_mask = train_mask & label_end.notna() & (label_end < pd.to_datetime(test_start))
    return train_mask, test_mask
