"""
Prediction targets and the matched naive baseline.

NextVolSpike labels whether tomorrow's 10-day rolling volatility exceeds its expanding
80th percentile; the rolling windows at t and t+1 overlap by nine days. FwdVolRegime
labels whether volatility over the next 10 trading days, a window disjoint from the
features known at t, exceeds the same threshold. The matched naive baseline predicts a
regime whenever today's rolling volatility already sits above the per-ticker expanding
threshold that defines the labels, using only information available at t.
"""

import warnings

import numpy as np


def add_regime_columns(df, forward_window=10, quantile=0.8, min_periods=126):
    """
    Add the forward-disjoint target and the matched naive baseline signal from the persisted
    expanding volatility threshold. Assumes the frame is date-ordered within each ticker.

    The threshold that defines both labels is stored in the vol_threshold column (built and
    verified by scripts/verify_vol_threshold.py). Recomputing it from a warm-up-trimmed frame
    would start every expanding window late and drift from the stored labels, so the persisted
    column is authoritative and recomputation is only a fallback for frames that lack it.
    """
    df = df.copy()
    ret = df.groupby("Ticker")["Return"]

    if "vol_threshold" not in df.columns:
        warnings.warn(
            "vol_threshold not found in the data, recomputing from the loaded frame. "
            "If warm-up rows were trimmed, the recomputed threshold will not match the "
            "persisted labels.",
            RuntimeWarning,
        )
        df["vol_threshold"] = df.groupby("Ticker")["RollingVol"].transform(
            lambda s: s.shift(1).expanding(min_periods=min_periods).quantile(quantile)
        )

    # Matched naive baseline signal, known at t. NaN where no threshold exists yet, so a
    # missing threshold surfaces as an excluded row rather than a silent "no regime".
    df["currently_high"] = np.where(
        df["vol_threshold"].isna(),
        np.nan,
        (df["RollingVol"] > df["vol_threshold"]).astype(float),
    )

    # Forward realized volatility over [t+1, t+forward_window], disjoint from RollingVol(t).
    df["forward_vol"] = ret.transform(
        lambda r: r.rolling(forward_window).std().shift(-forward_window)
    )
    df["FwdVolRegime"] = np.where(
        df["forward_vol"].isna() | df["vol_threshold"].isna(),
        np.nan,
        (df["forward_vol"] > df["vol_threshold"]).astype(float),
    )

    # Date of the last return entering each target's forward window, used to purge training
    # rows whose label reaches into the test period, exact in trading days. The forward
    # target looks forward_window trading days ahead, the next-day spike target one ahead.
    dates = df.groupby("Ticker")["Date"]
    df["label_end_forward"] = dates.shift(-forward_window)
    df["label_end_nextday"] = dates.shift(-1)
    return df


def matched_naive_predict(df_slice):
    """
    Return (prediction, score) for the matched naive baseline on a set of rows.

    The prediction flags a regime when today's volatility is already above threshold. The
    score is the volatility-to-threshold ratio, used for ranking metrics such as PR-AUC.
    """
    pred = df_slice["currently_high"].to_numpy()
    ratio = (df_slice["RollingVol"] / df_slice["vol_threshold"]).to_numpy()
    return pred, ratio
