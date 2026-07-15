# Dataset Card

## Overview

- Dataset name: Merged equity features with volatility targets
- Version: 1.1 (adds the persisted `vol_threshold` column)
- Purpose: forecast whether the next 10 trading days are a high-volatility regime
  (`FwdVolRegime`), and study why volatility is more persistent and forecastable than next-day
  return direction. A next-day target (`NextVolSpike`) is retained as a
  reference case.

## Provenance

- Sources: daily OHLCV from Yahoo Finance (via `yfinance`); VIX index as a market-wide
  volatility signal.
- Collection period: raw download 2015-01-02 to 2026-01-22 (warm-up rows included);
  model-ready window 2015-07-20 to 2026-01-21 after trimming incomplete lookbacks.
- Tickers: AAPL, MSFT, AMZN, GOOGL, NVDA, TSLA (technology), JPM (finance), WMT (retail),
  DAL, UAL (airlines), LMT, RTX, NOC (defense), XOM (energy).
- Rights: public market data used for research and educational purposes.

## Structure

- Two files. `data/raw/merged_features_full.csv` (38,920 rows) keeps warm-up rows with
  incomplete indicators for EDA. `data/processed/merged_features_clean.csv` (37,002 rows) is
  the model-ready set after dropping rows with incomplete lookback windows; 1,918 rows are
  dropped.
- Features: `Return`, `RollingVol`, `RSI`, `Price_to_SMA20`, `Price_to_SMA50`,
  `SMA20_to_SMA50`, `Volume_Z`, `VIX`, `Lagged_Return`. All are known at time `t`.
- Targets in the CSV: `NextReturn`, `NextDirection`, `NextVolSpike`, all describing `t+1`.
  `NextReturn` and `NextDirection` are the direction baseline (the negative control that
  motivates predicting volatility instead of direction) and are not modeled.
- Stored threshold: `vol_threshold`, the per-ticker expanding 80th percentile of past
  `RollingVol` (`shift(1)`, minimum 126 observations) that defines both volatility labels.
  It is computed on the full download (warm-up rows included), persisted by notebook 00, and
  written into both files by `scripts/verify_vol_threshold.py` (which asserts it reproduces
  the stored labels exactly), so downstream code never re-derives it from the trimmed clean file.
- Derived target: `FwdVolRegime`, the primary modeling target, is computed at runtime in
  `src/targets.py` from the stored threshold (the target itself is not stored in the CSV)
  together with the matched naive baseline signal.
- Missingness: the clean file has none.

## Labeling

- `NextVolSpike` (demonstration only): 1 if tomorrow's 10-day rolling volatility
  exceeds an expanding 80th-percentile threshold of past volatility (`shift(1)` so the present
  is excluded), otherwise 0. Positive rate 25.2%.
- `FwdVolRegime` (primary): 1 if volatility over the next 10 trading days (disjoint from the
  10-day feature window) exceeds the same expanding threshold. Positive rate ~25.4%. Because
  feature and label windows do not overlap, this is a genuine forecast rather than a
  near-restatement of a feature.
- `NextDirection`: 1 if tomorrow's return is positive. The positive rate of 52.8% is a
  persistent upward drift with no exploitable conditional signal in this data, which
  motivates using volatility rather than direction.

## Risk and sensitivity

- No personal or sensitive data; market prices only.
- Representation gaps: US large caps across six sectors; airline tickers are highly
  correlated, reducing effective independence.
- Foreseeable misuse: treating these forecasts as a tradable signal without a cost and
  execution model.

## Recommended use

- Suitable: volatility-clustering studies, leakage-aware walk-forward benchmarking, teaching
  time-series evaluation.
- Unsuitable: live trading or investment decisions.
- Maintenance: append new daily data and rerun `notebooks/00_data_collection.ipynb` to
  refresh features and targets; the notebook persists `vol_threshold` alongside them.
