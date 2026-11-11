# Dataset Card

## Dataset overview

- **Name:** Merged equity features with volatility targets
- **Version:** 1.1
- **Domain:** Daily U.S. equity market data
- **Primary file:** `data/processed/merged_features_clean.csv`
- **Primary target:** `FwdVolRegime`, derived at runtime
- **Purpose:** Forecast whether the next 10 trading days form a high-volatility regime and demonstrate how overlapping target windows can overstate next-day forecasting skill
- **Sensitive data:** None

## Provenance

Daily OHLCV data were collected through `yfinance`. VIX is included as a market-wide implied-volatility signal.

### Time coverage

- **Raw download:** 2015-01-02 through 2026-01-22
- **Model-ready period:** 2015-07-20 through 2026-01-21

The raw dataset keeps the warm-up rows that the rolling indicators and the expanding label threshold need. The processed dataset drops every row with a missing value: each ticker's first 136 rows, where the threshold does not yet have its 126 prior observations, and the final trading day, whose next-day fields cannot be computed.

### Equity universe

| Industry group | Tickers |
|---|---|
| Technology | AAPL, MSFT, AMZN, GOOGL, NVDA, TSLA |
| Finance | JPM |
| Retail | WMT |
| Airlines | DAL, UAL |
| Defense | LMT, RTX, NOC |
| Energy | XOM |

The universe contains 14 U.S. large-cap equities in six industry groups. The groups are this project's own, not an official sector classification, and the universe is not intended to represent the full equity market.

## Files and row counts

| File | Rows | Purpose |
|---|---:|---|
| `data/raw/merged_features_full.csv` | 38,920 | Full dataset, including each ticker's warm-up rows and the final trading day |
| `data/processed/merged_features_clean.csv` | 37,002 | Model-ready dataset, with every row that has a missing value removed |

A total of 1,918 rows are removed between the raw and processed files: 1,904 warm-up rows (136 per ticker) and the 14 rows of 2026-01-22. 1,218 of the warm-up rows already have every feature, but no label threshold yet. The processed model feature columns contain no missing values.

## Features

The model uses nine features, all available by the end of trading day `t`:

| Feature | Description |
|---|---|
| `Return` | Current daily return |
| `RollingVol` | Current 10-day rolling volatility |
| `RSI` | Relative Strength Index |
| `Price_to_SMA20` | Price divided by the 20-day simple moving average |
| `Price_to_SMA50` | Price divided by the 50-day simple moving average |
| `SMA20_to_SMA50` | 20-day moving average divided by the 50-day moving average |
| `Volume_Z` | Standardized trading volume |
| `VIX` | Market-wide implied-volatility index, its daily close, which Cboe disseminates until 4:15 p.m. ET (Cboe Volatility Index Methodology, section 4) |
| `Lagged_Return` | Previous daily return |

Ratio and standardized features improve comparability across equities with different price and volume scales.

## Threshold and labels

### Persisted regime threshold

`vol_threshold` is the per-ticker expanding 80th percentile of historical `RollingVol`.

It is:

- computed with `shift(1)`, excluding the present and future;
- initialized after a minimum of 126 historical observations;
- calculated on the full dataset including warm-up rows;
- persisted in both CSV files;
- verified by `scripts/verify_vol_threshold.py`;
- reused by the targets, matched naive baseline, and HAR benchmark.

Persisting the threshold prevents downstream code from recalculating it on a trimmed sample and drifting from the stored labels.

### `NextVolSpike`

`NextVolSpike` is positive when tomorrow's 10-day rolling volatility exceeds `vol_threshold`.

- **Role:** negative-control target
- **Full-sample positive rate:** approximately 25.2%
- **Important caveat:** tomorrow's rolling-volatility window overlaps today's feature window by nine of ten days

The target is retained to show how overlapping windows can make persistence appear to be strong forward prediction.

### `FwdVolRegime`

`FwdVolRegime` is derived at runtime in `src/targets.py`. It is positive when volatility over the next 10 trading days exceeds `vol_threshold`.

- **Role:** primary modeling target
- **Full-sample positive rate:** approximately 25.4%
- **Key property:** the future label window is disjoint from the current 10-day rolling-volatility feature window

The target itself is not stored in the CSV. It is recreated through shared target-construction code from future returns and the persisted historical threshold.

### Direction fields

The CSV also contains `NextReturn` and `NextDirection`. `NextDirection` is positive on 52.8% of observations. These fields support exploratory analysis that motivates forecasting volatility rather than next-day direction. They are not modeled in the final comparison.

## Data preparation

The data-collection notebook:

1. downloads daily price and volume histories;
2. constructs technical indicators and lagged fields;
3. computes the past-only volatility threshold;
4. writes the raw and processed CSV files.

The modeling pipeline then:

1. loads the processed dataset;
2. derives HAR features;
3. derives `FwdVolRegime` and label-end dates;
4. excludes rows without complete features, thresholds, or targets;
5. applies expanding-window splits with horizon-exact purging.

## Missingness and exclusions

- The raw file retains the warm-up rows, where indicators or the label threshold are still undefined.
- The processed file removes those rows and the final trading day, whose next-day fields are missing.
- Final observations for each ticker may lack a complete future target and are excluded from target-specific fitting or evaluation.
- The processed model feature columns contain no missing values.

## Representation gaps and dependencies

- The universe contains only 14 U.S. large-cap equities.
- Six equities in the technology group make the sample imbalanced across groups.
- DAL and UAL are strongly correlated, reducing effective cross-sectional breadth.
- VIX is repeated across equities on the same trading date.
- Observations are dependent over time because volatility clusters.
- The date range covers multiple market conditions but does not guarantee performance in future regimes.
- The data exclude international markets, smaller companies, other asset classes, intraday behavior, transaction costs, and execution conditions.

The significance analyses account for temporal and cross-sectional dependence by resampling blocks of folds or of whole trading dates, but they do not expand the underlying market coverage.

## Appropriate uses

The dataset is suitable for:

- research on volatility clustering;
- leakage-aware time-series classification;
- walk-forward validation demonstrations;
- comparisons with naive and econometric baselines;
- educational reproduction of the reported analysis.

## Inappropriate uses

The dataset and labels should not be used as direct support for:

- live trading or investment recommendations;
- automated portfolio decisions;
- claims of profitability;
- evaluation without time-aware splitting;
- threshold construction that uses present or future data;
- claims of broad equity-market coverage;
- decisions that assume the included files are real-time, complete, or free of provider limitations.

## Maintenance

To refresh the dataset:

1. rerun `notebooks/00_data_collection.ipynb`, which overwrites both CSVs in `data/` with a new download, and review `git diff --stat data/` against the committed files;
2. run `make verify_vol_threshold`, which removes `build/vol_threshold/`, re-derives the threshold from the raw file and checks that it reproduces the labels in both new files and that the clean file holds the right raw rows. Only then does it write both files with the re-derived threshold to `build/vol_threshold/`, reporting whether each is identical to the file now in `data/`;
3. if a file differs, review the difference and copy the verified file into `data/`;
4. regenerate `data/SHA256SUMS`, which step 1 made stale, with `shasum -a 256 data/processed/merged_features_clean.csv data/raw/merged_features_full.csv > data/SHA256SUMS`;
5. run `make results`, `make sensitivity_grid` and `make model`, then refresh the golden references the tests compare against:

   ```bash
   gzip -9 -n -c reports/metrics/oof_predictions.csv > tests/golden/reference_oof.csv.gz
   shasum -a 256 reports/metrics/walkforward_per_fold.csv reports/metrics/walkforward_summary.csv \
       reports/metrics/significance.csv reports/metrics/significance_crosssectional.csv \
       reports/metrics/operating_points.csv reports/metrics/sensitivity_grid.csv \
       tests/golden/reference_oof.csv.gz models/artifacts/final_model_fwdvolregime.joblib \
       > tests/golden/SHA256SUMS
   ```

   `-n` leaves the name and timestamp out of the gzip header, so with the same gzip build the same predictions give the same hash.

Updating the data changes the evaluation period and may change thresholds, class rates, model rankings, significance results, explainability, and calibration.

## Licensing and data use

The repository's MIT license applies to the source code.

The included CSV files are derivatives of market data obtained through their original providers. The MIT license does not grant rights to the underlying market data. Use, redistribution, and downstream publication remain subject to the applicable provider terms.

The files are included for research reproduction and educational use, not as an independently licensed market-data product.
