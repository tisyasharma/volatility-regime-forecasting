# Forecasting Volatility Regimes in US Equities

A leakage-aware, walk-forward machine learning pipeline that forecasts whether the next 10
trading days will be a high-volatility regime across 14 US equities. The target is volatility
rather than direction because the EDA finds no next-day direction signal to exploit beyond
always predicting an up-day, while volatility is persistent enough to forecast. The project
tests whether that persistence amounts to measurable skill once leakage and naive baselines are
ruled out.

![PR-AUC by model on the next-day and forward volatility targets](reports/figures/target_reversal.png)

## Overview

Two targets are evaluated against a **matched naive baseline** that predicts a regime whenever
today's volatility already exceeds the per-ticker threshold defining the label. The next-day
target (`NextVolSpike`) overlaps its feature window by nine of ten days, so it mostly restates
today's regime and serves as a negative control. The forward target (`FwdVolRegime`) measures
volatility over the next 10 trading days, disjoint from the features, and is the genuine forecast
the saved model uses.

The result splits cleanly. On the next-day target a one-line naive rule reaches PR-AUC near 0.90
and beats every learned model, which is exactly why that target overstates skill. On the forward
target every score drops toward the base rate, and only the logistic regression adds a lift that
survives dependence-aware significance testing.

## Results

Forward target, mean across 28 walk-forward folds, with a no-skill PR-AUC floor near 0.27.

| Model | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| Majority | 0.00 | 0.00 | 0.00 | n/a |
| Persistence (pooled fixed threshold) | 0.33 | 0.33 | 0.32 | n/a |
| HAR | 0.30 | 0.26 | 0.26 | 0.32 |
| Matched naive | 0.32 | 0.32 | 0.32 | 0.33 |
| XGBoost | 0.33 | 0.44 | 0.36 | 0.35 |
| LightGBM | 0.33 | 0.50 | 0.38 | 0.36 |
| **Logistic Regression** | 0.32 | 0.55 | **0.38** | **0.37** |

The logistic regression adds +0.034 PR-AUC over the matched naive baseline, and the edge survives
a date-block bootstrap that clusters the 14 co-moving tickers by trading date (95% CI [0.009,
0.047], p=0.009, Holm p=0.028 across the three learned models). It is the only model that clears
that cross-sectional-aware bar. LightGBM is marginal, XGBoost is not significant, and HAR trails
the naive rule. The full comparison, the next-day results, the threshold and horizon grid,
operating points, explainability, and calibration are in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Leakage controls

Financial time-series models fail when future information leaks into training. Three
controls carry most of the weight, with the full account in
[docs/METHODOLOGY.md](docs/METHODOLOGY.md).

- **The label threshold uses only the past.** An expanding 80th-percentile threshold computed
  with `shift(1)` is persisted once and reused by the labels, the naive baseline, and HAR, so
  nothing re-derives it (`scripts/verify_vol_threshold.py`).
- **Feature and label windows do not overlap.** The forward target measures the next 10 days,
  disjoint from the 10-day feature window, which drops the mean feature-to-label correlation from
  ~0.96 for the next-day target to ~0.47 for the forward one.
- **Walk-forward with a horizon-exact purge.** Expanding-window training and non-overlapping
  63-day test windows, with a trading-day purge that drops any training row whose label window
  reaches into the test period (López de Prado, 2018), enforced in `src/split.py` and asserted in
  `tests/test_split.py`.

## Quickstart

```bash
pip install -r requirements.txt

python scripts/run_evaluation.py     # walk-forward metrics, both targets -> reports/metrics/
python scripts/significance.py        # dependence-aware significance -> reports/metrics/
python scripts/operating_points.py    # alert-budget operating points -> reports/metrics/
python scripts/make_figures.py        # metrics-derived figures -> reports/figures/
python scripts/sensitivity_grid.py    # threshold and horizon robustness grid -> reports/
python scripts/save_model.py          # train and save the final model -> models/artifacts/
pytest tests/                         # split and purge invariants
```

The notebooks reproduce the same pipeline interactively, from raw download through EDA, modeling,
and explainability. The SHAP, coefficient, and calibration figures are rendered by `03_explainability.ipynb`.

## Repository structure

```
volatility-regime-forecasting/
  config.yaml                 Central configuration (data, split, model params)
  Makefile                    Entry points for the pipeline stages
  data/
    raw/                      Full dataset including warm-up rows
    processed/                Model-ready clean dataset
  src/
    pipeline.py               Data loading and feature/label preparation
    split.py                  Expanding-window walk-forward splits with purge
    targets.py                Forward-disjoint target and the matched naive baseline
    modeling.py               Baselines, models, HAR, and evaluation
  scripts/                    Evaluation, significance, operating points, figures, model, sensitivity grid
  tests/                      Split and purge invariants
  notebooks/                  Data collection, EDA, modeling, explainability
  reports/
    metrics/                  Per-fold and aggregate metrics
    figures/                  Rendered figures
    model_card.md             Model summary, evaluation, deployment
    dataset_card.md           Provenance, structure, labeling, risks
  docs/
    METHODOLOGY.md            Full results, significance, and leakage detail
  models/artifacts/           Saved model
```

## Data

14 tickers across technology, finance, retail, airlines, defense, and energy, from 2015-07-20 to
2026-01-21 (37,002 model-ready rows). Prices come from Yahoo Finance and the VIX provides a
market-wide volatility signal. Features are technical indicators expressed as scale-free ratios
and z-scores so they compare across tickers. The [dataset card](reports/dataset_card.md) has the
full column list and the [model card](reports/model_card.md) has the modeling detail. The code is MIT-licensed, and the CSVs are derived from Yahoo Finance market data and included for research reproduction only.

## Limitations

- **The lift is modest.** About 0.03 PR-AUC over the matched naive baseline for the best model,
  statistically significant but small. This is a hard forecasting problem.
- **Cross-sectional dependence is in the primary interval, breadth is still limited.** The
  headline significance clusters the 14 co-moving tickers by trading date, so they count as one
  cross-section rather than 14 draws. The universe is still 14 names, so a wider one would sharpen
  the test.
- **Scope.** US large caps only, with no trading-cost model. The headline regime is the 80th
  percentile over 10 days, and the sensitivity grid shows the edge holds for short horizons at
  moderate thresholds and fades by the 21-day horizon.

The [methodology](docs/METHODOLOGY.md) has the full limitations.

## Learn more

- **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)** covers the full results, significance, operating
  points, explainability, calibration, and leakage detail.
- **[Model card](reports/model_card.md)** covers the model summary, intended uses, evaluation, and
  deployment.
- **[Dataset card](reports/dataset_card.md)** covers provenance, structure, labeling, and risks.
