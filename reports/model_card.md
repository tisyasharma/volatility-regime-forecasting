# Model Card

## Model overview

- **Model name:** Forward volatility-regime classifier
- **Version:** 2.1
- **Task:** Rank whether a U.S. equity will be in a high-volatility regime over the next 10 trading days
- **Primary target:** `FwdVolRegime`
- **Saved model:** Logistic regression bundled with its fitted `StandardScaler`
- **Artifact:** `models/artifacts/final_model_fwdvolregime.joblib`
- **Context:** Research and educational forecasting study, not a production trading system

The model is selected from logistic regression, LightGBM, and XGBoost using expanding-window performance on the forward-disjoint target. Logistic regression has the highest mean walk-forward average precision and is the only learned model whose improvement over the matched naive baseline remains significant under both dependence-aware bootstraps after a Holm correction across the three. It was chosen on the same 28 quarters that report its edge, so that edge is likely optimistic (see Limitations and risks).

A separate next-day target, `NextVolSpike`, is retained as a negative control. Its rolling-volatility label window overlaps the current feature window by nine of ten days, so a matched naive rule scores near 0.90. That rule uses the per-ticker threshold the learned models do not receive, so its lead over every learned model is not evidence that persistence beats them.

## Intended use

Appropriate uses include:

- studying volatility clustering and target construction;
- comparing machine learning models with matched naive and econometric baselines;
- demonstrating purged expanding-window validation;
- teaching dependence-aware inference for panel time series;
- reproducing the results reported in this repository.

The model is not intended for:

- live trading or investment advice;
- automated position sizing or execution;
- profitability claims;
- decisions that assume forecasts are net of transaction costs;
- probability-based decisions without new out-of-sample calibration;
- claims that performance generalizes beyond the tested equities, horizons, and market period.

## Data and target

The source dataset is `data/processed/merged_features_clean.csv`, containing 37,002 model-ready rows for 14 U.S. equities from 2015-07-20 through 2026-01-21. Rows without a complete future target are excluded before fitting.

`FwdVolRegime` is positive when volatility over the next 10 trading days exceeds a per-ticker expanding 80th-percentile threshold computed from past volatility only. The target window covers `t+1` through `t+10` and is disjoint from the current 10-day rolling-volatility feature window.

The full-sample positive rate is approximately 25.4%. The universe contains U.S. large-cap equities in six industry groups, with correlated names that reduce effective cross-sectional breadth.

See the [dataset card](dataset_card.md) for complete provenance and label construction.

## Features

The saved model uses nine features known by the end of trading day `t`:

1. `Return`
2. `RollingVol`
3. `RSI`
4. `Price_to_SMA20`
5. `Price_to_SMA50`
6. `SMA20_to_SMA50`
7. `Volume_Z`
8. `VIX`
9. `Lagged_Return`

## Training procedure

The saved logistic-regression artifact contains both the classifier and a `StandardScaler`. During walk-forward evaluation, the scaler is fitted separately on each training fold and applied to the corresponding test fold. The final artifact is then trained on all complete rows available for the configured target.

Models included in the comparison:

- majority classifier;
- persistence classifier;
- matched naive volatility-threshold baseline;
- HAR-style econometric volatility benchmark (Corsi, 2009);
- logistic regression;
- LightGBM;
- XGBoost.

Logistic regression and LightGBM use balanced class weights. XGBoost derives `scale_pos_weight` from each training fold's class ratio. The tree-model hyperparameters are stored in [`config.yaml`](../config.yaml). How the tree-model hyperparameters were chosen is not recorded, so tuning on the same quarters that report the results cannot be ruled out. Logistic regression uses scikit-learn's defaults (C = 1, lbfgs) with `max_iter=1000`, set in `src/modeling.py`. Random seed 42 is used for the learned models.

## Evaluation

### Validation protocol

- expanding-window walk-forward validation;
- 28 usable folds for `FwdVolRegime`;
- non-overlapping 63-trading-day full test windows;
- training begins on 2015-07-20 and expands after each fold;
- horizon-exact purging removes training rows whose label windows reach the test period;
- scaling is fitted on the training fold only.

[`tests/test_split.py`](../tests/test_split.py) verifies temporal ordering, train-test separation, purge boundaries, incomplete-label handling, and final-fold behavior.

### Metrics

Reported metrics are precision, recall, F1, and **average precision (AP)**.

### Forward-target results

Mean across 28 walk-forward folds:

| Model | Precision | Recall | F1 | Average precision |
|---|---:|---:|---:|---:|
| HAR | 0.2956 | 0.2563 | 0.2644 | 0.3226 |
| Matched naive | 0.3157 | 0.3213 | 0.3155 | 0.3342 |
| XGBoost | 0.3276 | 0.4400 | 0.3620 | 0.3499 |
| LightGBM | 0.3300 | 0.5017 | 0.3805 | 0.3604 |
| **Logistic regression** | **0.3164** | **0.5450** | **0.3849** | **0.3677** |

The logistic-regression improvement over the matched naive baseline is **0.0335 AP points**.

### Significance

Two dependence-aware bootstraps test the logistic-regression edge over the matched naive baseline, and neither is designated primary.

- **AP edge:** 0.0335
- **Fold-block bootstrap** (blocks of 4 consecutive folds, 20,000 resamples): 95% CI [0.0136, 0.0605], p=0.0022, Holm-adjusted p=0.0066
- **Date-block bootstrap** (21-trading-day blocks of dates with their full cross-section, 3,000 resamples, 24,682 out-of-fold rows on 1,763 trading dates): 95% CI [0.0093, 0.0475], p=0.0087, Holm-adjusted p=0.026

Holm adjusts across logistic regression, LightGBM, and XGBoost. Logistic regression is the only learned model whose improvement remains significant under both tests after correction. LightGBM's unadjusted p-values are below 0.05 under both tests but neither survives Holm correction, and XGBoost is not significant under either. Only the fold-block test resamples across market periods, since the date-block test keeps the 28 test windows fixed. Both block lengths are set in the code rather than estimated from the data.

### Sensitivity

Under the fold-block test, unadjusted across the nine cells, the logistic-regression lift is significant at every tested threshold at the 5-day horizon and at the 70th and 80th percentile thresholds for the 10-day horizon. At 21 days every interval includes zero, so the grid neither establishes nor rules out an edge there. Each cell also rebuilds the threshold and the naive baseline from that horizon's trailing volatility, and the number of scored folds varies from 26 to 29.

See [`docs/METHODOLOGY.md`](../docs/METHODOLOGY.md) and `reports/metrics/sensitivity_grid.csv` for the full grid.

### Operating points

At a pooled 20% alert budget:

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Logistic regression | 0.6226 | 0.4651 | 0.5325 |
| Matched naive | 0.5285 | 0.3948 | 0.4519 |

Logistic regression has higher precision and recall at every tested pooled alert budget from 5% through 30%. These are retrospective comparisons, not a selected deployment threshold. The pooled cutoff is one score quantile over all 28 quarters together, so it depends on later data.

## Interpretation and calibration

Logistic regression, the saved model's specification, is interpreted through its standardized coefficients refit in each walk-forward training window. VIX and current rolling volatility receive positive weights in every fold, though the folds' nested training windows make that agreement weaker evidence than independent samples would give.

A separate XGBoost model is trained through 2022 and explained with SHAP on the 2023-onward rows, which it never trained on. They are not untouched, since the walk-forward comparison that chose the final model includes those quarters. VIX is the dominant nonlinear feature, and current rolling volatility is another major contributor. These SHAP values explain XGBoost, not the saved logistic-regression model.

The learned models use class weighting to prioritize minority-class recall. Raw scores are not assumed to be calibrated probabilities. On those 2023-onward rows, XGBoost and LightGBM overpredict the observed regime rate. An isotonic-calibrated LightGBM, trained through 2021 and calibrated on 2022, has a lower Brier score than LightGBM trained through 2022, but the two differ in training window as well as calibration, and the calibration remains sensitive to market-regime shift. None of these Brier scores beats the roughly 0.14 that a constant forecast at the evaluation period's own rate (about 0.17) would score, a reference that uses that rate in hindsight.

Any probability-based use would require new prospective calibration on unseen data.

## Limitations and risks

- **Modest effect size:** the logistic-regression AP improvement is 0.0335.
- **Selection on the reported quarters:** logistic regression was chosen as the learned model with the highest mean average precision on the same 28 quarters that report its edge. The Holm correction accounts for testing three candidates, but the chosen model's edge is likely optimistic, and no untouched period has been evaluated.
- **Bootstrap settings:** both bootstraps use block lengths set in the code (4 folds and 21 trading days) rather than estimated from the data, and their blocks do not wrap around, so the folds and dates within one block length of either end are drawn less often than the rest. Results near 0.05, such as LightGBM's unadjusted p-values, may change with these settings.
- **Equal weighting:** the headline average precision is an equal-weighted mean over 28 quarters whose positive rates range from 4.2% to 69.7%.
- **Concentration:** logistic regression's AP is above the naive rule's in 19 of the 28 quarters, and the five largest per-quarter differences in `walkforward_per_fold.csv` sum to about 62% of the total.
- **Limited breadth:** the universe contains 14 U.S. large-cap equities in six industry groups, including correlated names.
- **Horizon dependence:** in the sensitivity grid's fold-block test, unadjusted across its nine cells, the edge is significant at every tested threshold for the 5-day horizon and at the 70th and 80th percentiles for 10 days. At 21 days the intervals include zero, which neither shows nor rules out an edge.
- **No trading-cost model:** transaction costs, slippage, borrow, execution, portfolio construction, and risk-adjusted returns are not modeled.
- **Calibration drift:** raw class-weighted scores are not trustworthy probabilities without fresh calibration.
- **No live serving environment:** the saved model is a research and reproducibility artifact.
- **Human oversight:** outputs should be interpreted as analytical forecasts, not automated decisions.

## Artifact use

```python
import joblib

from src.modeling import predict_with_model

artifact = joblib.load("models/artifacts/final_model_fwdvolregime.joblib")
y_pred, y_score = predict_with_model(
    artifact["model"],
    X[artifact["features"]],
)
```

The artifact contains:

```python
{
    "model_name": "LogisticRegression",
    "model": ...,       # LogisticRegression and StandardScaler bundle
    "features": [...],
    "target": "FwdVolRegime",
    "provenance": {...},  # data path and SHA-256, data filters, training rows and window,
                          # forward window, library versions
}
```

The committed artifact was saved under the locked environment, and its provenance records the library versions it was built with. `tests/test_model_artifact.py` loads it with scikit-learn's version-mismatch warning raised as an error and compares its predictions with a fresh fit. The golden tests check that `make model` reproduces it: the same metadata exactly, and the same scores to within 1e-12. scikit-learn warns that a model loaded under a different version may break or give invalid results, so after any change to `uv.lock`, regenerate the artifact with `make model` and update its line in `tests/golden/SHA256SUMS`. A joblib file is a pickle and can run code when loaded, so load only artifacts from a trusted source. The returned score is suitable for ranking within this research workflow and should not be treated as a calibrated probability without additional validation.

## Maintenance

To refresh the artifact:

1. update the source data;
2. rerun data collection and threshold verification;
3. rerun walk-forward evaluation and significance testing;
4. confirm that model selection still clears the matched baseline;
5. run `make model` and update the artifact's line in `tests/golden/SHA256SUMS`. When the source data changed, the dataset card's Maintenance section also updates `data/SHA256SUMS` and the other golden references.

No rollback procedure is defined because the model is not deployed.
