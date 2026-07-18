# Model Card

## Model overview

- **Model name:** Forward volatility-regime classifier
- **Version:** 2.1
- **Task:** Rank whether a U.S. equity will enter a high-volatility regime over the next 10 trading days
- **Primary target:** `FwdVolRegime`
- **Saved model:** Logistic regression bundled with its fitted `StandardScaler`
- **Artifact:** `models/artifacts/final_model_fwdvolregime.joblib`
- **Context:** Research and educational forecasting study, not a production trading system

The model is selected from logistic regression, LightGBM, and XGBoost using expanding-window performance on the forward-disjoint target. Logistic regression has the highest mean walk-forward average precision and is the only learned model whose improvement over the matched naive baseline remains significant under the primary cross-sectional-aware test after multiple-comparison correction.

A separate next-day target, `NextVolSpike`, is retained as a negative control. Its rolling-volatility label window overlaps the current feature window by nine of ten days, allowing a matched naive rule to outperform every learned model.

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

The full-sample positive rate is approximately 25.4%. The universe contains U.S. large-cap equities across six sectors, with correlated names that reduce effective cross-sectional breadth.

See the [dataset card](dataset_card.md) for complete provenance and label construction.

## Features

The saved model uses nine features known at time `t`:

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

Logistic regression and LightGBM use balanced class weights. XGBoost derives `scale_pos_weight` from each training fold's class ratio. Hyperparameters are stored in [`config.yaml`](../config.yaml), and random seed 42 is used for the learned models.

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

### Primary significance result

The primary inference uses a 21-trading-day date-block bootstrap that resamples complete trading-date cross-sections and preserves co-movement among the 14 equities.

For logistic regression versus the matched naive baseline:

- **AP edge:** 0.0335
- **95% CI:** [0.0093, 0.0474]
- **Unadjusted p:** 0.0093
- **Holm-adjusted p across logistic regression, LightGBM, and XGBoost:** 0.028
- **Out-of-fold rows:** 24,682
- **Distinct trading dates:** 1,763
- **Bootstrap resamples:** 3,000

Logistic regression is the only learned model whose improvement remains significant after the primary dependence-aware test and multiple-comparison correction. LightGBM has an unadjusted date-block p-value of 0.0420 but does not remain significant after correction. XGBoost is not significant.

### Sensitivity

The logistic-regression lift is significant across all tested thresholds at the 5-day horizon and at the 70th and 80th percentile thresholds for the 10-day horizon. No tested 21-day configuration is significant.

See [`docs/METHODOLOGY.md`](../docs/METHODOLOGY.md) and `reports/metrics/sensitivity_grid.csv` for the full grid.

### Operating points

At a pooled 20% alert budget:

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Logistic regression | 0.6226 | 0.4651 | 0.5325 |
| Matched naive | 0.5285 | 0.3948 | 0.4519 |

Logistic regression has higher precision and recall at every tested pooled alert budget from 5% through 30%. These are retrospective comparisons, not a selected deployment threshold.

## Interpretation and calibration

The logistic-regression model is interpreted through standardized coefficients across walk-forward folds. VIX and current rolling volatility receive stable positive weights.

A separate XGBoost model is trained through 2022 and explained with SHAP on a 2023-onward holdout. VIX is the dominant nonlinear feature, and current rolling volatility is another major contributor. These SHAP values explain XGBoost, not the saved logistic-regression model.

The learned models use class weighting to prioritize minority-class recall. Raw scores are not assumed to be calibrated probabilities. On the 2023-onward holdout, XGBoost and LightGBM overpredict the observed regime rate. Isotonic calibration improves LightGBM's Brier score on that analysis but remains sensitive to market-regime shift.

Any probability-based use would require new prospective calibration on unseen data.

## Limitations and risks

- **Modest effect size:** the primary AP improvement is 0.0335.
- **Limited breadth:** the universe contains 14 U.S. large-cap equities across six sectors, including correlated names.
- **Horizon dependence:** the lift holds at short horizons and does not remain significant at 21 trading days.
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
    "model": ...,      # LogisticRegression and StandardScaler bundle
    "features": [...],
    "target": "FwdVolRegime",
}
```

The committed artifact was serialized with scikit-learn 1.8.0. Regenerate it with `make model` when using a materially different scikit-learn version. The returned score is suitable for ranking within this research workflow and should not be treated as a calibrated probability without additional validation.

## Maintenance

To refresh the artifact:

1. update the source data;
2. rerun data collection and threshold verification;
3. rerun walk-forward evaluation and significance testing;
4. confirm that model selection still clears the matched baseline;
5. run `make model`.

No rollback procedure is defined because the model is not deployed.
