"""
Baselines and models for volatility-regime prediction (next-day spike and forward-disjoint
regime targets).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score
)


class MajorityClassifier(BaseEstimator, ClassifierMixin):
    """Always predict the majority class (normal = 0)."""

    def fit(self, X, y):
        self.majority_class_ = pd.Series(y).mode()[0]
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_class_)

    def predict_proba(self, X):
        proba = np.zeros((len(X), 2))
        proba[:, int(self.majority_class_)] = 1.0
        return proba


class PersistenceClassifier(BaseEstimator, ClassifierMixin):
    """
    Predict spike if current volatility is high.

    Uses 80th percentile of training RollingVol as threshold.
    """

    def __init__(self, threshold_quantile=0.8):
        self.threshold_quantile = threshold_quantile
        self.threshold_ = None

    def fit(self, X, y):
        if "RollingVol" not in X.columns:
            raise ValueError("PersistenceClassifier requires 'RollingVol' feature")
        self.threshold_ = X["RollingVol"].quantile(self.threshold_quantile)
        return self

    def predict(self, X):
        return (X["RollingVol"] > self.threshold_).astype(int).values

    def predict_proba(self, X):
        preds = self.predict(X)
        proba = np.zeros((len(X), 2))
        proba[preds == 0, 0] = 1.0
        proba[preds == 1, 1] = 1.0
        return proba


def train_baseline_majority(X_train, y_train):
    """Train a majority-class baseline and return the fitted model."""
    model = MajorityClassifier()
    model.fit(X_train, y_train)
    return model


def train_baseline_persistence(X_train, y_train, threshold_quantile=0.8):
    """Train a volatility threshold baseline and return the fitted model."""
    model = PersistenceClassifier(threshold_quantile=threshold_quantile)
    model.fit(X_train, y_train)
    return model


HAR_FEATURES = ["har_1", "har_5", "har_22"]


def add_har_features(df):
    """
    Add Corsi HAR components of rolling (historical) volatility to the dataframe.

    Builds the 1-, 5-, and 22-day views of RollingVol per ticker plus the continuous next-day
    volatility used as the HAR regression target. RollingVol (a 10-day rolling std of returns)
    stands in for the intraday realized volatility of classic HAR-RV. Assumes the frame is
    date-ordered within each ticker.
    """
    df = df.copy()
    vol = df.groupby("Ticker")["RollingVol"]
    df["har_1"] = df["RollingVol"]
    df["har_5"] = vol.transform(lambda s: s.rolling(5).mean())
    df["har_22"] = vol.transform(lambda s: s.rolling(22).mean())
    df["har_next_vol"] = vol.shift(-1)
    return df


def train_har(X_har, target_vol):
    """
    Fit the HAR volatility forecaster.

    OLS regresses the target volatility (next-day or forward, depending on the task) on the
    HAR components. The forecast is later thresholded against the same expanding threshold
    that defines the classification label, so HAR competes on identical terms.
    """
    X = np.asarray(X_har, dtype=float)
    y = np.asarray(target_vol, dtype=float)
    design = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    scale = float(np.std(y)) or 1.0
    return {"coef": coef, "scale": scale}


def har_forecast(model, X_har):
    """Return the HAR volatility forecast for a set of rows."""
    X = np.asarray(X_har, dtype=float)
    design = np.column_stack([np.ones(len(X)), X])
    return design @ model["coef"]


def threshold_to_prediction(forecast_vol, threshold, scale):
    """
    Turn a volatility forecast into a (prediction, score) pair against a threshold.

    The score is the sigmoid of the forecast margin, uncalibrated but monotone in
    the margin, so it ranks days for PR-AUC. It should not be read as a probability.
    """
    forecast_vol = np.asarray(forecast_vol, dtype=float)
    threshold = np.asarray(threshold, dtype=float)
    y_pred = (forecast_vol > threshold).astype(int)
    score = 1.0 / (1.0 + np.exp(-(forecast_vol - threshold) / scale))
    return y_pred, score


def train_logistic_regression(X_train, y_train, **params):
    """Train scaled logistic regression and return a model/scaler bundle."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    default_params = {
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": 42,
    }
    default_params.update(params)

    model = LogisticRegression(**default_params)
    model.fit(X_scaled, y_train)
    return {"model": model, "scaler": scaler}


def train_lightgbm(X_train, y_train, **params):
    """Train a LightGBM classifier and return the fitted model."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        raise ImportError("LightGBM not installed. Run: pip install lightgbm")

    default_params = {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 5,
        "num_leaves": 31,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
        "random_state": 42,
        "verbose": -1,
    }
    default_params.update(params)

    model = LGBMClassifier(**default_params)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, **params):
    """
    Train an XGBoost classifier and return the fitted model.

    Unless the caller supplies scale_pos_weight, it is derived from the training class
    ratio (negatives over positives) so XGBoost handles imbalance on the same terms as
    the class-weighted logistic regression and LightGBM it is compared against.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")

    default_params = {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": 42,
    }
    default_params.update(params)

    if "scale_pos_weight" not in default_params:
        y_arr = np.asarray(y_train)
        n_pos = float((y_arr == 1).sum())
        n_neg = float(len(y_arr) - n_pos)
        if n_pos > 0 and n_neg > 0:
            default_params["scale_pos_weight"] = n_neg / n_pos

    model = XGBClassifier(**default_params)
    model.fit(X_train, y_train)
    return model


# One builder per evaluated model, shared by the walk-forward driver and the artifact
# saver so both train through identical code paths. Each takes (X_train, y_train, cfg)
# where cfg is the models section of config.yaml.
MODEL_BUILDERS = {
    "Majority": lambda Xtr, ytr, cfg: train_baseline_majority(Xtr, ytr),
    "Persistence": lambda Xtr, ytr, cfg: train_baseline_persistence(Xtr, ytr),
    "LogisticRegression": lambda Xtr, ytr, cfg: train_logistic_regression(Xtr, ytr),
    "LightGBM": lambda Xtr, ytr, cfg: train_lightgbm(Xtr, ytr, **cfg.get("lightgbm", {})),
    "XGBoost": lambda Xtr, ytr, cfg: train_xgboost(Xtr, ytr, **cfg.get("xgboost", {})),
}


def predict_with_model(model, X_test):
    """
    Return (y_pred, y_proba) for a fitted model.

    Accepts either a fitted estimator or the dict bundle train_logistic_regression
    returns, which holds the classifier under "model" and its fitted StandardScaler
    under "scaler". Bundle inputs have their features scaled before predicting.
    """
    if isinstance(model, dict):
        scaler = model["scaler"]
        clf = model["model"]
        X_scaled = scaler.transform(X_test)
        y_pred = clf.predict(X_scaled)
        y_proba = clf.predict_proba(X_scaled)[:, 1]
        return y_pred, y_proba

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return y_pred, y_proba


def evaluate_model(y_true, y_pred, y_proba, pos_label=1):
    """
    Compute precision/recall/F1/PR-AUC and return a metrics dict.

    Pass y_proba=None for models that only emit hard labels (majority, persistence).
    Average precision needs a graded score to rank by, so pr_auc is NaN in that case
    rather than a fabricated number. It is likewise NaN when y_true has a single class.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_true_bin = (y_true_arr == pos_label).astype(int)

    if y_proba is None or len(np.unique(y_true_bin)) < 2:
        pr_auc = float("nan")
    else:
        y_proba_arr = np.asarray(y_proba)
        proba_pos = y_proba_arr if pos_label == 1 else 1 - y_proba_arr
        pr_auc = average_precision_score(y_true_bin, proba_pos)

    return {
        "precision": precision_score(y_true_arr, y_pred_arr, pos_label=pos_label, zero_division=0),
        "recall": recall_score(y_true_arr, y_pred_arr, pos_label=pos_label, zero_division=0),
        "f1": f1_score(y_true_arr, y_pred_arr, pos_label=pos_label, zero_division=0),
        "pr_auc": pr_auc,
    }
