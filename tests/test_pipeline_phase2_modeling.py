import sys
import os
import pytest
import pandas as pd
from unittest.mock import patch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.models.tune import tune_model
from src.models.train import train_model

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")


@pytest.fixture(scope="module")
def feature_df():
    df = load_data(DATA_PATH)
    df = preprocess_data(df)
    df = build_features(df)
    return df


@pytest.fixture(scope="module")
def X_y(feature_df):
    X = feature_df.drop(columns=["Churn"])
    y = feature_df["Churn"]
    return X, y


# ── Tune tests ────────────────────────────────────────────────────────────────

def test_tune_model_returns_dict(X_y):
    X, y = X_y
    # Use 2 trials to keep the test fast
    with patch("src.models.tune.optuna.create_study") as mock_study:
        mock_study.return_value.best_params = {
            "n_estimators": 300,
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
        mock_study.return_value.optimize = lambda *a, **kw: None
        best_params = tune_model(X, y)
    assert isinstance(best_params, dict)

def test_tune_model_has_expected_keys(X_y):
    X, y = X_y
    with patch("src.models.tune.optuna.create_study") as mock_study:
        mock_study.return_value.best_params = {
            "n_estimators": 300,
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
        mock_study.return_value.optimize = lambda *a, **kw: None
        best_params = tune_model(X, y)
    expected_keys = {"n_estimators", "learning_rate", "max_depth", "subsample", "colsample_bytree"}
    assert expected_keys.issubset(best_params.keys())


# ── Train tests ───────────────────────────────────────────────────────────────

def test_train_model_runs(feature_df):
    with patch("src.models.train.mlflow"):
        train_model(feature_df.copy(), target_col="Churn")

def test_train_model_missing_target_raises(feature_df):
    with pytest.raises(KeyError):
        with patch("src.models.train.mlflow"):
            train_model(feature_df.copy(), target_col="NonExistentColumn")
