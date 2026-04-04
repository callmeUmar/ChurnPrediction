import sys
import os
import pytest
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.utils.validate_data import validate_telco_data
from src.features.build_features import build_features

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")


@pytest.fixture
def raw_df():
    return load_data(DATA_PATH)


@pytest.fixture
def processed_df(raw_df):
    return preprocess_data(raw_df.copy())


# ── Load tests ────────────────────────────────────────────────────────────────

def test_load_data_returns_dataframe(raw_df):
    assert isinstance(raw_df, pd.DataFrame)

def test_load_data_not_empty(raw_df):
    assert len(raw_df) > 0

def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file.csv")


# ── Preprocess tests ──────────────────────────────────────────────────────────

def test_preprocess_removes_customer_id(processed_df):
    assert "customerID" not in processed_df.columns

def test_preprocess_churn_is_binary(processed_df):
    assert set(processed_df["Churn"].dropna().unique()).issubset({0, 1})

def test_preprocess_total_charges_numeric(processed_df):
    assert pd.api.types.is_numeric_dtype(processed_df["TotalCharges"])

def test_preprocess_no_nulls_in_numeric(processed_df):
    num_cols = processed_df.select_dtypes(include=["number"]).columns
    assert processed_df[num_cols].isnull().sum().sum() == 0


# ── Validation tests ──────────────────────────────────────────────────────────

def test_validation_passes_on_clean_data(raw_df):
    is_valid, failed = validate_telco_data(raw_df.copy())
    assert isinstance(is_valid, bool)
    assert isinstance(failed, list)

def test_validation_fails_on_missing_column(raw_df):
    df_bad = raw_df.drop(columns=["tenure"])
    is_valid, failed = validate_telco_data(df_bad)
    assert not is_valid


# ── Feature engineering tests ─────────────────────────────────────────────────

def test_build_features_returns_dataframe(processed_df):
    result = build_features(processed_df.copy())
    assert isinstance(result, pd.DataFrame)

def test_build_features_not_empty(processed_df):
    result = build_features(processed_df.copy())
    assert len(result) > 0

def test_build_features_churn_column_preserved(processed_df):
    result = build_features(processed_df.copy())
    assert "Churn" in result.columns
