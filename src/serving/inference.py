"""
Inference module — loads the trained model and exposes run_inference().
Used by app/main.py for both the FastAPI endpoint and Gradio UI.
"""

import os
import sys
import json
import joblib
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

# ── Artifact paths ────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
MODEL_PATH    = os.path.join(ARTIFACTS_DIR, "model.pkl")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "feature_columns.json")

# ── Lazy globals (loaded on first request, not at import time) ────────────────
_model        = None
_feature_cols = None


def _load_artifacts():
    global _model, _feature_cols
    if _model is not None:
        return
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run scripts/run_pipeline.py first.")
    if not os.path.exists(FEATURES_PATH):
        raise RuntimeError(f"Feature columns not found at {FEATURES_PATH}. Run scripts/run_pipeline.py first.")
    _model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH) as f:
        _feature_cols = json.load(f)
    print(f"Model loaded from {MODEL_PATH}")
    print(f"Feature columns loaded: {len(_feature_cols)} features")


# ── Main inference function ───────────────────────────────────────────────────
def run_inference(data: dict, threshold: float = 0.50) -> dict:
    """
    Takes a raw customer dict, runs the full preprocessing + feature
    engineering pipeline, and returns churn prediction + probability.

    Args:
        data:      Raw customer data matching CustomerData schema.
        threshold: Classification threshold (default 0.35).

    Returns:
        {
            "churn_prediction":  0 or 1,
            "churn_probability": float (0.0 – 1.0)
        }
    """
    _load_artifacts()

    df = pd.DataFrame([data])
    df = preprocess_data(df)
    df = build_features(df, target_col="Churn") if "Churn" in df.columns else build_features(df)

    for c in df.select_dtypes(include=["bool"]).columns:
        df[c] = df[c].astype(int)

    df = df.reindex(columns=_feature_cols, fill_value=0)

    proba      = float(_model.predict_proba(df)[0][1])
    prediction = int(proba >= threshold)

    return {
        "churn_prediction":  prediction,
        "churn_probability": round(proba, 4),
    }
