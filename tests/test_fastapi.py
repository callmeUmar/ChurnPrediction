import sys
import os
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

SAMPLE_INPUT = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 55.0,
    "TotalCharges": 660.0
}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_200():
    response = client.post("/predict", json=SAMPLE_INPUT)
    assert response.status_code == 200


def test_predict_response_has_churn_key():
    response = client.post("/predict", json=SAMPLE_INPUT)
    data = response.json()
    assert "churn_prediction" in data


def test_predict_churn_is_binary():
    response = client.post("/predict", json=SAMPLE_INPUT)
    data = response.json()
    assert data["churn_prediction"] in [0, 1]


def test_predict_missing_field_returns_422():
    bad_input = SAMPLE_INPUT.copy()
    del bad_input["MonthlyCharges"]
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422
