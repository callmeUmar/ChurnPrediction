# 🔮 Telco Customer Churn Prediction

A production-ready end-to-end Machine Learning system that predicts whether a telecom customer is likely to churn. Built with a full MLOps pipeline — from data validation and feature engineering to model training, experiment tracking, REST API serving, and an interactive web UI.

---

## 🚀 Live Demo

> **[churnprediction-1zdd.onrender.com](https://churnprediction-1zdd.onrender.com/ui)**

---

## 📌 Project Overview

The pipeline takes raw Telco customer data and:
1. Validates data quality with custom checks
2. Preprocesses and engineers features
3. Trains an XGBoost classifier with class imbalance handling
4. Tracks all experiments, metrics, and artifacts with MLflow
5. Serves predictions via a FastAPI REST API
6. Provides an interactive Gradio UI for real-time predictions
7. Containerised with Docker for deployment

---

## 🏗️ Project Structure

```
ChurnPrediction/
├── app/
│   └── main.py                  # FastAPI + Gradio serving app
├── src/
│   ├── data/
│   │   ├── load_data.py         # CSV loading with error handling
│   │   └── preprocess.py        # Data cleaning and type fixes
│   ├── features/
│   │   └── build_features.py    # Binary encoding + one-hot encoding
│   ├── models/
│   │   ├── train.py             # XGBoost training with MLflow logging
│   │   ├── tune.py              # Hyperparameter tuning with Optuna
│   │   └── evaluate.py          # Model evaluation metrics
│   ├── serving/
│   │   └── inference.py         # Inference pipeline (lazy model loading)
│   └── utils/
│       └── validate_data.py     # Data quality validation
├── scripts/
│   └── run_pipeline.py          # End-to-end training pipeline
├── tests/
│   ├── test_pipeline_phase1_data_features.py
│   ├── test_pipeline_phase2_modeling.py
│   └── test_fastapi.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── artifacts/                   # Saved model and feature columns
├── data/
│   └── raw/                     # Raw Telco CSV dataset
└── requirements.txt
```

---

## 🧰 Tech Stack

| Category | Library |
|---|---|
| **ML Model** | `xgboost` |
| **ML Utilities** | `scikit-learn`, `numpy`, `pandas`, `scipy` |
| **Hyperparameter Tuning** | `optuna` |
| **Experiment Tracking** | `mlflow` |
| **API Framework** | `fastapi`, `uvicorn`, `starlette` |
| **Data Validation** | `pydantic` |
| **Web UI** | `gradio` |
| **Model Serialisation** | `joblib` |
| **Environment** | `python-dotenv` |
| **Testing** | `pytest` |
| **Containerisation** | `docker`, `docker-compose` |
| **CI/CD** | GitHub Actions → Docker Hub → Render |

---

## ⚙️ How to Run Locally

### 1. Clone and set up environment
```bash
git clone https://github.com/callmeUmar/ChurnPrediction.git
cd ChurnPrediction
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the model
```bash
python scripts/run_pipeline.py --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 3. Start the API + UI
```bash
uvicorn app.main:app --reload
```

- Gradio UI → `http://localhost:8000/ui`
- API docs → `http://localhost:8000/docs`
- Health check → `http://localhost:8000/health`

---

## 🐳 Run with Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

---

## 📊 Model Details

- **Algorithm:** XGBoost Classifier
- **Class imbalance:** Handled via `scale_pos_weight`
- **Threshold:** 0.35 (optimised for recall — catching churners matters more than precision)
- **Hyperparameter tuning:** Optuna with 100 trials optimising recall via 3-fold cross-validation
- **Experiment tracking:** All runs logged to MLflow (params, metrics, model artifacts)

**Metrics logged:**
- Precision, Recall, F1 Score, ROC AUC
- Training time, Inference time
- Data quality pass/fail

---

## 🔌 API Usage

```bash
curl -X POST https://churnprediction-1zdd.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 85.0
  }'
```

**Response:**
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.7423
}
```

---

## 🗂️ Dataset

[Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

7,043 customers × 21 features including demographics, services, contract type, and billing information.

---

## 👤 Author

**Umar Turdumambetov**  
[github.com/callmeUmar](https://github.com/callmeUmar)
