"""
Telco Churn Prediction API
FastAPI backend + Gradio UI mounted at /ui
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

# ── Artifact paths ────────────────────────────────────────────────────────────
PROJECT_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS_DIR  = os.path.join(PROJECT_ROOT, "artifacts")
MODEL_PATH     = os.path.join(ARTIFACTS_DIR, "model.pkl")
FEATURES_PATH  = os.path.join(ARTIFACTS_DIR, "feature_columns.json")

# ── Load model & feature list once at startup ─────────────────────────────────
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Run the pipeline first.")
if not os.path.exists(FEATURES_PATH):
    raise RuntimeError(f"Feature columns not found at {FEATURES_PATH}. Run the pipeline first.")

model         = joblib.load(MODEL_PATH)
feature_cols  = json.load(open(FEATURES_PATH))

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Telco Churn Predictor",
    description="XGBoost-based churn prediction API with Gradio UI at /ui",
    version="1.0.0",
)


# ── Pydantic schema ───────────────────────────────────────────────────────────
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int = 0
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# ── Shared inference helper ───────────────────────────────────────────────────
def run_inference(data: dict, threshold: float = 0.35) -> dict:
    df = pd.DataFrame([data])
    df = preprocess_data(df)
    df = build_features(df, target_col="Churn") if "Churn" in df.columns else build_features(df)

    # Convert booleans
    for c in df.select_dtypes(include=["bool"]).columns:
        df[c] = df[c].astype(int)

    # Align to training feature columns
    df = df.reindex(columns=feature_cols, fill_value=0)

    proba = float(model.predict_proba(df)[0][1])
    prediction = int(proba >= threshold)
    return {"churn_prediction": prediction, "churn_probability": round(proba, 4)}


# ── API endpoints ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: CustomerData):
    try:
        result = run_inference(customer.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Gradio inference wrapper ──────────────────────────────────────────────────
def gradio_predict(
    gender, partner, dependents, phone_service, multiple_lines,
    internet_service, online_security, online_backup, device_protection,
    tech_support, streaming_tv, streaming_movies, contract,
    paperless_billing, payment_method, tenure, monthly_charges, total_charges
):
    data = {
        "gender": gender,
        "SeniorCitizen": 0,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "tenure": int(tenure),
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }

    result      = run_inference(data)
    proba       = result["churn_probability"]
    prediction  = result["churn_prediction"]
    pct         = proba * 100

    # Risk tier
    if pct >= 70:
        risk_emoji = "🔴"
        risk_label = "HIGH RISK"
        risk_tip   = "Immediate retention action recommended."
    elif pct >= 40:
        risk_emoji = "🟡"
        risk_label = "MEDIUM RISK"
        risk_tip   = "Consider proactive outreach or discount offer."
    else:
        risk_emoji = "🟢"
        risk_label = "LOW RISK"
        risk_tip   = "Customer appears stable. Standard engagement."

    # Churn drivers hint
    drivers = []
    if contract == "Month-to-month":
        drivers.append("month-to-month contract")
    if internet_service == "Fiber optic" and online_security == "No":
        drivers.append("fiber optic with no online security")
    if payment_method == "Electronic check":
        drivers.append("electronic check payment")
    if tenure <= 6:
        drivers.append(f"short tenure ({int(tenure)} months)")
    if monthly_charges >= 70:
        drivers.append(f"high monthly charge (${monthly_charges:.0f})")

    drivers_text = (
        "⚠️  Risk factors detected: " + ", ".join(drivers) + "."
        if drivers else
        "✅ No major risk factors detected."
    )

    verdict = "WILL CHURN" if prediction == 1 else "WILL NOT CHURN"

    output = (
        f"{risk_emoji}  {risk_label}  —  {verdict}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Churn probability : {pct:.1f}%\n"
        f"Threshold used    : 35%\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{drivers_text}\n"
        f"💡 {risk_tip}"
    )

    return output, round(pct, 1)


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate",
    ),
    title="Telco Churn Predictor"
) as demo:

    gr.Markdown("""
    # 🔮 Telco Customer Churn Predictor
    **Powered by XGBoost + MLflow**  |  Predict which customers are at risk of leaving.

    > Fill in the customer profile below and click **Predict Churn** to get an instant risk assessment.
    """)

    with gr.Row():
        # ── Left column: inputs ───────────────────────────────────────────────
        with gr.Column(scale=2):

            with gr.Tab("👤 Demographics & Account"):
                gender           = gr.Dropdown(["Male", "Female"], label="Gender", value="Male")
                partner          = gr.Dropdown(["Yes", "No"], label="Partner", value="No")
                dependents       = gr.Dropdown(["Yes", "No"], label="Dependents", value="No")
                tenure           = gr.Slider(0, 72, value=1, step=1, label="Tenure (months)")
                monthly_charges  = gr.Slider(0, 200, value=85.0, step=0.5, label="Monthly Charges ($)")
                total_charges    = gr.Number(value=85.0, label="Total Charges ($)")

            with gr.Tab("📞 Phone & Internet Services"):
                phone_service    = gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes")
                multiple_lines   = gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines", value="No")
                internet_service = gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic")
                online_security  = gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security", value="No")
                online_backup    = gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup", value="No")
                device_protection = gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection", value="No")
                tech_support     = gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support", value="No")
                streaming_tv     = gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV", value="Yes")
                streaming_movies = gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies", value="Yes")

            with gr.Tab("💳 Contract & Billing"):
                contract         = gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month")
                paperless_billing = gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="Yes")
                payment_method   = gr.Dropdown([
                    "Electronic check", "Mailed check",
                    "Bank transfer (automatic)", "Credit card (automatic)"
                ], label="Payment Method", value="Electronic check")

        # ── Right column: outputs ─────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Prediction Result")
            result_text = gr.Textbox(
                label="Risk Assessment",
                lines=8,
                interactive=False,
                placeholder="Results will appear here after prediction..."
            )
            churn_prob = gr.Number(label="Churn Probability (%)", interactive=False)

            predict_btn = gr.Button("🔍 Predict Churn", variant="primary", size="lg")

            gr.Markdown("""
            ---
            **Risk Tiers**
            - 🔴 **High** — ≥ 70% probability
            - 🟡 **Medium** — 40–69% probability
            - 🟢 **Low** — < 40% probability
            """)

    # ── Examples ──────────────────────────────────────────────────────────────
    gr.Markdown("### 💡 Example Profiles")
    gr.Examples(
        examples=[
            ["Female", "No",  "No",  "Yes", "No",  "Fiber optic", "No",  "No",  "No",  "No",  "Yes", "Yes", "Month-to-month", "Yes", "Electronic check",          1,  85.0,   85.0],
            ["Male",   "Yes", "Yes", "Yes", "Yes", "DSL",         "Yes", "Yes", "Yes", "Yes", "No",  "No",  "Two year",        "No",  "Credit card (automatic)",  60,  45.0, 2700.0],
            ["Male",   "No",  "No",  "Yes", "No",  "Fiber optic", "No",  "No",  "No",  "No",  "No",  "No",  "Month-to-month", "Yes", "Electronic check",          3,  70.0,  210.0],
        ],
        inputs=[
            gender, partner, dependents, phone_service, multiple_lines,
            internet_service, online_security, online_backup, device_protection,
            tech_support, streaming_tv, streaming_movies, contract,
            paperless_billing, payment_method, tenure, monthly_charges, total_charges
        ],
        label="Click any row to load it"
    )

    # ── Wire button ───────────────────────────────────────────────────────────
    predict_btn.click(
        fn=gradio_predict,
        inputs=[
            gender, partner, dependents, phone_service, multiple_lines,
            internet_service, online_security, online_backup, device_protection,
            tech_support, streaming_tv, streaming_movies, contract,
            paperless_billing, payment_method, tenure, monthly_charges, total_charges
        ],
        outputs=[result_text, churn_prob]
    )


# ── Mount Gradio into FastAPI at /ui ──────────────────────────────────────────
app = gr.mount_gradio_app(app, demo, path="/ui")
