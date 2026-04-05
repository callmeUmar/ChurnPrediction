"""
Telco Churn Prediction — FastAPI + Gradio
API  → /predict  (POST)
UI   → /ui
Docs → /docs
"""

import os
import sys
import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Support both local dev and Docker (PYTHONPATH=/app/src)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.serving.inference import run_inference

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Telco Churn Predictor",
    description="XGBoost churn prediction API. UI available at /ui",
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


# ── API endpoints ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ui")

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: CustomerData):
    try:
        return run_inference(customer.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Gradio helpers ────────────────────────────────────────────────────────────
def probability_gauge(pct: float) -> str:
    """Returns an HTML visual gauge bar for churn probability."""
    if pct >= 70:
        color = "#ef4444"   # red
    elif pct >= 30:
        color = "#f59e0b"   # amber
    else:
        color = "#22c55e"   # green

    return f"""
    <div style="font-family:sans-serif; padding:12px;">
        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
            <span style="font-weight:600; font-size:14px;">Churn Probability</span>
            <span style="font-weight:700; font-size:18px; color:{color};">{pct:.1f}%</span>
        </div>
        <div style="background:#e5e7eb; border-radius:999px; height:14px; width:100%;">
            <div style="background:{color}; width:{min(pct,100):.1f}%; height:14px;
                        border-radius:999px; transition:width 0.4s ease;"></div>
        </div>
        <div style="display:flex; justify-content:space-between;
                    font-size:11px; color:#6b7280; margin-top:4px;">
            <span>0%</span><span>35% threshold</span><span>100%</span>
        </div>
    </div>
    """


def retention_advice(contract, internet_service, online_security,
                     payment_method, tenure, monthly_charges) -> str:
    """Returns targeted retention recommendations."""
    tips = []

    if contract == "Month-to-month":
        tips.append(" <b>Offer a discounted annual contract</b> — long-term contracts reduce churn by ~3×")
    if internet_service == "Fiber optic" and online_security == "No":
        tips.append(" <b>Bundle Online Security</b> — fiber customers without security churn significantly more")
    if payment_method == "Electronic check":
        tips.append(" <b>Incentivise auto-pay</b> — electronic check users have the highest churn rate")
    if tenure <= 6:
        tips.append(" <b>Trigger onboarding programme</b> — first 6 months is the highest-risk window")
    if monthly_charges >= 70:
        tips.append(" <b>Review pricing or add perks</b> — high-spend customers expect more value")

    if not tips:
        return "<p style='color:#22c55e; font-weight:600;'> No urgent retention actions needed.</p>"

    items = "".join(f"<li style='margin-bottom:8px;'>{t}</li>" for t in tips)
    return f"<ul style='padding-left:18px; font-family:sans-serif; font-size:14px;'>{items}</ul>"


def gradio_predict(
    gender, partner, dependents, phone_service, multiple_lines,
    internet_service, online_security, online_backup, device_protection,
    tech_support, streaming_tv, streaming_movies, contract,
    paperless_billing, payment_method, tenure, monthly_charges, total_charges
):
    data = {
        "gender": gender, "SeniorCitizen": 0,
        "Partner": partner, "Dependents": dependents,
        "tenure": int(tenure), "PhoneService": phone_service,
        "MultipleLines": multiple_lines, "InternetService": internet_service,
        "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_protection, "TechSupport": tech_support,
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        "Contract": contract, "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }

    result     = run_inference(data)
    proba      = result["churn_probability"]
    prediction = result["churn_prediction"]
    pct        = proba * 100

    # Verdict badge
    if pct >= 50:
        badge = "HIGH RISK — WILL CHURN"
        badge_color = "#ef4444"
    elif pct >= 30:
        badge = "MEDIUM RISK — MONITOR CLOSELY"
        badge_color = "#f59e0b"
    else:
        badge = "LOW RISK — STABLE CUSTOMER"
        badge_color = "#22c55e"

    verdict_html = f"""
    <div style="text-align:center; padding:16px; border-radius:12px;
                background:{badge_color}18; border:2px solid {badge_color};
                font-family:sans-serif;">
        <div style="font-size:22px; font-weight:700; color:{badge_color};">{badge}</div>
    </div>
    """

    gauge_html   = probability_gauge(pct)
    advice_html  = retention_advice(contract, internet_service, online_security,
                                    payment_method, tenure, monthly_charges)

    return verdict_html, gauge_html, advice_html


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
    title="Churn Predictor"
) as demo:

    gr.Markdown("""
    # 🔮 Telco Customer Churn Predictor
    Fill in the customer profile across the tabs, then click **Predict**.
    """)

    with gr.Row():

        # ── Input panel ───────────────────────────────────────────────────────
        with gr.Column(scale=2):
            with gr.Tab("👤 Demographics & Account"):
                gender          = gr.Dropdown(["Male", "Female"], label="Gender", value="Male")
                partner         = gr.Dropdown(["Yes", "No"], label="Partner", value="No")
                dependents      = gr.Dropdown(["Yes", "No"], label="Dependents", value="No")
                tenure          = gr.Slider(0, 72, value=1, step=1, label="Tenure (months)")
                monthly_charges = gr.Slider(0, 200, value=85.0, step=0.5, label="Monthly Charges ($)")
                total_charges   = gr.Number(value=85.0, label="Total Charges ($)")

            with gr.Tab("Services"):
                phone_service     = gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes")
                multiple_lines    = gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines", value="No")
                internet_service  = gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic")
                online_security   = gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security", value="No")
                online_backup     = gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup", value="No")
                device_protection = gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection", value="No")
                tech_support      = gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support", value="No")
                streaming_tv      = gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV", value="Yes")
                streaming_movies  = gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies", value="Yes")

            with gr.Tab("Contract & Billing"):
                contract          = gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month")
                paperless_billing = gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="Yes")
                payment_method    = gr.Dropdown([
                    "Electronic check", "Mailed check",
                    "Bank transfer (automatic)", "Credit card (automatic)"
                ], label="Payment Method", value="Electronic check")

        # ── Output panel ──────────────────────────────────────────────────────
        with gr.Column(scale=1):
            predict_btn  = gr.Button("🔍 Predict Churn", variant="primary", size="lg")
            verdict_out  = gr.HTML(label="Verdict")
            gauge_out    = gr.HTML(label="Probability")

            gr.Markdown("### 💡 Retention Recommendations")
            advice_out   = gr.HTML(label="Actions")

    # ── Example profiles ──────────────────────────────────────────────────────
    gr.Markdown("### 📋 Example Profiles")
    gr.Examples(
        label="Click a row to load it",
        examples=[
            # High churn risk example
            ["Female", "No",  "No",  "Yes", "No",  "Fiber optic",
             "No",  "No",  "No",  "No",  "Yes", "Yes", "Month-to-month", 
             "Yes", "Electronic check", 1,  85.0,   85.0],
            # Low churn risk example
            ["Male",   "Yes", "Yes", "Yes", "Yes", "DSL", 
             "Yes", "Yes", "Yes", "Yes", "No",  "No",  "Two year", 
             "No", "Credit card (automatic)", 60,  45.0, 2700.0],
        ],
        inputs=[
            gender, partner, dependents, phone_service, multiple_lines,
            internet_service, online_security, online_backup, device_protection,
            tech_support, streaming_tv, streaming_movies, contract,
            paperless_billing, payment_method, tenure, monthly_charges, total_charges
        ],
    )

    # ── Wire up ───────────────────────────────────────────────────────────────
    predict_btn.click(
        fn=gradio_predict,
        inputs=[
            gender, partner, dependents, phone_service, multiple_lines,
            internet_service, online_security, online_backup, device_protection,
            tech_support, streaming_tv, streaming_movies, contract,
            paperless_billing, payment_method, tenure, monthly_charges, total_charges
        ],
        outputs=[verdict_out, gauge_out, advice_out],
    )


# ── Mount Gradio into FastAPI ─────────────────────────────────────────────────
try:
    app = gr.mount_gradio_app(app, demo, path="/ui")
except Exception as e:
    print(f"Warning: Gradio UI could not be mounted: {e}")
