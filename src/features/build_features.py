import pandas as pd

def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Feature engineering for Telco churn dataset.
    - Binary Yes/No columns → 0/1
    - One-hot encode remaining categoricals
    - Drop original string columns replaced by encoding
    """
    df = df.copy()

    # Binary encode Yes/No columns
    binary_cols = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling",
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0, "No phone service": 0, "No internet service": 0})

    # Engineered features — high signal for churn
    if "tenure" in df.columns and "MonthlyCharges" in df.columns:
        # Total value of customer
        df["tenure_x_monthly"] = df["tenure"] * df["MonthlyCharges"]
        # New customers are high risk
        df["is_new_customer"] = (df["tenure"] <= 3).astype(int)
        # Long-term customers are low risk
        df["is_long_term"] = (df["tenure"] >= 24).astype(int)

    if "TotalCharges" in df.columns and "MonthlyCharges" in df.columns:
        # Avg monthly spend vs current — drift indicator
        df["avg_monthly_ratio"] = (
            df["TotalCharges"] / (df["tenure"].replace(0, 1)) / df["MonthlyCharges"].replace(0, 1)
        )

    # Number of services subscribed — more services = less likely to churn
    service_cols = [
        "PhoneService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    existing = [c for c in service_cols if c in df.columns]
    if existing:
        df["num_services"] = df[existing].sum(axis=1)

    # One-hot encode remaining categoricals (Gender, InternetService, Contract, PaymentMethod)
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df
