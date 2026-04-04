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

    # One-hot encode remaining categoricals (Gender, InternetService, Contract, PaymentMethod)
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df
