import pandas as pd
from typing import Tuple, List


def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates Telco churn dataset using plain pandas checks.
    Returns (is_valid, list_of_failed_checks).
    """

    print("Starting data validation...")
    failed = []

    # ── Schema: required columns ──────────────────────────────────────────────
    print("Validating schema and required columns...")
    required_cols = [
        "customerID", "gender", "Partner", "Dependents",
        "PhoneService", "InternetService", "Contract",
        "tenure", "MonthlyCharges", "TotalCharges"
    ]
    for col in required_cols:
        if col not in df.columns:
            failed.append(f"Missing required column: '{col}'")

    # Stop early — remaining checks need the columns to exist
    if failed:
        print(f"Schema validation failed: {failed}")
        return False, failed

    # ── Null checks ───────────────────────────────────────────────────────────
    for col in ["customerID", "tenure", "MonthlyCharges"]:
        if df[col].isnull().any():
            failed.append(f"Null values found in column '{col}'")

    # ── Allowed value checks ──────────────────────────────────────────────────
    print("Validating business logic constraints...")

    allowed = {
        "gender":          {"Male", "Female"},
        "Partner":         {"Yes", "No"},
        "Dependents":      {"Yes", "No"},
        "PhoneService":    {"Yes", "No"},
        "InternetService": {"DSL", "Fiber optic", "No"},
        "Contract":        {"Month-to-month", "One year", "Two year"},
    }
    for col, valid_set in allowed.items():
        bad = set(df[col].dropna().unique()) - valid_set
        if bad:
            failed.append(f"Invalid values in '{col}': {bad}")

    # ── Numeric range checks ──────────────────────────────────────────────────
    print("Validating numeric ranges...")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if (df["tenure"] < 0).any():
        failed.append("'tenure' contains negative values")
    if (df["tenure"] > 120).any():
        failed.append("'tenure' contains values above 120 months")
    if (df["MonthlyCharges"] < 0).any():
        failed.append("'MonthlyCharges' contains negative values")
    if (df["MonthlyCharges"] > 200).any():
        failed.append("'MonthlyCharges' contains values above $200")
    if (df["TotalCharges"] < 0).any():
        failed.append("'TotalCharges' contains negative values")

    # ── Business logic: TotalCharges >= MonthlyCharges (95% of rows) ─────────
    valid_rows = df["TotalCharges"].notna() & df["MonthlyCharges"].notna()
    pct_valid = (df.loc[valid_rows, "TotalCharges"] >= df.loc[valid_rows, "MonthlyCharges"]).mean()
    if pct_valid < 0.95:
        failed.append(f"TotalCharges < MonthlyCharges in more than 5% of rows ({(1-pct_valid)*100:.1f}%)")

    # ── Summary ───────────────────────────────────────────────────────────────
    total  = len(required_cols) + len(allowed) + 6
    passed = total - len(failed)

    if not failed:
        print(f"Data validation passed! {passed}/{total} checks successful.")
    else:
        print(f"Data validation failed! {len(failed)}/{total} checks failed.")
        for f in failed:
            print(f"  ✗ {f}")

    return len(failed) == 0, failed
