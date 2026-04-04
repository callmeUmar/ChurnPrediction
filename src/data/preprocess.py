import pandas as pd 

def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    
    """
    Basic cleaning for Telco churn.
    - trim column names
    - drop obvious ID cols
    - fix TotalCharges to be numeric
    - map target Churn to 0/1 if needed
    - simple NA handling 
    """

    #tidy headers
    df.columns = df.columns.str.strip() #Remove leading/Trailing whitespace from column names
    
    for col in ["customerID", "Unnamed: 0"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
            
    #target to 0/1 if it's Yes/No
    if target_col in df.columns and df[target_col].dtype == 'object':
        df[target_col] = df[target_col].map({'Yes': 1, 'No': 0})
        
    #TotalCharges ofthen has blanks in this dataset -> coerce to numeric and fill NA with 0
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce').fillna(0)
        
    #SeniorCitizen should be 0/1 ints if present
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)
        
    # simple NA strategy:
    # - numeric: fill with 0
    # - others: leave fpr encoders to handle (get_dummies will handle NA as a separate category)
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)
    
    return df           
           