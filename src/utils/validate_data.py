import great_expectations as ge
from typing import Tuple , List 

def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """ 
    Comprehensive data validation for Telco churn dataset using Great Expectations.
    
    This function implements critical data quality checks that must pass before model training can proceed. It ensures the integrity and consistency of the dataset by validating:.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.
    """
    
    print("Starting data validation with Great Expectations...")
    
    #Convert pandas DataFrame to Great Expectations DataSet
    ge_df = ge.dataset.PandasDataset(df)
    
    #Schema validation - The essential columns.
    print("Validating schema and required columns...")
    
    #Customer identifier must exist (required for business operations)
    ge_df.expect_column_to_exist("customerID")
    ge_df.expect_column_values_to_not_be_null("customerID")
    
    #Core demographic features
    ge_df.expect_column_to_exist("Gender")
    ge_df.expect_column_to_exist("Partner")
    ge_df.expect_column_to_exist("Dependents")
    
    #Service Features
    ge_df.expect_column_to_exist("PhoneService")
    ge_df.expect_column_to_exist("InternetService")
    ge_df.expect_column_to_exist("Contract")
    
    #Financial Features
    ge_df.expect_column_to_exist("tenure")
    ge_df.expect_column_to_exist("MonthlyCharges")
    ge_df.expect_column_to_exist("TotalCharges")
    
    #Business Logic Validation
    print("Validating business logic constraints...")
    
    #Gender must be one of the expected values
    ge_df.expect_column_values_to_be_in_set("Gender", ["Male", "Female"])
    
    #Yes/No fields must have valid values
    ge_df.expect_column_values_to_be_in_set("Partner", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("Dependents", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("PhoneService", ["Yes", "No"])
    
    #Contract types must be valid (bunsiness constraint)
    ge_df.expect_column_values_to_be_in_set("Contract", ["Month-to-month", "One year", "Two year"])
    
    #Internet service types (business constraint)
    ge_df.expect_column_values_to_be_in_set("InternetService", ["DSL", "Fiber optic", "No"])
    
    #Numeric Range Validation
    print("Validating numeric ranges and business constraints...")

    #Tenure must be non-negative (business logic - can't have negative tenure)
    ge_df.expect_column_values_to_be_between("tenure", min_value=0)
    
    #MonthlyCharges must be positive (business logic - no free service)
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0)
    
    #TotalCharges should be non-negative (business logic)
    ge_df.expect_column_values_to_be_between("TotalCharges", min_value=0)
    
    #Statical Validation
    print("Validating statistical properties...")
    
    #Tenure should be reasonable (max 10 years = 120 month to telecom)
    ge_df.expect_column_values_to_be_between("tenure", max_value=120)
    
    #MonthkyCharges should be withing a reasonable range
    ge_df.expect_column_values_to_be_between("MonthlyCharges", max_value=200)
    
    #No missing values in critical numeric features
    ge_df.expect_column_values_to_not_be_null("tenure")
    ge_df.expect_column_values_to_not_be_null("MonthlyCharges")
    
    # Total charges should generally be >= MOnthylCharges (expect for every new customers)
    # This is a business logic check to catch data entry errors
    ge_df.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="TotalCharges",
        column_B="MonthlyCharges",
        or_equal=True,
        mostly=0.95 #Allow 5% exceptions for edge cases
    )
    
    #Run validation SUITE
    print("running complete validation suite...")
    results = ge_df.validate()
    
    #Process results
    #Extract failed expectations for detailed error reporting
    failed_expectations = []
    for r in results["results"]:
        if not r["success"]:
            expectation_type = r["expectation_config"]["expectation_type"]
            column = r["expectation_config"]["kwargs"].get("column", "N/A")
            failed_expectations.append(f"{expectation_type} on column '{column}'")

    #Print validation summary
    total_checks = len(results["results"])
    passed_checks = sum(1 for r in results["results"] if r["success"])
    failed_checks = total_checks - passed_checks

    if results["success"]:
        print(f"Data validation passed! {passed_checks}/{total_checks} checks successful.")
    else:
        print(f"Data validation failed! {failed_checks}/{total_checks} checks failed.")
        print(f"Failed expectations: {failed_expectations}")

    is_valid = len(failed_expectations) == 0
    return is_valid, failed_expectations            
            
            
