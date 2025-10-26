import pandas as pd
import os

def load_data(file_path: str = "data/loan-recovery.csv") -> pd.DataFrame:
    """
    Load loan recovery dataset from CSV file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate the dataset structure and required columns.
    """
    required_columns = [
        'Borrower_ID', 'Age', 'Gender', 'Employment_Type', 'Monthly_Income',
        'Num_Dependents', 'Loan_ID', 'Loan_Amount', 'Loan_Tenure', 'Interest_Rate',
        'Loan_Type', 'Collateral_Value', 'Outstanding_Loan_Amount', 'Monthly_EMI',
        'Payment_History', 'Num_Missed_Payments', 'Days_Past_Due', 'Recovery_Status',
        'Collection_Attempts', 'Collection_Method', 'Legal_Action_Taken'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True