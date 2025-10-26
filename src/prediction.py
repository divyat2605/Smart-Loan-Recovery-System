import numpy as np
import pandas as pd
import joblib
from src.monitoring import log_event, log_error

import numpy as np
import pandas as pd
import joblib
from src.monitoring import log_event, log_error

def predict_outcome(input_data: dict):
    """
    Predict loan recovery outcome for single input.
    """
    try:
        model = joblib.load("models/model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        encoders = joblib.load("models/encoders.pkl")
        feature_names = joblib.load("models/feature_names.pkl")
        
        log_event("PREDICT_START", {"input": input_data})

        # Create a DataFrame from input (easier to handle)
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical features
        cat_cols = ['Gender', 'Employment_Type', 'Loan_Type', 'Payment_History',
                    'Collection_Method', 'Legal_Action_Taken']
        
        for col in cat_cols:
            if col in input_df.columns and col in encoders:
                # Convert to string
                input_df[col] = input_df[col].astype(str)
                
                # Handle unseen categories
                value = input_df[col].iloc[0]
                if value not in encoders[col].classes_:
                    print(f"Warning: '{value}' not in training data for {col}, using default")
                    input_df[col] = encoders[col].classes_[0]
                
                # Encode
                input_df[col] = encoders[col].transform(input_df[col])
        
        # Create engineered features if they exist in feature_names
        if 'DTI_Ratio' in feature_names:
            input_df['DTI_Ratio'] = np.where(
                input_df['Monthly_Income'] > 0,
                input_df['Monthly_EMI'] / input_df['Monthly_Income'],
                0
            )
        
        if 'LTV_Ratio' in feature_names:
            input_df['LTV_Ratio'] = np.where(
                input_df['Collateral_Value'] > 0,
                input_df['Outstanding_Loan_Amount'] / input_df['Collateral_Value'],
                0
            )
        
        if 'Payment_Burden' in feature_names:
            input_df['Payment_Burden'] = np.where(
                input_df['Monthly_Income'] > 0,
                input_df['Monthly_EMI'] / input_df['Monthly_Income'],
                0
            )
        
        if 'Risk_Score' in feature_names:
            max_missed = 10  # Reasonable defaults
            max_dpd = 180
            input_df['Risk_Score'] = (
                (input_df['Num_Missed_Payments'] / max_missed * 0.3) +
                (input_df['Days_Past_Due'] / max_dpd * 0.2) +
                (input_df.get('DTI_Ratio', 0) * 0.25) +
                (input_df.get('LTV_Ratio', 0) * 0.25)
            )
        
        if 'Financial_Stability' in feature_names:
            max_income = 200000  # Reasonable default
            input_df['Financial_Stability'] = (
                (input_df['Monthly_Income'] / max_income) * 0.4 +
                (1 - (input_df['Num_Missed_Payments'] / 10)) * 0.3 +
                (1 - (input_df['Days_Past_Due'] / 180)) * 0.3
            )
        
        # Ensure all required features are present
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Select only the features used in training, in the correct order
        X = input_df[feature_names]
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        prediction_proba = model.predict_proba(X_scaled)[0]
        
        log_event("PREDICT_SUCCESS", {
            "prediction": int(prediction),
            "probability": float(max(prediction_proba))
        })
        
        return {
            "prediction": int(prediction),
            "probability": float(max(prediction_proba)),
            "class_probabilities": prediction_proba.tolist()
        }
    except Exception as e:
        log_error("PREDICT_FAILED", {"exception": str(e)})
        import traceback
        traceback.print_exc()  # Print full error for debugging
        return {"error": str(e)}

def batch_predict(df: pd.DataFrame):
    """
    Predict loan recovery outcomes for batch data.
    """
    try:
        from src.preprocessing import preprocess_data
        
        # Preprocess the data
        X_processed = preprocess_data(df, is_training=False)
        
        # Load model and make predictions
        model = joblib.load("models/model.pkl")
        predictions = model.predict(X_processed)
        probabilities = model.predict_proba(X_processed)
        
        # Add predictions to dataframe
        df['Predicted_Recovery'] = predictions
        df['Prediction_Probability'] = np.max(probabilities, axis=1)
        
        # Generate recovery strategies
        from src.recovery_strategy import generate_recovery_strategy
        df['Recommended_Strategy'] = df.apply(
            lambda row: generate_recovery_strategy({
                'prediction': row['Predicted_Recovery'],
                'probability': row['Prediction_Probability'],
                'risk_factors': {
                    'Num_Missed_Payments': row.get('Num_Missed_Payments', 0),
                    'Days_Past_Due': row.get('Days_Past_Due', 0),
                    'Monthly_Income': row.get('Monthly_Income', 0),
                    'Outstanding_Loan_Amount': row.get('Outstanding_Loan_Amount', 0)
                }
            }), axis=1
        )
        
        log_event("BATCH_PREDICT_SUCCESS", {"batch_size": len(df)})
        return df
        
    except Exception as e:
        log_error("BATCH_PREDICT_FAILED", {"exception": str(e)})
        raise