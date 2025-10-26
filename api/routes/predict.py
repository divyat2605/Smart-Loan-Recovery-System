from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import sys
import os
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.prediction import predict_outcome, batch_predict
from src.recovery_strategy import generate_recovery_strategy, get_strategy_actions

router = APIRouter()

class PredictionRequest(BaseModel):
    Age: int
    Gender: str
    Employment_Type: str
    Monthly_Income: float
    Num_Dependents: int
    Loan_Amount: float
    Loan_Tenure: int
    Interest_Rate: float
    Loan_Type: str
    Collateral_Value: float
    Outstanding_Loan_Amount: float
    Monthly_EMI: float
    Payment_History: str
    Num_Missed_Payments: int
    Days_Past_Due: int
    Collection_Attempts: int
    Collection_Method: str
    Legal_Action_Taken: str

@router.post("/predict")
async def predict_api(request: PredictionRequest):
    """
    Predict loan recovery outcome for a single borrower.
    """
    try:
        input_data = request.dict()
        
        # Check if model exists
        import os
        if not os.path.exists("models/model.pkl"):
            raise HTTPException(
                status_code=400, 
                detail="Model not trained yet. Please train the model first by going to 'Train Model' page."
            )
        
        prediction_result = predict_outcome(input_data)
        
        if "error" in prediction_result:
            raise HTTPException(status_code=400, detail=f"Prediction error: {prediction_result['error']}")
        
        # Generate recovery strategy
        strategy = generate_recovery_strategy({
            'prediction': prediction_result['prediction'],
            'probability': prediction_result['probability'],
            'risk_factors': {
                'Num_Missed_Payments': input_data['Num_Missed_Payments'],
                'Days_Past_Due': input_data['Days_Past_Due'],
                'Monthly_Income': input_data['Monthly_Income'],
                'Outstanding_Loan_Amount': input_data['Outstanding_Loan_Amount']
            }
        })
        
        actions = get_strategy_actions(strategy)
        
        return {
            "status": "success",
            "prediction": prediction_result['prediction'],
            "probability": prediction_result['probability'],
            "recovery_strategy": strategy,
            "recommended_actions": actions,
            "risk_level": "High" if prediction_result['probability'] > 0.7 else "Medium" if prediction_result['probability'] > 0.5 else "Low"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Prediction error: {error_details}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/predict/batch")
async def batch_predict_api(file_data: Dict[str, Any]):
    """
    Batch prediction for multiple borrowers.
    """
    try:
        # Check if model exists
        import os
        if not os.path.exists("models/model.pkl"):
            raise HTTPException(
                status_code=400, 
                detail="Model not trained yet. Please train the model first."
            )
        
        # Convert input data to DataFrame
        df = pd.DataFrame(file_data['borrowers'])
        
        # Perform batch prediction
        results_df = batch_predict(df)
        
        return {
            "status": "success",
            "predictions": results_df.to_dict('records'),
            "total_processed": len(results_df)
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Batch prediction error: {error_details}")
        raise HTTPException(status_code=500, detail=str(e))