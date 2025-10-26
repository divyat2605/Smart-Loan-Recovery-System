from fastapi import APIRouter, HTTPException
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_loader import load_data, validate_data
from src.preprocessing import preprocess_data
from src.feature_engineering import create_financial_features, create_borrower_segments
from src.model_training import train_recovery_model, tune_hyperparameters  # Updated import
from src.evaluation import evaluate_model
from src.monitoring import log_event, log_error, log_training_metrics
import pandas as pd

router = APIRouter()

@router.post("/train")
async def train_model_api():
    """
    Train the loan recovery prediction model.
    """
    try:
        log_event("TRAINING_START", {})
        
        # Load and validate data
        df = load_data("data/loan-recovery.csv")
        validate_data(df)
        
        # Feature engineering
        df = create_financial_features(df)
        df = create_borrower_segments(df)
        
        # Preprocess data
        X_train, X_test, y_train, y_test, feature_names = preprocess_data(df, is_training=True)
        
        # Train model - using renamed function
        model, accuracy = train_recovery_model(X_train, y_train, X_test, y_test)
        
        # Evaluate model
        evaluation_results = evaluate_model(model, X_test, y_test, feature_names)
        
        # Log training completion
        log_training_metrics(evaluation_results)
        
        return {
            "status": "success",
            "message": "Model trained successfully",
            "accuracy": accuracy,
            "evaluation": evaluation_results,
            "feature_count": len(feature_names)
        }
        
    except Exception as e:
        log_error("TRAINING_FAILED", {"exception": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train/tune")
async def tune_model_api():
    """
    Tune model hyperparameters.
    """
    try:
        log_event("TUNING_START", {})
        
        df = load_data("data/loan-recovery.csv")
        df = create_financial_features(df)
        X_train, X_test, y_train, y_test, _ = preprocess_data(df, is_training=True)
        
        best_model = tune_hyperparameters(X_train, y_train)
        
        log_event("TUNING_COMPLETE", {"best_params": best_model.get_params()})
        
        return {
            "status": "success",
            "message": "Hyperparameter tuning completed",
            "best_params": best_model.get_params()
        }
        
    except Exception as e:
        log_error("TUNING_FAILED", {"exception": str(e)})
        raise HTTPException(status_code=500, detail=str(e))