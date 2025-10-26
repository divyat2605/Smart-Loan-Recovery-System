from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sys
import os
import pandas as pd
import numpy as np
import joblib

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.feature_engineering import create_financial_features, create_borrower_segments
from src.data_loader import load_data

router = APIRouter()

class SegmentPredictionRequest(BaseModel):
    Age: int
    Monthly_Income: float
    Loan_Amount: float
    Num_Missed_Payments: int
    Days_Past_Due: int

@router.get("/features/analysis")
async def get_feature_analysis():
    """
    Get statistical analysis of engineered features.
    """
    try:
        # Load data
        df = load_data("data/loan-recovery.csv")
        df = create_financial_features(df)
        
        # Calculate statistics for key features
        feature_cols = ['DTI_Ratio', 'LTV_Ratio', 'Payment_Burden', 'Risk_Score', 'Financial_Stability']
        
        stats = []
        distributions = {}
        
        for col in feature_cols:
            if col in df.columns:
                stats.append({
                    'feature': col,
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'median': float(df[col].median())
                })
                
                distributions[col] = df[col].dropna().tolist()[:1000]  # Limit to 1000 points
        
        return {
            "status": "success",
            "feature_stats": stats,
            "distributions": distributions
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/features/segments")
async def get_borrower_segments():
    """
    Get borrower segmentation results.
    """
    try:
        # Load data
        df = load_data("data/loan-recovery.csv")
        df = create_financial_features(df)
        df = create_borrower_segments(df)
        
        # Segment distribution
        segment_counts = df['Segment_Name'].value_counts().reset_index()
        segment_counts.columns = ['Segment_Name', 'Count']
        
        # Segment profiles
        segment_profiles = df.groupby('Segment_Name').agg({
            'Monthly_Income': 'mean',
            'Loan_Amount': 'mean',
            'Risk_Score': 'mean',
            'Num_Missed_Payments': 'mean',
            'Days_Past_Due': 'mean'
        }).reset_index()
        
        segment_profiles.columns = [
            'Segment_Name', 'Avg_Income', 'Avg_Loan_Amount',
            'Avg_Risk_Score', 'Avg_Missed_Payments', 'Avg_Days_Past_Due'
        ]
        
        return {
            "status": "success",
            "segments": segment_counts.to_dict('records'),
            "segment_profiles": segment_profiles.to_dict('records')
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/features/predict-segment")
async def predict_borrower_segment(request: SegmentPredictionRequest):
    """
    Predict which segment a borrower belongs to.
    """
    try:
        # Check if models exist
        if not os.path.exists("models/kmeans_model.pkl"):
            raise HTTPException(
                status_code=400, 
                detail="Segmentation model not found. Please train the model first by going to 'Train Model' page."
            )
        
        if not os.path.exists("models/cluster_scaler.pkl"):
            raise HTTPException(
                status_code=400,
                detail="Cluster scaler not found. Please train the model first."
            )
        
        # Load models
        kmeans = joblib.load("models/kmeans_model.pkl")
        cluster_scaler = joblib.load("models/cluster_scaler.pkl")
        
        # Load training data to get statistics for feature engineering
        df = load_data("data/loan-recovery.csv")
        df = create_financial_features(df)
        
        # Create a single-row dataframe with input data
        input_df = pd.DataFrame([{
            'Age': request.Age,
            'Monthly_Income': request.Monthly_Income,
            'Loan_Amount': request.Loan_Amount,
            'Interest_Rate': 12.0,  # Default value
            'Collateral_Value': request.Loan_Amount * 1.2,  # Estimate
            'Outstanding_Loan_Amount': request.Loan_Amount * 0.8,  # Estimate
            'Monthly_EMI': request.Monthly_Income * 0.3,  # Estimate
            'Num_Missed_Payments': request.Num_Missed_Payments,
            'Days_Past_Due': request.Days_Past_Due
        }])
        
        # Calculate engineered features
        input_df['DTI_Ratio'] = np.where(
            input_df['Monthly_Income'] > 0,
            input_df['Monthly_EMI'] / input_df['Monthly_Income'],
            0
        )
        
        input_df['LTV_Ratio'] = np.where(
            input_df['Collateral_Value'] > 0,
            input_df['Outstanding_Loan_Amount'] / input_df['Collateral_Value'],
            0
        )
        
        # Use training data statistics for normalization
        max_missed = df['Num_Missed_Payments'].max() if df['Num_Missed_Payments'].max() > 0 else 10
        max_dpd = df['Days_Past_Due'].max() if df['Days_Past_Due'].max() > 0 else 180
        
        input_df['Risk_Score'] = (
            (input_df['Num_Missed_Payments'] / max_missed * 0.3) +
            (input_df['Days_Past_Due'] / max_dpd * 0.2) +
            (input_df['DTI_Ratio'] * 0.25) +
            (input_df['LTV_Ratio'] * 0.25)
        )
        
        # Features used for clustering (must match training)
        cluster_features = [
            'Age', 'Monthly_Income', 'Loan_Amount', 'Interest_Rate',
            'Collateral_Value', 'Outstanding_Loan_Amount', 'Monthly_EMI',
            'Num_Missed_Payments', 'Days_Past_Due', 'DTI_Ratio', 'Risk_Score'
        ]
        
        # Select features
        X_input = input_df[cluster_features].fillna(0)
        
        # Scale features
        X_scaled = cluster_scaler.transform(X_input)
        
        # Predict segment
        segment = int(kmeans.predict(X_scaled)[0])
        
        segment_names = {
            0: 'Low Risk - Stable',
            1: 'Medium Risk - Watchlist',
            2: 'High Risk - Critical',
            3: 'Moderate Risk - Monitoring'
        }
        
        segment_descriptions = {
            0: "Financially stable borrower with low default risk. Good payment history and strong financial indicators.",
            1: "Requires monitoring but manageable risk level. Some missed payments but overall stable income.",
            2: "High risk of default, immediate attention needed. Multiple missed payments and high debt burden.",
            3: "Moderate risk, proactive management recommended. Mixed financial indicators requiring attention."
        }
        
        return {
            "status": "success",
            "segment": segment,
            "segment_name": segment_names.get(segment, f"Segment {segment}"),
            "segment_description": segment_descriptions.get(segment, "Custom segment"),
            "calculated_features": {
                "DTI_Ratio": float(input_df['DTI_Ratio'].iloc[0]),
                "Risk_Score": float(input_df['Risk_Score'].iloc[0])
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Segment prediction error: {error_details}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.get("/features/risk-profiles")
async def get_risk_profiles():
    """
    Get risk profile analysis.
    """
    try:
        # Load data
        df = load_data("data/loan-recovery.csv")
        df = create_financial_features(df)
        
        # Categorize by risk level
        df['Risk_Level'] = pd.cut(
            df['Risk_Score'],
            bins=[0, 0.3, 0.5, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        # Risk distribution
        risk_counts = df['Risk_Level'].value_counts().reset_index()
        risk_counts.columns = ['Risk_Level', 'Count']
        
        # Sample risk factors for visualization
        risk_factors = df[['DTI_Ratio', 'Risk_Score', 'Outstanding_Loan_Amount',
                           'Num_Missed_Payments', 'Days_Past_Due', 'Risk_Level']].sample(min(500, len(df)))
        
        risk_factors['Risk_Level'] = risk_factors['Risk_Level'].astype(str).str.replace(' Risk', '')
        
        return {
            "status": "success",
            "risk_distribution": risk_counts.to_dict('records'),
            "risk_factors": risk_factors.to_dict('records')
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))