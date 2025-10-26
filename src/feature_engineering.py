import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

def create_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create advanced financial health features.
    """
    df = df.copy()
    
    # Debt-to-Income Ratio (avoid division by zero)
    df['DTI_Ratio'] = np.where(
        df['Monthly_Income'] > 0,
        df['Monthly_EMI'] / df['Monthly_Income'],
        0
    )
    
    # Loan-to-Value Ratio (avoid division by zero)
    df['LTV_Ratio'] = np.where(
        df['Collateral_Value'] > 0,
        df['Outstanding_Loan_Amount'] / df['Collateral_Value'],
        0
    )
    df['LTV_Ratio'] = df['LTV_Ratio'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Payment Burden
    df['Payment_Burden'] = np.where(
        df['Monthly_Income'] > 0,
        df['Monthly_EMI'] / df['Monthly_Income'],
        0
    )
    
    # Risk Score based on multiple factors (normalize to prevent large values)
    max_missed = df['Num_Missed_Payments'].max() if df['Num_Missed_Payments'].max() > 0 else 1
    max_dpd = df['Days_Past_Due'].max() if df['Days_Past_Due'].max() > 0 else 1
    
    df['Risk_Score'] = (
        (df['Num_Missed_Payments'] / max_missed * 0.3) +
        (df['Days_Past_Due'] / max_dpd * 0.2) +
        (df['DTI_Ratio'] * 0.25) +
        (df['LTV_Ratio'] * 0.25)
    )
    
    # Financial Stability Index
    max_income = df['Monthly_Income'].max() if df['Monthly_Income'].max() > 0 else 1
    max_missed_norm = max_missed if max_missed > 0 else 1
    max_dpd_norm = max_dpd if max_dpd > 0 else 1
    
    df['Financial_Stability'] = (
        (df['Monthly_Income'] / max_income) * 0.4 +
        (1 - (df['Num_Missed_Payments'] / max_missed_norm)) * 0.3 +
        (1 - (df['Days_Past_Due'] / max_dpd_norm)) * 0.3
    )
    
    return df

def create_borrower_segments(df: pd.DataFrame, n_clusters=4):
    """
    Create borrower segments using K-means clustering.
    """
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Features for clustering
    cluster_features = [
        'Age', 'Monthly_Income', 'Loan_Amount', 'Interest_Rate',
        'Collateral_Value', 'Outstanding_Loan_Amount', 'Monthly_EMI',
        'Num_Missed_Payments', 'Days_Past_Due', 'DTI_Ratio', 'Risk_Score'
    ]
    
    # Only use features that exist in the dataframe
    available_features = [f for f in cluster_features if f in df.columns]
    
    if not available_features:
        print("⚠️ Warning: No clustering features available, skipping segmentation")
        df['Borrower_Segment'] = 0
        df['Segment_Name'] = 'Unknown'
        return df
    
    # Select and scale features
    X_cluster = df[available_features].fillna(0)
    
    # Check if we have enough samples for clustering
    if len(X_cluster) < n_clusters:
        print(f"⚠️ Warning: Not enough samples for {n_clusters} clusters, using 2 clusters")
        n_clusters = 2
    
    cluster_scaler = StandardScaler()
    X_scaled = cluster_scaler.fit_transform(X_cluster)
    
    # Apply K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Borrower_Segment'] = kmeans.fit_predict(X_scaled)
    
    # Save models
    joblib.dump(kmeans, "models/kmeans_model.pkl")
    joblib.dump(cluster_scaler, "models/cluster_scaler.pkl")
    print(f"✅ K-means model saved with {n_clusters} clusters")
    
    # Define segment names based on characteristics
    segment_names = {
        0: 'Low Risk - Stable',
        1: 'Medium Risk - Watchlist', 
        2: 'High Risk - Critical',
        3: 'Moderate Risk - Monitoring'
    }
    
    # Map segment names (only for existing segments)
    df['Segment_Name'] = df['Borrower_Segment'].map(
        lambda x: segment_names.get(x, f'Segment {x}')
    )
    
    print(f"✅ Borrower segmentation complete:")
    print(df['Segment_Name'].value_counts())
    
    return df