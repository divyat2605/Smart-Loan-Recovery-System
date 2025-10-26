import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

SCALER_PATH = "models/scaler.pkl"
ENCODERS_PATH = "models/encoders.pkl"
FEATURE_NAMES_PATH = "models/feature_names.pkl"
TARGET_ENCODER_PATH = "models/target_encoder.pkl"

def preprocess_data(df: pd.DataFrame, is_training=True):
    """
    Preprocess the loan recovery data for training or prediction.
    """
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    df = df.copy()
    
    # Handle categorical features
    cat_cols = ['Gender', 'Employment_Type', 'Loan_Type', 'Payment_History',
                'Collection_Method', 'Legal_Action_Taken']
    
    encoders = {}
    
    if is_training:
        # Training: create and save encoders
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col])
                encoders[col] = encoder
        
        # Save encoders
        joblib.dump(encoders, ENCODERS_PATH)
        print(f"✅ Encoders saved to {ENCODERS_PATH}")
    else:
        # Prediction: load and use existing encoders
        if not os.path.exists(ENCODERS_PATH):
            raise FileNotFoundError("Encoders not found. Please train the model first.")
        encoders = joblib.load(ENCODERS_PATH)
        
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
                # Handle unseen categories
                df[col] = df[col].apply(lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0])
                df[col] = encoders[col].transform(df[col])
    
    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))
    
    # Columns to exclude from features
    exclude_cols = [
        'Recovery_Status',      # Target variable
        'Borrower_ID',          # ID column
        'Loan_ID',              # ID column
        'Segment_Name'          # Text label (we use Borrower_Segment instead)
    ]
    
    # Select features and target
    if is_training:
        # Encode target variable
        if 'Recovery_Status' in df.columns:
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(df['Recovery_Status'].astype(str))
            joblib.dump(target_encoder, TARGET_ENCODER_PATH)
            print(f"✅ Target encoder saved. Classes: {target_encoder.classes_}")
        else:
            raise ValueError("Recovery_Status column not found in training data")
        
        # Drop excluded columns
        X = df.drop(exclude_cols, axis=1, errors='ignore')
        
        # Ensure we only have numeric columns
        X = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32'])
        
    else:
        # For prediction, exclude only ID columns
        X = df.drop(['Borrower_ID', 'Loan_ID', 'Segment_Name'], axis=1, errors='ignore')
        
        # Ensure we only have numeric columns
        X = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32'])
        
        y = None
    
    # Save feature names during training
    if is_training:
        feature_names = X.columns.tolist()
        joblib.dump(feature_names, FEATURE_NAMES_PATH)
        print(f"✅ Feature names saved ({len(feature_names)} features): {feature_names}")
    else:
        # Load feature names and ensure order matches
        if not os.path.exists(FEATURE_NAMES_PATH):
            raise FileNotFoundError("Feature names not found. Please train the model first.")
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        
        # Reorder columns to match training
        missing_features = set(feature_names) - set(X.columns)
        if missing_features:
            print(f"⚠️ Warning: Missing features: {missing_features}")
            # Add missing features with zeros
            for feature in missing_features:
                X[feature] = 0
        
        # Reorder columns
        X = X[feature_names]
    
    # Scale numeric features
    if is_training:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, SCALER_PATH)
        print(f"✅ Scaler saved to {SCALER_PATH}")
    else:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError("Scaler not found. Please train the model first.")
        scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(X)
    
    if is_training:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test, feature_names
    else:
        return X_scaled