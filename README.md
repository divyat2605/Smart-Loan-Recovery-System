# Smart Loan Recovery System with Machine Learning

A machine learning system that predicts loan default risk and assigns recovery strategies based on borrower profiles and payment behavior.

## 🎯 Features

- **Risk Prediction**: Predicts probability of loan default
- **Borrower Segmentation**: K-Means clustering to group borrowers
- **Dynamic Recovery Strategies**: Assigns strategies based on risk level
- **RESTful API**: FastAPI endpoints for training and prediction
- **Monitoring & Logging**: Tracks all operations

## 📂 Project Structure

```
smart_loan_recovery/
│
├── data/
│   └── loan_recovery.csv          # Dataset (IMPORTANT: must be named exactly this)
│
├── src/
│   ├── data_loader.py             # Load CSV data
│   ├── preprocessing.py           # Encoding, scaling, clustering
│   ├── feature_engineering.py     # K-Means borrower segments
│   ├── model.py                   # RandomForest training
│   ├── recovery_strategy.py       # Risk-based recovery plans
│   └── monitoring.py              # Logging system
│
├── api/
│   ├── main.py                    # FastAPI application
│   ├── routes/
│   │   ├── training.py           # POST /api/train
│   │   ├── prediction.py         # POST /api/predict
│   │   └── monitoring.py         # GET /api/logs, /api/status
│   └── services/
│       ├── training_service.py   # Training pipeline
│       └── pipeline_service.py   # Prediction pipeline
│
└── requirements.txt
```

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

**CRITICAL**: Your dataset MUST be named `loan_recovery.csv` and placed in the `data/` folder:

```
data/loan_recovery.csv
```

Required columns:
- Borrower_ID, Loan_ID (identifiers)
- Age, Gender, Monthly_Income, Num_Dependents, Employment_Type
- Loan_Amount, Loan_Tenure, Interest_Rate, Collateral_Value
- Outstanding_Loan_Amount, Monthly_EMI
- Payment_History, Num_Missed_Payments, Days_Past_Due
- Collection_Method, Collection_Attempts, Legal_Action_Taken
- Recovery_Status (target for analysis)

### 3. Run the API Server

```bash
uvicorn api.main:app --reload
```

Server starts at: `http://localhost:8000`

## 📡 API Endpoints

### 1. Train Model
```bash
POST /api/train
```

**Response:**
```json
{
  "message": "✅ Model trained successfully",
  "accuracy": 0.923,
  "target": "High_Risk_Flag (Default Risk Prediction)",
  "models_saved": [
    "model.pkl (RandomForest)",
    "kmeans_model.pkl (Clustering)",
    "scaler.pkl (Feature Scaler)",
    "encoders.pkl (Label Encoders)"
  ]
}
```

### 2. Predict Default Risk
```bash
POST /api/predict
Content-Type: application/json

{
  "Age": 35,
  "Gender": "Male",
  "Monthly_Income": 116520,
  "Num_Dependents": 1,
  "Employment_Type": "Salaried",
  "Loan_Type": "Personal",
  "Loan_Amount": 1923410,
  "Loan_Tenure": 72,
  "Interest_Rate": 7.74,
  "Collateral_Value": 2622540,
  "Outstanding_Loan_Amount": 1031372,
  "Monthly_EMI": 14324.61,
  "Payment_History": "Delayed",
  "Num_Missed_Payments": 2,
  "Days_Past_Due": 124,
  "Collection_Method": "Legal Notice",
  "Legal_Action_Taken": "No",
  "Collection_Attempts": 2
}
```

**Response:**
```json
{
  "high_risk_flag": 1,
  "risk_score": 0.8234,
  "risk_level": "High",
  "recovery_strategy": "Immediate legal notices & aggressive recovery attempts",
  "priority": "Urgent",
  "recommended_actions": [
    "Send legal notice immediately",
    "Initiate aggressive collection calls",
    "Consider asset seizure if applicable",
    "Escalate to legal team"
  ],
  "interpretation": {
    "is_high_risk": true,
    "risk_percentage": "82.34%",
    "message": "This borrower has a 82.3% probability of default"
  }
}
```

### 3. Check System Status
```bash
GET /api/status
```

### 4. View Logs
```bash
GET /api/logs
```

## 🧠 How It Works

### Training Pipeline:
1. **Load Data**: Reads `loan_recovery.csv`
2. **Clustering**: K-Means creates 4 borrower segments
3. **Feature Engineering**: Adds segment labels and high-risk flags
4. **Preprocessing**: Encodes categorical features, scales numerical features
5. **Model Training**: RandomForest predicts High_Risk_Flag
6. **Save Models**: Saves all models for inference

### Prediction Pipeline:
1. **Load Input**: Receives borrower details
2. **Clustering**: Assigns borrower to segment
3. **Preprocessing**: Transforms features using saved encoders/scalers
4. **Prediction**: Calculates default probability (risk score)
5. **Strategy Assignment**: Recommends recovery strategy based on risk level

## 📊 Risk Levels & Strategies

| Risk Score | Risk Level | Strategy |
|------------|-----------|----------|
| > 0.75 | High | Immediate legal notices & aggressive recovery |
| 0.50 - 0.75 | Medium | Settlement offers & repayment plans |
| < 0.50 | Low | Automated reminders & monitoring |

## 🔍 Borrower Segments

1. **High Income, Low Default Risk**: Stable, low risk
2. **Moderate Income, Medium Risk**: Average risk profile
3. **Moderate Income, High Loan Burden**: Financial strain, higher risk
4. **High Loan, Higher Default Risk**: Large loans, highest risk

## 🐛 Troubleshooting

### Error: "Dataset not found"
- Ensure file is named exactly `loan_recovery.csv` (not `loan_data.csv`)
- Check it's in the `data/` folder

### Error: "Model not found"
- Run `POST /api/train` first before making predictions

### Error: "Encoders not found"
- Training didn't complete successfully
- Check logs at `GET /api/logs`

## 📝 License
MIT License

