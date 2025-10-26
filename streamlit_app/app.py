import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Smart Loan Recovery System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .strategy-high { background-color: #ff6b6b; color: white; padding: 0.5rem; border-radius: 5px; }
    .strategy-medium { background-color: #ffa500; color: white; padding: 0.5rem; border-radius: 5px; }
    .strategy-low { background-color: #51cf66; color: white; padding: 0.5rem; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# API base URL
API_BASE = "http://localhost:8000/api/v1"

def main():
    st.markdown('<h1 class="main-header">💰 Smart Loan Recovery System</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Dashboard", 
        "Train Model", 
        "Make Predictions", 
        "Model Evaluation",
        "System Monitoring"
    ])
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Train Model":
        show_training_page()
    elif page == "Make Predictions":
        show_prediction_page()
    elif page == "Model Evaluation":
        show_evaluation_page()
    elif page == "System Monitoring":
        show_monitoring_page()

def show_dashboard():
    st.header("📊 System Dashboard")
    
    try:
        # Get system status
        response = requests.get(f"{API_BASE}/monitoring/status")
        if response.status_code == 200:
            status_data = response.json()
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("System Status", status_data['status'].upper())
            
            with col2:
                model_status = "Trained" if status_data['components']['model'] else "Not Trained"
                st.metric("Model Status", model_status)
            
            with col3:
                st.metric("API Status", "Online")
            
            with col4:
                st.metric("Last Update", datetime.now().strftime("%H:%M:%S"))
            
            # Component status
            st.subheader("Component Status")
            components_df = pd.DataFrame.from_dict(status_data['components'], orient='index', columns=['Status'])
            components_df['Status'] = components_df['Status'].map({True: '✅ Healthy', False: '❌ Missing'})
            st.dataframe(components_df)
            
        else:
            st.error("Unable to connect to API")
            
    except Exception as e:
        st.error(f"Error connecting to system: {str(e)}")
    
    # Quick actions
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Train Model", use_container_width=True):
            try:
                with st.spinner("Training model..."):
                    response = requests.post(f"{API_BASE}/train")
                    if response.status_code == 200:
                        st.success("Model trained successfully!")
                    else:
                        st.error("Training failed")
            except Exception as e:
                st.error(f"Training error: {str(e)}")
    
    with col2:
        if st.button("📈 View Evaluation", use_container_width=True):
            st.session_state.page = "Model Evaluation"
            st.rerun()
    
    with col3:
        if st.button("🔍 System Check", use_container_width=True):
            st.session_state.page = "System Monitoring"
            st.rerun()

def show_training_page():
    st.header("🎯 Train Prediction Model")
    
    st.info("""
    This will train a new machine learning model using your loan recovery data.
    The model will learn to predict recovery outcomes and generate strategies.
    """)
    
    if st.button("🚀 Start Model Training", type="primary"):
        try:
            with st.spinner("Training in progress... This may take a few minutes."):
                response = requests.post(f"{API_BASE}/train")
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Model training completed successfully!")
                    
                    # Display training results
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{result['accuracy']:.3f}")
                    with col2:
                        st.metric("Features Used", result['feature_count'])
                    with col3:
                        st.metric("Status", "Success")
                    with col4:
                        st.metric("Training Time", "Completed")
                    
                    # Show evaluation metrics if available
                    if 'evaluation' in result:
                        eval_data = result['evaluation']
                        st.subheader("Training Metrics")
                        
                        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                        with metrics_col1:
                            st.metric("Precision", f"{eval_data['precision']:.3f}")
                        with metrics_col2:
                            st.metric("Recall", f"{eval_data['recall']:.3f}")
                        with metrics_col3:
                            st.metric("F1-Score", f"{eval_data['f1_score']:.3f}")
                        with metrics_col4:
                            st.metric("ROC AUC", f"{eval_data['roc_auc']:.3f}" if eval_data['roc_auc'] else "N/A")
                
                else:
                    st.error(f"Training failed: {response.text}")
                    
        except Exception as e:
            st.error(f"Training error: {str(e)}")
    
    # Hyperparameter tuning section
    st.subheader("Advanced Options")
    if st.button("⚙️ Tune Hyperparameters"):
        with st.spinner("Performing hyperparameter tuning..."):
            try:
                response = requests.post(f"{API_BASE}/train/tune")
                if response.status_code == 200:
                    st.success("Hyperparameter tuning completed!")
                    st.json(response.json())
                else:
                    st.error("Tuning failed")
            except Exception as e:
                st.error(f"Tuning error: {str(e)}")

def show_prediction_page():
    st.header("🔮 Make Predictions")
    
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
    
    with tab1:
        st.subheader("Single Borrower Prediction")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", min_value=18, max_value=100, value=35)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Business", "Unemployed"])
                monthly_income = st.number_input("Monthly Income ($)", min_value=0, value=50000)
                num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1)
                loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=100000)
                loan_tenure = st.number_input("Loan Tenure (months)", min_value=1, value=36)
                interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=50.0, value=12.5)
            
            with col2:
                loan_type = st.selectbox("Loan Type", ["Home", "Auto", "Personal", "Business", "Education"])
                collateral_value = st.number_input("Collateral Value ($)", min_value=0, value=120000)
                outstanding_loan = st.number_input("Outstanding Loan Amount ($)", min_value=0, value=80000)
                monthly_emi = st.number_input("Monthly EMI ($)", min_value=0, value=3000)
                payment_history = st.selectbox("Payment History", ["On-Time", "Delayed", "Missed"])
                num_missed_payments = st.number_input("Number of Missed Payments", min_value=0, value=0)
                days_past_due = st.number_input("Days Past Due", min_value=0, value=0)
                collection_attempts = st.number_input("Collection Attempts", min_value=0, value=0)
                collection_method = st.selectbox("Collection Method", ["Calls", "Emails", "Settlement Offer", "Legal Notice", "Debt Collectors"])
                legal_action_taken = st.selectbox("Legal Action Taken", ["No", "Yes"])
            
            submitted = st.form_submit_button("Predict Recovery Outcome", type="primary")
            
            if submitted:
                prediction_data = {
                    "Age": age,
                    "Gender": gender,
                    "Employment_Type": employment_type,
                    "Monthly_Income": monthly_income,
                    "Num_Dependents": num_dependents,
                    "Loan_Amount": loan_amount,
                    "Loan_Tenure": loan_tenure,
                    "Interest_Rate": interest_rate,
                    "Loan_Type": loan_type,
                    "Collateral_Value": collateral_value,
                    "Outstanding_Loan_Amount": outstanding_loan,
                    "Monthly_EMI": monthly_emi,
                    "Payment_History": payment_history,
                    "Num_Missed_Payments": num_missed_payments,
                    "Days_Past_Due": days_past_due,
                    "Collection_Attempts": collection_attempts,
                    "Collection_Method": collection_method,
                    "Legal_Action_Taken": legal_action_taken
                }
                
                try:
                    with st.spinner("Analyzing borrower data..."):
                        response = requests.post(f"{API_BASE}/predict", json=prediction_data)
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Display results
                            st.success("Prediction completed successfully!")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                prediction_map = {0: "Not Recovered", 1: "Partially Recovered", 2: "Fully Recovered"}
                                prediction_text = prediction_map.get(result['prediction'], "Unknown")
                                st.metric("Predicted Outcome", prediction_text)
                            
                            with col2:
                                confidence = result['probability']
                                st.metric("Confidence", f"{confidence:.1%}")
                            
                            with col3:
                                risk_level = result['risk_level']
                                risk_color = {
                                    "High": "strategy-high",
                                    "Medium": "strategy-medium", 
                                    "Low": "strategy-low"
                                }.get(risk_level, "")
                                st.markdown(f'<div class="{risk_color}">Risk Level: {risk_level}</div>', unsafe_allow_html=True)
                            
                            # Recovery strategy
                            st.subheader("🎯 Recommended Recovery Strategy")
                            st.info(f"**{result['recovery_strategy']}**")
                            
                            st.subheader("📋 Recommended Actions")
                            for i, action in enumerate(result['recommended_actions'], 1):
                                st.write(f"{i}. {action}")
                        
                        else:
                            st.error(f"Prediction failed: {response.text}")
                
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
    
    with tab2:
        st.subheader("Batch Prediction")
        st.info("Upload a CSV file with multiple borrower records for batch processing.")
        
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())
                
                if st.button("Process Batch Prediction"):
                    with st.spinner("Processing batch prediction..."):
                        batch_data = {"borrowers": df.to_dict('records')}
                        response = requests.post(f"{API_BASE}/predict/batch", json=batch_data)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"Batch processing completed! Processed {result['total_processed']} records.")
                            
                            # Convert results to DataFrame for display
                            results_df = pd.DataFrame(result['predictions'])
                            st.dataframe(results_df)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                "Download Results",
                                csv,
                                "batch_predictions.csv",
                                "text/csv",
                                key='download-csv'
                            )
                        else:
                            st.error("Batch prediction failed")
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

def show_evaluation_page():
    st.header("📈 Model Evaluation")
    
    try:
        # Get evaluation results
        response = requests.get(f"{API_BASE}/monitoring/evaluation")
        
        if response.status_code == 200:
            eval_data = response.json()
            
            if eval_data['status'] == 'success':
                metrics = eval_data['evaluation']
                
                # Display key metrics
                st.subheader("Model Performance Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                with col2:
                    st.metric("Precision", f"{metrics['precision']:.3f}")
                with col3:
                    st.metric("Recall", f"{metrics['recall']:.3f}")
                with col4:
                    st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = metrics['confusion_matrix']
                fig = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Not Recovered', 'Partially Recovered', 'Fully Recovered'],
                    y=['Not Recovered', 'Partially Recovered', 'Fully Recovered']
                )
                st.plotly_chart(fig)
                
                # Feature Importance
                if metrics['top_features']:
                    st.subheader("Top 10 Feature Importances")
                    features_df = pd.DataFrame(metrics['top_features'], columns=['Feature', 'Importance'])
                    fig = px.bar(
                        features_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Feature Importance"
                    )
                    st.plotly_chart(fig)
                
                # Show detailed report
                if st.button("View Detailed Report"):
                    report_response = requests.get(f"{API_BASE}/monitoring/evaluation/report")
                    if report_response.status_code == 200:
                        report_data = report_response.json()
                        if report_data['status'] == 'success':
                            st.text_area("Detailed Evaluation Report", report_data['report'], height=400)
            
            else:
                st.warning("No evaluation results found. Please train the model first.")
        
        else:
            st.error("Unable to fetch evaluation results")
    
    except Exception as e:
        st.error(f"Error loading evaluation: {str(e)}")

def show_monitoring_page():
    st.header("🔍 System Monitoring")
    
    tab1, tab2 = st.tabs(["System Logs", "Component Status"])
    
    with tab1:
        st.subheader("Application Logs")
        
        if st.button("Refresh Logs"):
            st.rerun()
        
        try:
            response = requests.get(f"{API_BASE}/monitoring/logs")
            if response.status_code == 200:
                logs_data = response.json()
                
                if 'logs' in logs_data and logs_data['logs']:
                    st.text_area("Recent Logs", "\n".join(logs_data['logs']), height=400)
                else:
                    st.info("No logs available yet.")
            else:
                st.error("Unable to fetch logs")
        
        except Exception as e:
            st.error(f"Error fetching logs: {str(e)}")
    
    with tab2:
        st.subheader("Component Status")
        
        try:
            response = requests.get(f"{API_BASE}/monitoring/status")
            if response.status_code == 200:
                status_data = response.json()
                
                # Create status visualization
                components = status_data['components']
                
                st.metric("Overall System Status", status_data['status'].upper())
                
                # Component status table
                status_df = pd.DataFrame.from_dict(components, orient='index', columns=['Available'])
                status_df['Status'] = status_df['Available'].map({True: '✅ Healthy', False: '❌ Missing'})
                status_df = status_df.drop('Available', axis=1)
                
                st.dataframe(status_df)
                
                # Health visualization
                healthy_count = sum(components.values())
                total_count = len(components)
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = healthy_count,
                    title = {'text': "System Health Score"},
                    gauge = {
                        'axis': {'range': [0, total_count]},
                        'bar': {'color': "green" if healthy_count == total_count else "orange" if healthy_count > total_count/2 else "red"},
                        'steps': [
                            {'range': [0, total_count/2], 'color': "lightgray"},
                            {'range': [total_count/2, total_count], 'color': "gray"}
                        ]
                    }
                ))
                st.plotly_chart(fig)
            
            else:
                st.error("Unable to fetch system status")
        
        except Exception as e:
            st.error(f"Error fetching status: {str(e)}")

if __name__ == "__main__":
    main()