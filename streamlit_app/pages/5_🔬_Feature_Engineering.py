import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Feature Engineering - Smart Loan Recovery",
    page_icon="🔬",
    layout="wide"
)

# API base URL
API_BASE = "http://localhost:8000/api/v1"

st.title("🔬 Feature Engineering & Borrower Segmentation")

st.markdown("""
This page shows the advanced features created from raw data and how borrowers are segmented using K-means clustering.
""")

# Add tabs
tab1, tab2, tab3 = st.tabs(["📊 Feature Analysis", "👥 Borrower Segments", "🎯 Risk Profiles"])

with tab1:
    st.header("Financial Health Features")
    
    st.info("""
    **Engineered Features:**
    - **DTI Ratio**: Debt-to-Income ratio (Monthly EMI / Monthly Income)
    - **LTV Ratio**: Loan-to-Value ratio (Outstanding Loan / Collateral Value)
    - **Payment Burden**: Proportion of income used for loan payments
    - **Risk Score**: Composite score based on multiple risk factors
    - **Financial Stability**: Overall financial health indicator
    """)
    
    # Try to get feature engineering results
    try:
        response = requests.get(f"{API_BASE}/features/analysis")
        
        if response.status_code == 200:
            data = response.json()
            
            if 'feature_stats' in data:
                stats_df = pd.DataFrame(data['feature_stats'])
                
                st.subheader("Feature Statistics")
                st.dataframe(stats_df.style.format({
                    'mean': '{:.4f}',
                    'std': '{:.4f}',
                    'min': '{:.4f}',
                    'max': '{:.4f}'
                }))
                
                # Feature distributions
                if 'distributions' in data:
                    st.subheader("Feature Distributions")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # DTI Ratio distribution
                        fig = go.Figure(data=[go.Histogram(
                            x=data['distributions'].get('DTI_Ratio', []),
                            nbinsx=30,
                            name='DTI Ratio'
                        )])
                        fig.update_layout(
                            title="DTI Ratio Distribution",
                            xaxis_title="DTI Ratio",
                            yaxis_title="Count"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Risk Score distribution
                        fig = go.Figure(data=[go.Histogram(
                            x=data['distributions'].get('Risk_Score', []),
                            nbinsx=30,
                            name='Risk Score'
                        )])
                        fig.update_layout(
                            title="Risk Score Distribution",
                            xaxis_title="Risk Score",
                            yaxis_title="Count"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Feature analysis not available. Please train the model first.")
    
    except Exception as e:
        st.error(f"Error loading feature analysis: {str(e)}")
    
    # Feature calculator
    st.subheader("🧮 Feature Calculator")
    st.write("Calculate financial features for a borrower:")
    
    with st.form("feature_calculator"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            monthly_income = st.number_input("Monthly Income ($)", min_value=0, value=50000)
            monthly_emi = st.number_input("Monthly EMI ($)", min_value=0, value=3000)
        
        with col2:
            outstanding_loan = st.number_input("Outstanding Loan ($)", min_value=0, value=80000)
            collateral_value = st.number_input("Collateral Value ($)", min_value=0, value=120000)
        
        with col3:
            num_missed = st.number_input("Missed Payments", min_value=0, value=0)
            days_past_due = st.number_input("Days Past Due", min_value=0, value=0)
        
        calculate = st.form_submit_button("Calculate Features", type="primary")
        
        if calculate:
            # Calculate features
            dti_ratio = monthly_emi / monthly_income if monthly_income > 0 else 0
            ltv_ratio = outstanding_loan / collateral_value if collateral_value > 0 else 0
            payment_burden = monthly_emi / monthly_income if monthly_income > 0 else 0
            risk_score = (num_missed * 0.3) + (days_past_due * 0.2) + (dti_ratio * 0.25) + (ltv_ratio * 0.25)
            financial_stability = (monthly_income / 100000 * 0.4) + ((1 - num_missed/10) * 0.3) + ((1 - days_past_due/180) * 0.3)
            
            st.success("✅ Features Calculated!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("DTI Ratio", f"{dti_ratio:.3f}")
                st.metric("LTV Ratio", f"{ltv_ratio:.3f}")
            
            with col2:
                st.metric("Payment Burden", f"{payment_burden:.3f}")
                st.metric("Risk Score", f"{risk_score:.3f}")
            
            with col3:
                st.metric("Financial Stability", f"{financial_stability:.3f}")
                
                # Risk assessment
                if risk_score > 0.5:
                    st.error("🔴 High Risk")
                elif risk_score > 0.3:
                    st.warning("🟡 Medium Risk")
                else:
                    st.success("🟢 Low Risk")

with tab2:
    st.header("Borrower Segmentation")
    
    try:
        response = requests.get(f"{API_BASE}/features/segments")
        
        if response.status_code == 200:
            data = response.json()
            
            if 'segments' in data:
                segments_df = pd.DataFrame(data['segments'])
                
                # Segment distribution
                st.subheader("Segment Distribution")
                
                fig = px.pie(
                    segments_df,
                    names='Segment_Name',
                    values='Count',
                    title="Borrower Distribution by Segment",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Segment characteristics
                st.subheader("Segment Characteristics")
                
                if 'segment_profiles' in data:
                    profiles_df = pd.DataFrame(data['segment_profiles'])
                    
                    # Create radar chart for each segment
                    st.write("**Average Characteristics by Segment:**")
                    
                    for segment in profiles_df['Segment_Name'].unique():
                        segment_data = profiles_df[profiles_df['Segment_Name'] == segment]
                        
                        with st.expander(f"📊 {segment}"):
                            cols = st.columns(4)
                            
                            metrics = ['Avg_Income', 'Avg_Loan_Amount', 'Avg_Risk_Score', 'Avg_Missed_Payments']
                            for idx, metric in enumerate(metrics):
                                if metric in segment_data.columns:
                                    with cols[idx]:
                                        st.metric(
                                            metric.replace('Avg_', '').replace('_', ' '),
                                            f"{segment_data[metric].iloc[0]:,.2f}"
                                        )
            else:
                st.warning("Segment data not available. Please train the model first.")
        else:
            st.warning("Segmentation results not available. Please train the model first.")
    
    except Exception as e:
        st.error(f"Error loading segments: {str(e)}")
    
    # Segment predictor
    st.subheader("🎯 Predict Borrower Segment")
    st.write("Enter borrower details to predict which segment they belong to:")
    
    with st.form("segment_predictor"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            monthly_income = st.number_input("Monthly Income ($)", min_value=0, value=50000, key="seg_income")
            loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=100000, key="seg_loan")
        
        with col2:
            num_missed = st.number_input("Missed Payments", min_value=0, value=0, key="seg_missed")
            days_past_due = st.number_input("Days Past Due", min_value=0, value=0, key="seg_dpd")
        
        predict_segment = st.form_submit_button("Predict Segment", type="primary")
        
        if predict_segment:
            try:
                segment_data = {
                    "Age": age,
                    "Monthly_Income": monthly_income,
                    "Loan_Amount": loan_amount,
                    "Num_Missed_Payments": num_missed,
                    "Days_Past_Due": days_past_due
                }
                
                response = requests.post(f"{API_BASE}/features/predict-segment", json=segment_data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success(f"✅ Predicted Segment: **{result['segment_name']}**")
                    st.info(result['segment_description'])
                    
                    # Show calculated features
                    if 'calculated_features' in result:
                        st.subheader("📊 Calculated Features")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("DTI Ratio", f"{result['calculated_features']['DTI_Ratio']:.3f}")
                        
                        with col2:
                            st.metric("Risk Score", f"{result['calculated_features']['Risk_Score']:.3f}")
                else:
                    error_detail = response.json().get('detail', 'Unknown error')
                    st.error(f"Segment prediction failed: {error_detail}")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")

with tab3:
    st.header("Risk Profile Analysis")
    
    try:
        response = requests.get(f"{API_BASE}/features/risk-profiles")
        
        if response.status_code == 200:
            data = response.json()
            
            if 'risk_distribution' in data:
                risk_df = pd.DataFrame(data['risk_distribution'])
                
                # Risk level distribution
                st.subheader("Risk Level Distribution")
                
                fig = px.bar(
                    risk_df,
                    x='Risk_Level',
                    y='Count',
                    color='Risk_Level',
                    title="Borrowers by Risk Level",
                    color_discrete_map={
                        'Low Risk': '#51cf66',
                        'Medium Risk': '#ffa500',
                        'High Risk': '#ff6b6b'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk factors correlation
                if 'risk_factors' in data:
                    st.subheader("Risk Factors Correlation")
                    
                    factors_df = pd.DataFrame(data['risk_factors'])
                    
                    fig = px.scatter(
                        factors_df,
                        x='DTI_Ratio',
                        y='Risk_Score',
                        color='Risk_Level',
                        size='Outstanding_Loan_Amount',
                        hover_data=['Num_Missed_Payments', 'Days_Past_Due'],
                        title="DTI Ratio vs Risk Score",
                        color_discrete_map={
                            'Low': '#51cf66',
                            'Medium': '#ffa500',
                            'High': '#ff6b6b'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Risk profile data not available. Please train the model first.")
    
    except Exception as e:
        st.error(f"Error loading risk profiles: {str(e)}")
    
    # Risk assessment tool
    st.subheader("🎯 Quick Risk Assessment")
    
    with st.form("risk_assessment"):
        col1, col2 = st.columns(2)
        
        with col1:
            dti = st.slider("DTI Ratio", 0.0, 1.0, 0.3, 0.01)
            ltv = st.slider("LTV Ratio", 0.0, 2.0, 0.8, 0.01)
        
        with col2:
            missed = st.slider("Missed Payments", 0, 20, 2)
            dpd = st.slider("Days Past Due", 0, 365, 30)
        
        assess = st.form_submit_button("Assess Risk", type="primary")
        
        if assess:
            # Calculate risk score
            risk_score = (missed / 20 * 0.3) + (dpd / 365 * 0.2) + (dti * 0.25) + (ltv * 0.25)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Risk Score", f"{risk_score:.3f}")
            
            with col2:
                if risk_score > 0.5:
                    risk_level = "High Risk"
                    color = "🔴"
                elif risk_score > 0.3:
                    risk_level = "Medium Risk"
                    color = "🟡"
                else:
                    risk_level = "Low Risk"
                    color = "🟢"
                
                st.metric("Risk Level", f"{color} {risk_level}")
            
            with col3:
                recovery_prob = max(0, min(100, (1 - risk_score) * 100))
                st.metric("Recovery Probability", f"{recovery_prob:.1f}%")
            
            # Recommendations
            st.subheader("📋 Recommendations")
            
            if risk_score > 0.5:
                st.error("""
                **High Risk Alert!**
                - Immediate intervention required
                - Consider legal action
                - Increase collection frequency
                - Review collateral value
                """)
            elif risk_score > 0.3:
                st.warning("""
                **Medium Risk - Monitor Closely**
                - Schedule payment plan review
                - Send payment reminders
                - Consider restructuring options
                """)
            else:
                st.success("""
                **Low Risk - Standard Monitoring**
                - Continue regular monitoring
                - Maintain current payment schedule
                - Annual review recommended
                """)