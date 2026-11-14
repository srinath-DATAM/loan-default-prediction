import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="🏦",
    layout="wide"
)

# 1. Load saved model, scaler, and exact training features
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load('load_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('features.pkl') # Automatically loads exact training columns
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

# 2. Web App Header
st.title("🏦 Loan Default Risk Prediction App")
st.write("Enter all the applicant's details in the sidebar to analyze the credit default risk.")

# 3. Sidebar Layout for Inputs
st.sidebar.header("📋 Applicant Complete Details")

# Numerical Inputs
applicant_income = st.sidebar.number_input("Applicant Income (₹)", min_value=0, value=50000, step=1000)
coapplicant_income = st.sidebar.number_input("Coapplicant Income (₹)", min_value=0, value=0, step=1000)
loan_amount = st.sidebar.number_input("Loan Amount (₹)", min_value=0, value=150000, step=5000)
loan_amount_term = st.sidebar.number_input("Loan Amount Term (Months)", min_value=0, value=360, step=12)
dependents = st.sidebar.selectbox("Number of Dependents", options=[0, 1, 2, 3], format_func=lambda x: f"{x} Dependents" if x < 3 else "3+ Dependents")

# Categorical & Binary Inputs
credit_history = st.sidebar.selectbox(
    "Credit History Status", 
    options=[1.0, 0.0], 
    format_func=lambda x: "Good (1.0)" if x == 1.0 else "Bad (0.0)"
)

education = st.sidebar.selectbox(
    "Education Status",
    options=[0, 1],
    format_func=lambda x: "Graduate" if x == 0 else "Not Graduate"
)

property_area = st.sidebar.selectbox(
    "Property Area",
    options=["Semiurban", "Urban", "Rural"]
)

self_employed = st.sidebar.selectbox(
    "Self Employed?",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

married = st.sidebar.selectbox(
    "Marital Status",
    options=[0, 1],
    format_func=lambda x: "Unmarried" if x == 0 else "Married"
)

gender = st.sidebar.selectbox(
    "Gender",
    options=[0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

# Values calculation
edu_not_grad = 1 if education == 1 else 0
prop_semiurban = 1 if property_area == "Semiurban" else 0
prop_urban = 1 if property_area == "Urban" else 0
self_emp_yes = 1 if self_employed == 1 else 0
married_yes = 1 if married == 1 else 0
gender_male = 1 if gender == 1 else 0

# 4. Main Screen Prediction Trigger
if st.sidebar.button("Predict Default Risk", type="primary"):
    
    # User inputs mapped into a DataFrame using training script's exact columns list
    user_values = [
        credit_history, applicant_income, loan_amount, coapplicant_income, 
        loan_amount_term, dependents, edu_not_grad, prop_semiurban, 
        prop_urban, self_emp_yes, married_yes, gender_male
    ]
    
    input_data = pd.DataFrame([user_values], columns=feature_names)
    
    # Loading animation
   # Loading animation
    with st.spinner("Analyzing all parameters..."):
        # Scale input data safely
        scaled_data = scaler.transform(input_data)
        
        # Get prediction probabilities instead of direct 0 or 1
        # [Probability of Class 0 (Low Risk), Probability of Class 1 (High Risk)]
        probabilities = model.predict_proba(scaled_data)
        default_prob = probabilities[0][1] # Probability of defaulting
        
        # Custom threshold (e.g., if default probability > 35%, mark as High Risk)
        threshold = 0.35
        prediction = 1 if default_prob >= threshold else 0
    
    st.markdown("---")
    st.subheader("📊 Prediction Result:")
    
    # Result Display based on strict custom logic
    if prediction == 0:
        st.success(f"✅ **Low Risk / Will Repay (Loan Approved)** (Confidence: {(1-default_prob)*100:.1f}%)")
        st.balloons()
    else:
        st.error(f"❌ **High Risk / Default Risk (Loan Rejected)** (Default Risk: {default_prob*100:.1f}%)")