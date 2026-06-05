import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load('../Models/model.pkl')
scaler = joblib.load('../Models/scaler.pkl')

st.set_page_config(page_title="Churn Predictor", page_icon="📡")
st.title("📡 Customer Churn Predictor")
st.markdown("Fill in customer details to predict if they will churn.")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 120.0, 65.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col2:
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment = st.selectbox("Payment Method", [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ])

if st.button("Predict Churn"):
    input_dict = {
        'gender': 0, 'SeniorCitizen': 1 if senior == "Yes" else 0,
        'Partner': 1 if partner == "Yes" else 0, 'Dependents': 0,
        'tenure': tenure, 'PhoneService': 1, 'PaperlessBilling': 1 if paperless == "Yes" else 0,
        'MonthlyCharges': monthly_charges, 'TotalCharges': tenure * monthly_charges,
        'MultipleLines_No phone service': 0, 'MultipleLines_Yes': 0,
        'InternetService_Fiber optic': 1 if internet == "Fiber optic" else 0,
        'InternetService_No': 1 if internet == "No" else 0,
        'OnlineSecurity_No internet service': 0, 'OnlineSecurity_Yes': 0,
        'OnlineBackup_No internet service': 0, 'OnlineBackup_Yes': 0,
        'DeviceProtection_No internet service': 0, 'DeviceProtection_Yes': 0,
        'TechSupport_No internet service': 0, 'TechSupport_Yes': 0,
        'StreamingTV_No internet service': 0, 'StreamingTV_Yes': 0,
        'StreamingMovies_No internet service': 0, 'StreamingMovies_Yes': 0,
        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if payment == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if payment == "Mailed check" else 0,
    }

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    prob = model.predict_proba(input_scaled)[0][1]

    st.divider()
    st.metric("Churn Probability", f"{prob:.0%}")

    if prob > 0.6:
        st.error("🔴 High churn risk — consider offering a discount or contract upgrade")
    elif prob > 0.4:
        st.warning("🟡 Medium risk — worth monitoring this customer")
    else:
        st.success("🟢 Low churn risk — customer looks stable")