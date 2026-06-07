import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

@st.cache_resource
def train_model():
    df = pd.read_csv('Data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop('customerID', axis=1, inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})

    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                  'OnlineBackup', 'DeviceProtection', 'TechSupport',
                  'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    # Convert all bool columns to int
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_scaled, y)
    return model, scaler, list(X.columns)

model, scaler, feature_cols = train_model()

st.set_page_config(page_title="Churn Predictor", page_icon="📡")
st.title("📡 Customer Churn Predictor")
st.markdown("Fill in customer details to predict if they will churn.")

col1, col2 = st.columns(2)

with col1:
    tenure          = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 120.0, 65.0)
    contract        = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet        = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col2:
    senior    = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner   = st.selectbox("Has Partner", ["No", "Yes"])
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment   = st.selectbox("Payment Method", [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ])

if st.button("Predict Churn"):
    input_dict = {col: 0 for col in feature_cols}

    input_dict['tenure']           = tenure
    input_dict['MonthlyCharges']   = monthly_charges
    input_dict['TotalCharges']     = tenure * monthly_charges
    input_dict['SeniorCitizen']    = 1 if senior == "Yes" else 0
    input_dict['Partner']          = 1 if partner == "Yes" else 0
    input_dict['PaperlessBilling'] = 1 if paperless == "Yes" else 0

    input_dict['InternetService_Fiber optic'] = 1 if internet == "Fiber optic" else 0
    input_dict['InternetService_No']          = 1 if internet == "No" else 0
    input_dict['Contract_One year']           = 1 if contract == "One year" else 0
    input_dict['Contract_Two year']           = 1 if contract == "Two year" else 0
    input_dict['PaymentMethod_Credit card (automatic)'] = 1 if payment == "Credit card (automatic)" else 0
    input_dict['PaymentMethod_Electronic check']        = 1 if payment == "Electronic check" else 0
    input_dict['PaymentMethod_Mailed check']            = 1 if payment == "Mailed check" else 0

    input_df     = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    prob         = model.predict_proba(input_scaled)[0][1]

    st.divider()
    st.metric("Churn Probability", f"{prob:.0%}")
    if prob > 0.6:
        st.error("🔴 High churn risk")
    elif prob > 0.4:
        st.warning("🟡 Medium risk")
    else:
        st.success("🟢 Low churn risk")