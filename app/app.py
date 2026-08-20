from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"
MODEL_PATH = PROJECT_ROOT / "models" / "gradient_boosting_churn_model.joblib"

preprocessor = joblib.load(PREPROCESSOR_PATH)
model = joblib.load(MODEL_PATH)


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")

st.markdown(
    """
    Predict customer churn risk using a tuned **Gradient Boosting**
    machine learning model.

    Enter the customer's account, service, and billing information below
    to estimate **churn probability**, assign a **risk segment**, and
    generate a recommended **retention action**.
    """
)

st.caption(
    "Model: Tuned Gradient Boosting | "
    "Decision Threshold: 0.30 | "
    "Objective: Customer Retention"
)

st.divider()

st.subheader("Customer Profile")

profile_col1, profile_col2 = st.columns(2)

with profile_col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen_label = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )
    senior_citizen = 1 if senior_citizen_label == "Yes" else 0
    partner = st.selectbox("Partner", ["Yes", "No"])

with profile_col2:
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

st.subheader("Services")

service_col1, service_col2 = st.columns(2)

with service_col1:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

with service_col2:
    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

tech_support = st.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)

st.subheader("Account & Billing")

billing_col1, billing_col2 = st.columns(2)

with billing_col1:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

with billing_col2:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0
    )

customer_data = pd.DataFrame({
"gender": [gender],
"SeniorCitizen": [senior_citizen],
"Partner": [partner],
"Dependents": [dependents],
"tenure": [tenure],
"PhoneService": [phone_service],
"MultipleLines": [multiple_lines],
"InternetService": [internet_service],
"OnlineSecurity": [online_security],
"OnlineBackup": [online_backup],
"DeviceProtection": [device_protection],
"TechSupport": [tech_support],
"StreamingTV": [streaming_tv],
"StreamingMovies": [streaming_movies],
"Contract": [contract],
"PaperlessBilling": [paperless_billing],
"PaymentMethod": [payment_method],
"MonthlyCharges": [monthly_charges],
"TotalCharges": [total_charges]
})

st.subheader("Churn Prediction")

if st.button("Predict Churn"):
    customer_processed = preprocessor.transform(customer_data)

    churn_probability = model.predict_proba(
        customer_processed
    )[:, 1][0]

    threshold = 0.30

    churn_prediction = (
        "Likely to Churn"
        if churn_probability >= threshold
        else "Likely to Stay"
    )

    if churn_probability < 0.20:
        risk_segment = "Low Risk"
        recommendation = "Standard customer engagement"

    elif churn_probability < 0.30:
        risk_segment = "Moderate Risk"
        recommendation = "Monitor customer and consider low-cost outreach"

    elif churn_probability < 0.60:
        risk_segment = "High Risk"
        recommendation = "Targeted retention campaign recommended"

    else:
        risk_segment = "Very High Risk"
        recommendation = "Priority personalized retention intervention"

    st.divider()
    st.subheader("Prediction Results")

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric(
            "Churn Probability",
            f"{churn_probability:.1%}"
        )

    with metric_col2:
        st.metric(
            "Risk Segment",
            risk_segment
        )

    if churn_probability >= threshold:
        st.warning(f"⚠️ {churn_prediction}")
    else:
        st.success(f"✅ {churn_prediction}")

    st.info(
        f"**Recommended Action:** {recommendation}"
    )

    st.caption(
        "Prediction uses the tuned Gradient Boosting model "
        "with a classification threshold of 0.30."
    )

    