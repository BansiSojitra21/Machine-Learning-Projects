import streamlit as st
import pandas as pd
import pickle
from pathlib import Path


# ============================================================
# Load Model and Preprocessor
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "churn_model.pkl", "rb") as file:
    model = pickle.load(file)

with open(BASE_DIR / "preprocessor.pkl", "rb") as file:
    preprocessor = pickle.load(file)


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# Title
# ============================================================

st.title("📊 Telco Customer Churn Prediction")

st.write(
    "Enter customer details below to predict whether "
    "the customer is likely to churn."
)


# ============================================================
# Input Columns
# ============================================================

col1, col2, col3 = st.columns(3)


# ============================================================
# Column 1
# ============================================================

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["Yes", "No"]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )


# ============================================================
# Column 2
# ============================================================

with col2:

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )


# ============================================================
# Column 3
# ============================================================

with col3:

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

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

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=800.0
    )


# ============================================================
# Prediction
# ============================================================

st.divider()

if st.button("🔮 Predict Churn", use_container_width=True):

    # Convert Senior Citizen Yes/No back to 1/0
    senior_citizen_value = 1 if senior_citizen == "Yes" else 0


    # --------------------------------------------------------
    # Create Input DataFrame
    # --------------------------------------------------------

    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen_value],
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


    # --------------------------------------------------------
    # Apply Preprocessor
    # --------------------------------------------------------

    input_transformed = preprocessor.transform(input_data)


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(input_transformed)[0]


    # --------------------------------------------------------
    # Churn Probability
    # --------------------------------------------------------

    probability = model.predict_proba(input_transformed)[0][1]


    # --------------------------------------------------------
    # Display Result
    # --------------------------------------------------------

    if prediction == 1:
        st.error("⚠️ Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")


    st.metric(
        "Churn Probability",
        f"{probability:.2%}"
    )
