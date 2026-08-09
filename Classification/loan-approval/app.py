import streamlit as st
import pickle
import pandas as pd
from pathlib import Path

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Loan Approval Prediction", page_icon="🏦", layout="centered"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
<style>

    /* Main application background */
    .stApp {
        background-color: #f4f7fb;
    }

    /* Main content width */
    .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Remove Streamlit menu/footer */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* ---------------- HEADER ---------------- */

    .hero {
        background: linear-gradient(135deg, #172554, #2563eb);
        padding: 32px 25px;
        border-radius: 18px;
        text-align: center;
        margin: 25px 0px;
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.20);
    }

    .hero-title {
        color: white !important;
        font-size: 36px !important;
        font-weight: 700 !important;
        margin: 0 !important;
    }

    .info-text {
        color: #dbeafe !important;
        font-size: 15px !important;
        margin: 0 !important;
    }

    /* ---------------- SECTION HEADINGS ---------------- */

    .section-title {
        color: #111827 !important;
        font-size: 21px !important;
        font-weight: 700 !important;
        margin-top: 10px !important;
        margin-bottom: 15px !important;
    }

    /* ---------------- INPUT CARDS ---------------- */

    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: white;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
        padding: 10px;
    }

    /* Input labels */
    label {
        color: #1f2937 !important;
        font-weight: 600 !important;
    }

    /* Slider text */
    [data-testid="stSlider"] {
        color: #1f2937 !important;
    }

    /* Select box text */
    [data-baseweb="select"] {
        color: #111827 !important;
    }

    /* Number input text */
    input {
        color: #fff !important;
    }

    /* ---------------- BUTTON ---------------- */

    .stButton > button {
        width: 100%;
        height: 52px;
        border-radius: 12px;
        border: none;
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white !important;
        font-size: 18px;
        font-weight: 700;
        box-shadow: 0 5px 15px rgba(37, 99, 235, 0.25);
        transition: 0.2s;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
        transform: translateY(-1px);
    }

    /* ---------------- RESULT CARDS ---------------- */

    .approved-box {
        background-color: #ecfdf5;
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        margin-top: 25px;
    }

    .approved-title {
        color: #047857 !important;
        font-size: 27px !important;
        font-weight: 700 !important;
    }

    .approved-text {
        color: #065f46 !important;
        font-size: 16px !important;
    }

    .rejected-box {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        margin-top: 25px;
    }

    .rejected-title {
        color: #b91c1c !important;
        font-size: 27px !important;
        font-weight: 700 !important;
    }

    .rejected-text {
        color: #991b1b !important;
        font-size: 16px !important;
    }


/* Normal text after prediction */
.result-text {
    color: #1f2937 !important;
    font-size: 16px;
    font-weight: 500;
}

/* Probability label */
.probability-label {
    color: #1f2937 !important;
    font-size: 15px;
    font-weight: 600;
    margin-top: 15px;
}

/* Input box */
div[data-testid="stNumberInput"] input {
    background-color: #00000 !important;
    color: #fff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #111827 !important;
}
/* =========================================================
   SELECT BOX
   ========================================================= */

/* Selectbox label */
div[data-testid="stSelectbox"] label {
    color: #111827 !important;
    font-weight: 500 !important;
}

/* Selectbox container */
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    min-height: 42px !important;
}

/* Selected value */
div[data-testid="stSelectbox"] div[data-baseweb="select"] span {
    color: #111827 !important;
}

/* Dropdown arrow */
div[data-testid="stSelectbox"] svg {
    fill: #111827 !important;
    color: #111827 !important;
}

/* Selectbox focus */
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 1px #2563eb !important;
}
</style>
""",
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent

# =========================================================
# LOAD MODEL
# =========================================================

with open(BASE_DIR / "loan_model.pkl", "rb") as file:
    model = pickle.load(file)


# =========================================================
# LOAD SCALER
# =========================================================

with open(BASE_DIR / "scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🏦 Loan Approval Predictor</div>
        <div class="info-text">
            Enter the applicant's personal and financial information to predict whether the loan is likely to be approved.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# TWO COLUMN INPUT SECTION
# =========================================================

left_col, right_col = st.columns(2, gap="large")


# =========================================================
# LEFT COLUMN
# =========================================================

with left_col:

    with st.container(border=True):

        st.markdown(
            '<div class="section-title">👤 Personal Information</div>',
            unsafe_allow_html=True,
        )

        # Number of dependents
        no_of_dependents = st.number_input(
            "Number of Dependents", min_value=0, max_value=10, value=2
        )

        # Education
        education_option = st.selectbox("Education", ["Graduate", "Not Graduate"])

        # Self employed
        self_employed_option = st.selectbox("Self Employed", ["Yes", "No"])

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================================================
# LOAN INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">🏠 Loan Information</div>', unsafe_allow_html=True
)

loan_col1, loan_col2 = st.columns(2, gap="large")


with loan_col1:

    with st.container(border=True):

        loan_term = st.slider("Loan Term (Years)", min_value=1, max_value=50, value=10)


with loan_col2:

    with st.container(border=True):

        cibil_score = st.slider("CIBIL Score", min_value=300, max_value=900, value=700)

# =========================================================
# RIGHT COLUMN
# =========================================================

with right_col:

    with st.container(border=True):

        st.markdown(
            '<div class="section-title">💰 Financial Information</div>',
            unsafe_allow_html=True,
        )

        # Annual income
        income_annum = st.slider(
            "Annual Income (₹)",
            min_value=0,
            max_value=100000000,
            value=5000000,
            step=100000,
            format="₹%d",
        )

        # Loan amount
        loan_amount = st.slider(
            "Loan Amount (₹)",
            min_value=0,
            max_value=50000000,
            value=10000000,
            step=100000,
            format="₹%d",
        )

        # Assets
        assets = st.slider(
            "Total Assets (₹)",
            min_value=0,
            max_value=100000000,
            value=10000000,
            step=100000,
            format="₹%d",
        )


# =========================================================
# CONVERT CATEGORICAL VALUES
# =========================================================

education = 1 if education_option == "Graduate" else 0

self_employed = 1 if self_employed_option == "Yes" else 0


# =========================================================
# PREDICTION
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Predict Loan Status"):

    input_data = pd.DataFrame(
        [
            [
                no_of_dependents,
                education,
                self_employed,
                income_annum,
                loan_amount,
                loan_term,
                cibil_score,
                assets,
            ]
        ],
        columns=[
            "no_of_dependents",
            "education",
            "self_employed",
            "income_annum",
            "loan_amount",
            "loan_term",
            "cibil_score",
            "assets",
        ],
    )

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0]

    if prediction == 1:

        st.success("✅ Loan Approved")

    else:

        st.error("❌ Loan Rejected")
