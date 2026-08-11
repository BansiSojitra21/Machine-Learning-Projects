import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)


# --------------------------------------------------
# Load Model and Preprocessor
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

with open("student_performance.pkl", "rb") as file:
    model = pickle.load(file)

with open("preprocessor.pkl", "rb") as file:
    preprocessor = pickle.load(file)


# --------------------------------------------------
# Student Performance Predictor
# --------------------------------------------------

st.title("🎓 Student Performance Predictor")

st.write(
    "Enter the student's information below to predict "
    "the Performance Index."
)


# --------------------------------------------------
# Input Section
# --------------------------------------------------

st.subheader("Student Information")


col1, col2 = st.columns(2)


with col1:

    hours_studied = st.number_input(
        "Hours Studied",
        min_value=1,
        max_value=9,
        value=5,
        step=1
    )

    previous_scores = st.number_input(
        "Previous Scores",
        min_value=40,
        max_value=100,
        value=70,
        step=1
    )

    sleep_hours = st.number_input(
        "Sleep Hours",
        min_value=4,
        max_value=9,
        value=7,
        step=1
    )


with col2:

    extracurricular = st.selectbox(
        "Extracurricular Activities",
        ["Yes", "No"]
    )

    sample_papers = st.number_input(
        "Sample Question Papers Practiced",
        min_value=0,
        max_value=9,
        value=4,
        step=1
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Performance", use_container_width=True):

    # Convert Yes / No to 1 / 0
    extracurricular_value = (
        1 if extracurricular == "Yes" else 0
    )


    # Create DataFrame
    input_data = pd.DataFrame({
        "Hours Studied": [hours_studied],
        "Previous Scores": [previous_scores],
        "Extracurricular Activities": [extracurricular_value],
        "Sleep Hours": [sleep_hours],
        "Sample Question Papers Practiced": [sample_papers]
    })


    # --------------------------------------------------
    # Apply Preprocessor
    # --------------------------------------------------

    transformed_data = preprocessor.transform(input_data)


    # --------------------------------------------------
    # Make Prediction
    # --------------------------------------------------

    prediction = model.predict(transformed_data)[0]


    # --------------------------------------------------
    # Display Result
    # --------------------------------------------------

    st.success(
        f"Predicted Performance Index: **{prediction:.2f}**"
    )
