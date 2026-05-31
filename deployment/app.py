import os

import joblib
import pandas as pd
import streamlit as st

from huggingface_hub import hf_hub_download


MODEL_REPO_ID = 'ChaithuML/MLOps-Tourism-Prediction-Model'
FEATURES = [
    "Age",
    "MonthlyIncome",
    "Passport",
    "NumberOfTrips",
    "PitchSatisfactionScore",
    "Designation",
]

st.set_page_config(page_title="Visit With Us Predictor", layout="centered")
st.title("Visit With Us: Wellness Package Predictor")
st.info(
    "Fill in the customer details below and click **Predict Purchase** "
    "to see the likelihood of buying the Wellness Tourism Package."
)


@st.cache_resource
def load_model():
    # Model is pulled from Hugging Face Hub so app and training stay aligned.
    model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename="model.joblib",
        repo_type="model",
    )
    return joblib.load(model_path)


try:
    model = load_model()
    st.success("Model loaded successfully.")
except Exception as ex:
    st.error(f"Model loading failed: {ex}")
    model = None

st.divider()

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=75, value=None, placeholder="Enter age")
    income = st.number_input(
        "Monthly Income (₹)",
        min_value=0,
        max_value=200000,
        value=None,
        placeholder="Enter monthly income",
    )
    passport = st.selectbox("Has Passport?", options=["No", "Yes"])
with col2:
    trips = st.number_input(
        "Number Of Trips Per Year",
        min_value=0,
        max_value=20,
        value=None,
        placeholder="Enter number of trips",
    )
    pitch_sat = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    designation = st.selectbox(
        "Designation",
        options=["AVP", "Executive", "Manager", "Senior Manager", "VP"],
    )

if st.button("Predict Purchase", type="primary"):
    if model is None:
        st.error("Model is not loaded. Cannot make predictions.")
        st.stop()

    # Validate required inputs before sending data to the model.
    errors = []
    if age is None:
        errors.append("Age is required.")
    if income is None:
        errors.append("Monthly Income is required.")
    if trips is None:
        errors.append("Number Of Trips is required.")
    if errors:
        for err in errors:
            st.warning(err)
        st.stop()

    designation_map = {
        "AVP": 0,
        "Executive": 1,
        "Manager": 2,
        "Senior Manager": 3,
        "VP": 4,
    }
    passport_val = 1 if passport == "Yes" else 0

    row = {
        "Age": int(age),
        "MonthlyIncome": float(income),
        "Passport": passport_val,
        "NumberOfTrips": int(trips),
        "PitchSatisfactionScore": pitch_sat,
        "Designation": designation_map[designation],
    }
    input_df = pd.DataFrame([row])[FEATURES]

    pred = int(model.predict(input_df)[0])
    prob = float(model.predict_proba(input_df)[0][1])

    st.divider()
    if pred == 1:
        st.success(
            f"**Likely to buy the Wellness Package**\n\n"
            f"Purchase probability: **{prob * 100:.1f}%**"
        )
    else:
        st.warning(
            f"**Unlikely to buy the Wellness Package**\n\n"
            f"Purchase probability: **{prob * 100:.1f}%**"
        )
