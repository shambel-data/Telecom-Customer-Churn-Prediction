import streamlit as st
import joblib
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("churn_model.joblib")

bundle = load_model()

model = bundle["model"]
feature_names = bundle["feature_names"]
categorical_mapping = bundle["categorical_mapping"]
# Feature importance
feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance", ascending=True
)
# Application title
# Application title
st.title("📊 Telecom Customer Churn Prediction")



# Dashboard metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Features", "9")

with col2:
    st.metric("Selected Model", "Random Forest")

with col3:
    st.metric("Test Accuracy", "94.75%")

st.divider()
with st.form("customer_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        account_length = st.number_input("Account Length", min_value=0, value=100)
        international_plan = st.selectbox("International Plan", ["No", "Yes"])
        voice_mail_plan = st.selectbox("Voice Mail Plan", ["No", "Yes"])

    with col2:
        number_vmail_messages = st.number_input("Voicemail Messages", min_value=0, value=0)
        total_day_charge = st.number_input("Total Day Charge", min_value=0.0, value=30.0)
        total_eve_charge = st.number_input("Total Evening Charge", min_value=0.0, value=15.0)

    with col3:
        total_night_charge = st.number_input("Total Night Charge", min_value=0.0, value=10.0)
        total_intl_charge = st.number_input("Total International Charge", min_value=0.0, value=2.0)
        customer_service_calls = st.number_input("Customer Service Calls", min_value=0, value=1)

    predict_button = st.form_submit_button("Predict Customer Churn")
if predict_button:

    customer_data = {
        "Account length": account_length,
        "International plan": categorical_mapping["International plan"][international_plan],
        "Voice mail plan": categorical_mapping["Voice mail plan"][voice_mail_plan],
        "Number vmail messages": number_vmail_messages,
        "Total day charge": total_day_charge,
        "Total eve charge": total_eve_charge,
        "Total night charge": total_night_charge,
        "Total intl charge": total_intl_charge,
        "Customer service calls": customer_service_calls
    }
    input_data = pd.DataFrame([customer_data], columns=feature_names)

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    churn_probability = probabilities[list(model.classes_).index(1)]


