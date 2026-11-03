from pathlib import Path
import streamlit as st
import joblib
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load improved trained model
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "churn_model.joblib"

    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        st.write("Files available in the app directory:")
        st.write([file.name for file in Path(__file__).parent.iterdir()])
        st.stop()

    bundle = joblib.load(model_path)

    model = bundle["model"]
    feature_names = bundle["feature_names"]
    categorical_mapping = bundle["categorical_mapping"]

    return model, feature_names, categorical_mapping


model, feature_names, categorical_mapping = load_model()

# Application title
st.title("📊 Telecom Customer Churn Prediction")
st.write("Predict whether a telecom customer is likely to leave the company.")

st.divider()

st.subheader("Enter Customer Information")

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

# Prediction section

    # Put the corrected prediction section here
prediction = model.predict(input_data)[0]
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
st.divider()
st.subheader("Prediction Result")

if prediction == 1:
    st.error("⚠️ This customer is likely to churn.")
else:
    st.success("✅ This customer is likely to stay.")

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(input_data)[0]
    churn_index = list(model.classes_).index(1)
    churn_probability = probabilities[churn_index]

    st.metric(
        "Churn Probability",
        f"{churn_probability * 100:.2f}%"
    )

    st.progress(float(churn_probability))
    # Predict churn probability
    probabilities = model.predict_proba(input_data)[0]

    churn_probability = probabilities[list(model.classes_).index(1)]

    # Apply selected threshold
    prediction = int(churn_probability >= threshold)

    # Display prediction result
    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")

    st.metric("Churn Probability", f"{churn_probability * 100:.2f}%")
    st.progress(float(churn_probability))
