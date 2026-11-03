
import streamlit as st
import joblib
import pandas as pd

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 2. LOAD TRAINED MODEL
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load("churn_model.joblib")


try:
    bundle = load_model()

    model = bundle["model"]
    feature_names = bundle["feature_names"]
    categorical_mapping = bundle["categorical_mapping"]

except Exception as e:
    st.error("Unable to load the trained model.")
    st.exception(e)
    st.stop()


# =====================================================
# 3. FEATURE IMPORTANCE
# =====================================================

if not hasattr(model, "feature_importances_"):
    st.error("The loaded model does not support feature importance.")
    st.stop()

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=True
)


# =====================================================
# 4. SIDEBAR
# =====================================================

with st.sidebar:

    st.title("📊 Churn Predictor")

    st.write(
        "An intelligent machine learning system "
        "for predicting telecom customer churn."
    )

    st.divider()

    st.subheader("About the Project")

    st.write(
        "This application uses a Random Forest "
        "classification model to estimate whether "
        "a telecom customer is likely to leave."
    )

    st.divider()

    st.subheader("Model Information")

    st.write("**Algorithm:** Random Forest")
    st.write("**Task:** Binary Classification")
    st.write("**Number of Features:** 9")
    st.write("**Number of Classes:** 2")

    st.divider()

    st.caption("Telecom Customer Churn Prediction Project")


# =====================================================
# 5. APPLICATION HEADER
# =====================================================

st.title("📊 Telecom Customer Churn Prediction")

st.markdown(
    "Predict customer churn using machine learning "
    "and identify customers who may need retention support."
)

st.divider()


# =====================================================
# 6. DASHBOARD METRICS
# =====================================================

st.subheader("📈 Model Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Input Features",
        value="9"
    )

with col2:
    st.metric(
        label="Selected Algorithm",
        value="Random Forest"
    )

with col3:
    st.metric(
        label="Test Accuracy",
        value="94.75%"
    )

st.caption(
    "Test accuracy is based on the previously evaluated "
    "test dataset."
)

st.divider()


# =====================================================
# 7. CUSTOMER INPUT FORM
# =====================================================

st.subheader("👤 Customer Information")

st.write(
    "Enter the customer's information below "
    "to generate a churn prediction."
)

with st.form("customer_form"):

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("**Customer Details**")

        account_length = st.number_input(
            "Account Length",
            min_value=0,
            value=100
        )

        international_plan = st.selectbox(
            "International Plan",
            ["No", "Yes"]
        )

        voice_mail_plan = st.selectbox(
            "Voice Mail Plan",
            ["No", "Yes"]
        )

    with col2:

        st.markdown("**Communication Details**")

        number_vmail_messages = st.number_input(
            "Voicemail Messages",
            min_value=0,
            value=0
        )

        customer_service_calls = st.number_input(
            "Customer Service Calls",
            min_value=0,
            value=1
        )

        total_intl_charge = st.number_input(
            "Total International Charge",
            min_value=0.0,
            value=2.0,
            step=0.1
        )

    with col3:

        st.markdown("**Usage Charges**")

        total_day_charge = st.number_input(
            "Total Day Charge",
            min_value=0.0,
            value=30.0,
            step=0.1
        )

        total_eve_charge = st.number_input(
            "Total Evening Charge",
            min_value=0.0,
            value=15.0,
            step=0.1
        )

        total_night_charge = st.number_input(
            "Total Night Charge",
            min_value=0.0,
            value=10.0,
            step=0.1
        )

    st.divider()

    predict_button = st.form_submit_button(
        "🔍 Predict Customer Churn",
        use_container_width=True
    )


# =====================================================
# 8. CUSTOMER CHURN PREDICTION
# =====================================================

if predict_button:

    # Prepare customer information

    customer_data = {
        "Account length": account_length,

        "International plan":
            categorical_mapping["International plan"][
                international_plan
            ],

        "Voice mail plan":
            categorical_mapping["Voice mail plan"][
                voice_mail_plan
            ],

        "Number vmail messages": number_vmail_messages,
        "Total day charge": total_day_charge,
        "Total eve charge": total_eve_charge,
        "Total night charge": total_night_charge,
        "Total intl charge": total_intl_charge,
        "Customer service calls": customer_service_calls
    }

    # Convert input into a DataFrame

    input_data = pd.DataFrame(
        [customer_data],
        columns=feature_names
    )

    # Generate prediction

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    # Get probability for churn class (1)

    churn_class_index = list(model.classes_).index(1)

    churn_probability = probabilities[churn_class_index]

    # =================================================
    # 9. DISPLAY PREDICTION RESULTS
    # =================================================

    st.divider()

    st.subheader("🎯 Customer Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        if prediction == 1:

            st.error(
                "⚠️ This customer is likely to churn."
            )

        else:

            st.success(
                "✅ This customer is likely to stay."
            )

    # =================================================
    # 10. CUSTOMER RISK CLASSIFICATION
    # =================================================

    if churn_probability < 0.30:

        risk_level = "Low Risk"
        st.success("🟢 Low Customer Churn Risk")

    elif churn_probability < 0.70:

        risk_level = "Medium Risk"
        st.warning("🟡 Medium Customer Churn Risk")

    else:

        risk_level = "High Risk"
        st.error("🔴 High Customer Churn Risk")

    # =================================================
    # 11. DISPLAY PROBABILITY AND RISK
    # =================================================

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Churn Probability",
            f"{churn_probability * 100:.2f}%"
        )

    with result_col2:

        st.metric(
            "Customer Risk Level",
            risk_level
        )

    st.write("### Churn Probability Indicator")

    st.progress(float(churn_probability))

    st.caption(
        "The probability represents the model's estimated "
        "likelihood of the customer belonging to the churn class."
    )

    # =================================================
    # 12. CUSTOMER RECOMMENDATIONS
    # =================================================

    st.divider()

    st.subheader("💡 Suggested Customer Action")

    if prediction == 1:

        st.info(
            "Consider contacting this customer, reviewing "
            "their service experience, and offering suitable "
            "retention support."
        )

    else:

        st.info(
            "Continue providing reliable service and "
            "monitor customer satisfaction."
        )


# =====================================================
# 13. FEATURE IMPORTANCE ANALYSIS
# =====================================================

st.divider()

st.subheader("📊 Feature Importance Analysis")

st.write(
    "This chart shows the relative importance of the "
    "nine input features in the Random Forest model."
)

chart_data = feature_importance.set_index("Feature")

st.bar_chart(
    chart_data,
    horizontal=True
)

with st.expander("View Feature Importance Values"):

    importance_table = feature_importance.sort_values(
        by="Importance",
        ascending=False
    ).copy()

    importance_table["Importance"] = (
        importance_table["Importance"] * 100
    ).round(2)

    importance_table = importance_table.rename(
        columns={"Importance": "Importance (%)"}
    )

    st.dataframe(
        importance_table,
        use_container_width=True,
        hide_index=True
    )

st.caption(
    "Feature importance describes how much each feature "
    "contributes to the Random Forest's overall decision-making. "
    "It does not establish causation or explain an individual "
    "prediction."
)


# =====================================================
# 14. FOOTER
# =====================================================

st.divider()

st.caption(
    "Telecom Customer Churn Prediction | "
    "Machine Learning Project | Random Forest"
)
