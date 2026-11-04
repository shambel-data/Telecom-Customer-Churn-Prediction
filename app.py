
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ==================================================
# 2. LOAD TRAINED MODEL
# ==================================================

@st.cache_resource
def load_model():
    return joblib.load("churn_model.joblib")


try:
    bundle = load_model()

    model = bundle["model"]
    feature_names = list(bundle["feature_names"])
    categorical_mapping = bundle["categorical_mapping"]

except Exception as e:
    st.error(f"Unable to load the trained model: {e}")
    st.stop()


# ==================================================
# 3. FEATURE CONFIGURATION
# ==================================================

numeric_features = [
    "Account length",
    "Number vmail messages",
    "Total day charge",
    "Total eve charge",
    "Total night charge",
    "Total intl charge",
    "Customer service calls"
]

categorical_features = [
    "International plan",
    "Voice mail plan"
]

required_features = numeric_features + categorical_features

if (
    len(feature_names) != 9
    or set(feature_names) != set(required_features)
):
    st.error(
        "The model's feature names do not match "
        "the expected nine customer features."
    )
    st.stop()

if not hasattr(model, "predict_proba"):
    st.error("The loaded model does not support probability prediction.")
    st.stop()

if 1 not in list(model.classes_):
    st.error("The model does not contain the churn class (1).")
    st.stop()


# ==================================================
# 4. HELPER FUNCTIONS
# ==================================================

def classify_risk(probability):

    if probability < 0.30:
        return "Low"

    elif probability < 0.70:
        return "Medium"

    return "High"


def prepare_model_input(data):

    model_input = data.copy()

    # Convert categorical Yes/No values to 0/1.
    for col in categorical_features:

        model_input[col] = model_input[col].map(
            categorical_mapping[col]
        )

    # Ensure exact feature order.
    return model_input[feature_names]


def make_predictions(data):

    model_input = prepare_model_input(data)

    predictions = model.predict(model_input)

    probabilities = model.predict_proba(model_input)

    churn_index = list(model.classes_).index(1)

    churn_probabilities = probabilities[:, churn_index]

    return predictions, churn_probabilities


# ==================================================
# 5. APPLICATION HEADER
# ==================================================

st.title("📊 Telecom Customer Churn Prediction")

st.write(
    "A machine-learning application that predicts "
    "whether telecom customers are likely to leave "
    "a telecommunications company."
)

st.divider()


# ==================================================
# 6. DASHBOARD METRICS
# ==================================================

col1, col2, col3 = st.columns(3)

col1.metric("Total Features", "9")

col2.metric("Selected Algorithm", "Random Forest")

col3.metric("Test Accuracy", "94.75%")

st.divider()


# ==================================================
# 7. SIDEBAR
# ==================================================

with st.sidebar:

    st.title("📊 About the Project")

    st.write(
        "This application uses a trained Random Forest "
        "classification model to predict telecom customer churn."
    )

    st.divider()

    st.subheader("Model Information")

    st.write("Algorithm: Random Forest")
    st.write("Problem: Binary Classification")
    st.write("Class 0: Customer stays")
    st.write("Class 1: Customer churns")

    st.divider()

    st.subheader("Risk Categories")

    st.write("🟢 Low: Below 30%")
    st.write("🟡 Medium: 30% to below 70%")
    st.write("🔴 High: 70% or above")

    st.caption("Risk thresholds are illustrative.")


# ==================================================
# 8. APPLICATION TABS
# ==================================================

individual_tab, batch_tab = st.tabs(
    [
        "👤 Individual Prediction",
        "📁 Batch Prediction"
    ]
)


# ==================================================
# TAB 1: INDIVIDUAL PREDICTION
# ==================================================

with individual_tab:

    st.header("Individual Customer Prediction")

    st.write(
        "Enter the information of one customer "
        "to predict their likelihood of leaving."
    )

    with st.form("customer_form"):

        col1, col2, col3 = st.columns(3)

        with col1:

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

            number_vmail_messages = st.number_input(
                "Voicemail Messages",
                min_value=0,
                value=0
            )

            total_day_charge = st.number_input(
                "Total Day Charge",
                min_value=0.0,
                value=30.0
            )

            total_eve_charge = st.number_input(
                "Total Evening Charge",
                min_value=0.0,
                value=15.0
            )

        with col3:

            total_night_charge = st.number_input(
                "Total Night Charge",
                min_value=0.0,
                value=10.0
            )

            total_intl_charge = st.number_input(
                "Total International Charge",
                min_value=0.0,
                value=2.0
            )

            customer_service_calls = st.number_input(
                "Customer Service Calls",
                min_value=0,
                value=1
            )

        predict_button = st.form_submit_button(
            "Predict Customer Churn",
            use_container_width=True
        )

    if predict_button:

        customer_data = pd.DataFrame([{

            "Account length": account_length,

            "International plan": international_plan,

            "Voice mail plan": voice_mail_plan,

            "Number vmail messages": number_vmail_messages,

            "Total day charge": total_day_charge,

            "Total eve charge": total_eve_charge,

            "Total night charge": total_night_charge,

            "Total intl charge": total_intl_charge,

            "Customer service calls": customer_service_calls

        }])

        prediction, churn_probability = make_predictions(
            customer_data
        )

        prediction = int(prediction[0])

        churn_probability = float(churn_probability[0])

        st.divider()

        st.subheader("Prediction Result")

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

        with result_col2:

            st.metric(
                "Churn Probability",
                f"{churn_probability * 100:.2f}%"
            )

        st.progress(churn_probability)

        risk_level = classify_risk(churn_probability)

        st.subheader("Customer Risk Level")

        if risk_level == "High":

            st.error("🔴 High Risk")

            st.write(
                "Consider reviewing this customer's "
                "service experience and retention needs."
            )

        elif risk_level == "Medium":

            st.warning("🟡 Medium Risk")

            st.write(
                "Consider monitoring this customer's "
                "satisfaction and service usage."
            )

        else:

            st.success("🟢 Low Risk")

            st.write(
                "The model estimates a relatively low "
                "likelihood of churn."
            )


# ==================================================
# TAB 2: BATCH PREDICTION
# ==================================================

with batch_tab:

    st.header("Batch Customer Prediction")

    st.write(
        "Upload a CSV file containing multiple customers. "
        "The trained Random Forest model will predict "
        "churn for every customer."
    )

    st.info(
        "Download the sample template first, fill it with "
        "customer information, and then upload your CSV file."
    )

    # --------------------------------------------------
    # 9. SAMPLE CSV TEMPLATE
    # --------------------------------------------------

    sample_data = pd.DataFrame({

        "Account length": [100, 45, 80, 120, 30],

        "International plan": [
            "No", "Yes", "No", "Yes", "No"
        ],

        "Voice mail plan": [
            "Yes", "No", "Yes", "No", "No"
        ],

        "Number vmail messages": [25, 0, 30, 0, 0],

        "Total day charge": [
            30.0, 45.0, 20.0, 50.0, 15.0
        ],

        "Total eve charge": [
            15.0, 20.0, 18.0, 25.0, 10.0
        ],

        "Total night charge": [
            10.0, 12.0, 9.0, 15.0, 8.0
        ],

        "Total intl charge": [
            2.0, 3.0, 1.5, 4.0, 1.0
        ],

        "Customer service calls": [1, 5, 2, 4, 0]

    })

    template_csv = sample_data.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="⬇️ Download Sample CSV Template",
        data=template_csv,
        file_name="customer_data_template.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------
    # 10. CSV UPLOAD
    # --------------------------------------------------

    st.subheader("Upload Customer Data")

    uploaded_file = st.file_uploader(
        "Choose your customer CSV file",
        type=["csv"],
        key="batch_csv_upload"
    )

    if uploaded_file is not None:

        try:

            uploaded_data = pd.read_csv(uploaded_file)

            st.success("CSV file uploaded successfully!")

            if uploaded_data.empty:

                st.error(
                    "The uploaded CSV file contains no customer records."
                )

            else:

                st.subheader("Uploaded Customer Data")

                st.write(
                    f"Total customer records: {len(uploaded_data)}"
                )

                st.dataframe(
                    uploaded_data.head(10),
                    use_container_width=True
                )

                # --------------------------------------
                # 11. COLUMN VALIDATION
                # --------------------------------------

                missing_columns = [
                    col for col in feature_names
                    if col not in uploaded_data.columns
                ]

                if missing_columns:

                    st.error(
                        "Missing required columns: "
                        + ", ".join(missing_columns)
                    )

                else:

                    processed_data = uploaded_data.copy()

                    validation_errors = []

                    # Validate numeric columns.
                    for col in numeric_features:

                        processed_data[col] = pd.to_numeric(
                            processed_data[col],
                            errors="coerce"
                        )

                        if processed_data[col].isna().any():

                            validation_errors.append(
                                f"{col} contains missing "
                                "or non-numeric values."
                            )

                        elif not np.isfinite(
                            processed_data[col].to_numpy(
                                dtype=float
                            )
                        ).all():

                            validation_errors.append(
                                f"{col} contains infinite values."
                            )

                        elif (processed_data[col] < 0).any():

                            validation_errors.append(
                                f"{col} contains negative values."
                            )

                    # Validate categorical columns.
                    for col in categorical_features:

                        processed_data[col] = (
                            processed_data[col]
                            .astype("string")
                            .str.strip()
                            .str.title()
                        )

                        valid_values = list(
                            categorical_mapping[col].keys()
                        )

                        invalid_values = (
                            processed_data[col].isna()
                            | ~processed_data[col].isin(valid_values)
                        )

                        if invalid_values.any():

                            validation_errors.append(
                                f"{col} must contain only "
                                + " or ".join(valid_values)
                                + "."
                            )

                    # Validate whole-number columns.
                    integer_features = [
                        "Account length",
                        "Number vmail messages",
                        "Customer service calls"
                    ]

                    for col in integer_features:

                        if (
                            processed_data[col].notna().all()
                            and not np.equal(
                                processed_data[col],
                                np.floor(processed_data[col])
                            ).all()
                        ):

                            validation_errors.append(
                                f"{col} must contain whole numbers."
                            )

                    # ----------------------------------
                    # 12. DISPLAY VALIDATION RESULTS
                    # ----------------------------------

                    if validation_errors:

                        st.error(
                            "Please correct the following problems "
                            "in your CSV file:"
                        )

                        for error in validation_errors:

                            st.write(f"• {error}")

                    else:

                        st.success(
                            "All required customer data is valid."
                        )

                        # ----------------------------------
                        # 13. RUN BATCH PREDICTION
                        # ----------------------------------

                        if st.button(
                            "🚀 Predict All Customers",
                            use_container_width=True
                        ):

                            try:

                                predictions, churn_probabilities = (
                                    make_predictions(processed_data)
                                )

                                results = uploaded_data.copy()

                                results["Churn Prediction"] = [
                                    "Churn" if int(p) == 1 else "Stay"
                                    for p in predictions
                                ]

                                results["Churn Probability (%)"] = (
                                    churn_probabilities * 100
                                ).round(2)

                                results["Risk Level"] = [
                                    classify_risk(float(p))
                                    for p in churn_probabilities
                                ]

                                # ----------------------------------
                                # 14. SUMMARY METRICS
                                # ----------------------------------

                                st.divider()

                                st.subheader(
                                    "Batch Prediction Summary"
                                )

                                total_customers = len(results)

                                churn_count = int(
                                    (predictions == 1).sum()
                                )

                                stay_count = (
                                    total_customers - churn_count
                                )

                                churn_rate = (
                                    churn_count / total_customers
                                ) * 100

                                metric1, metric2, metric3, metric4 = (
                                    st.columns(4)
                                )

                                metric1.metric(
                                    "Total Customers",
                                    total_customers
                                )

                                metric2.metric(
                                    "Predicted Churn",
                                    churn_count
                                )

                                metric3.metric(
                                    "Predicted Stay",
                                    stay_count
                                )

                                metric4.metric(
                                    "Churn Rate",
                                    f"{churn_rate:.2f}%"
                                )

                                # ----------------------------------
                                # 15. PREDICTION RESULTS TABLE
                                # ----------------------------------

                                st.divider()

                                st.subheader(
                                    "Customer Prediction Results"
                                )

                                st.dataframe(
                                    results,
                                    use_container_width=True
                                )

                                # ----------------------------------
                                # 16. RISK DISTRIBUTION
                                # ----------------------------------

                                st.subheader(
                                    "Customer Risk Analysis"
                                )

                                risk_counts = (
                                    results["Risk Level"]
                                    .value_counts()
                                    .reindex(
                                        ["Low", "Medium", "High"],
                                        fill_value=0
                                    )
                                )

                                st.bar_chart(risk_counts)

                                for risk, count in risk_counts.items():

                                    percentage = (
                                        count / total_customers
                                    ) * 100

                                    st.write(
                                        f"**{risk} Risk:** "
                                        f"{count} customers "
                                        f"({percentage:.2f}%)"
                                    )

                                # ----------------------------------
                                # 17. DOWNLOAD RESULTS
                                # ----------------------------------

                                st.divider()

                                st.subheader(
                                    "Download Prediction Results"
                                )

                                output_csv = results.to_csv(
                                    index=False
                                ).encode("utf-8-sig")

                                st.download_button(
                                    label=(
                                        "⬇️ Download Prediction "
                                        "Results CSV"
                                    ),
                                    data=output_csv,
                                    file_name=(
                                        "customer_churn_predictions.csv"
                                    ),
                                    mime="text/csv",
                                    use_container_width=True
                                )

                                st.success(
                                    "Batch prediction completed successfully!"
                                )

                            except Exception as e:

                                st.error(
                                    f"Prediction failed: {e}"
                                )

        except Exception as e:

            st.error(
                f"Unable to read the uploaded CSV file: {e}"
            )


# ==================================================
# 18. FEATURE IMPORTANCE
# ==================================================

st.divider()

st.header("📈 Model Feature Importance")

st.write(
    "This chart shows the relative importance of each feature "
    "in the Random Forest model. Feature importance does not "
    "establish that a feature causes customer churn."
)

if hasattr(model, "feature_importances_"):

    feature_importance = pd.DataFrame({

        "Feature": feature_names,

        "Importance": model.feature_importances_

    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    st.bar_chart(
        feature_importance.set_index("Feature")
    )

    with st.expander("View Feature Importance Table"):

        st.dataframe(
            feature_importance,
            use_container_width=True
        )

else:

    st.info(
        "Feature importance is not available for this model."
    )


# ==================================================
# 19. FOOTER
# ==================================================

st.divider()

st.caption(
    "Telecom Customer Churn Prediction | "
    "Powered by Random Forest and Streamlit"
)
