import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# Telecom Customer Churn Prediction | Professional Streamlit App
# Uses the existing trained model bundle and threshold file.
# ============================================================

st.set_page_config(
    page_title="Telecom Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------- Styling ----------------------
st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1500px;
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30,136,229,.09), rgba(42,157,143,.04));
        border: 1px solid rgba(100,116,139,.20);
        padding: 14px 17px;
        border-radius: 14px;
    }
    [data-testid="stMetricLabel"] { font-weight: 650; }
    .hero {
        background: linear-gradient(120deg, #102A43 0%, #176B87 58%, #2A9D8F 100%);
        padding: 1.35rem 1.55rem;
        border-radius: 18px;
        margin-bottom: 1rem;
        color: white;
    }
    .hero h1 { color: white; margin: 0; font-size: 2.15rem; }
    .hero p { color: #EAF6F6; margin: .45rem 0 0; font-size: 1.02rem; }
    div[data-testid="stTabs"] button { font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------- Files ----------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "churn_model.joblib"
THRESHOLD_PATH = APP_DIR / "churn_threshold.pkl"

@st.cache_resource
def load_model_bundle(path):
    return joblib.load(path)

@st.cache_data
def load_threshold(path):
    return float(joblib.load(path))

if not MODEL_PATH.is_file():
    st.error("Model file not found: churn_model.joblib")
    st.info("Place churn_model.joblib in the same repository folder as app.py.")
    st.stop()

if not THRESHOLD_PATH.is_file():
    st.error("Threshold file not found: churn_threshold.pkl")
    st.info("Place churn_threshold.pkl in the same repository folder as app.py.")
    st.stop()

try:
    bundle = load_model_bundle(str(MODEL_PATH))
    model = bundle["model"]
    feature_names = list(bundle["feature_names"])
    categorical_mapping = bundle.get("categorical_mapping", {})
    threshold = load_threshold(str(THRESHOLD_PATH))
except Exception:
    st.error("Unable to load the model bundle or threshold. Verify that the files are valid Joblib files.")
    st.exception(Exception("Model loading failed. Check the Streamlit logs for the original exception."))
    st.stop()

if not hasattr(model, "predict_proba"):
    st.error("The loaded estimator does not support predict_proba.")
    st.stop()

if not 0 <= threshold <= 1:
    st.error(f"Invalid threshold value: {threshold}. The threshold must be between 0 and 1.")
    st.stop()

# Metrics are from the user's reported held-out test evaluation at threshold 0.29.
TEST_METRICS = {
    "Recall": 83.16,
    "Accuracy": 92.35,
    "Precision": 69.30,
    "F1-score": 75.60,
}
CONFUSION_MATRIX = pd.DataFrame(
    [[537, 35], [16, 79]],
    index=["Actual: No Churn", "Actual: Churn"],
    columns=["Predicted: No Churn", "Predicted: Churn"],
)

# ---------------------- Input preparation ----------------------
FORM_FEATURES = [
    "Account length",
    "International plan",
    "Voice mail plan",
    "Number vmail messages",
    "Total day charge",
    "Total eve charge",
    "Total night charge",
    "Total intl charge",
    "Customer service calls",
]

def prepare_model_input(raw_df):
    """Validate, encode categories, and align input columns to saved model order."""
    df = raw_df.copy()

    missing_columns = [col for col in feature_names if col not in df.columns]
    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))

    for column, mapping in categorical_mapping.items():
        if column not in df.columns:
            continue
        if df[column].dtype == object or pd.api.types.is_string_dtype(df[column]):
            normalized = df[column].astype(str).str.strip().str.title()
            df[column] = normalized.map(mapping)
            if df[column].isna().any():
                allowed = ", ".join(mapping.keys())
                raise ValueError(f"'{column}' must contain only: {allowed}.")

    for column in feature_names:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[feature_names].isna().any().any():
        bad = df[feature_names].columns[df[feature_names].isna().any()].tolist()
        raise ValueError("Missing or non-numeric values in: " + ", ".join(bad))

    values = df[feature_names].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Input contains infinite or invalid numeric values.")

    return df[feature_names]

def predict_customers(raw_df):
    """Predict churn using class-1 probability and the saved decision threshold."""
    model_input = prepare_model_input(raw_df)
    probabilities = model.predict_proba(model_input)
    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("The trained model does not contain class 1 (Churn).")

    churn_idx = classes.index(1)
    churn_probs = probabilities[:, churn_idx]
    predictions = (churn_probs >= threshold).astype(int)

    results = raw_df.copy()
    results["Churn Probability (%)"] = np.round(churn_probs * 100, 2)
    results["Predicted Churn"] = np.where(predictions == 1, "Yes", "No")
    results["Risk Level"] = [
        "High" if p >= 0.70 else "Medium" if p >= threshold else "Low"
        for p in churn_probs
    ]
    results["Decision Threshold (%)"] = round(threshold * 100, 2)
    return results, predictions, churn_probs

def build_single_customer(
    account_length, international_plan, voice_mail_plan,
    number_vmail_messages, total_day_charge, total_eve_charge,
    total_night_charge, total_intl_charge, customer_service_calls
):
    return pd.DataFrame([{
        "Account length": account_length,
        "International plan": international_plan,
        "Voice mail plan": voice_mail_plan,
        "Number vmail messages": number_vmail_messages,
        "Total day charge": total_day_charge,
        "Total eve charge": total_eve_charge,
        "Total night charge": total_night_charge,
        "Total intl charge": total_intl_charge,
        "Customer service calls": customer_service_calls,
    }])

# ---------------------- Sidebar ----------------------
with st.sidebar:
    st.markdown("## 📡 Churn Intelligence")
    st.caption("Telecom Customer Churn Prediction")
    st.divider()
    st.markdown("### Project overview")
    st.write(
        "A machine-learning decision-support prototype that estimates which "
        "telecom customers may discontinue their service."
    )
    st.markdown("### Model information")
    st.write("**Algorithm:** Optimized Random Forest")
    st.write("**Task:** Binary classification")
    st.write(f"**Features:** {len(feature_names)}")
    st.write(f"**Threshold:** {threshold:.2f}")
    st.write("**Class 0:** Customer stays")
    st.write("**Class 1:** Customer churns")
    st.divider()
    st.markdown("### Decision rule")
    st.write(f"Probability ≥ **{threshold:.2f}** → flagged as likely churn.")
    st.caption("Predictions are estimates, not guarantees.")
    st.divider()
    st.caption("Reported metrics are based on the held-out test set.")

# ---------------------- Header ----------------------
st.markdown(
    """
    <div class="hero">
        <h1>📊 Telecom Customer Churn Prediction</h1>
        <p>Identify customers at risk of leaving and support data-informed retention decisions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

dashboard_tab, individual_tab, batch_tab, insights_tab = st.tabs(
    ["📈 Dashboard", "👤 Individual Prediction", "📁 Batch Prediction", "🔎 Model Insights"]
)

# ---------------------- Dashboard ----------------------
with dashboard_tab:
    st.subheader("Model Performance")
    st.caption("Held-out test-set evaluation of the optimized Random Forest using the selected threshold.")
    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Churn Recall", f"{TEST_METRICS['Recall']:.2f}%",
        help="Share of actual churners correctly identified. This is the primary project metric."
    )
    metric_cols[1].metric("Test Accuracy", f"{TEST_METRICS['Accuracy']:.2f}%")
    metric_cols[2].metric("Precision", f"{TEST_METRICS['Precision']:.2f}%")
    metric_cols[3].metric("F1-score", f"{TEST_METRICS['F1-score']:.2f}%")

    st.divider()
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Confusion Matrix")
        st.caption("Rows show actual classes; columns show predicted classes.")
        st.dataframe(CONFUSION_MATRIX, use_container_width=True)
        st.markdown(
            "**Test-set counts**\n\n"
            "- True negatives: **537**\n"
            "- False positives: **35**\n"
            "- False negatives: **16**\n"
            "- True positives: **79**"
        )
    with right:
        st.subheader("Threshold interpretation")
        st.metric("Selected threshold", f"{threshold:.2f}")
        st.write(
            "A customer is flagged when the model's estimated churn probability "
            "is at least this threshold. A lower threshold tends to increase recall "
            "but can also increase false-positive alerts."
        )
        st.info("Recall is emphasized because the project aims to identify as many potential churners as possible.")

    st.divider()
    st.subheader("Project summary")
    summary_cols = st.columns(3)
    summary_cols[0].metric("Algorithm", "Random Forest")
    summary_cols[1].metric("Input features", str(len(feature_names)))
    summary_cols[2].metric("Test records", "667")

# ---------------------- Individual prediction ----------------------
with individual_tab:
    st.subheader("Individual Customer Assessment")
    st.write("Enter a customer's service and billing information, then run the prediction.")

    with st.form("individual_prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            account_length = st.number_input("Account length", min_value=0, max_value=5000, value=100, step=1)
            international_plan = st.selectbox("International plan", ["No", "Yes"])
            voice_mail_plan = st.selectbox("Voice mail plan", ["No", "Yes"])
        with c2:
            number_vmail_messages = st.number_input("Number of voicemail messages", min_value=0, max_value=1000, value=0, step=1)
            total_day_charge = st.number_input("Total day charge", min_value=0.0, max_value=10000.0, value=30.0, step=0.5)
            total_eve_charge = st.number_input("Total evening charge", min_value=0.0, max_value=10000.0, value=15.0, step=0.5)
        with c3:
            total_night_charge = st.number_input("Total night charge", min_value=0.0, max_value=10000.0, value=10.0, step=0.5)
            total_intl_charge = st.number_input("Total international charge", min_value=0.0, max_value=10000.0, value=2.0, step=0.1)
            customer_service_calls = st.number_input("Customer service calls", min_value=0, max_value=100, value=1, step=1)

        submitted = st.form_submit_button("🔍 Predict churn", type="primary", use_container_width=True)

    if submitted:
        try:
            customer_df = build_single_customer(
                account_length, international_plan, voice_mail_plan,
                number_vmail_messages, total_day_charge, total_eve_charge,
                total_night_charge, total_intl_charge, customer_service_calls
            )
            result_df, predictions, probs = predict_customers(customer_df)
            prob = float(probs[0])
            risk = result_df.iloc[0]["Risk Level"]

            st.divider()
            st.subheader("Prediction result")
            result_col, probability_col = st.columns([1, 1.35])
            with result_col:
                if predictions[0] == 1:
                    st.warning("⚠️ **Customer flagged as likely to churn**")
                    st.markdown("**Suggested retention action**")
                    st.write("Contact the customer, check for service concerns, and consider an appropriate retention offer.")
                else:
                    st.success("✅ **Customer not flagged as likely to churn**")
                    st.markdown("**Suggested action**")
                    st.write("Continue normal customer care and monitor service satisfaction.")
                st.metric("Risk category", risk)
            with probability_col:
                st.metric("Estimated churn probability", f"{prob * 100:.2f}%")
                st.progress(float(np.clip(prob, 0, 1)))
                st.caption(f"Threshold: {threshold:.2f} ({threshold * 100:.0f}%). This is a model estimate.")
            with st.expander("View input data"):
                st.dataframe(customer_df, use_container_width=True)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

# ---------------------- Batch prediction ----------------------
with batch_tab:
    st.subheader("Batch Customer Prediction")
    st.write("Upload a CSV file with one row per customer and all required model features.")
    st.markdown("**Required feature columns**")
    st.code(", ".join(feature_names), language="text")

    template = pd.DataFrame([{
        "Account length": 100,
        "International plan": "No",
        "Voice mail plan": "Yes",
        "Number vmail messages": 25,
        "Total day charge": 30.0,
        "Total eve charge": 15.0,
        "Total night charge": 10.0,
        "Total intl charge": 2.0,
        "Customer service calls": 1,
    }])
    st.download_button(
        "⬇️ Download sample CSV template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="churn_input_template.csv",
        mime="text/csv",
    )

    upload = st.file_uploader("Upload customer CSV", type=["csv"], key="batch_upload")
    if upload is not None:
        try:
            batch_df = pd.read_csv(upload)
            if batch_df.empty:
                st.warning("The uploaded CSV is empty.")
            else:
                st.write(f"Records uploaded: **{len(batch_df):,}**")
                with st.expander("Preview uploaded records"):
                    st.dataframe(batch_df.head(10), use_container_width=True)

                if st.button("🚀 Run batch prediction", type="primary"):
                    with st.spinner("Predicting churn for uploaded records..."):
                        results, batch_predictions, batch_probs = predict_customers(batch_df)

                    flagged = int(batch_predictions.sum())
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Customers analyzed", f"{len(results):,}")
                    k2.metric("Flagged as likely churners", f"{flagged:,}")
                    k3.metric("Flagged share", f"{flagged / len(results) * 100:.2f}%")

                    st.success("Batch prediction completed.")
                    st.dataframe(results, use_container_width=True)
                    st.download_button(
                        "⬇️ Download prediction results",
                        data=results.to_csv(index=False).encode("utf-8"),
                        file_name="telecom_churn_predictions.csv",
                        mime="text/csv",
                    )
        except Exception as exc:
            st.error(f"Could not process the CSV: {exc}")
            st.info("Check the required column names and ensure numeric fields contain valid values.")

# ---------------------- Model insights ----------------------
with insights_tab:
    st.subheader("Feature Importance")
    st.caption("Random Forest feature importance indicates contribution to tree splits; it does not prove causation.")

    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        st.info("This estimator does not provide built-in feature importance.")
    elif len(importances) != len(feature_names):
        st.warning("The importance array length does not match the saved feature list.")
    else:
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        }).sort_values("Importance", ascending=False)
        st.bar_chart(importance_df.set_index("Feature"), horizontal=True)
        st.dataframe(
            importance_df.assign(Importance=importance_df["Importance"].round(4)),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("Model interpretation")
    st.markdown(
        f"""
        - **Algorithm:** Optimized Random Forest
        - **Target:** `0 = No Churn`, `1 = Churn`
        - **Decision threshold:** `{threshold:.2f}`
        - **Primary metric:** Churn Recall ({TEST_METRICS['Recall']:.2f}% on the reported test set)
        - **Trade-off:** A lower threshold can detect more churners while increasing false-positive alerts.
        """
    )
    st.warning(
        "This app is a decision-support prototype. Predictions are estimates and should be validated "
        "with current, representative data before use in real customer-retention campaigns."
    )

st.divider()
st.caption("Telecom Customer Churn Prediction • Optimized Random Forest • Streamlit")
