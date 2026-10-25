import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)

st.set_page_config(page_title="FraudGuard | Transaction Intelligence", page_icon="🛡️", layout="wide")

MODEL_PATH = Path(__file__).parent / "fraud_model.joblib"
METRICS_PATH = Path(__file__).parent / "metrics.json"
DATA_PATH = Path(__file__).parent / "credit_card_fraud_balanced.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def make_features(frame):
    x = frame.copy()
    dt = pd.to_datetime(x["TransactionDate"], errors="coerce")
    x["TransactionHour"] = dt.dt.hour
    x["TransactionDayOfWeek"] = dt.dt.dayofweek
    x["TransactionMonth"] = dt.dt.month
    x["TransactionDay"] = dt.dt.day
    return x.drop(columns=["TransactionID", "TransactionDate"], errors="ignore")

def normalize_input(frame):
    required = ["TransactionID", "TransactionDate", "Amount", "MerchantID",
                "TransactionType", "Location"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    return frame[required].copy()

model = load_model()

# CSS for a clean dashboard
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#101827,#263b5b); color: white; margin-bottom: 1rem;}
.hero h1 {margin: 0; font-size: 2.2rem;}
.hero p {color: #d8e4f4; margin: .45rem 0 0 0;}
div[data-testid="stMetric"] {background: rgba(120,140,160,.09); padding: 14px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🛡️ FraudGuard</h1><p>Interactive credit-card transaction risk screening • ML-powered demo</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.title("Control Center")
    page = st.radio("Navigate", ["Overview", "Check a transaction", "Batch screening", "Model evaluation"])
    st.divider()
    threshold = st.slider("Fraud alert threshold", 0.05, 0.95, 0.50, 0.05,
                          help="Lower threshold flags more transactions but can increase false alarms.")
    st.caption("A transaction is flagged when its estimated fraud probability meets or exceeds this threshold.")

st.warning("Prototype warning: this dataset contains only 2,000 balanced rows. Its held-out test performance is close to chance, so this app is for learning/demo purposes—not real payment decisions.")

if page == "Overview":
    st.subheader("Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset rows", "2,000")
    c2.metric("Fraud / legitimate", "50% / 50%")
    c3.metric("Model", "Random Forest")
    st.markdown("### What you can do")
    left, right = st.columns(2)
    with left:
        st.info("**Single transaction**\n\nEnter amount, merchant, date/time, transaction type and location to estimate risk.")
        st.info("**Batch screening**\n\nUpload a CSV with the same transaction columns to score many transactions at once.")
    with right:
        st.info("**Adjust sensitivity**\n\nChange the alert threshold in the sidebar and observe how flags change.")
        st.info("**Evaluate the model**\n\nReview precision, recall, F1, ROC-AUC and confusion matrix on the saved holdout metrics.")
    if DATA_PATH.exists():
        data = pd.read_csv(DATA_PATH)
        st.markdown("### Dataset snapshot")
        a, b = st.columns(2)
        with a:
            fig = px.histogram(data, x="Amount", color="IsFraud", nbins=30,
                               barmode="overlay", opacity=0.75,
                               title="Transaction amount distribution",
                               labels={"IsFraud":"Fraud label"})
            st.plotly_chart(fig, use_container_width=True)
        with b:
            counts = data["TransactionType"].value_counts().rename_axis("Transaction type").reset_index(name="Count")
            fig = px.bar(counts, x="Transaction type", y="Count", title="Transaction types")
            st.plotly_chart(fig, use_container_width=True)

elif page == "Check a transaction":
    st.subheader("Check a transaction")
    st.write("Fill in the transaction details, then select **Estimate risk**.")
    with st.form("single_transaction"):
        col1, col2, col3 = st.columns(3)
        with col1:
            amount = st.number_input("Amount", min_value=0.0, value=1250.0, step=50.0)
            merchant = st.number_input("Merchant ID", min_value=0, value=101, step=1)
        with col2:
            date_time = st.datetime_input("Transaction date & time", value=pd.Timestamp("2024-10-25 14:30").to_pydatetime())
            transaction_type = st.selectbox("Transaction type", ["purchase", "refund"])
        with col3:
            location = st.selectbox("Location", ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
                                                 "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose", "Other"])
            transaction_id = st.number_input("Transaction ID (not used by model)", min_value=0, value=999999, step=1)
        submitted = st.form_submit_button("🔎 Estimate risk", use_container_width=True)
    if submitted:
        row = pd.DataFrame([{
            "TransactionID": int(transaction_id),
            "TransactionDate": pd.Timestamp(date_time).isoformat(),
            "Amount": float(amount),
            "MerchantID": int(merchant),
            "TransactionType": transaction_type,
            "Location": location
        }])
        probability = float(model.predict_proba(make_features(row))[0, 1])
        flagged = probability >= threshold
        c1, c2 = st.columns([1,2])
        c1.metric("Estimated fraud probability", f"{probability:.1%}")
        if flagged:
            c2.error("🚩 FLAGGED FOR REVIEW — probability meets the selected threshold.")
        else:
            c2.success("✅ Not flagged at the current threshold.")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption("This is a model score, not proof that a transaction is fraudulent. Review context before taking action.")

elif page == "Batch screening":
    st.subheader("Batch screening")
    st.write("Upload a CSV containing `TransactionID`, `TransactionDate`, `Amount`, `MerchantID`, `TransactionType`, and `Location`. An optional `IsFraud` column lets you compare predictions with labels.")
    uploaded = st.file_uploader("Upload transaction CSV", type=["csv"])
    if uploaded:
        try:
            batch = pd.read_csv(uploaded)
            st.write("Preview")
            st.dataframe(batch.head(10), use_container_width=True)
            clean = normalize_input(batch)
            scores = model.predict_proba(make_features(clean))[:, 1]
            result = batch.copy()
            result["FraudProbability"] = scores
            result["FlaggedForReview"] = scores >= threshold
            result["RiskLevel"] = pd.cut(scores, bins=[-0.001, 0.3, 0.7, 1.001],
                                         labels=["Lower", "Medium", "Higher"])
            a, b, c = st.columns(3)
            a.metric("Transactions scored", f"{len(result):,}")
            b.metric("Flagged for review", f"{int(result['FlaggedForReview'].sum()):,}")
            c.metric("Flag rate", f"{result['FlaggedForReview'].mean():.1%}" if len(result) else "0%")
            st.dataframe(result.sort_values("FraudProbability", ascending=False), use_container_width=True)
            st.download_button("⬇️ Download scored CSV", result.to_csv(index=False).encode("utf-8"),
                               file_name="fraud_screening_results.csv", mime="text/csv")
            if "IsFraud" in result.columns and result["IsFraud"].nunique() == 2:
                y_true = result["IsFraud"].astype(int)
                y_hat = (scores >= threshold).astype(int)
                p, r, f = st.columns(3)
                p.metric("Precision", f"{precision_score(y_true, y_hat, zero_division=0):.3f}")
                r.metric("Recall", f"{recall_score(y_true, y_hat, zero_division=0):.3f}")
                f.metric("F1-score", f"{f1_score(y_true, y_hat, zero_division=0):.3f}")
        except Exception as exc:
            st.error(f"Could not score this file: {exc}")

elif page == "Model evaluation":
    st.subheader("Model evaluation")
    if METRICS_PATH.exists():
        saved = json.loads(METRICS_PATH.read_text())
        cols = st.columns(5)
        for col, key, label in zip(cols, ["accuracy","precision","recall","f1","roc_auc"],
                                   ["Accuracy","Precision","Recall","F1-score","ROC-AUC"]):
            col.metric(label, f"{saved[key]:.3f}")
        cm = np.array(saved["confusion_matrix"])
        fig = px.imshow(cm, text_auto=True, x=["Predicted legitimate","Predicted fraud"],
                        y=["Actual legitimate","Actual fraud"], aspect="auto",
                        title="Confusion matrix at 0.50 threshold",
                        labels={"x":"Prediction","y":"Actual"})
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Average precision", f"{saved['average_precision']:.3f}")
        st.caption("Metrics come from a stratified 20% holdout split. Results may change with different data or splits.")
    else:
        st.info("metrics.json not found. Run train_model.py or the Colab notebook and add the generated metrics.json to this project.")

st.divider()
st.caption("FraudGuard • Educational prototype. Validate on representative, time-separated, real-world data before any production use.")
