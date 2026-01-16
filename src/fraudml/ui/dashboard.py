import streamlit as st
import requests
import plotly.graph_objects as go
import time

import os

# --- Configuration ---
API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/predict")
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #1f2937;
    }
    .stButton>button {
        width: 100%;
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        border-color: #2563eb;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("🛡️ Fraud Guard")
st.sidebar.markdown("---")
st.sidebar.subheader("Transaction Details")

amount = st.sidebar.slider("Transaction Amount ($)", 0.0, 10000.0, 500.0, step=10.0)
hour = st.sidebar.slider("Hour of Day", 0, 23, 14)
device_score = st.sidebar.slider("Device Reliability (0-1)", 0.0, 1.0, 0.9, step=0.01)
country_risk = st.sidebar.selectbox("Country Risk Level", [1, 2, 3, 4, 5], index=0)

st.sidebar.markdown("---")
st.sidebar.caption(f"Model Endpoint: `{API_URL}`")


# --- Main Content ---
st.title("Fraud Detection Analysis")
st.markdown("Real-time scoring using our **Balanced Logistic Regression** model with **Precision-Focused** dynamic thresholding.")

col1, col2 = st.columns([1, 1])

# Run Prediction
fraud_prob = 0.0
fraud_label = 0
threshold = 0.5
model_version = ""

# Auto-run prediction on change? Or button? Let's do button for "Action" feel, or auto. 
# Let's do auto for smooth feel, but maybe a button is clearer. Let's do auto-update.

payload = {
    "amount": amount,
    "hour": hour,
    "device_score": device_score,
    "country_risk": country_risk
}

try:
    with st.spinner("Analyzing transaction..."):
        # Small delay to simulate network/processing for UX
        # time.sleep(0.3) 
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            fraud_prob = data["fraud_probability"]
            fraud_label = data["fraud_label"]
            threshold = data.get("threshold", 0.5)
            model_version = data.get("model_version", "N/A")
        else:
            st.error(f"API Error: {response.status_code}")
except requests.exceptions.ConnectionError:
    st.error("⚠️ Could not connect to API. Is the backend running?")
    st.code("uvicorn src.fraudml.api.app:app --reload")

# --- Visuals ---

with col1:
    st.subheader("Risk Analysis")
    
    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = fraud_prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fraud Probability (%)"},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "rgba(0,0,0,0)"}, # Hide default bar, use threshold
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold*100], 'color': "#10b981"}, # Green
                {'range': [threshold*100, 100], 'color': "#ef4444"} # Red
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': fraud_prob * 100
            }
        }
    ))
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Decision & Explanation")
    
    # Status Card
    if fraud_label == 1:
        st.error(f"🚨 **FRAUD SUSPECTED**")
        st.markdown(f"Probability **{fraud_prob:.2%}** exceeds dynamic threshold **{threshold:.2%}**.")
        
        # Explain why
        reasons = []
        if amount > 2000: reasons.append("Large Transaction Amount")
        if device_score < 0.3: reasons.append("Low Device Trust Score")
        if country_risk >= 4: reasons.append("High Risk Country")
        if hour < 6 or hour > 22: reasons.append("Unusual Transaction Hour")
        
        if reasons:
            st.markdown("**Primary Risk Factors:**")
            for r in reasons:
                st.markdown(f"- {r}")
        else:
            st.markdown("*Complex pattern detected by model logic.*")
            
    else:
        st.success(f"✅ **LEGITIMATE**")
        st.markdown(f"Probability **{fraud_prob:.2%}** is below threshold **{threshold:.2%}**.")
        st.markdown("*Transaction appears normal based on current risk model.*")

    st.markdown("---")
    st.caption(f"Model Version: {model_version}")

# --- Footer ---
st.markdown("---")
cols = st.columns(4)
cols[0].metric("Target Precision", "95%")
cols[1].metric("Current Threshold", f"{threshold:.4f}")
cols[2].metric("Model Type", "Logistic Regression")
cols[3].metric("Training Date", model_version.split("T")[0] if "T" in model_version else "N/A")
