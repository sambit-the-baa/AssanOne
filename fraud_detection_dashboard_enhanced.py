import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import json

st.set_page_config(page_title="TPA Fraud Detection System", layout="wide")

# Configure API endpoint
API_URL = "http://localhost:8000/api"

st.title("Insurance Claims Fraud Detection System")
st.markdown("---")

# ============================================================================
# SIDEBAR - Configuration
# ============================================================================

with st.sidebar:
    st.header(" Configuration")
    
    execution_mode = st.selectbox(
        "Execution Mode",
        ["parallel", "sequential", "mixed"],
        help="Parallel: Fast (all agents). Sequential: Slow (one-by-one). Mixed: Balanced"
    )
    
    st.markdown("---")
    st.write("**Detection Agents**")
    st.write(" Overbilling Protection")
    st.write(" Fraud Diagnostic Analysis")
    st.write(" Unbundling/Upcoding Detection")
    st.write(" Identity Theft Protection")

# ============================================================================
# MAIN CONTENT - Tabs
# ============================================================================

tab1, tab2, tab3 = st.tabs(["Dashboard", "Single Claim Analysis", "Batch Processing"])

# ============================================================================
# TAB 1: REAL-TIME DASHBOARD
# ============================================================================

with tab1:
    st.header(" Real-Time Fraud Detection Dashboard")
    
    # Fetch metrics from API
    try:
        response = requests.get(f"{API_URL}/dashboard/metrics", timeout=10)
        metrics = response.json()
    except Exception as e:
        st.error(f"Failed to fetch dashboard metrics: {e}")
        metrics = None
    
    if metrics:
        col1, col2, col3 = st.columns(3)
        col1.metric("Fraud Cases Detected", metrics.get("fraud_cases_detected", 0))
        col2.metric("Amount Protected", f"${metrics.get('amount_protected', 0):,.0f}")
        col3.metric("Fraud Detection Rate", f"{metrics.get('fraud_detection_rate', 0):.2f}%")
        st.markdown("---")
        st.subheader("Risk Distribution")
        risk_dist = metrics.get("risk_distribution", {})
        st.bar_chart(pd.DataFrame(list(risk_dist.items()), columns=["Risk Level", "Count"]).set_index("Risk Level"))
        st.subheader("Recent Claims")
        recent_claims = metrics.get("recent_claims", [])
        if recent_claims:
            st.dataframe(pd.DataFrame(recent_claims))
        st.subheader("Fraud Trend (Last 30 Days)")
        fraud_trend = metrics.get("fraud_trend", [])
        if fraud_trend:
            df_trend = pd.DataFrame(fraud_trend)
            fig = px.line(df_trend, x="date", y=["total_claims", "fraud_cases"], markers=True)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 2: SINGLE CLAIM ANALYSIS
# ============================================================================

with tab2:
    st.header("Analyze Individual Claim")
    uploaded_file = st.file_uploader("Upload Claim PDF", type=["pdf"])
    user_id = st.text_input("User ID (optional)")
    if st.button("Analyze Claim") and uploaded_file:
        with st.spinner("Uploading and analyzing claim..."):
            files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            data = {"execution_mode": execution_mode, "user_id": user_id}
            try:
                response = requests.post(f"{API_URL}/claims/analyze", files=files, data=data, timeout=60)
                result = response.json()
                if result.get("status") == "success":
                    st.success(f"Claim analyzed! Claim ID: {result['claim_id']}")
                    st.json(result)
                else:
                    st.error(f"Analysis failed: {result}")
            except Exception as e:
                st.error(f"Error: {e}")

# ============================================================================
# TAB 3: BATCH PROCESSING (placeholder)
# ============================================================================

with tab3:
    st.header("Batch Processing (Coming Soon)")
    st.info("Batch processing via API will be available in a future update.")
