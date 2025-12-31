"""
TPA Fraud Detection Dashboard
Streamlit app for claims processing and fraud detection.
"""
import streamlit as st
import json
from pathlib import Path
from datetime import datetime

from agent.pipeline import process_claim_full_pipeline
from agent.database import ClaimsDatabase
from agent.orchestrator import ExecutionMode

st.set_page_config(page_title="TPA Fraud Detection System", layout="wide")

st.title("🛡️ Insurance Claims Fraud Detection System")
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    execution_mode = st.selectbox(
        "Execution Mode",
        ["parallel", "sequential", "mixed"],
        help="Parallel: Fast, all agents run together. Sequential: Slower, agents run one-by-one. Mixed: Identity check first, then parallel."
    )
    st.markdown("---")
    st.write("**Detection Agents:**")
    st.write("✓ Overbilling Protection")
    st.write("✓ Fraud Diagnostic Analysis")
    st.write("✓ Unbundling/Upcoding Detection")
    st.write("✓ Identity Theft Protection")

# Initialize database
db = ClaimsDatabase()

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    execution_mode = st.selectbox(
        "Execution Mode",
        ["parallel", "sequential", "mixed"],
        help="Parallel: Fast, all agents run together. Sequential: Slower, agents run one-by-one. Mixed: Identity check first, then parallel."
    )
    st.markdown("---")
    st.subheader("Detection Agents")
    st.write("✅ Overbilling Protection")
    st.write("✅ Fraud Diagnostic Analysis")
    st.write("✅ Unbundling/Upcoding Detection")
    st.write("✅ Identity Theft Protection")
    st.markdown("---")
    st.subheader("Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("System Status", "🟢 Active")
    with col2:
        st.metric("Agents Ready", "4/4")

# Main tabs
tab1, tab2, tab3 = st.tabs(["📊 Single Claim Analysis", "📈 Batch Processing", "📋 Dashboard"])

# ============================================================================
# TAB 1: SINGLE CLAIM ANALYSIS
# ============================================================================
with tab1:
    st.header("Analyze Individual Claim")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Upload or Select PDF")
        
        # File upload
        uploaded_file = st.file_uploader("Upload Claim PDF", type=["pdf"])
        
        # Or select from Data folder
        data_dir = Path("Data")
        if data_dir.exists():
            st.write("Or select from Data folder:")
            pdf_files = sorted(data_dir.glob("*.pdf"))
            selected_file = st.selectbox("Available PDFs", [f.name for f in pdf_files], key="single_select")
            if selected_file:
                uploaded_file = data_dir / selected_file

    with col2:
        if uploaded_file:
            if isinstance(uploaded_file, str):
                pdf_path = uploaded_file
            else:
                # Save uploaded file temporarily
                import tempfile
                tmp_dir = Path("Data/uploads")
                tmp_dir.mkdir(parents=True, exist_ok=True)
                pdf_path = tmp_dir / uploaded_file.name
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                pdf_path = str(pdf_path)

            # Ensure analysis_type is always defined before use
            analysis_type = st.selectbox(
                "Analysis Type",
                [
                    "Full Analysis",      # All 4 agents
                    "Quick Fraud Check",  # Only identity + overbilling
                    "OCR Only",           # Just extract text/fields
                    "Pattern Detection",  # Focus on fraud patterns
                ],
                help="Choose which agents to run for this claim"
            )
            priority_level = st.selectbox(
                "Priority Level",
                ["Normal", "High", "Critical"],
                help="Affects review queue priority"
            )
            st.markdown("---")
            if st.button("🔍 Analyze Claim", key="analyze_single"):
                with st.spinner("Processing claim..."):
                    try:
                        # Map analysis_type to pipeline mode
                        if analysis_type == "OCR Only":
                            mode = "ocr_only"
                        else:
                            mode = "full_analysis"
                        result = process_claim_full_pipeline(
                            pdf_path,
                            execution_mode=execution_mode,
                            save_results=True,
                            mode=mode
                        )
                        if "error" in result:
                            st.error(f"❌ Error: {result['error']}")
                        else:
                            st.success("✅ Analysis Complete")
                            if mode == "ocr_only":
                                st.subheader("📝 OCR & Field Extraction Results")
                                st.write("**Extracted Text:**")
                                st.text_area("OCR Text", result.get("ocr_text", ""), height=200)
                                st.write("**Extracted Fields:**")
                                claim_data = result.get("claim_data", {})
                                st.json(claim_data, expanded=False)
                                st.markdown("---")
                                st.subheader("💾 Download Results")
                                report_json = json.dumps(result, indent=2, default=str)
                                st.download_button(
                                    "Download OCR Results (JSON)",
                                    data=report_json,
                                    file_name=f"{Path(pdf_path).stem}_ocr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
                            else:
                                # Display summary for full analysis
                                col_score, col_risk, col_action = st.columns(3)
                                with col_score:
                                    score = result.get("fraud_risk_score", 0)
                                    st.metric("Fraud Risk Score", f"{score}/100")
                                with col_risk:
                                    risk = result.get("overall_risk_level", "N/A")
                                    color = "🔴" if risk == "CRITICAL" else "🟠" if risk == "HIGH" else "🟡" if risk == "MEDIUM" else "🟢"
                                    st.metric("Risk Level", f"{color} {risk}")
                                with col_action:
                                    action = result.get("recommended_action", "N/A")
                                    st.metric("Action", action[:20] + "..." if len(action) > 20 else action)
                                st.markdown("---")
                                st.subheader("🤖 Agent Findings")
                                agent_results = result.get("agent_results", {})
                                for agent_name, agent_result in agent_results.items():
                                    if agent_result:
                                        with st.expander(f"**{agent_name.title()}** - {agent_result.get('risk_level', 'N/A')}"):
                                            col_left, col_right = st.columns([1, 1])
                                            with col_left:
                                                st.write(f"**Confidence:** {agent_result.get('confidence', 0):.0%}")
                                                st.write(f"**Risk Level:** {agent_result.get('risk_level', 'N/A')}")
                                            with col_right:
                                                findings = agent_result.get("findings", [])
                                                if findings:
                                                    st.write("**Findings:**")
                                                    for finding in findings:
                                                        st.write(f"• {finding}")
                                                else:
                                                    st.write("No findings")
                                            st.json(agent_result.get("details", {}), expanded=False)
                                st.markdown("---")
                                st.subheader("📋 All Findings Summary")
                                all_findings = result.get("all_findings", [])
                                if all_findings:
                                    for i, finding in enumerate(all_findings, 1):
                                        st.write(f"{i}. {finding}")
                                else:
                                    st.write("✓ No suspicious findings")
                                st.markdown("---")
                                st.subheader("💾 Download Results")
                                report_json = json.dumps(result, indent=2, default=str)
                                st.download_button(
                                    "Download Full Report (JSON)",
                                    data=report_json,
                                    file_name=f"{Path(pdf_path).stem}_fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {e}")

        with col2:
            st.subheader("⚙️ Analysis Options")
            analysis_type = st.selectbox(
                "Analysis Type",
                [
                    "Full Analysis",      # All 4 agents
                    "Quick Fraud Check",  # Only identity + overbilling
                    "OCR Only",           # Just extract text/fields
                    "Pattern Detection",  # Focus on fraud patterns
                ],
                help="Choose which agents to run for this claim"
            )
            priority_level = st.selectbox(
                "Priority Level",
                ["Normal", "High", "Critical"],
                help="Affects review queue priority"
            )
            st.markdown("---")


# ============================================================================
# TAB 2: BATCH PROCESSING
# ============================================================================
with tab2:
    st.header("Batch Process Multiple Claims")

    col1, col2 = st.columns([1, 2])

    with col1:
        data_dir = Path("Data")
        if data_dir.exists():
            pdf_files = list(data_dir.glob("*.pdf"))
            st.write(f"Found {len(pdf_files)} PDF(s) in Data/ folder:")
            for f in pdf_files:
                st.write(f"• {f.name}")
        else:
            st.write("No Data/ folder found")

    with col2:
        if st.button("🔄 Process All Claims", key="batch_process"):
            data_dir = Path("Data")
            pdf_files = list(data_dir.glob("*.pdf"))

            if not pdf_files:
                st.warning("No PDFs found in Data folder")
            else:
                progress_bar = st.progress(0)
                results_list = []

                for idx, pdf_file in enumerate(pdf_files):
                    with st.spinner(f"Processing {pdf_file.name} ({idx + 1}/{len(pdf_files)})..."):
                        try:
                            result = process_claim_full_pipeline(
                                str(pdf_file),
                                execution_mode=execution_mode,
                                save_results=True
                            )
                            results_list.append({
                                "file": pdf_file.name,
                                "fraud_score": result.get("fraud_risk_score", 0),
                                "risk_level": result.get("overall_risk_level", "N/A"),
                                "action": result.get("recommended_action", "N/A")
                            })
                        except Exception as e:
                            results_list.append({
                                "file": pdf_file.name,
                                "error": str(e)
                            })
                    
                    progress_bar.progress((idx + 1) / len(pdf_files))

                st.success("✅ Batch processing complete!")

                # Results table
                st.subheader("Batch Results Summary")
                results_df = pd.DataFrame(results_list) if results_list else None
                if results_df is not None:
                    st.dataframe(results_df, use_container_width=True)

                # Export batch results
                batch_json = json.dumps(results_list, indent=2)
                st.download_button(
                    "Download Batch Results (JSON)",
                    data=batch_json,
                    file_name=f"batch_fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )


# ============================================================================
# TAB 3: SYSTEM DASHBOARD
# ============================================================================
with tab3:
    st.header("Fraud Detection System Dashboard")

    # System info
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Execution Mode", execution_mode.upper())

    with col2:
        outputs_dir = Path("Data/outputs")
        report_count = len(list(outputs_dir.glob("*_fraud_report.json"))) if outputs_dir.exists() else 0
        st.metric("Reports Generated", report_count)

    with col3:
        st.metric("Detection Agents", "4")

    with col4:
        st.metric("Status", "🟢 Active")

    st.markdown("---")

    # Agent descriptions
    st.subheader("🤖 Fraud Detection Agents")

    col1, col2 = st.columns(2)

    with col1:
        with st.container():
            st.write("### 💰 Overbilling Protection")
            st.write("""
            - Compares billed amounts with local market prices
            - Detects unnecessary medical tests
            - Identifies excessive line items
            """)

        with st.container():
            st.write("### 🔍 Fraud Diagnostic Analysis")
            st.write("""
            - Cross-document diagnosis consistency checks
            - Detects diagnostic-procedure mismatches
            - Identifies diagnostic overkill patterns
            """)

    with col2:
        with st.container():
            st.write("### 📦 Unbundling/Upcoding Detection")
            st.write("""
            - Identifies bundled services billed separately
            - Detects unusual pricing patterns
            - Estimates overbilling amounts
            """)

        with st.container():
            st.write("### 🪪 Identity Theft Protection")
            st.write("""
            - Government ID verification
            - Deepfake detection (mock)
            - Duplicate claim detection
            """)

    st.markdown("---")

    # Recent reports
    st.subheader("📊 Recent Fraud Reports")
    outputs_dir = Path("Data/outputs")
    if outputs_dir.exists():
        report_files = sorted(outputs_dir.glob("*_fraud_report.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        if report_files:
            for report_file in report_files[:5]:  # Show last 5
                with open(report_file) as f:
                    report = json.load(f)
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**{report_file.stem.replace('_fraud_report', '')}**")
                    with col2:
                        score = report.get("fraud_risk_score", 0)
                        st.write(f"Score: {score}")
                    with col3:
                        risk = report.get("overall_risk_level", "N/A")
                        st.write(f"{risk}")
        else:
            st.info("No reports generated yet. Process a claim to generate reports.")
    else:
        st.info("No Data/outputs folder found.")

    st.markdown("---")

    # Instructions
    st.subheader("📖 How to Use")
    st.write("""
    1. **Single Claim**: Upload a PDF in Tab 1 and click "Analyze Claim"
    2. **Batch Processing**: Place PDFs in the Data/ folder and use Tab 2
    3. **Results**: JSON outputs saved to Data/outputs/ automatically
    4. **Execution Modes**:
       - **Parallel**: All agents run simultaneously (fastest)
       - **Sequential**: Agents run one-by-one (slowest, best for debugging)
       - **Mixed**: Identity check first, then parallel (balanced)
    """)

import pandas as pd
