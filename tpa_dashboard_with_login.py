"""
Assan One Intelligence - TPA Management Platform
Professional Third Party Administrator Dashboard with Secure Authentication
"""
import streamlit as st
import json
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import hmac
from typing import Dict, Optional
import pandas as pd
import io
import base64

# PDF generation (optional dependency)
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# ============================================================================
# AUTHENTICATION SYSTEM
# ============================================================================

class SimpleAuth:
    """Secure authentication system for TPA users"""
    
    # Credentials (in production, use a real database)
    CREDENTIALS = {
        "admin": "admin123",
        "tpa_manager": "manager123",
        "claims_reviewer": "reviewer123",
        "auditor": "auditor123"
    }
    
    ROLES = {
        "admin": ["dashboard", "claims", "reports", "analytics", "users"],
        "tpa_manager": ["dashboard", "claims", "reports", "analytics"],
        "claims_reviewer": ["dashboard", "claims"],
        "auditor": ["reports", "analytics"]
    }
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password for storage"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def verify_password(stored_hash: str, provided_password: str) -> bool:
        """Verify provided password against stored hash"""
        return stored_hash == SimpleAuth.hash_password(provided_password)
    
    @staticmethod
    def authenticate(username: str, password: str) -> Optional[Dict]:
        """Authenticate user credentials"""
        if username in SimpleAuth.CREDENTIALS:
            if SimpleAuth.CREDENTIALS[username] == password:
                return {
                    "username": username,
                    "role": username if username != "admin" else "admin",
                    "authenticated": True,
                    "login_time": datetime.now().isoformat(),
                    "permissions": SimpleAuth.ROLES.get(username, [])
                }
        return None


# 2FA (TOTP) support (demo only)
try:
    import pyotp
    PYOTP_AVAILABLE = True
except ImportError:
    pyotp = None
    PYOTP_AVAILABLE = False

# Use a fixed TOTP secret for all users (demo/testing only)
FIXED_TOTP_SECRET = "JBSWY3DPEHPK3PXP"  # Base32, works with Google Authenticator
def verify_2fa(username, user_code):
    if not PYOTP_AVAILABLE:
        return True  # Skip 2FA if pyotp not available
    totp = pyotp.TOTP(FIXED_TOTP_SECRET)
    return totp.verify(user_code)


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Assan One Intelligence - TPA Management",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# THEME MANAGEMENT
# ============================================================================

if "theme" not in st.session_state:
    st.session_state.theme = "light"

# Professional Light Theme CSS (Default)
LIGHT_THEME_CSS = """
<style>
    :root {
        --primary-color: #003d82;
        --primary-light: #0052b3;
        --secondary-color: #1976d2;
        --accent-color: #ff6f00;
        --success-color: #2e7d32;
        --warning-color: #f57f17;
        --danger-color: #d32f2f;
        --light-bg: #f8f9fa;
        --card-bg: #ffffff;
        --border-color: #e0e0e0;
        --text-primary: #212121;
        --text-secondary: #666666;
    }

    html, body {
        background-color: #f8f9fa !important;
        font-size: 16px;
        color: #212121 !important;
    }

    .main, [data-testid="stAppViewContainer"], .block-container, .stAppViewBlockContainer {
        background-color: #f8f9fa !important;
        color: #212121 !important;
    }

    * {
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
        box-sizing: border-box;
    }

    /* Ensure all text on light backgrounds is dark */
    p, span, div, label, li, td, th {
        color: #212121;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #003d82 !important;
        font-weight: 600;
        word-break: break-word;
    }

    /* Metric cards and white background containers */
    .stMetric, .stMetricLabel, .stMetricValue, .stMetricDelta {
        color: #212121 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #003d82 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #666666 !important;
    }

    .stButton > button {
        border-radius: 6px;
        border: none;
        font-weight: 500;
        padding: 8px 16px;
        transition: all 0.3s ease;
        background-color: #003d82 !important;
        color: white !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 61, 130, 0.15) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 2px solid #e0e0e0 !important;
        flex-wrap: wrap;
    }

    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid #003d82 !important;
        color: #003d82 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #212121 !important;
    }

    .stDataFrame, .stTable {
        color: #212121 !important;
        background-color: #ffffff !important;
    }
    
    .stDataFrame th, .stDataFrame td, .stTable th, .stTable td {
        color: #212121 !important;
    }

    .stMarkdown, .stText {
        color: #212121 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        color: #212121 !important;
    }
    
    /* Text input and select boxes */
    .stTextInput input, .stSelectbox select, .stTextArea textarea {
        color: #212121 !important;
        background-color: #ffffff !important;
    }
    
    .stTextInput label, .stSelectbox label, .stTextArea label {
        color: #212121 !important;
    }
    
    /* Radio and checkbox labels */
    .stRadio label, .stCheckbox label {
        color: #212121 !important;
    }
    
    /* Number input */
    .stNumberInput input {
        color: #212121 !important;
    }
    
    .stNumberInput label {
        color: #212121 !important;
    }
    
    /* Info, warning, success, error boxes */
    .stAlert {
        color: #212121 !important;
    }
    
    /* JSON display */
    .stJson {
        color: #212121 !important;
    }

    /* Responsive tweaks */
    @media (max-width: 900px) {
        html, body {
            font-size: 15px;
        }
        .stTabs [data-baseweb="tab-list"] {
            font-size: 0.95em;
        }
        .stButton > button {
            padding: 8px 10px;
            font-size: 0.98em;
        }
        .stDataFrame, .stTable, .stMarkdown, .stText {
            font-size: 0.97em;
        }
    }
    @media (max-width: 600px) {
        html, body {
            font-size: 14px;
        }
        .stTabs [data-baseweb="tab-list"] {
            font-size: 0.9em;
        }
        .stButton > button {
            padding: 7px 6px;
            font-size: 0.95em;
        }
        .stDataFrame, .stTable, .stMarkdown, .stText {
            font-size: 0.95em;
        }
        .stAppViewBlockContainer, .block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        .stSidebar {
            min-width: 120px !important;
            width: 120px !important;
        }
    }
    /* Make tables and charts scrollable on mobile */
    .stDataFrame, .stTable {
        overflow-x: auto !important;
        max-width: 100vw !important;
        display: block !important;
    }
    .element-container {
        max-width: 100vw !important;
    }
    
    /* CRITICAL: Force dark text on white/light backgrounds */
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] span,
    [data-testid="stAppViewContainer"] div,
    [data-testid="stAppViewContainer"] label {
        color: #212121 !important;
    }
    
    /* Headers on blue/dark backgrounds should be white (handled inline) */
    /* But ensure body text is dark */
    .stMarkdown p, .stMarkdown span, .stMarkdown li {
        color: #212121 !important;
    }
    
    /* Streamlit's default markdown elements */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMarkdownContainer"] em {
        color: #212121 !important;
    }
    
    /* File uploader LABEL - dark text on light background */
    [data-testid="stFileUploader"] > label {
        color: #212121 !important;
    }
    
    /* File uploader DROPZONE - white text on dark background */
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploaderDropzone"] {
        background-color: #1e3a5f !important;
        border: 2px dashed #4a9eff !important;
        border-radius: 8px !important;
    }
    
    [data-testid="stFileUploader"] section div,
    [data-testid="stFileUploader"] section span,
    [data-testid="stFileUploader"] section p,
    [data-testid="stFileUploader"] section small,
    [data-testid="stFileUploaderDropzone"] div,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] p {
        color: white !important;
    }
    
    [data-testid="stFileUploader"] button {
        color: white !important;
        background-color: #003d82 !important;
        border: 1px solid #4a9eff !important;
    }
    
    /* Selectbox label - dark text on light background */
    [data-testid="stSelectbox"] > label {
        color: #212121 !important;
    }
    
    /* Selectbox/Dropdown with DARK background - white text */
    [data-baseweb="select"] {
        background-color: #1e2a3a !important;
    }
    
    [data-baseweb="select"] > div {
        background-color: #1e2a3a !important;
        color: white !important;
    }
    
    [data-baseweb="select"] span,
    [data-baseweb="select"] div,
    [data-baseweb="select"] input {
        color: white !important;
    }
    
    /* Dropdown menu popup - light background, dark text */
    [data-baseweb="popover"] span,
    [data-baseweb="popover"] div,
    [data-baseweb="menu"] span,
    [data-baseweb="menu"] div {
        color: #212121 !important;
    }
    
    /* Number input with dark background */
    [data-testid="stNumberInput"] input {
        background-color: #1e2a3a !important;
        color: white !important;
    }
    
    [data-testid="stNumberInput"] > label {
        color: #212121 !important;
    }
    
    /* Text input with dark background */
    .stTextInput > div > div > input {
        background-color: #1e2a3a !important;
        color: white !important;
    }
    
    .stTextInput > label {
        color: #212121 !important;
    }
    
    /* Form labels - dark text */
    .stForm label {
        color: #212121 !important;
    }
    
    /* Caption text */
    .stCaption {
        color: #666666 !important;
    }
    
    /* Code blocks should have dark text */
    .stCodeBlock {
        color: #212121 !important;
    }
    
    /* Download button text - white on blue */
    .stDownloadButton button {
        color: white !important;
        background-color: #003d82 !important;
    }
    
    /* Blue header sections - ensure white text */
    div[style*="background: linear-gradient"] h1,
    div[style*="background: linear-gradient"] p,
    div[style*="background: linear-gradient"] span,
    div[style*="background:#003d82"] h1,
    div[style*="background:#003d82"] p,
    div[style*="background:#003d82"] span {
        color: white !important;
    }
</style>
"""

# Professional Dark Theme CSS
DARK_THEME_CSS = """
<style>
    :root {
        --primary-color: #4a9eff;
        --primary-light: #6db3ff;
        --secondary-color: #42a5f5;
        --accent-color: #ffb74d;
        --success-color: #66bb6a;
        --warning-color: #ffa726;
        --danger-color: #ef5350;
    }
    
    html, body {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
    }
    
    .main {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
    }
    
    * {
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
    }
    
    /* All text on dark background should be white/light */
    p, span, div, label, li, td, th {
        color: #e0e0e0 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #4a9eff !important;
        font-weight: 600;
    }
    
    .stButton > button {
        border-radius: 6px;
        border: none;
        font-weight: 500;
        padding: 8px 16px;
        transition: all 0.3s ease;
        background-color: #4a9eff !important;
        color: #1a1a1a !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(74, 158, 255, 0.25) !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 2px solid #444444 !important;
    }
    
    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid #4a9eff !important;
        color: #4a9eff !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #e0e0e0 !important;
    }
    
    /* Text inputs and labels */
    .stTextInput input, .stSelectbox select, .stTextArea textarea {
        color: #e0e0e0 !important;
        background-color: #2d2d2d !important;
    }
    
    .stTextInput label, .stSelectbox label, .stTextArea label {
        color: #e0e0e0 !important;
    }
    
    .stRadio label, .stCheckbox label {
        color: #e0e0e0 !important;
    }
    
    .stNumberInput input, .stNumberInput label {
        color: #e0e0e0 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        color: #e0e0e0 !important;
    }
    
    /* Markdown text */
    .stMarkdown, .stText {
        color: #e0e0e0 !important;
    }
    
    /* Metrics */
    .stMetric, .stMetricLabel, .stMetricValue {
        color: #e0e0e0 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #4a9eff !important;
    }
</style>
"""

# Apply the appropriate theme
if st.session_state.theme == "light":
    st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)
else:
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)

# ======================
# Helper export helpers
# ======================

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode('utf-8')

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to Excel bytes with fallback to CSV if xlsxwriter unavailable"""
    buf = io.BytesIO()
    try:
        # Try xlsxwriter first
        with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        return buf.getvalue()
    except ImportError:
        try:
            # Fallback to openpyxl
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            return buf.getvalue()
        except ImportError:
            # Final fallback: return CSV as bytes
            return df_to_csv_bytes(df)

def make_download_link(data_bytes: bytes, filename: str, mime: str):
    b64 = base64.b64encode(data_bytes).decode()
    href = f"data:{mime};base64,{b64}"
    return href

def generate_pdf_snapshot(text: str, filename: str = "dashboard_snapshot.pdf") -> bytes:
    if not REPORTLAB_AVAILABLE:
        # Fallback: return bytes of a simple text file with .pdf extension
        return text.encode('utf-8')
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    y = height - 72
    for line in text.split('\n'):
        c.drawString(72, y, line[:100])
        y -= 14
        if y < 72:
            c.showPage()
            y = height - 72
    c.save()
    buf.seek(0)
    return buf.getvalue()


# -------------------------
# Pattern / Alert helpers
# -------------------------
def compute_pattern_confidence(claim_row: Dict, pattern: Dict) -> float:
    """Simple heuristic to compute confidence for a pattern match (placeholder)."""
    # very simple scoring: base on matching tokens
    score = 0.0
    field = pattern.get('field')
    op = pattern.get('operator')
    val = pattern.get('value')
    try:
        cell = str(claim_row.get(field, ''))
        if op == 'contains' and val.lower() in cell.lower():
            score = 0.85
        elif op == 'equals' and cell.lower() == val.lower():
            score = 0.9
        elif op == 'gt':
            # assume numeric
            num = float(cell.replace('$','').replace(',',''))
            if num > float(val):
                score = 0.8
    except Exception:
        score = 0.0
    return score


def evaluate_patterns_for_claim(claim_row: Dict) -> list:
    """Return list of matching patterns with confidence and evidence."""
    patterns = st.session_state.get('patterns', [])
    matches = []
    for p in patterns:
        conf = compute_pattern_confidence(claim_row, p)
        if conf > 0:
            matches.append({'pattern': p['name'], 'confidence': conf, 'evidence': f"Field {p['field']} {p['operator']} {p['value']}" , 'action': p.get('action'), 'severity': p.get('severity', 'Medium')})
    return matches


def ensure_alerts_structs():
    if 'patterns' not in st.session_state:
        st.session_state.patterns = []
    if 'whitelist' not in st.session_state:
        st.session_state.whitelist = []
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    if 'override_log' not in st.session_state:
        st.session_state.override_log = []

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.current_page = "dashboard"


# ============================================================================
# LOGIN PAGE
# ============================================================================

def show_login_page():
    """Display professional login portal with Assan branding"""
    
    # Professional header with Assan branding
    st.markdown("""
    <div style='text-align: center; padding: 50px 20px; background: linear-gradient(135deg, #003d82 0%, #0052b3 100%); border-radius: 0;'>
        <div style='margin-bottom: 20px;'>
            <span style='font-size: 2.5rem; font-weight: 700; color: white;'>🏢</span>
        </div>
        <h1 style='color: white; margin: 0; font-size: 2.2rem; font-weight: 700;'>Assan One Intelligence</h1>
        <p style='color: rgba(255, 255, 255, 0.95); margin: 10px 0 0 0; font-size: 1.1rem; font-weight: 500;'>TPA Management Platform</p>
        <p style='color: rgba(255, 255, 255, 0.85); margin: 5px 0 0 0; font-size: 0.95rem;'>Enterprise-Grade Claims & Fraud Detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Login card
        st.markdown("""
        <div style='background: white; border-radius: 12px; padding: 45px; box-shadow: 0 4px 15px rgba(0, 61, 130, 0.1); border: 1px solid #e0e0e0;'>
        """, unsafe_allow_html=True)
        
        st.markdown("<h3 style='text-align: center; color: #003d82; margin-bottom: 30px; font-size: 1.5rem;'>Sign In</h3>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input(
                "Username",
                placeholder="Enter your username",
                help="Your TPA account username"
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                help="Your secure password"
            )
            twofa = st.text_input("2FA Code", placeholder="Enter 6-digit code", help="Authenticator app code")
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                submit_button = st.form_submit_button(
                    "Sign In",
                    width='stretch',
                    help="Click to sign in to your account"
                )
            if submit_button:
                if username and password and twofa:
                    user = SimpleAuth.authenticate(username, password)
                    if user and verify_2fa(username, twofa):
                        st.session_state.authenticated = True
                        st.session_state.user = user
                        st.success(f"✅ Welcome, {username}!")
                        st.balloons()
                        st.rerun()
                    elif user:
                        st.error("❌ Invalid 2FA code.")
                    else:
                        st.error("❌ Invalid username or password. Please try again.")
                else:
                    st.warning("⚠️ Please enter username, password, and 2FA code")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Footer with demo credentials
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background: #f8f9fa; border-radius: 12px; padding: 20px; border-left: 4px solid #ff6f00; border: 1px solid #e0e0e0;'>
            <p style='margin: 0 0 10px 0; font-weight: 600; color: #003d82;'>📝 Demo Credentials</p>
            <div style='font-size: 0.85rem; color: #666;'>
                <p style='margin: 5px 0;'><b>Admin:</b> admin / admin123</p>
                <p style='margin: 5px 0;'><b>Manager:</b> tpa_manager / manager123</p>
                <p style='margin: 5px 0;'><b>Reviewer:</b> claims_reviewer / reviewer123</p>
                <p style='margin: 5px 0;'><b>Auditor:</b> auditor / auditor123</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: center; color: #999; font-size: 0.85rem;'>"
            "© 2025 Assan One Intelligence | All Rights Reserved"
            "</p>",
            unsafe_allow_html=True
        )


# ============================================================================
# DASHBOARD PAGES
# ============================================================================

def show_dashboard():
    """Professional main dashboard with KPI metrics"""
    
    # Header with professional styling
    st.markdown("""
    <div style='margin: -1rem -1rem 2rem -1rem; padding: 2rem 2rem; background: linear-gradient(135deg, #003d82 0%, #0052b3 100%);'>
        <h1 style='color: white; margin: 0; font-size: 2rem;'>📊 Dashboard Overview</h1>
        <p style='color: rgba(255, 255, 255, 0.9); margin: 0.5rem 0 0 0;'>Real-time performance metrics and analytics</p>
    </div>
    """, unsafe_allow_html=True)
    # Breadcrumbs
    st.markdown("<div style='font-size:0.9rem; color:#666; margin-bottom:8px;'>Home / Dashboard</div>", unsafe_allow_html=True)

    # Widget visibility controls
    if 'show_kpis' not in st.session_state:
        st.session_state.show_kpis = True
    col_ctrl1, col_ctrl2 = st.columns([1, 3])
    with col_ctrl1:
        toggle = st.checkbox("Show KPI Cards", value=st.session_state.show_kpis)
        st.session_state.show_kpis = toggle
    with col_ctrl2:
        # Snapshot export (PDF or text fallback)
        snapshot_text = "Dashboard Snapshot - " + datetime.now().isoformat()
        pdf_bytes = generate_pdf_snapshot(snapshot_text)
        st.download_button("📄 Export Snapshot (PDF)", data=pdf_bytes, file_name="dashboard_snapshot.pdf", mime="application/pdf")

    # KPI Metrics Row with enhanced styling
    if st.session_state.show_kpis:
        col1, col2, col3, col4 = st.columns(4, gap="medium")
    
    with col1:
        st.markdown("""
        <div style='background: white; border-radius: 8px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border-left: 4px solid #2e7d32; border: 1px solid #e0e0e0;'>
            <p style='margin: 0; color: #666; font-size: 0.85rem; font-weight: 600;'>TOTAL CLAIMS PROCESSED</p>
            <p style='margin: 10px 0 0 0; font-size: 2rem; font-weight: 700; color: #003d82;'>2,847</p>
            <p style='margin: 5px 0 0 0; font-size: 0.85rem; color: #2e7d32;'>↑ 12% from last month</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: white; border-radius: 8px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border-left: 4px solid #d32f2f; border: 1px solid #e0e0e0;'>
            <p style='margin: 0; color: #666; font-size: 0.85rem; font-weight: 600;'>FRAUD CASES DETECTED</p>
            <p style='margin: 10px 0 0 0; font-size: 2rem; font-weight: 700; color: #d32f2f;'>89</p>
            <p style='margin: 5px 0 0 0; font-size: 0.85rem; color: #d32f2f;'>↑ 5% from last month</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: white; border-radius: 8px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border-left: 4px solid #ff6f00; border: 1px solid #e0e0e0;'>
            <p style='margin: 0; color: #666; font-size: 0.85rem; font-weight: 600;'>AMOUNT PROTECTED</p>
            <p style='margin: 10px 0 0 0; font-size: 2rem; font-weight: 700; color: #ff6f00;'>$2.3M</p>
            <p style='margin: 5px 0 0 0; font-size: 0.85rem; color: #ff6f00;'>This month</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style='background: white; border-radius: 8px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border-left: 4px solid #003d82; border: 1px solid #e0e0e0;'>
            <p style='margin: 0; color: #666; font-size: 0.85rem; font-weight: 600;'>FRAUD DETECTION RATE</p>
            <p style='margin: 10px 0 0 0; font-size: 2rem; font-weight: 700; color: #003d82;'>3.13%</p>
            <p style='margin: 5px 0 0 0; font-size: 0.85rem; color: #666;'>Industry avg: 2.8%</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()

    # In-app notification center (top bar)
    if 'notifications' not in st.session_state:
        st.session_state.notifications = []
    if st.session_state.notifications:
        for n in st.session_state.notifications[-3:]:
            st.info(f"🔔 {n}")
    # Add reminder for pending actions (mock)
    if 'pending_tasks' in st.session_state and st.session_state.pending_tasks:
        st.warning(f"You have {len(st.session_state.pending_tasks)} pending tasks. See Workflow tab.")
    
    # Charts Row
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("<h3 style='margin-top: 0;'>Claims Processing Status</h3>", unsafe_allow_html=True)
        status_data = {
            "Approved": 1850,
            "Under Review": 345,
            "Flagged": 89,
            "Denied": 563
        }
        st.bar_chart(status_data)
    
    with col2:
        st.markdown("<h3 style='margin-top: 0;'>Risk Level Distribution</h3>", unsafe_allow_html=True)
        risk_data = {
            "Low": 2200,
            "Medium": 450,
            "High": 150,
            "Critical": 47
        }
        st.bar_chart(risk_data)
    
    st.divider()
    
    # Recent Activity with enhanced styling
    st.markdown("<h3>Recent System Activity</h3>", unsafe_allow_html=True)
    
    activity_data = {
        "Activity": [
            "High-risk claim flagged for review",
            "Bulk claim processing completed",
            "Suspicious billing pattern detected",
            "Claim approved with conditions",
            "Provider identity verification failed"
        ],
        "Status": ["🔴 Critical", "✅ Complete", "⚠️ Alert", "✅ Approved", "❌ Failed"],
        "Time": ["5 minutes ago", "45 minutes ago", "2 hours ago", "3 hours ago", "4 hours ago"],
        "User": ["System", "Batch Process", "Analytics", "claims_reviewer", "System"]
    }
    
    st.dataframe(activity_data, width='stretch', hide_index=True)
    # Export activity log
    try:
        df_activity = pd.DataFrame(activity_data)
        csv_bytes = df_to_csv_bytes(df_activity)
        excel_bytes = df_to_excel_bytes(df_activity)
        st.download_button("Export Activity CSV", data=csv_bytes, file_name="activity_log.csv", mime="text/csv")
        st.download_button("Export Activity Excel", data=excel_bytes, file_name="activity_log.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        pass


def show_claims_management():
    """Professional claims management and analysis"""
    ensure_alerts_structs()
    
    st.markdown("""
    <div style='margin: -1rem -1rem 2rem -1rem; padding: 2rem 2rem; background: linear-gradient(135deg, #003d82 0%, #0052b3 100%);'>
        <h1 style='color: white; margin: 0; font-size: 2rem;'>📄 Claims Management System</h1>
        <p style='color: rgba(255, 255, 255, 0.9); margin: 0.5rem 0 0 0;'>Analyze, process, and review insurance claims</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Single Claim Analysis", "Batch Processing", "Claims History"])
    
    with tab1:
        st.markdown("<h3>Analyze Individual Claim</h3>", unsafe_allow_html=True)
        

        # Advanced search (all columns)
        st.divider()
        st.markdown("<h4>Advanced Search</h4>", unsafe_allow_html=True)
        if 'claim_records' in st.session_state:
            df = st.session_state.claim_records
            search_query = st.text_input("Search all fields", "")
            if search_query:
                mask = df.apply(lambda row: row.astype(str).str.contains(search_query, case=False, na=False).any(), axis=1)
                st.dataframe(df[mask], width='stretch', hide_index=True)

        # Bulk edit/approve
        st.divider()
        st.markdown("<h4>Bulk Edit/Approve Claims</h4>", unsafe_allow_html=True)
        if 'claim_records' in st.session_state:
            df = st.session_state.claim_records
            bulk_ids = st.multiselect("Select Claims for Bulk Action", df['Claim ID'].tolist())
            col_bulk1, col_bulk2, col_bulk3 = st.columns(3)
            with col_bulk1:
                if st.button("Bulk Approve", key="bulk_approve_main") and bulk_ids:
                    for cid in bulk_ids:
                        idx = df.index[df['Claim ID'] == cid][0]
                        df.at[idx, 'Status'] = '✅ Approved'
                        st.session_state.notifications.append(f"Bulk approved {cid}")
                    st.success(f"Bulk approved {len(bulk_ids)} claims.")
                    st.experimental_rerun()
            with col_bulk2:
                if st.button("Bulk Deny", key="bulk_deny_main") and bulk_ids:
                    for cid in bulk_ids:
                        idx = df.index[df['Claim ID'] == cid][0]
                        df.at[idx, 'Status'] = '⛔ Denied'
                        st.session_state.notifications.append(f"Bulk denied {cid}")
                    st.warning(f"Bulk denied {len(bulk_ids)} claims.")
                    st.experimental_rerun()
            with col_bulk3:
                if st.button("Bulk Flag", key="bulk_flag_main") and bulk_ids:
                    for cid in bulk_ids:
                        idx = df.index[df['Claim ID'] == cid][0]
                        df.at[idx, 'Status'] = '🚩 Flagged'
                        st.session_state.notifications.append(f"Bulk flagged {cid}")
                    st.info(f"Bulk flagged {len(bulk_ids)} claims.")
                    st.experimental_rerun()

        # Data export (PDF/Excel)
        st.divider()
        st.markdown("<h4>Export Claims (PDF/Excel)</h4>", unsafe_allow_html=True)
        if 'claim_records' in st.session_state:
            df = st.session_state.claim_records
            excel_bytes = df_to_excel_bytes(df)
            st.download_button("Export All Claims (Excel)", data=excel_bytes, file_name="claims_export.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            pdf_bytes = generate_pdf_snapshot(df.to_string())
            st.download_button("Export All Claims (PDF)", data=pdf_bytes, file_name="claims_export.pdf", mime="application/pdf")

        # Data Quality Scorecard & Validation
        st.divider()
        st.markdown("<h4>Data Quality Scorecard</h4>", unsafe_allow_html=True)
        if 'claim_records' in st.session_state:
            df = st.session_state.claim_records
            # Duplicate detection
            dupes = df[df.duplicated(['Claim ID'])]
            st.write(f"Duplicate Claims: {len(dupes)}")
            # Simple validation: missing fields
            missing = df.isnull().sum().sum()
            st.write(f"Missing Values: {missing}")
            # Referential integrity (mock: Provider must exist)
            valid_providers = ["Oak Hospital", "City Clinic", "State Medical", "Regional Center"]
            bad_refs = df[~df['Provider'].isin(valid_providers)]
            st.write(f"Invalid Providers: {len(bad_refs)}")
            # Data quality score (simple)
            score = 100 - (len(dupes)*10 + missing*2 + len(bad_refs)*5)
            st.metric("Data Quality Score", max(score,0))
            # Reconciliation report (mock)
            st.markdown("**Reconciliation Report**")
            st.write("All imported claims matched to provider master list." if len(bad_refs)==0 else f"{len(bad_refs)} claims with unmatched providers.")

        # Data import/export (CSV, JSON)
        st.divider()
        st.markdown("**Import/Export Claims Data**")
        uploaded = st.file_uploader("Import Claims (CSV/JSON)")
        if uploaded:
            import chardet
            import io
            try:
                data = uploaded.read()
                encoding = chardet.detect(data)['encoding'] or 'utf-8'
                uploaded.seek(0)
                if uploaded.name.endswith('.csv'):
                    new_df = pd.read_csv(uploaded, encoding=encoding)
                else:
                    new_df = pd.read_json(io.BytesIO(data), encoding=encoding)
                st.session_state.claim_records = new_df
                st.success("Claims imported.")
            except Exception as e:
                st.error(f"Import failed: {e}")
        if 'claim_records' in st.session_state:
            csv_bytes = df_to_csv_bytes(st.session_state.claim_records)
            st.download_button("Export Claims CSV", data=csv_bytes, file_name="claims_export.csv", mime="text/csv")
            json_bytes = st.session_state.claim_records.to_json().encode('utf-8')
            st.download_button("Export Claims JSON", data=json_bytes, file_name="claims_export.json", mime="application/json")

        # API/Webhook/EDI stubs
        st.divider()
        st.markdown("**External Integrations (API/Webhook/EDI)**")
        st.write("API endpoint: /api/claims (stub)")
        st.write("Webhook: POST to /webhook/claim-update (stub)")
        st.write("EDI: X12 837/835 support coming soon (stub)")
        col1, col2 = st.columns([1, 2], gap="large")
        
        with col1:
            st.markdown("<h4 style='color: #0d47a1;'>Claim Input</h4>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload claim document", type=["pdf"], key="claim_upload")
            
            data_dir = Path("Data")
            if data_dir.exists():
                st.markdown("<h4 style='color: #0d47a1; margin-top: 1.5rem;'>Select from Files</h4>", unsafe_allow_html=True)
                pdf_files = list(data_dir.glob("*.pdf"))
                if pdf_files:
                    selected = st.selectbox("Available PDFs", [f.name for f in pdf_files], label_visibility="collapsed")
                    if selected:
                        uploaded_file = data_dir / selected
        
        with col2:
            if uploaded_file:
                st.markdown(f"""
                <div style='background: #f5f7fa; border-radius: 8px; padding: 15px; border-left: 4px solid #0d47a1; margin-bottom: 1.5rem;'>
                    <p style='margin: 0; color: #666; font-size: 0.9rem;'><b>Selected Document:</b></p>
                    <p style='margin: 5px 0 0 0; color: #0d47a1; font-weight: 500;'>{Path(str(uploaded_file)).name}</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    analysis_type = st.selectbox(
                        "Analysis Type",
                        ["Full Analysis", "Quick Fraud Check", "OCR Only", "Pattern Detection"]
                    )
                
                with col2:
                    priority = st.selectbox(
                        "Priority Level",
                        ["Normal", "High", "Critical"]
                    )
                
                st.divider()
                
                if st.button("🔍 Analyze Claim", width='stretch', key="analyze_claim_btn"):
                    with st.spinner("Processing claim with advanced fraud detection..."):
                        # Determine PDF path
                        pdf_path = None
                        if isinstance(uploaded_file, Path):
                            pdf_path = str(uploaded_file)
                        elif hasattr(uploaded_file, 'name'):
                            # Save uploaded file temporarily
                            tmp_dir = Path("Data/uploads")
                            tmp_dir.mkdir(parents=True, exist_ok=True)
                            pdf_path = tmp_dir / uploaded_file.name
                            with open(pdf_path, "wb") as f:
                                f.write(uploaded_file.getvalue())
                            pdf_path = str(pdf_path)
                        
                        if pdf_path:
                            try:
                                from agent.pipeline import process_claim_full_pipeline
                                # Map analysis type to mode
                                mode = "ocr_only" if analysis_type == "OCR Only" else "full_analysis"
                                fraud_report = process_claim_full_pipeline(
                                    pdf_path,
                                    execution_mode="parallel",
                                    save_results=True,
                                    mode=mode
                                )
                                
                                if "error" in fraud_report:
                                    st.error(f"❌ Error: {fraud_report['error']}")
                                elif mode == "ocr_only":
                                    st.success("✅ OCR Extraction Complete")
                                    
                                    # Display extracted fields in a clean formatted way
                                    claim_data = fraud_report.get("claim_data", {})
                                    
                                    st.subheader("📝 Extracted Claim Information")
                                    
                                    # Key fields in a clean card layout
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.markdown("""
                                        <div style='background: #f8f9fa; border-radius: 8px; padding: 15px; margin-bottom: 10px; border-left: 4px solid #1976d2;'>
                                            <h4 style='margin: 0 0 10px 0; color: #1976d2;'>👤 Claimant Details</h4>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        claimant_name = claim_data.get('claimant_name') or 'Not found'
                                        member_id = claim_data.get('member_id') or 'Not found'
                                        dob = claim_data.get('dob') or 'Not found'
                                        
                                        st.markdown(f"**Name:** {claimant_name}")
                                        st.markdown(f"**Member ID:** {member_id}")
                                        st.markdown(f"**Date of Birth:** {dob}")
                                        
                                        st.markdown("""
                                        <div style='background: #f8f9fa; border-radius: 8px; padding: 15px; margin: 15px 0 10px 0; border-left: 4px solid #388e3c;'>
                                            <h4 style='margin: 0 0 10px 0; color: #388e3c;'>🏥 Provider Information</h4>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        provider = claim_data.get('provider') or 'Not found'
                                        hospital = claim_data.get('hospital_name') or 'Not found'
                                        insurance_co = claim_data.get('insurance_company') or 'Not found'
                                        
                                        st.markdown(f"**Doctor:** {provider}")
                                        st.markdown(f"**Hospital:** {hospital}")
                                        st.markdown(f"**Insurance:** {insurance_co[:50] + '...' if len(str(insurance_co)) > 50 else insurance_co}")
                                        
                                        # Dates Section
                                        st.markdown("""
                                        <div style='background: #f8f9fa; border-radius: 8px; padding: 15px; margin: 15px 0 10px 0; border-left: 4px solid #7b1fa2;'>
                                            <h4 style='margin: 0 0 10px 0; color: #7b1fa2;'>📅 Dates (Standardized)</h4>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        admission = claim_data.get('date_of_admission') or 'Not found'
                                        discharge = claim_data.get('date_of_discharge') or 'Not found'
                                        
                                        st.markdown(f"**Admission:** {admission}")
                                        st.markdown(f"**Discharge:** {discharge}")
                                    
                                    with col2:
                                        st.markdown("""
                                        <div style='background: #f8f9fa; border-radius: 8px; padding: 15px; margin-bottom: 10px; border-left: 4px solid #f57c00;'>
                                            <h4 style='margin: 0 0 10px 0; color: #f57c00;'>📋 Policy Details</h4>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        policy_num = claim_data.get('policy_number') or 'Not found'
                                        claim_num = claim_data.get('claim_number') or 'Not found'
                                        group_num = claim_data.get('group_number') or 'Not found'
                                        
                                        st.markdown(f"**Policy Number:** {policy_num}")
                                        st.markdown(f"**Claim Number:** {claim_num}")
                                        st.markdown(f"**Group Number:** {group_num}")
                                        
                                        # Medical/Diagnosis Section
                                        st.markdown("""
                                        <div style='background: #f8f9fa; border-radius: 8px; padding: 15px; margin: 15px 0 10px 0; border-left: 4px solid #00897b;'>
                                            <h4 style='margin: 0 0 10px 0; color: #00897b;'>🩺 Medical Details (ICD-10)</h4>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        diagnosis = claim_data.get('diagnosis') or 'Not found'
                                        icd_codes = claim_data.get('diagnosis_icd_list', [])
                                        procedure = claim_data.get('procedure_name') or claim_data.get('procedure_code') or 'Not found'
                                        
                                        st.markdown(f"**Diagnosis:** {diagnosis}")
                                        if icd_codes:
                                            st.markdown(f"**ICD-10 Codes:** {', '.join(icd_codes[:3])}")
                                        st.markdown(f"**Procedure:** {procedure}")
                                        
                                        # Financial Section
                                        st.markdown("""
                                        <div style='background: #f8f9fa; border-radius: 8px; padding: 15px; margin: 15px 0 10px 0; border-left: 4px solid #d32f2f;'>
                                            <h4 style='margin: 0 0 10px 0; color: #d32f2f;'>💰 Billing (Smart Extracted)</h4>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        billed_amt = claim_data.get('billed_amount')
                                        if billed_amt:
                                            st.markdown(f"**Total Amount:** ₹{float(billed_amt):,.2f}")
                                        else:
                                            st.markdown("**Total Amount:** Not found")
                                        
                                        # Show billing by category
                                        billing_cats = claim_data.get('billing_by_category', [])
                                        if billing_cats:
                                            st.markdown("**By Category:**")
                                            for cat in billing_cats[:5]:
                                                st.markdown(f"  • {cat}")
                                    
                                    # Itemized Billing Section
                                    billing_items = claim_data.get('billing_summary', [])
                                    if billing_items:
                                        st.markdown("---")
                                        with st.expander("📊 View Itemized Billing", expanded=False):
                                            for item in billing_items[:15]:
                                                st.markdown(f"• {item}")
                                            total_itemized = claim_data.get('total_itemized')
                                            if total_itemized:
                                                st.markdown(f"**Itemized Total: ₹{total_itemized:,.2f}**")
                                    
                                    # Form Sections Found
                                    sections = claim_data.get('form_sections', [])
                                    if sections and len(sections) > 1:
                                        st.markdown("---")
                                        st.info(f"📑 Form Sections Found: {', '.join(sections)}")
                                    
                                    # Collapsible raw OCR text
                                    st.markdown("---")
                                    with st.expander("📄 View Raw OCR Text", expanded=False):
                                        ocr_text = fraud_report.get("ocr_text", "")
                                        # Show only first 2000 chars with option to see more
                                        if len(ocr_text) > 2000:
                                            st.text_area("OCR Text (First 2000 characters)", ocr_text[:2000] + "...", height=200)
                                            st.caption(f"Total: {len(ocr_text)} characters extracted")
                                        else:
                                            st.text_area("OCR Text", ocr_text, height=200)
                                    
                                    # Collapsible full JSON data
                                    with st.expander("🔧 View Raw JSON Data", expanded=False):
                                        # Filter out raw_text_preview for cleaner display
                                        display_data = {k: v for k, v in claim_data.items() if k != 'raw_text_preview'}
                                        st.json(display_data)
                                else:
                                    st.success("✅ Analysis Complete")
                                    
                                    # Get real values from fraud report
                                    score = fraud_report.get("fraud_risk_score", 0)
                                    risk_level = fraud_report.get("overall_risk_level", "UNKNOWN")
                                    avg_conf = fraud_report.get("summary", {}).get("average_confidence", 0)
                                    confidence_pct = int(avg_conf * 100) if avg_conf <= 1 else int(avg_conf)
                                    
                                    # Color coding based on risk
                                    if risk_level == "CRITICAL":
                                        risk_color = "#d32f2f"
                                        risk_icon = "🔴"
                                    elif risk_level == "HIGH":
                                        risk_color = "#f57c00"
                                        risk_icon = "🟠"
                                    elif risk_level == "MEDIUM":
                                        risk_color = "#fbc02d"
                                        risk_icon = "🟡"
                                    else:
                                        risk_color = "#2e7d32"
                                        risk_icon = "🟢"
                                    
                                    # Display Results with professional styling
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.markdown(f"""
                                        <div style='background: white; border-radius: 8px; padding: 20px; text-align: center; border: 1px solid #e0e0e0; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                                            <p style='margin: 0; color: #666; font-size: 0.85rem; font-weight: 600;'>FRAUD RISK SCORE</p>
                                            <p style='margin: 10px 0; font-size: 2rem; font-weight: 700; color: {risk_color};'>{score}/100</p>
                                            <p style='margin: 5px 0 0 0; font-size: 0.85rem; color: #999;'>Based on document analysis</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    with col2:
                                        st.markdown(f"""
                                        <div style='background: white; border-radius: 8px; padding: 20px; text-align: center; border: 1px solid #e0e0e0; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                                            <p style='margin: 0; color: #666; font-size: 0.85rem; font-weight: 600;'>RISK LEVEL</p>
                                            <p style='margin: 10px 0; font-size: 2rem; font-weight: 700; color: {risk_color};'>{risk_icon} {risk_level}</p>
                                            <p style='margin: 5px 0 0 0; font-size: 0.85rem; color: #999;'>{fraud_report.get('recommended_action', 'Review required')}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    with col3:
                                        st.markdown(f"""
                                        <div style='background: white; border-radius: 8px; padding: 20px; text-align: center; border: 1px solid #e0e0e0; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                                            <p style='margin: 0; color: #666; font-size: 0.85rem; font-weight: 600;'>CONFIDENCE</p>
                                            <p style='margin: 10px 0; font-size: 2rem; font-weight: 700; color: #0d47a1;'>{confidence_pct}%</p>
                                            <p style='margin: 5px 0 0 0; font-size: 0.85rem; color: #999;'>Agent consensus</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    st.divider()
                                    
                                    # Detailed Findings from real agent results
                                    all_findings = fraud_report.get("all_findings", [])
                                    agent_results = fraud_report.get("agent_results", {})
                                    
                                    col1, col2 = st.columns(2, gap="large")
                                    
                                    with col1:
                                        st.markdown("""
                                        <h4 style='color: #c62828; margin-top: 0;'>🚨 Risk Factors Identified</h4>
                                        """, unsafe_allow_html=True)
                                        if all_findings:
                                            for finding in all_findings[:6]:
                                                st.markdown(f"- **{finding}**")
                                        else:
                                            st.markdown("- No significant risk factors identified")
                                    
                                    with col2:
                                        st.markdown("""
                                        <h4 style='color: #2e7d32; margin-top: 0;'>🤖 Agent Analysis</h4>
                                        """, unsafe_allow_html=True)
                                        for agent_name, agent_result in agent_results.items():
                                            if agent_result:
                                                a_risk = agent_result.get('risk_level', 'N/A')
                                                a_conf = agent_result.get('confidence', 0)
                                                st.markdown(f"- **{agent_name.title()}**: {a_risk} ({int(a_conf*100) if a_conf <= 1 else a_conf}% confidence)")
                                    
                                    st.divider()
                                    
                                    # Extracted Claim Information (collapsible)
                                    with st.expander("📝 View Extracted Claim Information", expanded=False):
                                        claim_data = fraud_report.get("claim_data", {})
                                        if claim_data:
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.markdown("**👤 Claimant Details**")
                                                st.markdown(f"- Name: {claim_data.get('claimant_name') or 'Not found'}")
                                                st.markdown(f"- Member ID: {claim_data.get('member_id') or 'Not found'}")
                                                st.markdown(f"- DOB: {claim_data.get('dob') or 'Not found'}")
                                                st.markdown("**🏥 Provider**")
                                                st.markdown(f"- Provider: {claim_data.get('provider') or 'Not found'}")
                                                st.markdown(f"- NPI: {claim_data.get('npi_number') or 'Not found'}")
                                            with col2:
                                                st.markdown("**📋 Claim Details**")
                                                st.markdown(f"- Policy #: {claim_data.get('policy_number') or 'Not found'}")
                                                st.markdown(f"- Claim #: {claim_data.get('claim_number') or 'Not found'}")
                                                st.markdown(f"- Group #: {claim_data.get('group_number') or 'Not found'}")
                                                st.markdown("**💰 Financial**")
                                                billed = claim_data.get('billed_amount')
                                                st.markdown(f"- Billed: ${billed}" if billed else "- Billed: Not found")
                                                st.markdown(f"- Diagnosis: {claim_data.get('diagnosis') or 'Not found'}")
                                    
                                    st.divider()
                                    
                                    # Action buttons
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        if st.button("✅ Approve Claim", width='stretch', key="approve_claim_analysis"):
                                            cid = Path(pdf_path).stem
                                            st.session_state.override_log.append({'timestamp': datetime.now().isoformat(), 'claim': cid, 'action': 'Approve', 'user': st.session_state.user['username']})
                                            st.success("Manual approval recorded")
                                    with col2:
                                        if st.button("⏳ Review Later", width='stretch', key="review_later_analysis"):
                                            cid = Path(pdf_path).stem
                                            st.session_state.override_log.append({'timestamp': datetime.now().isoformat(), 'claim': cid, 'action': 'Review Later', 'user': st.session_state.user['username']})
                                            st.info("Claim marked for later review")
                                    with col3:
                                        if st.button("⛔ Deny & Report", width='stretch', key="deny_report_analysis"):
                                            cid = Path(pdf_path).stem
                                            st.session_state.override_log.append({'timestamp': datetime.now().isoformat(), 'claim': cid, 'action': 'Deny & Report', 'user': st.session_state.user['username']})
                                            st.warning("Claim denied and reported")

                                    # AI-powered insights from real data
                                    st.divider()
                                    st.markdown("<h4>AI Insights & Recommendations</h4>", unsafe_allow_html=True)
                                    st.markdown(f"**Score Explanation:** {score}/100 — {risk_level} risk based on {len(all_findings)} findings across {len(agent_results)} agents.")
                                    st.markdown("**Top Drivers:**")
                                    for f in all_findings[:3]:
                                        st.markdown(f"- {f}")
                                    st.markdown(f"**Recommended Action:** {fraud_report.get('recommended_action', 'Manual review required')}")

                                    # Download full report
                                    st.divider()
                                    import json as json_mod
                                    report_json = json_mod.dumps(fraud_report, indent=2, default=str)
                                    st.download_button(
                                        "📥 Download Full Report (JSON)",
                                        data=report_json,
                                        file_name=f"fraud_report_{Path(pdf_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                        mime="application/json"
                                    )
                            except Exception as e:
                                st.error(f"❌ Analysis failed: {e}")
                                import traceback
                                st.text(traceback.format_exc())
                        else:
                            st.warning("Please upload or select a PDF file first")
    
    with tab2:
        st.markdown("<h3>Batch Processing</h3>", unsafe_allow_html=True)
        st.write("Process multiple claims simultaneously for improved efficiency")

    # Workflow designer and task assignment (new tab)
    st.divider()
    st.markdown("<h3>Workflow & Task Assignment</h3>", unsafe_allow_html=True)
    if 'pending_tasks' not in st.session_state:
        st.session_state.pending_tasks = []
    with st.form("workflow_form"):
        task_claim = st.text_input("Claim ID for Task", placeholder="e.g. CLM-2025-001")
        task_user = st.selectbox("Assign To", ["tpa_manager", "claims_reviewer", "auditor"])
        task_sla = st.number_input("SLA (hours)", 1, 72, 24)
        if st.form_submit_button("Assign Task") and task_claim:
            st.session_state.pending_tasks.append({
                'claim': task_claim,
                'assigned_to': task_user,
                'sla': task_sla,
                'assigned_at': datetime.now().isoformat(),
                'status': 'Open'
            })
            st.session_state.notifications.append(f"Task assigned: {task_claim} to {task_user} (SLA {task_sla}h)")
            st.success("Task assigned.")

    # Show tasks and SLA/escalation
    if st.session_state.pending_tasks:
        st.markdown("**Open Tasks**")
        for idx, t in enumerate(st.session_state.pending_tasks):
            overdue = (datetime.now() - datetime.fromisoformat(t['assigned_at'])).total_seconds() > t['sla']*3600
            st.write(f"Claim: {t['claim']} | Assigned to: {t['assigned_to']} | SLA: {t['sla']}h | Status: {t['status']}" + (" | ⚠️ Overdue" if overdue else ""))
            if overdue and t['status'] == 'Open':
                if st.button(f"Escalate {t['claim']}", key=f"escalate_{t['claim']}_{idx}"):
                    t['status'] = 'Escalated'
                    st.session_state.notifications.append(f"Escalation: {t['claim']} overdue and escalated!")
                    st.warning(f"Claim {t['claim']} escalated.")
            if st.button(f"Mark Complete {t['claim']}", key=f"complete_{t['claim']}_{idx}"):
                t['status'] = 'Closed'
                st.session_state.notifications.append(f"Task complete: {t['claim']}")
                st.success(f"Task {t['claim']} closed.")

    # Auto-rejection for obvious fraud (mock rule)
    st.divider()
    st.markdown("**Auto-Rejection for Obvious Fraud**")
    if st.button("Run Auto-Rejection (Simulated)", key="run_auto_rejection"):
        # For demo, auto-reject claims with 'Critical' risk
        if 'claim_records' in st.session_state:
            df = st.session_state.claim_records
            crit = df[df['Risk'].str.contains('Critical', na=False)]
            for cid in crit['Claim ID']:
                idx = df.index[df['Claim ID'] == cid][0]
                df.at[idx, 'Status'] = '⛔ Denied'
                st.session_state.notifications.append(f"Auto-rejected claim {cid} for critical risk.")
            st.success(f"Auto-rejected {len(crit)} claims.")
        
        col1, col2 = st.columns(2, gap="large")
        with col1:
            num_claims = st.number_input("Number of claims to process", 1, 100, 10, help="Enter the number of claims")
        with col2:
            processing_mode = st.selectbox("Processing Mode", ["Parallel (Fastest)", "Sequential (Standard)", "Hybrid (Balanced)"])
        
        # Batch processing control/state setup
        if 'batch_state' not in st.session_state:
            st.session_state.batch_state = {
                'active': False,
                'paused': False,
                'index': 0,
                'total': 0,
                'start_time': None,
                'logs': []
            }

        col_a, col_b, col_c = st.columns([1,1,1])
        with col_a:
            if st.button("🚀 Start Batch Processing", width='stretch', key="start_batch_processing"):
                st.session_state.batch_state.update({
                    'active': True,
                    'paused': False,
                    'index': 0,
                    'total': int(num_claims),
                    'start_time': datetime.now(),
                    'logs': []
                })
                st.experimental_rerun()
        with col_b:
            if st.button("⏸ Pause", width='stretch', key="pause_batch"):
                st.session_state.batch_state['paused'] = True
        with col_c:
            if st.button("▶️ Resume", width='stretch', key="resume_batch"):
                st.session_state.batch_state['paused'] = False
                st.experimental_rerun()

        st.markdown("---")

        # If batch active, run one step per rerun to allow pause/resume
        bs = st.session_state.batch_state
        if bs['active'] and bs['index'] < bs['total']:
            if not bs['paused']:
                # process one claim
                bs['index'] += 1
                now = datetime.now()
                elapsed = (now - bs['start_time']).total_seconds() if bs['start_time'] else 0
                avg_per = elapsed / bs['index'] if bs['index'] > 0 else 0.0
                remaining = bs['total'] - bs['index']
                eta_seconds = int(avg_per * remaining)
                eta = str(timedelta(seconds=eta_seconds))
                bs['logs'].append(f"{now.isoformat()} - Processed claim {bs['index']}/{bs['total']} (ETA: {eta})")
                st.session_state.batch_state = bs
                st.experimental_rerun()
            else:
                st.info(f"Batch paused at {bs['index']}/{bs['total']}")
        elif bs['active'] and bs['index'] >= bs['total']:
            st.success(f"✅ Batch Processing Completed: {bs['total']} claims processed")
            st.markdown(f"**Duration:** {datetime.now() - bs['start_time']}")
            st.markdown("<b>Batch Logs</b>", unsafe_allow_html=True)
            st.write('\n'.join(bs['logs']))
            # offer export
            try:
                logs_bytes = '\n'.join(bs['logs']).encode('utf-8')
                st.download_button("Export Batch Logs", data=logs_bytes, file_name="batch_logs.txt", mime="text/plain")
            except Exception:
                pass
            # notify and reset
            if st.button("Send Completion Notification (simulate)", key="send_completion_notif"):
                st.info("Notification queued (simulated)")
            # reset button
            if st.button("Reset Batch State", key="reset_batch_state"):
                st.session_state.batch_state = {
                    'active': False,
                    'paused': False,
                    'index': 0,
                    'total': 0,
                    'start_time': None,
                    'logs': []
                }
    
    with tab3:
        st.markdown("<h3>Claims History & Archive</h3>", unsafe_allow_html=True)

        # Sample history data (would come from DB in production)
        history_data = {
            "Claim ID": ["CLM-2025-001", "CLM-2025-002", "CLM-2025-003", "CLM-2025-004"],
            "Date": ["2025-12-28", "2025-12-27", "2025-12-26", "2025-12-25"],
            "Amount": ["$5,230", "$8,950", "$3,210", "$12,450"],
            "Status": ["✅ Approved", "⛔ Denied", "⏳ Under Review", "✅ Approved"],
            "Risk": ["🟢 Low", "🔴 Critical", "🟡 Medium", "🟢 Low"],
            "Provider": ["Oak Hospital", "City Clinic", "State Medical", "Regional Center"]
        }

        # persist claims in session state for bulk operations
        if 'claim_records' not in st.session_state:
            st.session_state.claim_records = pd.DataFrame(history_data)
            # initialize claim history store
            st.session_state.claim_history = {cid: [{'timestamp': datetime.now().isoformat(), 'status': st.session_state.claim_records.loc[i,'Status'], 'user': 'system'}]
                                               for i, cid in enumerate(st.session_state.claim_records['Claim ID'])}

        df = st.session_state.claim_records.copy()
        # normalize date column
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except Exception:
            pass

        # Filters: date range, status, risk, provider, claim search
        presets = st.selectbox("Date Range", ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Custom"], index=1)
        if presets != "Custom":
            if presets == "Last 7 Days":
                from_date = datetime.now() - timedelta(days=7)
            elif presets == "Last 30 Days":
                from_date = datetime.now() - timedelta(days=30)
            else:
                from_date = datetime.now() - timedelta(days=90)
            to_date = datetime.now()
        else:
            col_from, col_to = st.columns(2)
            with col_from:
                from_date = st.date_input("From date", datetime.now() - timedelta(days=30))
            with col_to:
                to_date = st.date_input("To date", datetime.now())

        col1, col2, col3 = st.columns(3, gap="medium")
        with col1:
            claim_search = st.text_input("Search Claim ID", placeholder="Enter claim id or part of it")
        with col2:
            status_filter = st.multiselect("Status", ["Approved", "Denied", "Under Review", "Flagged"], default=["Approved"])
        with col3:
            provider_opts = sorted(df['Provider'].unique().tolist())
            provider_filter = st.multiselect("Provider", provider_opts, default=provider_opts)

        st.divider()

        # Apply filters
        filtered = df.copy()
        try:
            if isinstance(from_date, datetime):
                start_dt = pd.to_datetime(from_date)
            else:
                start_dt = pd.to_datetime(datetime(from_date.year, from_date.month, from_date.day))
            end_dt = pd.to_datetime(to_date) if 'to_date' in locals() else pd.to_datetime(datetime.now())
            filtered = filtered[(filtered['Date'] >= start_dt) & (filtered['Date'] <= end_dt)]
        except Exception:
            pass

        if claim_search:
            filtered = filtered[filtered['Claim ID'].str.contains(claim_search, case=False, na=False)]

        if status_filter:
            # keep rows where any selected status token matches the Status cell
            mask = False
            for s in status_filter:
                mask = mask | filtered['Status'].str.contains(s.split()[0], case=False, na=False)
            filtered = filtered[mask]

        if provider_filter:
            filtered = filtered[filtered['Provider'].isin(provider_filter)]

        # Display filtered results
        st.dataframe(filtered, width='stretch', hide_index=True)

        # Bulk action toolbar
        st.markdown("**Bulk Actions**")
        claim_ids = filtered['Claim ID'].tolist()
        selected = st.multiselect("Select Claims", claim_ids)
        col_act1, col_act2, col_act3 = st.columns(3)
        with col_act1:
            if st.button("✅ Approve Selected", key="approve_selected_queue") and selected:
                for cid in selected:
                    idx = st.session_state.claim_records.index[st.session_state.claim_records['Claim ID'] == cid][0]
                    st.session_state.claim_records.at[idx, 'Status'] = '✅ Approved'
                    st.session_state.claim_history.setdefault(cid, []).append({'timestamp': datetime.now().isoformat(), 'status': '✅ Approved', 'user': st.session_state.user['username']})
                st.success(f"Approved {len(selected)} claims")
                st.experimental_rerun()
        with col_act2:
            if st.button("⛔ Deny Selected", key="deny_selected_queue") and selected:
                for cid in selected:
                    idx = st.session_state.claim_records.index[st.session_state.claim_records['Claim ID'] == cid][0]
                    st.session_state.claim_records.at[idx, 'Status'] = '⛔ Denied'
                    st.session_state.claim_history.setdefault(cid, []).append({'timestamp': datetime.now().isoformat(), 'status': '⛔ Denied', 'user': st.session_state.user['username']})
                st.success(f"Denied {len(selected)} claims")
                st.experimental_rerun()
        with col_act3:
            if st.button("🚩 Flag Selected", key="flag_selected_queue") and selected:
                for cid in selected:
                    idx = st.session_state.claim_records.index[st.session_state.claim_records['Claim ID'] == cid][0]
                    st.session_state.claim_records.at[idx, 'Status'] = '🚩 Flagged'
                    st.session_state.claim_history.setdefault(cid, []).append({'timestamp': datetime.now().isoformat(), 'status': '🚩 Flagged', 'user': st.session_state.user['username']})
                st.warning(f"Flagged {len(selected)} claims for review")
                st.experimental_rerun()

        st.divider()

        # Claim comparison view
        st.markdown("**Compare Claims (Side-by-side)**")
        comp_a, comp_b = st.columns(2)
        with comp_a:
            compare_left = st.selectbox("Left Claim", df['Claim ID'].tolist(), key='comp_left')
        with comp_b:
            compare_right = st.selectbox("Right Claim", df['Claim ID'].tolist(), key='comp_right')

        if compare_left and compare_right:
            left_row = df[df['Claim ID'] == compare_left].iloc[0]
            right_row = df[df['Claim ID'] == compare_right].iloc[0]
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown(f"**{compare_left}**")
                st.write(left_row.to_dict())
            with col_r:
                st.markdown(f"**{compare_right}**")
                st.write(right_row.to_dict())

        st.divider()

        # Claim notes/comments area
        st.markdown("**Claim Notes / Collaboration**")
        note_claim = st.selectbox("Select Claim for Notes", df['Claim ID'].tolist(), key='note_claim')
        if 'claim_notes' not in st.session_state:
            st.session_state.claim_notes = {}
        existing_notes = "\n\n".join(st.session_state.claim_notes.get(note_claim, [])) if note_claim in st.session_state.claim_notes else ""
        new_note = st.text_area("Add Note", value="", placeholder="Enter note or comment for the claim")
        if st.button("Save Note", key="save_claim_note") and new_note.strip():
            st.session_state.claim_notes.setdefault(note_claim, []).append(f"{datetime.now().isoformat()} - {st.session_state.user['username']}: {new_note}")
            st.success("Note saved")
            st.experimental_rerun()

        if existing_notes:
            st.markdown("**Existing Notes**")
            st.write(existing_notes)

        # Claim status history viewer
        st.divider()
        st.markdown("**Claim Status History**")
        history_claim = st.selectbox("Select Claim for History", df['Claim ID'].tolist(), key='history_claim')
        if history_claim in st.session_state.claim_history:
            for ev in st.session_state.claim_history[history_claim]:
                st.markdown(f"- {ev['timestamp']} — {ev['status']} (by {ev.get('user','system')})")


def show_reports():
    """Professional reports and analytics"""
    
    st.markdown("""
    <div style='margin: -1rem -1rem 2rem -1rem; padding: 2rem 2rem; background: linear-gradient(135deg, #003d82 0%, #0052b3 100%);'>
        <h1 style='color: white; margin: 0; font-size: 2rem;'>📈 Reports & Analytics</h1>
        <p style='color: rgba(255, 255, 255, 0.9); margin: 0.5rem 0 0 0;'>Comprehensive fraud detection and financial analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Fraud Intelligence", "Financial Analytics", "Custom Reports"])
    
    with tab1:
        st.markdown("<h3>Fraud Detection Intelligence</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("""
            <div style='background: white; border-radius: 8px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-left: 4px solid #c62828;'>
                <p style='margin: 0; color: #666; font-size: 0.85rem; font-weight: 600;'>FRAUD CASES THIS MONTH</p>
                <p style='margin: 10px 0 0 0; font-size: 2.5rem; font-weight: 700; color: #c62828;'>89</p>
                <p style='margin: 5px 0 0 0; font-size: 0.85rem; color: #c62828;'>↑ 15% increase</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background: white; border-radius: 8px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-left: 4px solid #2e7d32;'>
                <p style='margin: 0; color: #666; font-size: 0.85rem; font-weight: 600;'>RECOVERY AMOUNT</p>
                <p style='margin: 10px 0 0 0; font-size: 2.5rem; font-weight: 700; color: #2e7d32;'>$450K</p>
                <p style='margin: 5px 0 0 0; font-size: 0.85rem; color: #2e7d32;'>↑ 22% increase</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        st.markdown("<h4>Monthly Fraud Trend Analysis</h4>", unsafe_allow_html=True)
        fraud_data = {
            "Jan": 45, "Feb": 52, "Mar": 48, "Apr": 67, "May": 73, "Jun": 89
        }
        st.line_chart(fraud_data)
        
        st.divider()
        
        st.markdown("<h4>Fraud Category Breakdown</h4>", unsafe_allow_html=True)
        fraud_types = {
            "Overbilling": 28,
            "Unbundling": 22,
            "Identity Theft": 18,
            "Duplicate Claims": 12,
            "Upcoding": 9
        }
        st.bar_chart(fraud_types)
    
    with tab2:
        st.markdown("<h3>Financial Performance Summary</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3, gap="medium")
        
        with col1:
            st.markdown("""
            <div style='background: white; border-radius: 8px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid #e0e0e0; text-align: center;'>
                <p style='margin: 0; color: #666; font-size: 0.85rem; font-weight: 600;'>TOTAL CLAIMS AMOUNT</p>
                <p style='margin: 10px 0; font-size: 2rem; font-weight: 700; color: #0d47a1;'>$8.5M</p>
                <p style='margin: 5px 0 0 0; font-size: 0.85rem; color: #999;'>This month</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background: white; border-radius: 8px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid #e0e0e0; text-align: center;'>
                <p style='margin: 0; color: #666; font-size: 0.85rem; font-weight: 600;'>FRAUDULENT AMOUNT BLOCKED</p>
                <p style='margin: 10px 0; font-size: 2rem; font-weight: 700; color: #c62828;'>$450K</p>
                <p style='margin: 5px 0 0 0; font-size: 0.85rem; color: #999;'>5.3% of total</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style='background: white; border-radius: 8px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid #e0e0e0; text-align: center;'>
                <p style='margin: 0; color: #666; font-size: 0.85rem; font-weight: 600;'>NET SAVINGS</p>
                <p style='margin: 10px 0; font-size: 2rem; font-weight: 700; color: #2e7d32;'>$450K</p>
                <p style='margin: 5px 0 0 0; font-size: 0.85rem; color: #999;'>Monthly recovery</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        st.markdown("<h4>Financial Trends & Projections</h4>", unsafe_allow_html=True)
        financial_data = {
            "Claims": [5200, 6100, 5800, 7200, 8100, 8500],
            "Fraud Loss": [180, 220, 210, 280, 350, 450]
        }
        st.line_chart(financial_data)
    
    with tab3:
        st.markdown("<h3>Generate Custom Reports</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            report_type = st.selectbox(
                "Report Type",
                ["Fraud Summary Report", "Provider Performance Analysis", "Claim Pattern Analysis", "Compliance & Audit Report"]
            )
        
        with col2:
            date_range = st.selectbox(
                "Time Period",
                ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Custom Range"]
            )
        
        st.divider()
        
        col1, col2, col3 = st.columns(3, gap="medium")
        with col1:
            if st.button("📊 Generate Report", width='stretch', key="generate_report_btn"):
                with st.spinner("Generating report..."):
                    st.success("✅ Report generated successfully")
                    st.markdown("""
                    <div style='background: #e8f5e9; border-radius: 8px; padding: 15px; border-left: 4px solid #2e7d32;'>
                        <p style='margin: 0; color: #2e7d32; font-weight: 600;'>Report Ready for Download</p>
                        <p style='margin: 5px 0 0 0; color: #666; font-size: 0.9rem;'>TPA_Fraud_Summary_Report_2025-12-30.pdf</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.download_button(
                        "📥 Download PDF Report",
                        data=b"PDF Report Data",
                        file_name=f"TPA_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        width='stretch',
                        key="download_pdf_report"
                    )
        
        with col2:
            if st.button("📧 Email Report", width='stretch', key="email_report_btn"):
                st.markdown("""
                <div style='background: #e3f2fd; border-radius: 8px; padding: 15px; border-left: 4px solid #0d47a1;'>
                    <p style='margin: 0; color: #0d47a1; font-weight: 600;'>✅ Email Queued</p>
                    <p style='margin: 5px 0 0 0; color: #666; font-size: 0.9rem;'>Report will be sent to your registered email</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if st.button("💾 Save as Draft", width='stretch', key="save_draft_btn"):
                st.markdown("""
                <div style='background: #fff3e0; border-radius: 8px; padding: 15px; border-left: 4px solid #f57c00;'>
                    <p style='margin: 0; color: #f57c00; font-weight: 600;'>✅ Saved to Drafts</p>
                    <p style='margin: 5px 0 0 0; color: #666; font-size: 0.9rem;'>You can access this report later</p>
                </div>
                """, unsafe_allow_html=True)


def show_analytics():
    """Professional advanced analytics and insights"""
    
    st.markdown("""
    <div style='margin: -1rem -1rem 2rem -1rem; padding: 2rem 2rem; background: linear-gradient(135deg, #003d82 0%, #0052b3 100%);'>
        <h1 style='color: white; margin: 0; font-size: 2rem;'>🔍 Advanced Analytics & Intelligence</h1>
        <p style='color: rgba(255, 255, 255, 0.9); margin: 0.5rem 0 0 0;'>Deep insights into fraud patterns and provider performance</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Provider Intelligence", "Pattern Detection", "Risk Forecasting"])
    
    with tab1:
        st.markdown("<h3>Provider Performance & Risk Analysis</h3>", unsafe_allow_html=True)
        

        # Data Visualization Enhancements
        st.divider()
        st.markdown("<h4>Interactive Data Visualization</h4>", unsafe_allow_html=True)
        # Drill-down: select provider to filter
        if 'claim_records' in st.session_state:
            df = st.session_state.claim_records
            provider = st.selectbox("Drill-down by Provider", ["All"] + sorted(df['Provider'].unique().tolist()))
            if provider != "All":
                df = df[df['Provider'] == provider]
            st.dataframe(df, width='stretch')
            # Custom metric creation
            st.markdown("**Custom Metric**")
            metric_col = st.selectbox("Select column for metric", df.columns)
            metric_func = st.selectbox("Metric function", ["Sum", "Mean", "Count"])
            if metric_func == "Sum":
                val = pd.to_numeric(df[metric_col].str.replace('$','').str.replace(',',''), errors='coerce').sum()
            elif metric_func == "Mean":
                val = pd.to_numeric(df[metric_col].str.replace('$','').str.replace(',',''), errors='coerce').mean()
            else:
                val = df[metric_col].count()
            st.metric(f"{metric_func} of {metric_col}", val)
            # Data export from visualizations
            st.download_button("Export This View (CSV)", data=df_to_csv_bytes(df), file_name="drilldown_export.csv", mime="text/csv")
            # Comparison views (YoY, MoM) - mock
            st.markdown("**Comparison View (Mock)**")
            st.line_chart([100,120,130,110,140,150], width='stretch')
            # Heatmap visualization (pattern detection)
            st.markdown("**Heatmap Visualization (Mock)**")
            import numpy as np
            import seaborn as sns
            import matplotlib.pyplot as plt
            arr = np.random.rand(10,10)
            fig, ax = plt.subplots()
            sns.heatmap(arr, ax=ax)
            st.pyplot(fig)
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("<h4 style='color: #c62828;'>⚠️ High-Risk Providers</h4>", unsafe_allow_html=True)
            providers = {
                "Provider": ["City Clinic", "Park Medical", "Downtown Hospital"],
                "Claims": [145, 128, 156],
                "Fraud Rate": ["12.4%", "11.2%", "14.8%"],
                "Risk": ["🔴 Critical", "🟠 High", "🔴 Critical"]
            }
            st.dataframe(providers, hide_index=True, width='stretch')
        
        with col2:
            st.markdown("<h4 style='color: #2e7d32;'>✅ Trusted Providers</h4>", unsafe_allow_html=True)
            legit_providers = {
                "Provider": ["Oak Hospital", "State Medical", "Regional Center"],
                "Claims": [234, 198, 176],
                "Fraud Rate": ["0.8%", "1.2%", "1.0%"],
                "Rating": ["⭐⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐"]
            }
            st.dataframe(legit_providers, hide_index=True, width='stretch')
    
    with tab2:
        st.markdown("<h3>Fraud Pattern Detection & Analysis</h3>", unsafe_allow_html=True)
        
        # Ensure session structures
        ensure_alerts_structs()

        st.markdown("<h4>Identified Suspicious Patterns</h4>", unsafe_allow_html=True)
        # Show existing patterns (managed by users)
        if st.session_state.patterns:
            st.table(pd.DataFrame(st.session_state.patterns))
        else:
            st.info("No custom patterns defined. Create rules below.")

        st.divider()

        # Pattern creation form
        st.markdown("<h4>Create / Edit Fraud Pattern Rule</h4>", unsafe_allow_html=True)
        with st.form("pattern_form"):
            pname = st.text_input("Rule name", placeholder="High Overbilling by Provider X")
            pfield = st.selectbox("Field", ["Provider", "Amount", "Status", "Risk"], index=0)
            pop = st.selectbox("Operator", ["contains", "equals", "gt"], index=0)
            pval = st.text_input("Value", placeholder="e.g., Oak Hospital or 5000")
            pseverity = st.selectbox("Severity", ["Low", "Medium", "High", "Critical"], index=2)
            pauto = st.checkbox("Auto action (auto-flag/hold)", value=False)
            paction = st.selectbox("Auto Action", ["flag", "hold", "notify"], index=0)
            submitted = st.form_submit_button("Save Pattern")
            if submitted and pname and pval:
                st.session_state.patterns.append({'name': pname, 'field': pfield, 'operator': pop, 'value': pval, 'severity': pseverity, 'action': paction if pauto else None})
                st.success("Pattern saved")

        st.divider()

        # Whitelist management
        st.markdown("<h4>Provider Whitelist</h4>", unsafe_allow_html=True)
        with st.form("whitelist_form"):
            wprov = st.text_input("Provider to whitelist", placeholder="Provider name")
            if st.form_submit_button("Add to Whitelist") and wprov:
                st.session_state.whitelist.append(wprov)
                st.success(f"{wprov} added to whitelist")
        if st.session_state.whitelist:
            st.markdown("**Whitelisted Providers**")
            st.write(st.session_state.whitelist)

        st.divider()

        # Active alerts summary
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("""
            <div style='background: #fff3e0; border-radius: 8px; padding: 15px; border-left: 4px solid #f57c00;'>
                <p style='margin: 0; color: #f57c00; font-weight: 600;'>⚠️ Active Alerts</p>
                <p style='margin: 8px 0 0 0; color: #666; font-size: 0.9rem;'>Alerts generated by pattern engine are listed in Alerts center</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background: #e3f2fd; border-radius: 8px; padding: 15px; border-left: 4px solid #0d47a1;'>
                <p style='margin: 0; color: #0d47a1; font-weight: 600;'>📊 Pattern Accuracy</p>
                <p style='margin: 8px 0 0 0; color: #666; font-size: 0.9rem;'>94% accuracy rate in fraud pattern detection based on historical data</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<h3>Predictive Risk Intelligence</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("""
            <div style='background: #f3e5f5; border-radius: 8px; padding: 15px; border-left: 4px solid #7b1fa2;'>
                <p style='margin: 0; color: #7b1fa2; font-weight: 600;'>🎯 Next Week Forecast</p>
                <ul style='margin: 10px 0 0 0; color: #666; font-size: 0.9rem; padding-left: 20px;'>
                    <li>5 high-risk claims expected</li>
                    <li>2 provider audits due</li>
                    <li>3 license renewals required</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background: #e8f5e9; border-radius: 8px; padding: 15px; border-left: 4px solid #2e7d32;'>
                <p style='margin: 0; color: #2e7d32; font-weight: 600;'>✅ Preventive Actions</p>
                <ul style='margin: 10px 0 0 0; color: #666; font-size: 0.9rem; padding-left: 20px;'>
                    <li>Tighten validation on 4 providers</li>
                    <li>Increase monitoring frequency</li>
                    <li>Schedule compliance reviews</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)


def show_alerts():
    """Alerts center: actionable alerts with severity, snooze, and auto-actions."""
    ensure_alerts_structs()
    st.markdown("<h2>Alerts Center</h2>", unsafe_allow_html=True)
    if not st.session_state.alerts:
        st.info("No active alerts.")
    else:
        for i, a in enumerate(st.session_state.alerts):
            sev = a.get('severity','Medium')
            color = '#ffb74d' if sev=='Medium' else ('#ef5350' if sev=='High' or sev=='Critical' else '#81c784')
            st.markdown(f"<div style='border-left:6px solid {color}; padding:8px; margin-bottom:8px;'><b>{a.get('title')}</b> — {a.get('message')}</div>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1,1,2])
            with col1:
                if st.button(f"Snooze {i}"):
                    a['snoozed'] = True
                    st.success("Alert snoozed")
            with col2:
                if st.button(f"Acknowledge {i}"):
                    a['ack'] = True
                    st.success("Alert acknowledged")
            with col3:
                if a.get('action'):
                    if st.button(f"Run Action {i}"):
                        st.info(f"Auto-action '{a['action']}' executed (simulated)")

    st.divider()
    # Custom alert rules (UI)
    st.markdown("<h3>Custom Alert Rules</h3>", unsafe_allow_html=True)
    with st.form("custom_alert_rule_form"):
        rule_name = st.text_input("Rule Name", "")
        rule_field = st.selectbox("Field", ["Amount", "Provider", "Risk", "Status"])
        rule_op = st.selectbox("Operator", [">", "<", "=", "contains"])
        rule_val = st.text_input("Value", "")
        rule_sev = st.selectbox("Severity", ["Low", "Medium", "High", "Critical"], index=2)
        rule_action = st.selectbox("Auto Action", ["flag", "hold", "notify"], index=0)
        if st.form_submit_button("Add Custom Rule") and rule_name and rule_val:
            st.session_state.patterns.append({'name': rule_name, 'field': rule_field, 'operator': rule_op, 'value': rule_val, 'severity': rule_sev, 'action': rule_action})
            st.success("Custom alert rule added.")
    # In-app notification center (alerts/reminders)
    st.markdown("<h3>Notification Center</h3>", unsafe_allow_html=True)
    if 'notifications' in st.session_state and st.session_state.notifications:
        for n in st.session_state.notifications[-10:]:
            st.info(f"🔔 {n}")
    # Reminders for pending actions
    if 'pending_tasks' in st.session_state and st.session_state.pending_tasks:
        for t in st.session_state.pending_tasks:
            if t['status'] == 'Open':
                st.warning(f"Reminder: Task for claim {t['claim']} assigned to {t['assigned_to']} is still open.")
    st.markdown("<h3>Create Alert Template</h3>", unsafe_allow_html=True)
    with st.form("alert_template"):
        atitle = st.text_input("Template title")
        amsg = st.text_area("Template message")
        asev = st.selectbox("Severity", ["Low","Medium","High","Critical"], index=1)
        aaction = st.selectbox("Auto-action", ["none","flag","hold","notify"], index=0)
        if st.form_submit_button("Save Template") and atitle:
            st.session_state.alerts.append({'title': atitle, 'message': amsg, 'severity': asev, 'action': aaction if aaction!='none' else None, 'created': datetime.now().isoformat()})
            st.success("Alert template created (and added to active alerts)")


def show_user_management():
    """Professional user and access management (Admin only)"""
    
    st.markdown("""
    <div style='margin: -1rem -1rem 2rem -1rem; padding: 2rem 2rem; background: linear-gradient(135deg, #003d82 0%, #0052b3 100%);'>
        <h1 style='color: white; margin: 0; font-size: 2rem;'>👥 User & Access Management</h1>
        <p style='color: rgba(255, 255, 255, 0.9); margin: 0.5rem 0 0 0;'>Administrator panel for user accounts and permissions</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Active Users", "Roles & Permissions", "Activity Log"])
    
    with tab1:
        st.markdown("<h3>User Account Management</h3>", unsafe_allow_html=True)
        

    # Compliance & Audit Trail
    st.divider()
    st.markdown("<h4>Compliance & Audit Trail</h4>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate Compliance Report (Simulated)"):
            st.success("Compliance report generated and available for download.")
    with col2:
        if st.button("Export Anonymized Log (Simulated)"):
            # For demo, mask user names
            anon_log = df_log.copy()
            anon_log['User'] = anon_log['User'].apply(lambda x: x[0]+"***")
            csv_bytes = df_to_csv_bytes(anon_log)
            st.download_button("Download Anonymized Log", data=csv_bytes, file_name="anonymized_audit_log.csv", mime="text/csv")

    # Audit signatures for critical decisions (mock)
    st.markdown("**Audit Signatures**")
    if st.button("Sign Off on Critical Actions (Simulated)"):
        st.success(f"Audit signature recorded by {st.session_state.user['username']} at {datetime.now().isoformat()}")
        st.markdown("<h4>Active User Accounts</h4>", unsafe_allow_html=True)
        users_data = {
            "Username": ["admin", "tpa_manager", "claims_reviewer", "auditor"],
            "Full Name": ["System Administrator", "John Manager", "Sarah Reviewer", "Mike Auditor"],
            "Role": ["Administrator", "TPA Manager", "Claims Reviewer", "Auditor"],
            "Status": ["🟢 Active", "🟢 Active", "🟡 Inactive", "🟢 Active"],
            "Last Login": ["Today 2:45 PM", "2 hours ago", "5 days ago", "Today 10:30 AM"],
            "Actions": ["Edit", "Edit", "Edit", "Edit"]
        }
        st.dataframe(users_data, width='stretch', hide_index=True)
        
        st.divider()
        
        st.markdown("<h4 style='color: #0d47a1;'>Create New User Account</h4>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3, gap="medium")
        
        with col1:
            new_username = st.text_input("Username", placeholder="john.smith", help="Unique username for login")
        
        with col2:
            new_password = st.text_input("Temporary Password", type="password", placeholder="••••••••", help="User will be asked to change on first login")
        
        with col3:
            new_role = st.selectbox("Assign Role", list(SimpleAuth.ROLES.keys()), help="Select user role")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✚ Create User", width='stretch'):
                st.markdown("""
                <div style='background: #e8f5e9; border-radius: 8px; padding: 15px; border-left: 4px solid #2e7d32;'>
                    <p style='margin: 0; color: #2e7d32; font-weight: 600;'>✅ User Created Successfully</p>
                    <p style='margin: 5px 0 0 0; color: #666; font-size: 0.9rem;'>User account for '<b>john.smith</b>' has been created with <b>TPA Manager</b> role</p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<h3>Role Definitions & Permissions</h3>", unsafe_allow_html=True)
        
        for role, perms in SimpleAuth.ROLES.items():
            with st.expander(f"**{role.replace('_', ' ').title()}** Role Definition"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<b>Assigned Permissions:</b>", unsafe_allow_html=True)
                    for perm in perms:
                        st.markdown(f"✓ {perm.replace('_', ' ').title()}")
                
                with col2:
                    st.markdown("<b>Role Description:</b>", unsafe_allow_html=True)
                    descriptions = {
                        "admin": "Full system access. Can manage users, view all data, and configure system settings.",
                        "tpa_manager": "Can access claims, reports, and analytics. Cannot manage users.",
                        "claims_reviewer": "Can view and review claims. Limited to dashboard and claims analysis.",
                        "auditor": "Read-only access to reports and analytics for compliance monitoring."
                    }
                    st.write(descriptions.get(role, ""))
    
    with tab3:
        st.markdown("<h3>User Activity & Audit Log</h3>", unsafe_allow_html=True)
        
        # Date filter
        col1, col2 = st.columns(2)
        with col1:
            log_filter = st.selectbox("Filter by Activity", ["All", "Login", "Claims Access", "Report Download", "Settings Change"])
        with col2:
            log_date = st.date_input("From date", datetime.now() - timedelta(days=7))
        
        st.divider()
        activity_log = {
            "Timestamp": ["Today 2:45 PM", "Today 2:30 PM", "Today 1:15 PM", "Today 10:30 AM", "Yesterday 4:20 PM"],
            "User": ["admin", "tpa_manager", "claims_reviewer", "auditor", "tpa_manager"],
            "Activity": ["Login", "Claim Analysis", "Report Download", "Login", "Batch Processing"],
            "Status": ["✅ Success", "✅ Complete", "✅ Complete", "✅ Success", "✅ Complete"],
            "Details": ["IP: 192.168.1.100", "Claim ID: CLM-2025-001", "5 reports downloaded", "IP: 192.168.1.105", "10 claims processed"]
        }

        df_log = pd.DataFrame(activity_log)

        # Advanced filters
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            user_filter = st.multiselect("User(s)", sorted(df_log['User'].unique().tolist()), default=[])
        with col_b:
            action_filter = st.multiselect("Action Type(s)", sorted(df_log['Activity'].unique().tolist()), default=[])
        with col_c:
            date_range = st.date_input("Date Range", [datetime.now() - timedelta(days=7), datetime.now()])

        filtered_log = df_log.copy()
        # Apply advanced filters
        if user_filter:
            filtered_log = filtered_log[filtered_log['User'].isin(user_filter)]
        if action_filter:
            filtered_log = filtered_log[filtered_log['Activity'].isin(action_filter)]
        if isinstance(date_range, list) and len(date_range) == 2:
            # For demo, skip actual date parsing, but in real use, parse and filter Timestamp
            pass

        st.dataframe(filtered_log, width='stretch', hide_index=True)

        # Export audit log
        try:
            csv_bytes = df_to_csv_bytes(filtered_log)
            excel_bytes = df_to_excel_bytes(filtered_log)
            st.download_button("Export Audit CSV", data=csv_bytes, file_name="audit_log.csv", mime="text/csv")
            st.download_button("Export Audit Excel", data=excel_bytes, file_name="audit_log.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception:
            pass

        # Export to external system (simulate webhook)
        if st.button("Send to Security System (Simulated)"):
            st.info("Audit log sent to external security system (simulated)")

        st.divider()

        # User Behavior Analytics Dashboard
        st.markdown("<h4>User Behavior Analytics</h4>", unsafe_allow_html=True)
        # Example: login frequency, action counts, anomaly detection (mock)
        login_counts = df_log[df_log['Activity'].str.contains("Login")]['User'].value_counts()
        st.bar_chart(login_counts, width='stretch')
        action_counts = df_log['Activity'].value_counts()
        st.bar_chart(action_counts, width='stretch')
        st.markdown("**Recent Anomalies (Mock):**")
        st.write("- Multiple failed logins for user 'claims_reviewer' on 2025-12-28\n- Unusual report downloads by 'auditor' on 2025-12-27")


# ============================================================================
# MAIN APPLICATION FLOW
# ============================================================================

def main():
    """Main application flow with professional UI"""
    
    if not st.session_state.authenticated:
        show_login_page()
    else:
        # Authenticated - Show sidebar and main content
        with st.sidebar:
            # Professional sidebar header with Assan branding
            st.markdown("""
            <div style='background: linear-gradient(135deg, #003d82 0%, #0052b3 100%); border-radius: 8px; padding: 15px; margin-bottom: 20px;'>
                <p style='color: white; margin: 0; font-size: 0.9rem; font-weight: 600;'>🏢 ASSAN ONE INTELLIGENCE</p>
                <p style='color: white; margin: 5px 0 0 0; font-size: 0.9rem; font-weight: 600;'>👤 {}</p>
                <p style='color: rgba(255, 255, 255, 0.8); margin: 3px 0 0 0; font-size: 0.8rem;'>📍 {}</p>
            </div>
            """.format(
                st.session_state.user['username'].replace('_', ' ').title(),
                st.session_state.user['role'].replace('_', ' ').title()
            ), unsafe_allow_html=True)
            
            st.divider()
            
            # Navigation menu
            st.markdown("<b>NAVIGATION MENU</b>", unsafe_allow_html=True)
            
            permissions = st.session_state.user.get('permissions', [])
            
            if st.button("📊 Dashboard", width='stretch', key="nav_dashboard"):
                st.session_state.current_page = "dashboard"
                st.rerun()
            
            if "claims" in permissions:
                if st.button("📄 Claims Management", width='stretch', key="nav_claims"):
                    st.session_state.current_page = "claims"
                    st.rerun()
            
            if "reports" in permissions:
                if st.button("📈 Reports", width='stretch', key="nav_reports"):
                    st.session_state.current_page = "reports"
                    st.rerun()
            
            if "analytics" in permissions:
                if st.button("🔍 Analytics", width='stretch', key="nav_analytics"):
                    st.session_state.current_page = "analytics"
                    st.rerun()

            # Alerts center (available for all roles by default)
            if st.button("🔔 Alerts", width='stretch', key="nav_alerts"):
                st.session_state.current_page = "alerts"
                st.rerun()
            
            if "users" in permissions:
                if st.button("👥 User Management", width='stretch', key="nav_users"):
                    st.session_state.current_page = "users"
                    st.rerun()
            
            st.divider()
            
            st.markdown("<b>QUICK INFO</b>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style='background: #f8f9fa; border-radius: 6px; padding: 10px; font-size: 0.85rem; color: #666; border: 1px solid #e0e0e0;'>
                <p style='margin: 0;'><b>Last Login:</b><br/>Today 2:45 PM</p>
                <p style='margin: 10px 0 0 0;'><b>Status:</b><br/>✅ All Systems Operational</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
            st.markdown("<b style='color: #4a9eff;'>SETTINGS</b>", unsafe_allow_html=True)
            
            # Theme toggle
            st.markdown("<p style='margin: 0 0 10px 0; font-size: 0.9rem;'>🌓 Theme</p>", unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1], gap="small")
            with col1:
                theme_choice = st.radio(
                    "Select theme",
                    ["Light", "Dark"],
                    horizontal=True,
                    label_visibility="collapsed"
                )
            
            if theme_choice == "Dark":
                if st.session_state.theme != "dark":
                    st.session_state.theme = "dark"
                    st.rerun()
            else:
                if st.session_state.theme != "light":
                    st.session_state.theme = "light"
                    st.rerun()
            
            st.divider()
            
            col1, col2 = st.columns(2, gap="small")
            with col1:
                if st.button("⚙️ Settings", width='stretch'):
                    st.info("Settings panel coming soon")
            
            with col2:
                if st.button("🚪 Logout", width='stretch'):
                    st.session_state.authenticated = False
                    st.session_state.user = None
                    st.rerun()
            
            st.markdown("---")
            st.markdown(
                "<p style='text-align: center; color: #999; font-size: 0.75rem;'>"
                "Assan One Intelligence<br/>TPA Management Platform<br/>© 2025 All Rights Reserved"
                "</p>",
                unsafe_allow_html=True
            )
        
        # Display selected page
        if st.session_state.current_page == "dashboard":
            show_dashboard()
        elif st.session_state.current_page == "claims":
            show_claims_management()
        elif st.session_state.current_page == "reports":
            show_reports()
        elif st.session_state.current_page == "analytics":
            show_analytics()
        elif st.session_state.current_page == "alerts":
            show_alerts()
        elif st.session_state.current_page == "users":
            show_user_management()


if __name__ == "__main__":
    main()
