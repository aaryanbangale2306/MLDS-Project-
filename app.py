"""
app.py — Main Streamlit Dashboard Entry Point
XGBoost Classifier with Feature Interaction Constraints
for Low-Signature Polymorphic Malware Detection
"""

import streamlit as st
from pathlib import Path

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PolyShield — Polymorphic Malware Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/aaryanbangale2306/MLDS-Project",
        "Report a bug": "https://github.com/aaryanbangale2306/MLDS-Project/issues",
        "About": "XGBoost Malware Detection System v1.0",
    },
)

# ─── Load Global CSS ──────────────────────────────────────────────────────────
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─── Import Fonts ─────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    code, pre, .metric-value { font-family: 'JetBrains Mono', monospace !important; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo / Title
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem 0;">
        <div style="font-size: 2.5rem; margin-bottom: 0.3rem;">🛡️</div>
        <div style="background: linear-gradient(135deg, #00D4FF, #7C3AED);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    font-size: 1.4rem; font-weight: 800; letter-spacing: -0.5px;">
            PolyShield
        </div>
        <div style="color: #9CA3AF; font-size: 0.75rem; margin-top: 0.2rem;">
            Polymorphic Malware Detection v1.0
        </div>
    </div>
    <hr style="border: 1px solid #1F2937; margin: 0 0 1rem 0;">
    """, unsafe_allow_html=True)

    # Navigation
    st.markdown('<p style="color:#9CA3AF; font-size:0.75rem; font-weight:600; letter-spacing:1px; text-transform:uppercase; margin-bottom:0.5rem;">Navigation</p>', unsafe_allow_html=True)

    pages = {
        "🏠 Overview": "pages/01_overview.py",
        "📊 Model Comparison": "pages/02_model_comparison.py",
        "🔬 XGBoost Deep Dive": "pages/03_xgboost_deep_dive.py",
        "🧪 Live Scanner": "pages/04_live_scanner.py",
        "📈 Dataset & EDA": "pages/05_eda.py",
        "🏗️ Feature Interactions": "pages/06_feature_interactions.py",
        "ℹ️ About": "pages/07_about.py",
    }

    selected_page = st.radio(
        "Navigate",
        options=list(pages.keys()),
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border: 1px solid #1F2937; margin: 1rem 0;'>", unsafe_allow_html=True)

    # Status indicator
    from pathlib import Path
    models_dir = Path(__file__).parent / "models" / "saved"
    models_exist = (models_dir / "xgboost_model.pkl").exists()

    if models_exist:
        st.markdown("""
        <div style="background:#10B98120; border:1px solid #10B981; border-radius:8px; padding:0.6rem 0.8rem;">
            <span style="color:#10B981; font-size:0.8rem;">● Models Ready</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#EF444420; border:1px solid #EF4444; border-radius:8px; padding:0.6rem 0.8rem;">
            <span style="color:#EF4444; font-size:0.8rem;">⚠ Models Not Trained</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="color:#9CA3AF; font-size:0.75rem; margin-top:0.5rem;">
        Run <code>python models/train_all_models.py</code> to train models.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border: 1px solid #1F2937; margin: 1rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#9CA3AF; font-size:0.7rem; text-align:center;">
        Built with XGBoost + SHAP<br>
        <a href="https://github.com/aaryanbangale2306/MLDS-Project"
           style="color:#00D4FF; text-decoration:none;">GitHub ↗</a>
    </div>
    """, unsafe_allow_html=True)

# ─── Route to Selected Page ───────────────────────────────────────────────────
page_map = {
    "🏠 Overview": "pages.page_01_overview",
    "📊 Model Comparison": "pages.page_02_model_comparison",
    "🔬 XGBoost Deep Dive": "pages.page_03_xgboost_deep_dive",
    "🧪 Live Scanner": "pages.page_04_live_scanner",
    "📈 Dataset & EDA": "pages.page_05_eda",
    "🏗️ Feature Interactions": "pages.page_06_feature_interactions",
    "ℹ️ About": "pages.page_07_about",
}

import importlib
try:
    page_module = importlib.import_module(page_map[selected_page])
    page_module.render()
except ModuleNotFoundError as e:
    st.error(f"Page module not found: {e}")
except Exception as e:
    st.error(f"Error loading page: {e}")
    st.exception(e)
