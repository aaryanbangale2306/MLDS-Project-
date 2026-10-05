"""
PolyShield — Polymorphic Malware Detection System
Main entry point. Uses Streamlit's native multi-page system.
"""
import streamlit as st

st.set_page_config(
    page_title="PolyShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global styles — clean, professional, not over-designed
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  color: #e5e7eb;
}
.stApp { background: #0f1117; }

/* Sidebar */
[data-testid="stSidebar"] {
  background: #161b27;
  border-right: 1px solid #1f2937;
}
[data-testid="stSidebar"] .stRadio label {
  font-size: 0.9rem;
  padding: 0.35rem 0.5rem;
  border-radius: 5px;
  cursor: pointer;
}

/* Main content area */
section.main > div.block-container {
  padding: 1.5rem 2rem;
  max-width: 1200px;
}

/* Headings */
h1 { font-size: 1.7rem; font-weight: 700; color: #f9fafb; margin-bottom: 0.25rem; }
h2 { font-size: 1.2rem; font-weight: 600; color: #f3f4f6; }
h3 { font-size: 1rem; font-weight: 600; color: #d1d5db; }

/* Buttons */
.stButton > button {
  background: #2563eb;
  color: #fff;
  border: none;
  border-radius: 6px;
  font-weight: 500;
  padding: 0.45rem 1.2rem;
  font-size: 0.875rem;
  transition: background 0.2s;
}
.stButton > button:hover { background: #1d4ed8; }

/* Cards */
.card {
  background: #1e2433;
  border: 1px solid #2d3748;
  border-radius: 8px;
  padding: 1.1rem 1.3rem;
}
.stat-card {
  background: #1e2433;
  border: 1px solid #2d3748;
  border-radius: 8px;
  padding: 1rem 1.2rem;
  text-align: center;
}
.stat-val { font-size: 1.8rem; font-weight: 700; color: #60a5fa; font-family: 'JetBrains Mono'; }
.stat-lbl { font-size: 0.78rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 0.2rem; }

/* Divider */
.divider { border: none; border-top: 1px solid #2d3748; margin: 1.2rem 0; }

/* Status pills */
.pill-green { background: #052e16; color: #4ade80; border: 1px solid #16a34a; border-radius: 4px; padding: 0.15rem 0.6rem; font-size: 0.78rem; font-weight: 500; }
.pill-red   { background: #3b0a0a; color: #f87171; border: 1px solid #dc2626; border-radius: 4px; padding: 0.15rem 0.6rem; font-size: 0.78rem; font-weight: 500; }
.pill-blue  { background: #1e3a5f; color: #93c5fd; border: 1px solid #3b82f6; border-radius: 4px; padding: 0.15rem 0.6rem; font-size: 0.78rem; font-weight: 500; }
.pill-yellow{ background: #2d1f00; color: #fbbf24; border: 1px solid #d97706; border-radius: 4px; padding: 0.15rem 0.6rem; font-size: 0.78rem; font-weight: 500; }

/* Tables */
[data-testid="stDataFrame"] { border: 1px solid #2d3748; border-radius: 8px; }

/* Selectbox, slider */
[data-baseweb="select"] { background: #1e2433; }
.stSelectbox label { font-size: 0.85rem; color: #9ca3af; }
.stSlider label { font-size: 0.85rem; color: #9ca3af; }
.stCheckbox label { font-size: 0.85rem; color: #d1d5db; }
.stNumberInput label { font-size: 0.85rem; color: #9ca3af; }

/* Alerts */
[data-testid="stAlert"] { border-radius: 6px; }

/* Code */
code { background: #1e2433; color: #a5f3fc; padding: 0.1rem 0.3rem; border-radius: 3px; font-size: 0.82rem; }

/* Hide the auto-generated Streamlit MPA page list */
[data-testid="stSidebarNav"] { display: none !important; }
</style>""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1.2rem 0;">
      <div style="font-size:1.4rem; font-weight:700; color:#f9fafb;">🛡️ PolyShield</div>
      <div style="font-size:0.75rem; color:#6b7280; margin-top:0.2rem;">Malware Detection System</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Overview", "Scanner", "Analytics", "Models & Method"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#2d3748; margin:1rem 0'>", unsafe_allow_html=True)

    # Model status indicator
    from pathlib import Path
    trained = (Path(__file__).parent / "models" / "saved" / "xgboost_model.pkl").exists()
    if trained:
        st.markdown('<span class="pill-green">● Models ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill-red">⚠ Models not trained</span>', unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:auto; padding-top:2rem; font-size:0.72rem; color:#4b5563;">
      XGBoost · LightGBM · CatBoost<br>Random Forest · Hybrid (IF+GB)
    </div>
    """, unsafe_allow_html=True)

# Route to the selected page
if page == "Overview":
    import pages.pg1_overview as pg
    pg.render()
elif page == "Scanner":
    import pages.pg2_scanner as pg
    pg.render()
elif page == "Analytics":
    import pages.pg3_analytics as pg
    pg.render()
elif page == "Models & Method":
    import pages.pg4_models as pg
    pg.render()
