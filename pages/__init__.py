"""Pages package — shared helpers for all dashboard pages."""
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List

SAVED_DIR = Path(__file__).parent.parent / "models" / "saved"
DATA_DIR = Path(__file__).parent.parent / "data"

# ─── Color palette ────────────────────────────────────────────────────────────
CYAN = "#00D4FF"
PURPLE = "#7C3AED"
BENIGN_COLOR = "#10B981"
MALWARE_COLOR = "#EF4444"
WARNING_COLOR = "#F59E0B"
BG = "#0A0E1A"
CARD = "#111827"
TEXT_SEC = "#9CA3AF"

MODEL_COLORS = {
    "XGBoost": CYAN,
    "Random Forest": PURPLE,
    "LightGBM": BENIGN_COLOR,
    "CatBoost": WARNING_COLOR,
    "Hybrid (IF+GB)": MALWARE_COLOR,
}

# ─── Caching helpers ──────────────────────────────────────────────────────────

@st.cache_resource
def load_model(name: str):
    path = SAVED_DIR / f"{name}.pkl"
    if path.exists():
        return joblib.load(str(path))
    return None


@st.cache_data
def load_metrics() -> Optional[List[Dict]]:
    path = SAVED_DIR / "metrics.pkl"
    if path.exists():
        return joblib.load(str(path))
    return None


@st.cache_data
def load_metrics_df() -> Optional[pd.DataFrame]:
    metrics = load_metrics()
    if metrics:
        df = pd.DataFrame(metrics)
        if "model_name" in df.columns:
            df = df.set_index("model_name")
        return df
    return None


@st.cache_data
def load_roc_data() -> Optional[Dict]:
    path = SAVED_DIR / "roc_data.pkl"
    return joblib.load(str(path)) if path.exists() else None


@st.cache_data
def load_pr_data() -> Optional[Dict]:
    path = SAVED_DIR / "pr_data.pkl"
    return joblib.load(str(path)) if path.exists() else None


@st.cache_data
def load_shap_data() -> Optional[Dict]:
    path = SAVED_DIR / "shap_data.pkl"
    return joblib.load(str(path)) if path.exists() else None


@st.cache_data
def load_optuna_history() -> Optional[Dict]:
    path = SAVED_DIR / "optuna_history.pkl"
    return joblib.load(str(path)) if path.exists() else None


@st.cache_data
def load_feature_importance() -> Optional[Dict]:
    path = SAVED_DIR / "feature_importance.pkl"
    return joblib.load(str(path)) if path.exists() else None


@st.cache_data
def load_dataset(n_rows: int = 5000) -> Optional[pd.DataFrame]:
    """Load a subset of the dataset for EDA (avoids loading full 50k rows)."""
    path = DATA_DIR / "synthetic_malware_data.csv"
    if path.exists():
        return pd.read_csv(path, nrows=n_rows)
    # Try test sample
    path2 = SAVED_DIR / "test_sample.csv"
    if path2.exists():
        return pd.read_csv(path2)
    return None


@st.cache_data
def load_training_config() -> Optional[Dict]:
    import json
    path = SAVED_DIR / "training_config.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_ablation_results() -> Optional[Dict]:
    path = SAVED_DIR / "ablation_results.pkl"
    return joblib.load(str(path)) if path.exists() else None

# ─── UI Helper functions ──────────────────────────────────────────────────────

def gradient_divider():
    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)


def metric_card(label: str, value: str, delta: str = "", delta_positive: bool = True,
                icon: str = ""):
    delta_class = "positive" if delta_positive else "negative"
    delta_html = f'<div class="metric-delta {delta_class}">{delta}</div>' if delta else ""
    icon_html = f'<span style="font-size:1.5rem">{icon}</span>' if icon else ""
    st.markdown(f"""
    <div class="metric-card">
        {icon_html}
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def models_not_trained_warning():
    st.warning("""
    ⚠️ **Models have not been trained yet.**

    Please run the training pipeline first:
    ```bash
    python models/train_all_models.py
    ```
    This will generate and train all 5 models (~5-15 minutes depending on hardware).
    """)


def section_header(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div class="section-header gradient-text">{title}</div>
    {"<p style='color:#9CA3AF; margin-top:-0.5rem; margin-bottom:1rem;'>" + subtitle + "</p>" if subtitle else ""}
    """, unsafe_allow_html=True)
