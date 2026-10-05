"""Page 1 — Overview / Home"""
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

SAVED = Path(__file__).parent.parent / "models" / "saved"
DATA_DIR = Path(__file__).parent.parent / "data"


def _load_metrics():
    p = SAVED / "metrics.pkl"
    if p.exists():
        return pd.DataFrame(joblib.load(str(p)))
    return None


def _load_config():
    import json
    p = SAVED / "training_config.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def render():
    st.markdown("# PolyShield — Overview")
    st.markdown("Detect polymorphic malware using XGBoost with feature interaction constraints. "
                "Five trained models analyse 33 PE/behavioral/network features to classify executables.")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    config = _load_config()
    metrics_df = _load_metrics()

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n = f"{config['n_samples']:,}" if config else "—"
        st.markdown(f'<div class="stat-card"><div class="stat-val">{n}</div><div class="stat-lbl">Samples</div></div>', unsafe_allow_html=True)
    with c2:
        if config:
            mal = int(config["n_samples"] * config["malware_ratio"])
            st.markdown(f'<div class="stat-card"><div class="stat-val">{mal:,}</div><div class="stat-lbl">Malware Samples</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="stat-card"><div class="stat-val">—</div><div class="stat-lbl">Malware Samples</div></div>', unsafe_allow_html=True)
    with c3:
        if metrics_df is not None and "roc_auc" in metrics_df.columns:
            best = metrics_df["roc_auc"].max()
            st.markdown(f'<div class="stat-card"><div class="stat-val">{best:.4f}</div><div class="stat-lbl">Best AUC</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="stat-card"><div class="stat-val">—</div><div class="stat-lbl">Best AUC</div></div>', unsafe_allow_html=True)
    with c4:
        nf = config["n_features"] if config else "—"
        st.markdown(f'<div class="stat-card"><div class="stat-val">{nf}</div><div class="stat-lbl">Features</div></div>', unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Model results table ───────────────────────────────────────────────────
    st.markdown("### Trained Model Results")
    if metrics_df is not None and "model_name" in metrics_df.columns:
        show = metrics_df.set_index("model_name")
        cols = [c for c in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc", "mcc"] if c in show.columns]
        st.dataframe(show[cols].round(4).sort_values("roc_auc", ascending=False))
    else:
        st.info("No model results yet. Run `python models/train_all_models.py` to train.")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── How it works ──────────────────────────────────────────────────────────
    st.markdown("### How it works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card">
          <strong>1. Feature Extraction</strong><br>
          <span style="color:#9ca3af; font-size:0.85rem">33 features from PE headers, section entropy, imports, strings, behavioral and network indicators.</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
          <strong>2. Constrained Classification</strong><br>
          <span style="color:#9ca3af; font-size:0.85rem">XGBoost interaction constraints prevent cross-domain feature leakage — key to detecting polymorphic variants.</span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="card">
          <strong>3. Multi-Model Voting</strong><br>
          <span style="color:#9ca3af; font-size:0.85rem">Five models (XGBoost, RF, LightGBM, CatBoost, Hybrid IF+GB) all vote on every sample.</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Constraint groups ─────────────────────────────────────────────────────
    st.markdown("### Feature Interaction Constraints")
    st.markdown("XGBoost restricts feature interactions to within these 6 domain groups. "
                "This prevents overfitting to cross-domain correlations that break on novel malware variants.")

    from utils.preprocessing import FEATURE_GROUPS
    cols = st.columns(3)
    colors = ["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444", "#ec4899"]
    for i, (group, feats) in enumerate(FEATURE_GROUPS.items()):
        with cols[i % 3]:
            c = colors[i % len(colors)]
            tags = ", ".join(feats)
            st.markdown(f"""
            <div class="card" style="border-left:3px solid {c}; margin-bottom:0.8rem;">
              <div style="color:{c}; font-weight:600; font-size:0.85rem; margin-bottom:0.3rem;">{group}</div>
              <div style="color:#6b7280; font-size:0.75rem; line-height:1.5;">{tags}</div>
            </div>
            """, unsafe_allow_html=True)
