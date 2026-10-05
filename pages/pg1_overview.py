"""Page 1 — Overview"""
import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

SAVED = Path(__file__).parent.parent / "models" / "saved"


@st.cache_data
def _load_metrics():
    p = SAVED / "metrics.pkl"
    if p.exists():
        return pd.DataFrame(joblib.load(str(p)))
    return None


@st.cache_data
def _load_config():
    p = SAVED / "training_config.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def render():
    st.markdown("# PolyShield")
    st.markdown("**Polymorphic Malware Detection System**")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Problem statement ─────────────────────────────────────────────────────
    st.markdown("""
> **Problem:** Traditional signature-based malware detection struggles when malicious software
> changes its byte-level structure (polymorphism). Each variant looks different, evading fixed rules.
>
> **Approach:** PolyShield trains machine-learning classifiers on *behavioural and structural features*
> extracted from PE executables. Features are grouped by domain (PE headers, section entropy, imports,
> strings, behavioral, network) and XGBoost interaction constraints prevent cross-domain feature leakage
> — improving generalisation to novel variants.
    """)

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
            best_name = metrics_df.loc[metrics_df["roc_auc"].idxmax(), "model_name"]
            st.markdown(f'<div class="stat-card"><div class="stat-val">{best:.4f}</div><div class="stat-lbl">Best AUC ({best_name})</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="stat-card"><div class="stat-val">—</div><div class="stat-lbl">Best AUC</div></div>', unsafe_allow_html=True)
    with c4:
        nf = config["n_features"] if config else "—"
        st.markdown(f'<div class="stat-card"><div class="stat-val">{nf}</div><div class="stat-lbl">Features</div></div>', unsafe_allow_html=True)

    st.markdown("""<div style="font-size:0.75rem; color:#6b7280; margin-top:0.5rem;">
    Note: Dataset is a <strong>synthetic benchmark</strong> generated with EMBER-style feature distributions.
    Metrics reflect performance on this synthetic test set.
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Detection flow ─────────────────────────────────────────────────────────
    st.markdown("### Detection pipeline")
    steps = [
        ("PE Sample", "Executable or extracted features in CSV format"),
        ("Feature Extraction", "33 features across 6 domain groups"),
        ("Preprocessing", "RobustScaler normalisation on training distribution"),
        ("ML Models", "5 classifiers score the sample independently"),
        ("Prediction", "Malware probability per model"),
        ("Explanation", "SHAP values showing which features drove the decision"),
    ]
    cols = st.columns(len(steps))
    for col, (title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="card" style="min-height:100px; text-align:center;">
              <div style="color:#60a5fa; font-size:0.75rem; font-weight:600; margin-bottom:0.3rem;">{title}</div>
              <div style="color:#6b7280; font-size:0.72rem; line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Model results table ───────────────────────────────────────────────────
    st.markdown("### Trained model results")
    st.caption("Evaluated on 7,500-sample held-out test set (synthetic benchmark).")
    if metrics_df is not None and "model_name" in metrics_df.columns:
        show = metrics_df.set_index("model_name")
        cols_show = [c for c in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc", "mcc"] if c in show.columns]
        st.dataframe(show[cols_show].round(4).sort_values("roc_auc", ascending=False))
    else:
        st.info("No model results yet. Run `python models/train_all_models.py` to train.")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Feature groups ─────────────────────────────────────────────────────────
    st.markdown("### Feature groups and interaction constraints")
    st.markdown("""<span style="font-size:0.82rem; color:#9ca3af;">
    XGBoost is constrained so that decision splits can only combine features within the same domain group.
    This prevents the model from learning cross-domain shortcuts that break on polymorphic variants.
    </span>""", unsafe_allow_html=True)

    from utils.preprocessing import FEATURE_GROUPS
    cols = st.columns(3)
    colors = ["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444", "#ec4899"]
    for i, (group, feats) in enumerate(FEATURE_GROUPS.items()):
        with cols[i % 3]:
            c = colors[i % len(colors)]
            tags = ", ".join(feats)
            st.markdown(f"""
            <div class="card" style="border-left:3px solid {c}; margin-bottom:0.8rem;">
              <div style="color:{c}; font-weight:600; font-size:0.85rem; margin-bottom:0.3rem;">{group} ({len(feats)} features)</div>
              <div style="color:#6b7280; font-size:0.75rem; line-height:1.5;">{tags}</div>
            </div>
            """, unsafe_allow_html=True)



