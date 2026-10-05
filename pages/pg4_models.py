"""Page 4 — Models & Method"""
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

SAVED = Path(__file__).parent.parent / "models" / "saved"


def _load_config():
    import json
    p = SAVED / "training_config.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def _load_metrics():
    p = SAVED / "metrics.pkl"
    if p.exists():
        return pd.DataFrame(joblib.load(str(p)))
    return None


def render():
    st.markdown("# Models & Method")
    st.markdown("Architecture, training pipeline, and the reasoning behind interaction constraints.")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    config = _load_config()

    # ── Pipeline summary ──────────────────────────────────────────────────────
    st.markdown("### Training pipeline")
    steps = [
        ("Data generation", "50 000 synthetic EMBER-style samples with realistic correlations, ~20% malware ratio, and 1% label noise."),
        ("Feature engineering", "33 features across 6 domain groups (PE Header, Section, Import, String, Behavioral, Network)."),
        ("Optuna tuning", "25-trial Bayesian optimisation (TPE sampler) on XGBoost, maximising ROC-AUC on validation set."),
        ("Model training", "5 models trained with class-imbalance handling: XGBoost (constrained), Random Forest, LightGBM, CatBoost, Hybrid IF+GBC."),
        ("Evaluation", "Full metric suite: Accuracy, Precision, Recall, F1, F2, ROC-AUC, PR-AUC, MCC, FPR, FNR."),
        ("SHAP analysis", "TreeExplainer on XGBoost for feature importance, waterfall plots, and dependence analysis."),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div class="card" style="margin-bottom:0.6rem; border-left:3px solid #3b82f6;">
          <span style="color:#60a5fa; font-weight:600;">{i}.</span> <strong>{title}</strong><br>
          <span style="color:#9ca3af; font-size:0.84rem;">{desc}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Model descriptions ────────────────────────────────────────────────────
    st.markdown("### Model details")

    models_info = [
        ("XGBoost (primary)", "#3b82f6", "interaction_constraints",
         "Uses `interaction_constraints` to prevent cross-domain feature splits. "
         "Tuned with Optuna (25 trials). Handles class imbalance via `scale_pos_weight`."),
        ("Random Forest", "#8b5cf6", "Ensemble baseline",
         "500 trees, sqrt feature sampling, `class_weight='balanced'`. Provides stable baseline with low variance."),
        ("LightGBM", "#10b981", "Leaf-wise growth",
         "GBDT boosting with leaf-wise splits, 63 num_leaves, early stopping at 50 rounds. Fast training."),
        ("CatBoost", "#f59e0b", "Ordered boosting",
         "Ordered boosting prevents target leakage. 500 iterations, depth 6. No need for manual encoding."),
        ("Hybrid (IF+GB)", "#ef4444", "Two-stage detection",
         "Stage 1: Isolation Forest detects anomalies from benign baseline. "
         "Stage 2: GBC trained on original features + IF anomaly score as meta-feature."),
    ]
    for name, color, subtitle, desc in models_info:
        st.markdown(f"""
        <div class="card" style="border-left:3px solid {color}; margin-bottom:0.6rem;">
          <div style="color:{color}; font-weight:600; font-size:0.9rem;">{name}</div>
          <div style="color:#6b7280; font-size:0.78rem; margin-bottom:0.3rem;">{subtitle}</div>
          <div style="color:#d1d5db; font-size:0.84rem; line-height:1.6;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── XGBoost best params ───────────────────────────────────────────────────
    st.markdown("### XGBoost tuned hyperparameters")
    if config and "best_xgb_params" in config:
        params = config["best_xgb_params"]
        rows = [{"Parameter": k, "Value": f"{v:.6f}" if isinstance(v, float) else str(v)}
                for k, v in params.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        st.info("Training config not found.")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Full results ──────────────────────────────────────────────────────────
    st.markdown("### Full evaluation results")
    mdf = _load_metrics()
    if mdf is not None and "model_name" in mdf.columns:
        show = mdf.set_index("model_name")
        display_cols = [c for c in ["accuracy", "precision", "recall", "f1_score", "f2_score",
                                    "roc_auc", "pr_auc", "mcc", "false_positive_rate",
                                    "false_negative_rate", "train_time_s", "model_size_mb"]
                        if c in show.columns]
        st.dataframe(show[display_cols].round(4).sort_values("roc_auc", ascending=False))
    else:
        st.info("Model metrics not available. Train models first.")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Why constraints matter ────────────────────────────────────────────────
    st.markdown("### Why feature interaction constraints matter")
    st.markdown("""
Polymorphic malware changes its byte signature with each infection but retains
**internal consistency within behavioral domains**. For example:
- Section entropy is uniformly high across all sections (packing/encryption)
- Behavioral indicators (mutation rate, obfuscation) correlate internally
- Network traffic patterns (beaconing, C2 similarity) are self-consistent

Without constraints, XGBoost may learn cross-domain rules like:  
*"high section entropy AND low connection attempts → benign"*

This breaks on polymorphic variants that have high entropy AND high connections.
Constraints force the model to learn **within-domain patterns** that generalise.
    """)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── About ─────────────────────────────────────────────────────────────────
    st.markdown("### About")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **PolyShield** — Polymorphic Malware Detection System  
        Built by **Aaryan Bangale**  
        
        Python 3.11 · XGBoost · SHAP · Optuna  
        Streamlit · scikit-learn · Plotly
        """)
    with c2:
        st.markdown("""
        **References**  
        - Anderson & Roth (2018) — EMBER Dataset  
        - Chen & Guestrin (2016) — XGBoost  
        - Lundberg & Lee (2017) — SHAP  
        - Akiba et al. (2019) — Optuna  
        - Liu et al. (2008) — Isolation Forest
        """)
