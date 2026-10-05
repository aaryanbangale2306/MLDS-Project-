"""Page 4 - Models and Method"""
import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

SAVED = Path(__file__).parent.parent / "models" / "saved"


@st.cache_data
def _load_config():
    p = SAVED / "training_config.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def _load_metrics():
    p = SAVED / "metrics.pkl"
    return pd.DataFrame(joblib.load(str(p))) if p.exists() else None


def render():
    st.markdown("# Models & Method")
    st.markdown("Technical description of the dataset, feature engineering, models, explainability and limitations.")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    config = _load_config()

    # A. Dataset
    st.markdown("### A. Dataset")
    st.markdown("""
**Source:** Synthetic benchmark dataset generated using EMBER-style feature distributions.

**Important:** This is a *synthetic* dataset, not real-world malware samples.
It uses statistically realistic within-group correlations and adds 1% label noise.
Performance metrics should be interpreted as benchmark results on this synthetic distribution,
not as proof of real-world malware detection effectiveness.
    """)
    if config:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{config["n_samples"]:,}</div><div class="stat-lbl">Total samples</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{int(config["n_samples"]*config["malware_ratio"]):,}</div><div class="stat-lbl">Malware (~{config["malware_ratio"]*100:.0f}%)</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{config["n_features"]}</div><div class="stat-lbl">Features</div></div>', unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # B. Feature engineering
    st.markdown("### B. Feature engineering")
    st.markdown("""
33 static and behavioral features grouped into 6 domains:
    """)
    from utils.preprocessing import FEATURE_GROUPS
    for group, feats in FEATURE_GROUPS.items():
        st.markdown(f"**{group}** ({len(feats)} features): {', '.join(f'`{f}`' for f in feats)}")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # C. Training pipeline
    st.markdown("### C. Training pipeline")
    steps = [
        ("Data generation", "50,000 synthetic EMBER-style samples with realistic within-group correlations and 1% label noise."),
        ("Train/val/test split", "70% / 15% / 15% stratified split. All preprocessing fitted on training data only."),
        ("Preprocessing", "RobustScaler (resistant to outliers). Fitted on training set, applied to val/test."),
        ("Hyperparameter tuning", "25-trial Optuna (TPE sampler) on XGBoost, optimising ROC-AUC on validation set."),
        ("Model training", "5 models with class-imbalance handling (scale_pos_weight / class_weight=balanced)."),
        ("Evaluation", "Accuracy, Precision, Recall, F1, F2, ROC-AUC, PR-AUC, MCC, FPR, FNR on held-out test set."),
        ("SHAP analysis", "TreeExplainer on XGBoost for global and per-sample feature attribution."),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div class="card" style="margin-bottom:0.6rem; border-left:3px solid #3b82f6;">
          <span style="color:#60a5fa; font-weight:600;">{i}.</span> <strong>{title}</strong><br>
          <span style="color:#9ca3af; font-size:0.84rem;">{desc}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # D. Models
    st.markdown("### D. Models")
    models_info = [
        ("XGBoost (primary)", "#3b82f6", "interaction_constraints",
         "Primary classifier. Uses `interaction_constraints` to limit feature combinations to within-domain groups. "
         "Tuned with 25-trial Optuna search. Class imbalance handled via `scale_pos_weight`."),
        ("Random Forest", "#8b5cf6", "Ensemble baseline",
         "500 trees, sqrt feature sampling, `class_weight='balanced'`. Stable baseline with low variance."),
        ("LightGBM", "#10b981", "Leaf-wise GBDT",
         "Gradient boosting with leaf-wise splits, 63 num_leaves, early stopping at 50 rounds."),
        ("CatBoost", "#f59e0b", "Ordered boosting",
         "Ordered boosting prevents target leakage. 500 iterations, depth 6."),
        ("Hybrid (IF+GB)", "#ef4444", "Two-stage",
         "Stage 1: Isolation Forest trained on benign samples to score anomalies. "
         "Stage 2: GradientBoostingClassifier trained on original features + IF anomaly score as extra feature."),
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

    # E. Key idea: Interaction constraints
    st.markdown("### E. Key idea: interaction constraints")
    st.markdown("""
Polymorphic malware changes its byte-level signature but retains **internal consistency within behavioral domains**.
For example, a packed/encrypted binary will have uniformly high entropy across all sections, and its network behavior
(beaconing, C2 similarity) will be self-consistent.

Without constraints, a decision tree can learn cross-domain rules like:
*"high section entropy AND low connection attempts → benign"*.
This rule breaks on polymorphic variants that have both high entropy and high network activity.

**Interaction constraints** restrict XGBoost splits so that nodes within a subtree can only use features
from the same domain group. This forces the model to learn within-domain patterns that generalise better.
    """)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # F. Explainability
    st.markdown("### F. Explainability (SHAP)")
    st.markdown("""
SHAP (SHapley Additive exPlanations) decomposes each prediction into additive contributions from each feature.
For each sample, a SHAP value is computed for every feature, indicating how much that feature pushed the
prediction toward malware (positive) or benign (negative).

`TreeExplainer` from the SHAP library is used, which computes exact Shapley values for tree-based models
without sampling approximations. See the Scanner page for per-sample SHAP waterfall plots.
    """)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # G. Full results table
    st.markdown("### G. Full evaluation results")
    st.caption("All metrics on 7,500-sample held-out test set (synthetic benchmark).")
    mdf = _load_metrics()
    if mdf is not None and "model_name" in mdf.columns:
        show = mdf.set_index("model_name")
        display_cols = [c for c in ["accuracy", "precision", "recall", "f1_score",
                                    "roc_auc", "pr_auc", "mcc", "false_positive_rate",
                                    "false_negative_rate", "train_time_s"]
                        if c in show.columns]
        st.dataframe(show[display_cols].round(4).sort_values("roc_auc", ascending=False))
    else:
        st.info("Model metrics not available. Train models first.")

    # XGBoost tuned params
    if config and "best_xgb_params" in config:
        st.markdown("**XGBoost tuned hyperparameters (Optuna, 25 trials):**")
        params = config["best_xgb_params"]
        rows = [{"Parameter": k, "Value": f"{v:.6f}" if isinstance(v, float) else str(v)}
                for k, v in params.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # H. Limitations
    st.markdown("### H. Limitations")
    st.markdown("""
- **Synthetic data**: All results are on a synthetically generated benchmark. Real-world malware families
  may have different feature distributions; these results should not be taken as evidence of deployment readiness.
- **Feature extraction dependency**: The system requires pre-extracted PE features. It cannot directly
  analyse arbitrary binary files — a feature extraction pipeline (e.g., using tools like `pefile`, `lief`,
  or EMBER's feature extractor) would be needed in a real deployment.
- **Static/behavioral features only**: Network features in this dataset are simulated. Real network
  behavioral analysis would require dynamic execution (sandboxing).
- **No adversarial robustness testing**: The model has not been tested against adversarial feature
  manipulation, which is a known concern for ML-based malware detection.
- **Label noise**: 1% noise was deliberately added during generation to prevent overfit. Real-world
  labelling errors could differ significantly.
    """)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # I. About
    st.markdown("### I. About")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**PolyShield** — Polymorphic Malware Detection System  
Built by **Aaryan Bangale**

Python 3.11 · XGBoost · LightGBM · CatBoost  
SHAP · Optuna · scikit-learn · Streamlit · Plotly
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
