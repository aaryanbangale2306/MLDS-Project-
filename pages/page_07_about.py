"""Page 7 — About & Methodology"""
import streamlit as st
from pages import (
    load_training_config, load_metrics_df, gradient_divider, section_header,
    CYAN, PURPLE, BENIGN_COLOR, MALWARE_COLOR, BG, CARD, TEXT_SEC,
)


def render():
    st.markdown('<h1 class="gradient-text" style="font-size:2rem; font-weight:800;">ℹ️ About & Methodology</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#9CA3AF;">System architecture, methodology, references, and team information.</p>', unsafe_allow_html=True)
    gradient_divider()

    # ── Project Overview ──────────────────────────────────────────────────────
    section_header("🎯 Project Overview")
    st.markdown(f"""
    <div style="background:{CARD}; border:1px solid #1F2937; border-radius:12px; padding:1.5rem;">
        <h3 style="color:{CYAN}; margin-bottom:0.8rem;">Problem Statement</h3>
        <p style="color:#D1D5DB; line-height:1.7;">
            Polymorphic malware evades traditional signature-based antivirus by mutating its code
            with each infection while preserving its malicious functionality. Low-signature variants
            are particularly challenging because they closely mimic benign software in isolated
            feature domains while maintaining malicious behavior across all domains holistically.
        </p>
        <h3 style="color:{CYAN}; margin-top:1.2rem; margin-bottom:0.8rem;">Our Solution</h3>
        <p style="color:#D1D5DB; line-height:1.7;">
            We propose an XGBoost classifier with <strong style="color:{CYAN};">Feature Interaction
            Constraints</strong> that encodes domain knowledge about polymorphic malware into the
            model architecture itself. By restricting cross-domain feature interactions, the model
            learns domain-consistent detection rules that generalize to novel polymorphic variants.
        </p>
    </div>
    """, unsafe_allow_html=True)

    gradient_divider()

    # ── Methodology Flowchart ─────────────────────────────────────────────────
    section_header("🔄 Methodology Pipeline")
    steps = [
        ("📦", "Data Generation", "Generate 50,000 synthetic samples with EMBER-inspired features. Realistic distributions with polymorphic, obfuscated, and classic malware sub-types. 80/20 benign/malware split."),
        ("⚙️", "Feature Engineering", "30+ features across 6 domain groups: PE Header, Section Analysis, Import Features, String Features, Behavioral, and Network indicators."),
        ("🔧", "Optuna Tuning", "50-trial Bayesian optimization using TPE sampler with MedianPruner. Optimizes ROC-AUC on held-out validation set."),
        ("🧠", "Model Training", "Train all 5 models: XGBoost (primary), Random Forest, LightGBM, CatBoost, and Hybrid IF+GBC. Early stopping on all gradient boosting models."),
        ("📊", "Evaluation", "Comprehensive evaluation: Accuracy, Precision, Recall, F1, F2, ROC-AUC, PR-AUC, MCC, FPR, FNR, Detection@1%FPR, timing, and model size."),
        ("🔍", "Explainability", "SHAP TreeExplainer for XGBoost: summary plots, waterfall charts, dependence plots, and feature group risk scores."),
        ("🌐", "Dashboard", "7-page Streamlit dashboard with dark cybersecurity UI, live scanning, and full interactive visualization suite."),
    ]
    for i, (icon, title, desc) in enumerate(steps):
        arrow = "↓" if i < len(steps) - 1 else "✓"
        st.markdown(f"""
        <div style="display:flex; align-items:flex-start; margin-bottom:0.5rem;">
            <div style="background:{CARD}; border:1px solid #1F2937; border-left:3px solid {CYAN};
                         border-radius:0 10px 10px 0; padding:0.8rem 1.2rem; flex:1;">
                <div style="display:flex; align-items:center; gap:0.7rem;">
                    <span style="font-size:1.4rem;">{icon}</span>
                    <div>
                        <div style="color:{CYAN}; font-weight:700; font-size:0.9rem;">{i+1}. {title}</div>
                        <div style="color:#9CA3AF; font-size:0.8rem; margin-top:0.2rem;">{desc}</div>
                    </div>
                </div>
            </div>
            <div style="color:{PURPLE}; font-size:1.2rem; margin:0.5rem 0.8rem; align-self:center;">{arrow}</div>
        </div>
        """, unsafe_allow_html=True)

    gradient_divider()

    # ── Architecture Diagram ───────────────────────────────────────────────────
    section_header("🏗️ System Architecture")
    st.markdown(f"""
    <div style="background:{CARD}; border:1px solid #1F2937; border-radius:12px; padding:1.5rem; font-family:JetBrains Mono; font-size:0.8rem; color:#9CA3AF; overflow-x:auto;">
    <pre style="color:#9CA3AF; margin:0; line-height:1.6;">
┌─────────────────────────────────────────────────────────────────────┐
│                     PolyShield Architecture                         │
└─────────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────▼───────────────┐
              │       Data Generation          │
              │   data/generate_dataset.py     │
              │  50,000 samples │ 30+ features │
              └───────────────┬───────────────┘
                              │
              ┌───────────────▼───────────────┐
              │      Preprocessing             │
              │   utils/preprocessing.py       │
              │ RobustScaler │ Train/Val/Test  │
              └───────────────┬───────────────┘
                              │
     ┌────────────────────────▼─────────────────────────┐
     │              Model Training Layer                 │
     │           models/train_all_models.py              │
     │                                                   │
     │  ┌──────────┐  ┌──────┐  ┌─────────┐            │
     │  │ XGBoost  │  │  RF  │  │ LightGBM│            │
     │  │+Constraints  │      │  │  DART   │            │
     │  └──────────┘  └──────┘  └─────────┘            │
     │  ┌──────────┐  ┌──────────────────┐              │
     │  │ CatBoost │  │ Hybrid IF + GBC  │              │
     │  │ Ordered  │  │  2-stage pipeline│              │
     │  └──────────┘  └──────────────────┘              │
     └────────────────────────┬─────────────────────────┘
                              │
     ┌────────────────────────▼─────────────────────────┐
     │           Explainability & Evaluation             │
     │  SHAP TreeExplainer │ Optuna │ All Metrics        │
     └────────────────────────┬─────────────────────────┘
                              │
     ┌────────────────────────▼─────────────────────────┐
     │               Streamlit Dashboard                 │
     │                    app.py                         │
     │  7 pages │ Dark Theme │ Plotly │ Live Scanner     │
     └───────────────────────────────────────────────────┘
    </pre>
    </div>
    """, unsafe_allow_html=True)

    gradient_divider()

    # ── Models Described ──────────────────────────────────────────────────────
    section_header("🤖 Models Overview")
    models_info = [
        (CYAN, "🥇 XGBoost (Primary)", "Feature Interaction Constraints",
         "Our primary model uses XGBoost's interaction_constraints parameter to restrict "
         "cross-group feature interactions. Tuned with Optuna (50 trials, TPE sampler). "
         "Handles class imbalance via scale_pos_weight. SHAP TreeExplainer for full interpretability."),
        (PURPLE, "🥈 Random Forest", "Ensemble Baseline",
         "500 trees with sqrt feature sampling and class_weight='balanced'. "
         "OOB score for extra validation. Supports both MDI and permutation feature importance."),
        (BENIGN_COLOR, "🥉 LightGBM", "Leaf-wise + DART",
         "Microsoft's gradient boosting with leaf-wise tree growth for better accuracy. "
         "DART (Dropouts meet Multiple Additive Regression Trees) for regularization. "
         "Scale_pos_weight for imbalance handling."),
        ("#F59E0B", "4️⃣ CatBoost", "Ordered Boosting",
         "Yandex's ordered boosting prevents target leakage during training, making it "
         "robust to overfitting. Built-in symmetrical tree structure with built-in "
         "feature importance analysis."),
        (MALWARE_COLOR, "5️⃣ Hybrid (IF+GB)", "Two-Stage Detection",
         "Stage 1: Isolation Forest for anomaly pre-screening (identifies samples deviating "
         "from benign baseline). Stage 2: Gradient Boosting Classifier trained on original "
         "features + IF anomaly score as extra meta-feature."),
    ]
    for color, title, subtitle, desc in models_info:
        st.markdown(f"""
        <div style="background:{CARD}; border:1px solid {color}44; border-left:4px solid {color};
                     border-radius:0 12px 12px 0; padding:1.2rem; margin-bottom:0.8rem;">
            <div style="font-size:1rem; font-weight:700; color:{color}; margin-bottom:0.2rem;">{title}</div>
            <div style="font-size:0.8rem; color:#9CA3AF; margin-bottom:0.5rem;">{subtitle}</div>
            <div style="font-size:0.85rem; color:#D1D5DB; line-height:1.6;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    gradient_divider()

    # ── Results Table ─────────────────────────────────────────────────────────
    section_header("📊 Best Results Summary")
    metrics_df = load_metrics_df()
    if metrics_df is not None:
        display_metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc", "mcc"]
        display_metrics = [m for m in display_metrics if m in metrics_df.columns]
        st.dataframe(
            metrics_df[display_metrics].round(4).sort_values("roc_auc", ascending=False),
            use_container_width=True
        )
    else:
        st.info("Train models first to see results.")

    gradient_divider()

    # ── References ────────────────────────────────────────────────────────────
    section_header("📚 References & Dataset")
    st.markdown(f"""
    <div style="background:{CARD}; border:1px solid #1F2937; border-radius:12px; padding:1.5rem;">
        <ol style="color:#D1D5DB; line-height:2.0; font-size:0.85rem;">
            <li><strong>EMBER Dataset:</strong> Anderson, H. S., & Roth, P. (2018). EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models. <em>arXiv:1804.04637</em></li>
            <li><strong>XGBoost:</strong> Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. <em>KDD '16</em></li>
            <li><strong>Feature Interaction Constraints:</strong> XGBoost Documentation — interaction_constraints parameter. <em>xgboost.readthedocs.io</em></li>
            <li><strong>SHAP:</strong> Lundberg, S., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. <em>NeurIPS 2017</em></li>
            <li><strong>Optuna:</strong> Akiba, T. et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. <em>KDD '19</em></li>
            <li><strong>LightGBM:</strong> Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. <em>NeurIPS 2017</em></li>
            <li><strong>CatBoost:</strong> Prokhorenkova, L. et al. (2018). CatBoost: Unbiased Boosting with Categorical Features. <em>NeurIPS 2018</em></li>
            <li><strong>Isolation Forest:</strong> Liu, F. T. et al. (2008). Isolation Forest. <em>IEEE ICDM 2008</em></li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    gradient_divider()

    # ── Author & GitHub ───────────────────────────────────────────────────────
    section_header("👤 Author & Repository")
    config = load_training_config()
    col_author, col_links = st.columns([1, 1])
    with col_author:
        st.markdown(f"""
        <div style="background:{CARD}; border:1px solid #1F2937; border-radius:12px; padding:1.5rem; text-align:center;">
            <div style="font-size:3rem; margin-bottom:0.5rem;">👨‍💻</div>
            <div style="font-size:1.1rem; font-weight:700; color:#F9FAFB; margin-bottom:0.3rem;">Aaryan Bangale</div>
            <div style="color:#9CA3AF; font-size:0.85rem; margin-bottom:0.8rem;">ML Engineer & Data Scientist</div>
            <a href="https://github.com/aaryanbangale2306" target="_blank"
               style="background:{CYAN}22; border:1px solid {CYAN}44; color:{CYAN}; border-radius:6px;
                      padding:0.4rem 1rem; font-size:0.85rem; text-decoration:none; font-weight:600;">
                GitHub Profile ↗
            </a>
        </div>
        """, unsafe_allow_html=True)

    with col_links:
        st.markdown(f"""
        <div style="background:{CARD}; border:1px solid #1F2937; border-radius:12px; padding:1.5rem;">
            <div style="color:{CYAN}; font-weight:700; margin-bottom:1rem;">🔗 Repository</div>
            <div style="margin-bottom:0.8rem;">
                <a href="https://github.com/aaryanbangale2306/MLDS-Project" target="_blank"
                   style="color:{CYAN}; text-decoration:none; font-family:JetBrains Mono; font-size:0.85rem;">
                    github.com/aaryanbangale2306/MLDS-Project ↗
                </a>
            </div>
            <div style="color:#9CA3AF; font-size:0.8rem; line-height:1.6;">
                <div>📝 License: MIT</div>
                <div>🐍 Python 3.10+</div>
                <div>📊 Streamlit 1.35+</div>
                <div>🧠 XGBoost 2.0+ with interaction_constraints</div>
                <div>🔍 SHAP 0.45+ TreeExplainer</div>
                <div>⚙️ Optuna 3.6+ TPE Sampler</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
