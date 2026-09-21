"""Page 1 — Overview & Executive Summary"""
import streamlit as st
import numpy as np
from pages import (
    load_metrics_df, load_training_config, metric_card,
    gradient_divider, section_header, models_not_trained_warning,
    CYAN, PURPLE, BENIGN_COLOR, MALWARE_COLOR, BG, CARD,
)
from utils.visualizations import plot_confidence_gauge


def render():
    # ── Hero Banner ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-banner">
        <div style="position:relative; z-index:1;">
            <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1rem;">
                <span style="font-size:3rem;">🛡️</span>
                <div>
                    <div style="font-size:2.2rem; font-weight:800;
                                background:linear-gradient(135deg,#00D4FF,#7C3AED);
                                -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                        PolyShield
                    </div>
                    <div style="color:#9CA3AF; font-size:1rem; margin-top:-0.2rem;">
                        Polymorphic Malware Detection System
                    </div>
                </div>
            </div>
            <p style="color:#D1D5DB; font-size:1rem; max-width:700px; line-height:1.7;">
                A multi-model AI system using <strong style="color:#00D4FF;">XGBoost with Feature
                Interaction Constraints</strong> to detect low-signature polymorphic malware with
                state-of-the-art accuracy. Compare 5 ML models, explore SHAP explanations, and
                scan files in real-time.
            </p>
            <div style="display:flex; gap:1rem; margin-top:1.2rem; flex-wrap:wrap;">
                <span style="background:#00D4FF20; border:1px solid #00D4FF44; border-radius:6px;
                             padding:0.3rem 0.8rem; color:#00D4FF; font-size:0.8rem; font-weight:600;">
                    🧠 XGBoost + Optuna
                </span>
                <span style="background:#7C3AED20; border:1px solid #7C3AED44; border-radius:6px;
                             padding:0.3rem 0.8rem; color:#7C3AED; font-size:0.8rem; font-weight:600;">
                    🔍 SHAP Explainability
                </span>
                <span style="background:#10B98120; border:1px solid #10B98144; border-radius:6px;
                             padding:0.3rem 0.8rem; color:#10B981; font-size:0.8rem; font-weight:600;">
                    📊 5 Model Comparison
                </span>
                <span style="background:#F59E0B20; border:1px solid #F59E0B44; border-radius:6px;
                             padding:0.3rem 0.8rem; color:#F59E0B; font-size:0.8rem; font-weight:600;">
                    🧪 Live Scanner
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Cards ────────────────────────────────────────────────────────────
    metrics_df = load_metrics_df()
    config = load_training_config()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_samples = f"{config['n_samples']:,}" if config else "50,000"
        metric_card("Total Samples Analyzed", n_samples, "↑ 50k synthetic samples", True, "📁")
    with col2:
        if config:
            n_malware = int(config['n_samples'] * config['malware_ratio'])
            metric_card("Threats Detected", f"{n_malware:,}", f"{config['malware_ratio']:.0%} malware ratio", True, "⚠️")
        else:
            metric_card("Threats Detected", "10,000", "~20% malware ratio", True, "⚠️")
    with col3:
        if metrics_df is not None:
            best_acc = metrics_df["accuracy"].max()
            best_model = metrics_df["accuracy"].idxmax()
            metric_card("Best Model Accuracy", f"{best_acc:.1%}", f"🏆 {best_model}", True, "🎯")
        else:
            metric_card("Best Model Accuracy", "—", "Train models first", False, "🎯")
    with col4:
        if metrics_df is not None:
            best_f1 = metrics_df["f1_score"].max()
            metric_card("Best F1-Score", f"{best_f1:.4f}", "↑ vs baseline", True, "⚡")
        else:
            metric_card("Best F1-Score", "—", "Train models first", False, "⚡")

    gradient_divider()

    # ── Threat Level Gauge + Model Status ────────────────────────────────────
    col_gauge, col_status = st.columns([1, 1])

    with col_gauge:
        section_header("🎯 System Threat Level", "Current threat assessment based on best model")
        if metrics_df is not None:
            best_recall = float(metrics_df["recall"].max())
            fig = plot_confidence_gauge(best_recall, "Detection Rate")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Train models to view threat level gauge.")

    with col_status:
        section_header("📊 Model Status Overview")
        if metrics_df is not None:
            for model_name, row in metrics_df.iterrows():
                auc = row.get("roc_auc", 0)
                f1 = row.get("f1_score", 0)
                color = {"XGBoost": CYAN, "Random Forest": PURPLE, "LightGBM": BENIGN_COLOR,
                         "CatBoost": "#F59E0B", "Hybrid (IF+GB)": MALWARE_COLOR}.get(str(model_name), CYAN)
                st.markdown(f"""
                <div style="background:{CARD}; border:1px solid #1F2937; border-left:3px solid {color};
                             border-radius:8px; padding:0.7rem 1rem; margin-bottom:0.5rem;
                             display:flex; justify-content:space-between; align-items:center;">
                    <span style="color:#F9FAFB; font-weight:600; font-size:0.9rem;">{model_name}</span>
                    <div>
                        <span style="color:{CYAN}; font-family:JetBrains Mono; font-size:0.85rem;">
                            AUC {auc:.4f}
                        </span>
                        <span style="color:#6B7280; margin:0 0.5rem;">|</span>
                        <span style="color:#10B981; font-family:JetBrains Mono; font-size:0.85rem;">
                            F1 {f1:.4f}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            models_not_trained_warning()

    gradient_divider()

    # ── Project Methodology ───────────────────────────────────────────────────
    section_header("🏗️ Project Methodology")
    cols = st.columns(5)
    steps = [
        ("1", "📦 Data Generation", "50K synthetic EMBER-inspired samples with realistic correlations"),
        ("2", "⚙️ Feature Engineering", "30+ PE, Section, Import, Behavioral, Network features"),
        ("3", "🧠 Model Training", "5 models incl. XGBoost+Constraints, Optuna tuning"),
        ("4", "📊 Evaluation", "AUC, F1, MCC, PR-AUC, FPR/FNR, detection @ 1% FPR"),
        ("5", "🔍 Explainability", "SHAP TreeExplainer, feature interactions, waterfall"),
    ]
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div style="background:{CARD}; border:1px solid #1F2937; border-radius:12px;
                         padding:1rem; text-align:center; height:100%;">
                <div style="font-size:2rem; margin-bottom:0.5rem;">{title.split()[0]}</div>
                <div style="color:#00D4FF; font-weight:700; font-size:0.85rem; margin-bottom:0.5rem;">
                    {' '.join(title.split()[1:])}
                </div>
                <div style="color:#9CA3AF; font-size:0.75rem; line-height:1.4;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    gradient_divider()

    # ── Key Innovation: Interaction Constraints ───────────────────────────────
    section_header("💡 Key Innovation: Feature Interaction Constraints")
    st.markdown("""
    <div class="info-box">
        <strong style="color:#00D4FF;">Why Feature Interaction Constraints?</strong><br><br>
        Polymorphic malware changes its byte signature with each infection but retains
        <em>internal consistency within behavioral domains</em>. For example:
        <ul style="margin-top:0.7rem; color:#D1D5DB;">
            <li>PE header fields are self-consistent regardless of payload</li>
            <li>Section entropy is high across <em>all</em> sections (not just some)</li>
            <li>Behavioral indicators (API calls, memory patterns) correlate internally</li>
            <li>Network traffic patterns maintain their own consistency</li>
        </ul>
        By constraining XGBoost to only interact features <em>within</em> the same domain group,
        we prevent spurious cross-domain correlations that generalize poorly to novel variants.
        This is the core technical innovation of this system.
    </div>
    """, unsafe_allow_html=True)

    from utils.preprocessing import FEATURE_GROUPS
    group_colors = [CYAN, PURPLE, BENIGN_COLOR, "#F59E0B", MALWARE_COLOR, "#EC4899"]
    cols = st.columns(3)
    for i, (group_name, features) in enumerate(FEATURE_GROUPS.items()):
        with cols[i % 3]:
            color = group_colors[i % len(group_colors)]
            st.markdown(f"""
            <div class="constraint-group" style="border-left:3px solid {color};">
                <div style="color:{color}; font-weight:700; font-size:0.85rem; margin-bottom:0.5rem;">
                    {group_name}
                </div>
                <div style="color:#9CA3AF; font-size:0.75rem; line-height:1.6;">
                    {', '.join(features)}
                </div>
            </div>
            """, unsafe_allow_html=True)
