"""Page 3 — XGBoost Deep Dive"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pages import (
    load_shap_data, load_optuna_history, load_feature_importance,
    load_metrics_df, load_ablation_results,
    gradient_divider, section_header, models_not_trained_warning,
    CYAN, PURPLE, BG, CARD, TEXT_SEC,
)
from utils.visualizations import (
    plot_shap_summary, plot_shap_waterfall, plot_feature_importance,
    plot_optuna_history, plot_confusion_matrix,
)


def render():
    st.markdown('<h1 class="gradient-text" style="font-size:2rem; font-weight:800;">🔬 XGBoost Deep Dive</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#9CA3AF;">Detailed analysis of the primary XGBoost model with feature interaction constraints, SHAP explanations, and hyperparameter optimization.</p>', unsafe_allow_html=True)
    gradient_divider()

    # ── Interaction Constraints Visualization ─────────────────────────────────
    section_header("🔗 Feature Interaction Constraint Groups")
    st.markdown("""
    <div class="info-box">
        <strong style="color:#00D4FF;">How Interaction Constraints Work:</strong><br>
        XGBoost's <code>interaction_constraints</code> parameter restricts which features can appear
        together in a single tree branch. Features in different groups are never split together,
        which prevents overfitting to spurious cross-domain correlations in training data —
        a key vulnerability when detecting polymorphic variants that only mutate one domain.
    </div>
    """, unsafe_allow_html=True)

    from utils.preprocessing import FEATURE_GROUPS
    group_colors = [CYAN, PURPLE, "#10B981", "#F59E0B", "#EF4444", "#EC4899"]
    group_icons = ["🗂️", "📐", "📦", "📝", "🧬", "🌐"]
    cols = st.columns(3)
    for i, (group_name, features) in enumerate(FEATURE_GROUPS.items()):
        with cols[i % 3]:
            color = group_colors[i % len(group_colors)]
            icon = group_icons[i % len(group_icons)]
            st.markdown(f"""
            <div style="background:{CARD}; border:1px solid #1F2937; border-left:4px solid {color};
                         border-radius:10px; padding:1rem; margin-bottom:1rem;">
                <div style="font-size:1.2rem; margin-bottom:0.3rem;">{icon}</div>
                <div style="color:{color}; font-weight:700; margin-bottom:0.5rem;">{group_name}</div>
                {''.join(f'<span style="background:{color}22; color:{color}; border-radius:4px; padding:0.15rem 0.4rem; margin:0.15rem; display:inline-block; font-size:0.72rem;">{f}</span>' for f in features)}
            </div>
            """, unsafe_allow_html=True)

    gradient_divider()

    # ── Optuna Hyperparameter Optimization ────────────────────────────────────
    section_header("🔧 Hyperparameter Optimization (Optuna)")
    optuna_hist = load_optuna_history()
    if optuna_hist and optuna_hist.get("n_trials", 0) > 0:
        col_plot, col_params = st.columns([2, 1])
        with col_plot:
            fig_optuna = plot_optuna_history(optuna_hist)
            st.plotly_chart(fig_optuna, use_container_width=True)
        with col_params:
            st.markdown(f"""
            <div style="background:{CARD}; border:1px solid #1F2937; border-radius:10px; padding:1.2rem;">
                <div style="color:{CYAN}; font-weight:700; margin-bottom:0.8rem;">Best Parameters</div>
            """, unsafe_allow_html=True)
            for k, v in optuna_hist.get("best_params", {}).items():
                formatted = f"{v:.4f}" if isinstance(v, float) else str(v)
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; padding:0.3rem 0;
                             border-bottom:1px solid #1F2937;">
                    <span style="color:#9CA3AF; font-size:0.8rem;">{k}</span>
                    <span style="color:#F9FAFB; font-family:JetBrains Mono; font-size:0.8rem;">{formatted}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown(f"""
                <div style="margin-top:1rem; padding:0.5rem; background:#00D4FF20; border-radius:6px; text-align:center;">
                    <span style="color:{CYAN}; font-weight:700;">Best AUC: {optuna_hist.get('best_value', 0):.4f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Optuna history not found. Run training pipeline first.")

    gradient_divider()

    # ── Feature Importance ────────────────────────────────────────────────────
    section_header("📊 Feature Importance Analysis")
    fi_data = load_feature_importance()
    if fi_data:
        imp_type = st.selectbox("Importance type:", ["gain", "weight", "cover"], key="fi_type")
        if imp_type in fi_data:
            fi_dict = fi_data[imp_type]
            fi_df = pd.DataFrame([
                {"feature": k, "importance": v} for k, v in fi_dict.items()
            ]).sort_values("importance", ascending=False).head(20)
            fig_fi = plot_feature_importance(fi_df, f"XGBoost Feature Importance ({imp_type})")
            st.plotly_chart(fig_fi, use_container_width=True)

            # Annotate with group membership
            from utils.preprocessing import FEATURE_GROUPS
            feat_to_group = {f: g for g, feats in FEATURE_GROUPS.items() for f in feats}
            fi_df["group"] = fi_df["feature"].map(feat_to_group).fillna("Other")
            group_imp = fi_df.groupby("group")["importance"].sum().sort_values(ascending=False)
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.markdown(f'<div style="color:{TEXT_SEC}; font-size:0.8rem; margin-bottom:0.5rem;">Importance by Feature Group</div>', unsafe_allow_html=True)
                for group, imp_val in group_imp.items():
                    pct = imp_val / group_imp.sum() * 100
                    st.markdown(f"""
                    <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.3rem;">
                        <div style="flex:1; color:#9CA3AF; font-size:0.8rem;">{group}</div>
                        <div style="width:120px; background:#1F2937; border-radius:4px; height:8px;">
                            <div style="width:{pct}%; background:{CYAN}; border-radius:4px; height:100%;"></div>
                        </div>
                        <div style="color:{CYAN}; font-family:JetBrains Mono; font-size:0.75rem; width:40px;">{pct:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("Feature importance not available. Train models first.")

    gradient_divider()

    # ── SHAP Summary Plot ─────────────────────────────────────────────────────
    section_header("🔍 SHAP Explainability — Summary Plot")
    shap_data = load_shap_data()
    if shap_data and "summary" in shap_data:
        col_shap, col_info = st.columns([3, 1])
        with col_shap:
            fig_shap = plot_shap_summary(shap_data["summary"])
            st.plotly_chart(fig_shap, use_container_width=True)
        with col_info:
            st.markdown(f"""
            <div class="info-box" style="margin-top:2rem;">
                <strong style="color:{CYAN};">How to read this:</strong><br><br>
                <span style="color:#D1D5DB; font-size:0.85rem;">
                Each point is one test sample.<br><br>
                <strong>X-axis:</strong> SHAP value — how much this feature pushed the prediction 
                toward malware (positive) or benign (negative).<br><br>
                <strong>Color:</strong> Feature value — red = high, blue = low.<br><br>
                Features are sorted by <em>mean |SHAP|</em>, so the most influential features 
                appear at the top.
                </span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("SHAP data not available. Train models first.")

    gradient_divider()

    # ── SHAP Waterfall ────────────────────────────────────────────────────────
    section_header("💧 SHAP Waterfall Plot — Individual Sample Explanation")
    if shap_data and "waterfall" in shap_data and shap_data["waterfall"]:
        st.markdown(f'<p style="color:{TEXT_SEC}; font-size:0.85rem;">Showing SHAP breakdown for a malware sample from the test set:</p>', unsafe_allow_html=True)
        fig_wf = plot_shap_waterfall(shap_data["waterfall"])
        st.plotly_chart(fig_wf, use_container_width=True)
    else:
        st.info("Waterfall data not available.")

    gradient_divider()

    # ── SHAP Dependence Plots ─────────────────────────────────────────────────
    section_header("📉 SHAP Dependence Plots — Top 3 Features")
    if shap_data and "dependence" in shap_data and shap_data["dependence"]:
        dep_keys = list(shap_data["dependence"].keys())
        cols = st.columns(min(3, len(dep_keys)))
        for i, (feat, dep_d) in enumerate(list(shap_data["dependence"].items())[:3]):
            with cols[i]:
                fig_dep = go.Figure(go.Scatter(
                    x=dep_d["feature_values"],
                    y=dep_d["shap_values"],
                    mode="markers",
                    marker=dict(color=CYAN, size=4, opacity=0.5),
                    hovertemplate=f"<b>{feat}</b><br>Value: %{{x:.3f}}<br>SHAP: %{{y:.4f}}<extra></extra>",
                ))
                fig_dep.add_hline(y=0, line_color=TEXT_SEC, line_dash="dot")
                fig_dep.update_layout(
                    template="plotly_dark",
                    paper_bgcolor=BG,
                    plot_bgcolor=CARD,
                    title=dict(text=feat, font=dict(size=13, color="#F9FAFB")),
                    xaxis_title="Feature Value",
                    yaxis_title="SHAP Value",
                    margin=dict(l=30, r=10, t=40, b=30),
                    height=280,
                    font=dict(color="#F9FAFB"),
                )
                st.plotly_chart(fig_dep, use_container_width=True)
    else:
        st.info("Dependence plot data not available.")

    gradient_divider()

    # ── Ablation Study ────────────────────────────────────────────────────────
    section_header("🧪 Ablation: With vs Without Constraints")
    ablation = load_ablation_results()
    if ablation:
        with_m = ablation.get("with_constraints", {})
        without_m = ablation.get("without_constraints", {})
        metrics_to_compare = ["roc_auc", "f1_score", "pr_auc", "recall", "false_positive_rate"]
        cols = st.columns(len(metrics_to_compare))
        for col, metric in zip(cols, metrics_to_compare):
            with col:
                v_with = with_m.get(metric, 0)
                v_without = without_m.get(metric, 0)
                delta = v_with - v_without
                is_minimize = metric in {"false_positive_rate", "false_negative_rate"}
                delta_positive = (delta < 0) if is_minimize else (delta > 0)
                color = "#10B981" if delta_positive else "#EF4444"
                st.markdown(f"""
                <div style="background:{CARD}; border:1px solid #1F2937; border-radius:10px;
                             padding:1rem; text-align:center;">
                    <div style="color:#9CA3AF; font-size:0.72rem; text-transform:uppercase;
                                letter-spacing:0.5px; margin-bottom:0.5rem;">{metric.replace('_',' ')}</div>
                    <div style="font-family:JetBrains Mono; font-size:1.2rem; color:#F9FAFB; font-weight:700;">
                        {v_with:.4f}
                    </div>
                    <div style="color:{color}; font-size:0.75rem; margin-top:0.3rem;">
                        {'+' if delta >= 0 else ''}{delta:.4f} vs no constraints
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Ablation study data not available. Train models first.")
