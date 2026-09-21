"""Page 2 — Model Performance Comparison"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pages import (
    load_metrics_df, load_roc_data, load_pr_data,
    gradient_divider, section_header, models_not_trained_warning,
    MODEL_COLORS, CYAN, PURPLE, BG, CARD, TEXT_SEC,
)
from utils.visualizations import (
    plot_roc_curves, plot_pr_curves, plot_metrics_bar, plot_radar_chart, plot_confusion_matrix,
)


def render():
    st.markdown('<h1 class="gradient-text" style="font-size:2rem; font-weight:800;">📊 Model Performance Comparison</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#9CA3AF;">Side-by-side evaluation of all 5 malware detection models across all metrics.</p>', unsafe_allow_html=True)
    gradient_divider()

    metrics_df = load_metrics_df()
    if metrics_df is None:
        models_not_trained_warning()
        return

    # ── Leaderboard ───────────────────────────────────────────────────────────
    section_header("🏆 Model Leaderboard", "Ranked by ROC-AUC")

    sorted_df = metrics_df.sort_values("roc_auc", ascending=False)
    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    for i, (model_name, row) in enumerate(sorted_df.iterrows()):
        color = MODEL_COLORS.get(str(model_name), CYAN)
        medal = medals[i] if i < len(medals) else str(i + 1)
        st.markdown(f"""
        <div class="leaderboard-row" style="border-left: 3px solid {color};">
            <span style="font-size:1.5rem; margin-right:1rem;">{medal}</span>
            <div style="flex:1;">
                <div style="color:#F9FAFB; font-weight:700;">{model_name}</div>
            </div>
            <div style="display:flex; gap:2rem; font-family:JetBrains Mono; font-size:0.85rem;">
                <span>AUC <span style="color:{CYAN};">{row.get('roc_auc',0):.4f}</span></span>
                <span>F1 <span style="color:#10B981;">{row.get('f1_score',0):.4f}</span></span>
                <span>Recall <span style="color:#F59E0B;">{row.get('recall',0):.4f}</span></span>
                <span>FPR <span style="color:#EF4444;">{row.get('false_positive_rate',0):.4f}</span></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    gradient_divider()

    # ── Metrics Table ─────────────────────────────────────────────────────────
    section_header("📋 Full Metrics Table")
    display_cols = ["accuracy", "precision", "recall", "f1_score", "f2_score",
                    "roc_auc", "pr_auc", "mcc", "false_positive_rate",
                    "false_negative_rate", "detection_at_1pct_fpr",
                    "train_time_s", "inference_time_ms", "model_size_mb"]
    display_cols = [c for c in display_cols if c in metrics_df.columns]
    display_df = metrics_df[display_cols].copy()

    # Color-code: best=green, worst=red per column
    minimize_cols = {"false_positive_rate", "false_negative_rate", "train_time_s",
                     "inference_time_ms", "model_size_mb"}

    def _style_fn(col):
        if col.name in minimize_cols:
            best_idx, worst_idx = col.idxmin(), col.idxmax()
        else:
            best_idx, worst_idx = col.idxmax(), col.idxmin()
        styles = []
        for idx in col.index:
            if idx == best_idx:
                styles.append("background-color: #10B98133; color: #10B981; font-weight:700;")
            elif idx == worst_idx:
                styles.append("background-color: #EF444433; color: #EF4444;")
            else:
                styles.append("")
        return styles

    numeric_cols = display_df.select_dtypes(include=[np.number]).columns
    styled = display_df.style.apply(_style_fn, axis=0, subset=numeric_cols).format("{:.4f}", subset=numeric_cols)
    st.dataframe(styled, use_container_width=True)

    gradient_divider()

    # ── Bar Chart ─────────────────────────────────────────────────────────────
    section_header("📊 Grouped Metric Comparison")
    selected_metrics = st.multiselect(
        "Select metrics to compare:",
        options=["accuracy", "precision", "recall", "f1_score", "f2_score",
                 "roc_auc", "pr_auc", "mcc", "detection_at_1pct_fpr"],
        default=["accuracy", "precision", "recall", "f1_score", "roc_auc"],
    )
    if selected_metrics:
        fig_bar = plot_metrics_bar(metrics_df, selected_metrics)
        st.plotly_chart(fig_bar, use_container_width=True)

    gradient_divider()

    # ── ROC + PR Curves ───────────────────────────────────────────────────────
    col_roc, col_pr = st.columns(2)

    roc_data = load_roc_data()
    pr_data = load_pr_data()

    with col_roc:
        section_header("📈 ROC Curves")
        if roc_data:
            fig_roc = plot_roc_curves(roc_data)
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.info("ROC data not found. Train models first.")

    with col_pr:
        section_header("📉 Precision-Recall Curves")
        if pr_data:
            fig_pr = plot_pr_curves(pr_data)
            st.plotly_chart(fig_pr, use_container_width=True)
        else:
            st.info("PR data not found. Train models first.")

    gradient_divider()

    # ── Radar Chart ───────────────────────────────────────────────────────────
    section_header("🕸️ Performance Radar")
    fig_radar = plot_radar_chart(metrics_df)
    st.plotly_chart(fig_radar, use_container_width=True)

    gradient_divider()

    # ── Confusion Matrices ────────────────────────────────────────────────────
    section_header("🔢 Confusion Matrices")
    model_options = list(metrics_df.index)
    selected_model = st.selectbox("Select model:", model_options)
    if selected_model and selected_model in metrics_df.index:
        row = metrics_df.loc[selected_model]
        tp = int(row.get("tp", 0))
        tn = int(row.get("tn", 0))
        fp = int(row.get("fp", 0))
        fn = int(row.get("fn", 0))
        if any([tp, tn, fp, fn]):
            fig_cm = plot_confusion_matrix(tp, tn, fp, fn, str(selected_model))
            col_left, col_right = st.columns([1, 1])
            with col_left:
                st.plotly_chart(fig_cm, use_container_width=True)
            with col_right:
                # Additional derived metrics
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                st.markdown(f"""
                <div style="margin-top:2rem;">
                    <div style="color:#9CA3AF; font-size:0.8rem; margin-bottom:0.5rem;">Detailed Breakdown</div>
                    <table style="width:100%; font-family:JetBrains Mono; font-size:0.85rem; border-collapse:collapse;">
                        <tr style="border-bottom:1px solid #1F2937;">
                            <td style="padding:0.4rem; color:#9CA3AF;">True Positives (TP)</td>
                            <td style="padding:0.4rem; color:#10B981; font-weight:700;">{tp:,}</td>
                        </tr>
                        <tr style="border-bottom:1px solid #1F2937;">
                            <td style="padding:0.4rem; color:#9CA3AF;">True Negatives (TN)</td>
                            <td style="padding:0.4rem; color:#10B981; font-weight:700;">{tn:,}</td>
                        </tr>
                        <tr style="border-bottom:1px solid #1F2937;">
                            <td style="padding:0.4rem; color:#9CA3AF;">False Positives (FP)</td>
                            <td style="padding:0.4rem; color:#F59E0B; font-weight:700;">{fp:,}</td>
                        </tr>
                        <tr style="border-bottom:1px solid #1F2937;">
                            <td style="padding:0.4rem; color:#9CA3AF;">False Negatives (FN)</td>
                            <td style="padding:0.4rem; color:#EF4444; font-weight:700;">{fn:,}</td>
                        </tr>
                        <tr style="border-bottom:1px solid #1F2937;">
                            <td style="padding:0.4rem; color:#9CA3AF;">False Positive Rate</td>
                            <td style="padding:0.4rem; color:#F59E0B;">{fpr:.4f}</td>
                        </tr>
                        <tr>
                            <td style="padding:0.4rem; color:#9CA3AF;">False Negative Rate</td>
                            <td style="padding:0.4rem; color:#EF4444;">{fnr:.4f}</td>
                        </tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Confusion matrix data not available. Ensure training included TP/TN/FP/FN.")

    gradient_divider()

    # ── Speed & Size Comparison ───────────────────────────────────────────────
    section_header("⚡ Speed & Size Comparison")
    if "inference_time_ms" in metrics_df.columns:
        speed_cols = [c for c in ["train_time_s", "inference_time_ms", "model_size_mb"] if c in metrics_df.columns]
        speed_df = metrics_df[speed_cols].copy()

        fig_speed = go.Figure()
        for col in speed_cols:
            fig_speed.add_trace(go.Bar(
                name=col.replace("_", " ").title(),
                x=speed_df.index.tolist(),
                y=speed_df[col].tolist(),
            ))
        fig_speed.update_layout(
            template="plotly_dark",
            paper_bgcolor=BG,
            plot_bgcolor=CARD,
            barmode="group",
            title="Training Time, Inference Time, Model Size",
            font=dict(color="#F9FAFB"),
            legend=dict(bgcolor=CARD),
        )
        st.plotly_chart(fig_speed, use_container_width=True)
