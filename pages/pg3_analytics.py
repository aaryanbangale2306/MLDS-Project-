"""Page 3 - Analytics: ROC curves, confusion matrix, SHAP feature importance, ablation"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from pathlib import Path

SAVED = Path(__file__).parent.parent / "models" / "saved"
DATA_DIR = Path(__file__).parent.parent / "data"

_COLORS = ["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444"]


def _lyt(**kw):
    cfg = dict(
        template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#1e2433",
        font=dict(family="Inter, sans-serif", color="#e5e7eb", size=12),
        margin=dict(l=40, r=20, t=45, b=40),
        xaxis=dict(gridcolor="#2d3748"),
        yaxis=dict(gridcolor="#2d3748"),
    )
    for k, v in kw.items():
        if k in ("xaxis", "yaxis") and isinstance(v, dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    return cfg


@st.cache_data
def _load_metrics():
    p = SAVED / "metrics.pkl"
    return pd.DataFrame(joblib.load(str(p))) if p.exists() else None


@st.cache_data
def _load_roc():
    p = SAVED / "roc_data.pkl"
    return joblib.load(str(p)) if p.exists() else None


@st.cache_data
def _load_test_data():
    p = SAVED / "test_data.pkl"
    return joblib.load(str(p)) if p.exists() else None


@st.cache_data
def _load_shap_fi():
    p = SAVED / "shap_data.pkl"
    if p.exists():
        d = joblib.load(str(p))
        return d.get("feature_importance", None)
    return None


@st.cache_data
def _load_ablation():
    p = SAVED / "ablation_results.pkl"
    return joblib.load(str(p)) if p.exists() else None


@st.cache_data
def _load_dataset_sample():
    p = DATA_DIR / "synthetic_malware_data.csv"
    if p.exists():
        return pd.read_csv(p, nrows=3000)
    p2 = SAVED / "test_sample.csv"
    if p2.exists():
        return pd.read_csv(p2)
    return None


def render():
    st.markdown("# Analytics")
    st.caption("All charts are generated from actual training artifacts. Dataset is a synthetic benchmark.")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    metrics_df = _load_metrics()
    roc_data = _load_roc()
    test_data = _load_test_data()
    shap_fi = _load_shap_fi()
    ablation = _load_ablation()
    df = _load_dataset_sample()

    # ── KPI row from real metrics ─────────────────────────────────────────────
    if metrics_df is not None:
        best_row = metrics_df.loc[metrics_df["roc_auc"].idxmax()]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{best_row["roc_auc"]:.4f}</div><div class="stat-lbl">Best ROC-AUC ({best_row["model_name"]})</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{best_row["f1_score"]:.4f}</div><div class="stat-lbl">F1 Score</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{best_row["precision"]:.4f}</div><div class="stat-lbl">Precision</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{best_row["recall"]:.4f}</div><div class="stat-lbl">Recall</div></div>', unsafe_allow_html=True)
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Section 1: ROC curves ─────────────────────────────────────────────────
    st.markdown("### ROC curves — all models")
    st.caption("Receiver Operating Characteristic curves on the held-out test set (7,500 samples).")
    if roc_data:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     line=dict(dash="dash", color="#4b5563"), name="Random baseline",
                                     hoverinfo="skip"))
        for i, (name, rd) in enumerate(roc_data.items()):
            auc = rd.get("auc", 0)
            fig_roc.add_trace(go.Scatter(
                x=rd["fpr"], y=rd["tpr"],
                mode="lines", name=f"{name} (AUC={auc:.4f})",
                line=dict(color=_COLORS[i % len(_COLORS)], width=2),
                hovertemplate=f"<b>{name}</b><br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>",
            ))
        fig_roc.update_layout(_lyt(
            height=400,
            xaxis=dict(title="False Positive Rate", gridcolor="#2d3748", range=[0, 1]),
            yaxis=dict(title="True Positive Rate", gridcolor="#2d3748", range=[0, 1]),
            legend=dict(bgcolor="#1e2433", x=0.55, y=0.05),
        ))
        st.plotly_chart(fig_roc)
    else:
        st.info("ROC data not available.")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Section 2: Confusion matrix (XGBoost on test set) ────────────────────
    st.markdown("### Confusion matrix — XGBoost on test set")
    if metrics_df is not None:
        xgb_row = metrics_df[metrics_df["model_name"] == "XGBoost"]
        if not xgb_row.empty and all(c in xgb_row.columns for c in ["tp", "tn", "fp", "fn"]):
            r = xgb_row.iloc[0]
            tp, tn, fp, fn = int(r["tp"]), int(r["tn"]), int(r["fp"]), int(r["fn"])
            z = [[tn, fp], [fn, tp]]
            text = [[f"TN={tn}", f"FP={fp}"], [f"FN={fn}", f"TP={tp}"]]
            fig_cm = go.Figure(go.Heatmap(
                z=z, x=["Predicted Benign", "Predicted Malware"],
                y=["Actual Benign", "Actual Malware"],
                text=text, texttemplate="%{text}",
                colorscale=[[0, "#1e2433"], [1, "#2563eb"]],
                showscale=False,
                hovertemplate="<b>%{y} / %{x}</b><br>Count: %{z}<extra></extra>",
            ))
            fig_cm.update_layout(_lyt(height=320, margin=dict(l=120, r=20, t=40, b=80)))
            left, right = st.columns([2, 1])
            with left:
                st.plotly_chart(fig_cm)
            with right:
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                st.markdown(f"""
                <div class="card" style="margin-top:1rem;">
                  <div style="margin-bottom:0.8rem;"><span class="pill-green">True Positives</span> {tp} — correctly identified malware</div>
                  <div style="margin-bottom:0.8rem;"><span class="pill-green">True Negatives</span> {tn} — correctly identified benign</div>
                  <div style="margin-bottom:0.8rem;"><span class="pill-red">False Positives</span> {fp} — benign flagged as malware (FPR: {fpr:.2%})</div>
                  <div><span class="pill-red">False Negatives</span> {fn} — malware missed (FNR: {fnr:.2%})</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Confusion matrix data not in metrics file.")
    else:
        st.info("Metrics not available.")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Section 3: SHAP feature importance ───────────────────────────────────
    st.markdown("### SHAP feature importance — XGBoost")
    st.caption("Mean absolute SHAP values on the test set. Higher = feature has more impact on predictions.")
    if shap_fi is not None and isinstance(shap_fi, pd.DataFrame):
        fi_sorted = shap_fi.sort_values("shap_importance", ascending=True).head(20)
        fig_fi = go.Figure(go.Bar(
            y=fi_sorted["feature"],
            x=fi_sorted["shap_importance"],
            orientation="h",
            marker=dict(color=fi_sorted["shap_importance"],
                        colorscale=[[0, "#1e3a5f"], [1, "#ef4444"]], showscale=False),
            hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.5f}<extra></extra>",
        ))
        fig_fi.update_layout(_lyt(height=480, xaxis=dict(title="Mean absolute SHAP value", gridcolor="#2d3748")))
        st.plotly_chart(fig_fi)
    else:
        st.info("SHAP feature importance data not available.")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Section 4: Ablation study ─────────────────────────────────────────────
    st.markdown("### Ablation study: effect of interaction constraints")
    st.caption("Comparison of XGBoost trained with vs. without domain-group interaction constraints.")
    if ablation:
        with_m = ablation.get("with_constraints", {})
        without_m = ablation.get("without_constraints", {})
        metrics_list = ["roc_auc", "f1_score", "pr_auc", "precision", "recall"]
        avail = [m for m in metrics_list if m in with_m and m in without_m]
        if avail:
            fig_ab = go.Figure()
            fig_ab.add_trace(go.Bar(
                name="With constraints", x=[m.replace("_", " ").title() for m in avail],
                y=[with_m[m] for m in avail], marker_color="#3b82f6",
                text=[f"{with_m[m]:.4f}" for m in avail], textposition="outside",
            ))
            fig_ab.add_trace(go.Bar(
                name="Without constraints", x=[m.replace("_", " ").title() for m in avail],
                y=[without_m[m] for m in avail], marker_color="#6b7280",
                text=[f"{without_m[m]:.4f}" for m in avail], textposition="outside",
            ))
            fig_ab.update_layout(_lyt(barmode="group", height=360, legend=dict(bgcolor="#1e2433")))
            st.plotly_chart(fig_ab)

            delta_auc = with_m.get("roc_auc", 0) - without_m.get("roc_auc", 0)
            delta_fpr = without_m.get("false_positive_rate", 0) - with_m.get("false_positive_rate", 0)
            st.markdown(f"""
Interaction constraints improve ROC-AUC by **{delta_auc:+.4f}** and reduce false positive rate by **{delta_fpr:+.4f}**.
The constrained model learns within-domain patterns that are more stable across polymorphic variants.
            """)
    else:
        st.info("Ablation data not available.")
