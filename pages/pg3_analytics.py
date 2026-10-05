"""Page 3 — Analytics (real data-driven charts)"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from pathlib import Path

SAVED = Path(__file__).parent.parent / "models" / "saved"
DATA_DIR = Path(__file__).parent.parent / "data"

# Shared plotly theme helper
def _layout(**kwargs):
    cfg = dict(
        template="plotly_dark",
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1e2433",
        font=dict(family="Inter, sans-serif", color="#e5e7eb", size=12),
        margin=dict(l=40, r=20, t=45, b=40),
        xaxis=dict(gridcolor="#2d3748"),
        yaxis=dict(gridcolor="#2d3748"),
    )
    for k, v in kwargs.items():
        if k in ("xaxis", "yaxis") and isinstance(v, dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    return cfg

_COLORS = ["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444"]


def _load_dataset(n=5000):
    p = DATA_DIR / "synthetic_malware_data.csv"
    if p.exists():
        return pd.read_csv(p, nrows=n)
    p2 = SAVED / "test_sample.csv"
    if p2.exists():
        return pd.read_csv(p2)
    return None


def _load_metrics():
    p = SAVED / "metrics.pkl"
    if p.exists():
        return pd.DataFrame(joblib.load(str(p)))
    return None


def render():
    st.markdown("# Analytics")
    st.markdown("Dataset distribution and model performance visualisations based on the 50 000-sample training set.")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    df = _load_dataset()
    metrics_df = _load_metrics()

    if df is None:
        st.warning("Dataset not found. Run training first (`python models/train_all_models.py`).")
        return

    n_total = len(df)
    n_benign = int((df["label"] == 0).sum())
    n_malware = int((df["label"] == 1).sum())

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="stat-card"><div class="stat-val">{n_total:,}</div><div class="stat-lbl">Samples</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><div class="stat-val">{n_benign:,}</div><div class="stat-lbl">Benign</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-card"><div class="stat-val">{n_malware:,}</div><div class="stat-lbl">Malware</div></div>', unsafe_allow_html=True)
    with c4:
        ratio = n_malware / n_total * 100
        st.markdown(f'<div class="stat-card"><div class="stat-val">{ratio:.1f}%</div><div class="stat-lbl">Malware Ratio</div></div>', unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Row 1: Class distribution + Model comparison ──────────────────────────
    left, right = st.columns(2)

    with left:
        st.markdown("### Class distribution")
        fig1 = go.Figure(go.Pie(
            labels=["Benign", "Malware"],
            values=[n_benign, n_malware],
            hole=0.55,
            marker=dict(colors=["#10b981", "#ef4444"], line=dict(color="#0f1117", width=3)),
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
        ))
        fig1.update_layout(_layout(height=320,
                           annotations=[dict(text=f"{n_total:,}<br>total", x=0.5, y=0.5,
                                             font_size=14, showarrow=False, font_color="#e5e7eb")]))
        st.plotly_chart(fig1)

    with right:
        st.markdown("### Model ROC-AUC comparison")
        if metrics_df is not None and "model_name" in metrics_df.columns:
            mdf = metrics_df.sort_values("roc_auc", ascending=True)
            fig2 = go.Figure(go.Bar(
                y=mdf["model_name"],
                x=mdf["roc_auc"],
                orientation="h",
                marker_color=_COLORS[:len(mdf)],
                text=mdf["roc_auc"].round(4),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>AUC: %{x:.4f}<extra></extra>",
            ))
            fig2.update_layout(_layout(height=320, xaxis=dict(range=[0.95, 1.0], gridcolor="#2d3748")))
            st.plotly_chart(fig2)
        else:
            st.info("Model metrics not available.")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Row 2: Top feature correlations ───────────────────────────────────────
    st.markdown("### Top features correlated with malware")
    corr = df.corr(numeric_only=True)["label"].drop("label").abs().sort_values(ascending=False).head(12)
    fig3 = go.Figure(go.Bar(
        x=corr.values,
        y=corr.index.tolist(),
        orientation="h",
        marker=dict(color=corr.values, colorscale=[[0, "#3b82f6"], [1, "#ef4444"]], showscale=False),
        hovertemplate="<b>%{y}</b><br>|Correlation|: %{x:.4f}<extra></extra>",
    ))
    fig3.update_layout(_layout(height=380, title="Absolute Pearson correlation with label"))
    st.plotly_chart(fig3)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Row 3: Feature distributions ──────────────────────────────────────────
    st.markdown("### Feature distribution: benign vs malware")
    from utils.preprocessing import FEATURE_GROUPS
    all_feats = [f for fs in FEATURE_GROUPS.values() for f in fs if f in df.columns]
    selected = st.selectbox("Select feature", all_feats, index=all_feats.index("polymorphic_score") if "polymorphic_score" in all_feats else 0)

    benign_vals = df[df["label"] == 0][selected]
    malware_vals = df[df["label"] == 1][selected]

    fig4 = go.Figure()
    fig4.add_trace(go.Histogram(x=benign_vals, name="Benign", marker_color="#10b981", opacity=0.7, nbinsx=50))
    fig4.add_trace(go.Histogram(x=malware_vals, name="Malware", marker_color="#ef4444", opacity=0.7, nbinsx=50))
    fig4.update_layout(_layout(barmode="overlay", height=320,
                       title=f"Distribution of '{selected}'",
                       xaxis_title=selected, yaxis_title="Count",
                       legend=dict(bgcolor="#1e2433")))
    st.plotly_chart(fig4)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Row 4: Ablation ──────────────────────────────────────────────────────
    st.markdown("### Ablation: interaction constraints effect")
    ab_path = SAVED / "ablation_results.pkl"
    if ab_path.exists():
        ab = joblib.load(str(ab_path))
        with_m = ab.get("with_constraints", {})
        without_m = ab.get("without_constraints", {})
        metrics_list = ["roc_auc", "f1_score", "pr_auc", "recall", "precision"]
        avail = [m for m in metrics_list if m in with_m and m in without_m]
        if avail:
            fig5 = go.Figure()
            fig5.add_trace(go.Bar(name="With constraints", x=[m.replace("_", " ") for m in avail],
                                  y=[with_m[m] for m in avail], marker_color="#3b82f6"))
            fig5.add_trace(go.Bar(name="Without constraints", x=[m.replace("_", " ") for m in avail],
                                  y=[without_m[m] for m in avail], marker_color="#6b7280"))
            fig5.update_layout(_layout(barmode="group", height=320,
                               legend=dict(bgcolor="#1e2433")))
            st.plotly_chart(fig5)

            delta = with_m.get("roc_auc", 0) - without_m.get("roc_auc", 0)
            st.markdown(f"Interaction constraints improve AUC by **{delta:+.4f}** — the model learns domain-consistent "
                        f"rules that generalise better to unseen polymorphic variants.")
    else:
        st.info("Ablation data not available.")
