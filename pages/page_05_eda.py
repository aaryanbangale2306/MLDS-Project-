"""Page 5 — Dataset Analysis & EDA"""
import streamlit as st
import numpy as np
import pandas as pd
from pages import (
    load_dataset, gradient_divider, section_header, BG, CARD, CYAN, PURPLE,
    BENIGN_COLOR, MALWARE_COLOR, TEXT_SEC,
)
from utils.visualizations import (
    plot_class_distribution, plot_correlation_heatmap,
    plot_feature_distributions, plot_embedding_scatter,
)
from utils.preprocessing import ALL_FEATURES, FEATURE_GROUPS


def render():
    st.markdown('<h1 class="gradient-text" style="font-size:2rem; font-weight:800;">📈 Dataset Analysis & EDA</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#9CA3AF;">Exploratory data analysis of the synthetic EMBER-inspired malware detection dataset.</p>', unsafe_allow_html=True)
    gradient_divider()

    df = load_dataset(n_rows=5000)
    if df is None:
        st.warning("""
        Dataset not found. Either:
        - Run `python data/generate_dataset.py` to generate it, or
        - Run `python models/train_all_models.py` (generates data automatically).
        """)
        return

    n_total = len(df)
    n_benign = (df["label"] == 0).sum()
    n_malware = (df["label"] == 1).sum()
    n_features = len([c for c in df.columns if c != "label"])

    # ── Dataset Statistics Cards ──────────────────────────────────────────────
    section_header("📊 Dataset Overview")
    cols = st.columns(4)
    stats = [
        ("Total Samples", f"{n_total:,}", "📁"),
        ("Features", str(n_features), "⚙️"),
        ("Benign Samples", f"{n_benign:,} ({n_benign/n_total:.0%})", "✅"),
        ("Malware Samples", f"{n_malware:,} ({n_malware/n_total:.0%})", "⚠️"),
    ]
    for col, (label, value, icon) in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.5rem;">{icon}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size:1.4rem;">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    gradient_divider()

    # ── Class Distribution + Basic Stats ──────────────────────────────────────
    col_donut, col_stats = st.columns([1, 1])
    with col_donut:
        section_header("🍩 Class Distribution")
        fig_dist = plot_class_distribution(int(n_benign), int(n_malware))
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_stats:
        section_header("📋 Basic Statistics")
        feat_cols = [c for c in df.columns if c != "label"]
        stat_df = df[feat_cols].describe().T[["mean", "std", "min", "max"]]
        stat_df = stat_df.round(3)
        st.dataframe(stat_df.head(20), use_container_width=True)

    gradient_divider()

    # ── Top Polymorphic Indicators ────────────────────────────────────────────
    section_header("🧬 Top Polymorphic Indicators", "Features with highest correlation to malware label")
    corr = df.corr()["label"].drop("label").abs().sort_values(ascending=False)
    top_feats = corr.head(15)

    import plotly.graph_objects as go
    fig_corr_bar = go.Figure(go.Bar(
        x=top_feats.values,
        y=top_feats.index.tolist(),
        orientation="h",
        marker=dict(
            color=top_feats.values,
            colorscale=[[0, PURPLE], [1, CYAN]],
            showscale=False,
        ),
        hovertemplate="<b>%{y}</b><br>|Correlation|: %{x:.4f}<extra></extra>",
    ))
    fig_corr_bar.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        title="Feature Correlation with Malware Label (Absolute)",
        xaxis_title="|Pearson Correlation|",
        height=420,
        font=dict(color="#F9FAFB"),
        margin=dict(l=10, r=20, t=50, b=40),
    )
    st.plotly_chart(fig_corr_bar, use_container_width=True)

    gradient_divider()

    # ── Feature Correlation Heatmap ───────────────────────────────────────────
    section_header("🌡️ Feature Correlation Heatmap")
    fig_heat = plot_correlation_heatmap(df, max_features=20)
    st.plotly_chart(fig_heat, use_container_width=True)

    gradient_divider()

    # ── Feature Distributions by Class ───────────────────────────────────────
    section_header("📊 Feature Distributions by Class")
    selected_group = st.selectbox("Select feature group:", list(FEATURE_GROUPS.keys()), key="eda_group")
    group_features = FEATURE_GROUPS.get(selected_group, [])
    avail_features = [f for f in group_features if f in df.columns]
    if avail_features:
        fig_violin = plot_feature_distributions(df, avail_features, n_cols=3)
        st.plotly_chart(fig_violin, use_container_width=True)

    gradient_divider()

    # ── t-SNE / PCA Visualization ─────────────────────────────────────────────
    section_header("🔭 Feature Space Visualization (PCA + t-SNE)")
    st.markdown(f'<p style="color:{TEXT_SEC}; font-size:0.85rem;">2D projection of the {n_total:,}-sample feature space.</p>', unsafe_allow_html=True)

    vis_method = st.radio("Visualization method:", ["PCA", "t-SNE (slow)"], horizontal=True, key="vis_method")
    n_vis = st.slider("Samples to visualize:", 500, min(3000, n_total), 1000, 100, key="n_vis")

    if st.button("Generate Visualization", key="gen_vis"):
        with st.spinner(f"Computing {vis_method} embeddings on {n_vis} samples..."):
            subset = df.sample(n_vis, random_state=42)
            X_sub = subset[[c for c in ALL_FEATURES if c in subset.columns]].fillna(0).values
            y_sub = subset["label"].values

            from sklearn.preprocessing import StandardScaler
            X_scaled = StandardScaler().fit_transform(X_sub)

            if vis_method == "PCA":
                from sklearn.decomposition import PCA
                emb = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
                title = "PCA — Feature Space"
            else:
                from sklearn.manifold import TSNE
                emb = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=500).fit_transform(X_scaled)
                title = "t-SNE — Feature Space"

            fig_emb = plot_embedding_scatter(emb, y_sub, title)
            st.plotly_chart(fig_emb, use_container_width=True)

    gradient_divider()

    # ── Per-Group Statistics ──────────────────────────────────────────────────
    section_header("📦 Feature Group Statistics by Class")
    benign_df = df[df["label"] == 0]
    malware_df = df[df["label"] == 1]
    group_stats = []
    for group, features in FEATURE_GROUPS.items():
        avail = [f for f in features if f in df.columns]
        for feat in avail:
            group_stats.append({
                "Group": group,
                "Feature": feat,
                "Benign Mean": benign_df[feat].mean(),
                "Malware Mean": malware_df[feat].mean(),
                "Ratio (M/B)": malware_df[feat].mean() / (benign_df[feat].mean() + 1e-8),
            })
    gdf = pd.DataFrame(group_stats).round(3).sort_values("Ratio (M/B)", ascending=False)
    st.dataframe(gdf, use_container_width=True)
