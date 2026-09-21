"""
Reusable Plotly visualization functions with cybersecurity dark theme.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional, Tuple

# ─── Theme Constants ──────────────────────────────────────────────────────────
BG_COLOR = "#0A0E1A"
CARD_COLOR = "#111827"
ACCENT_CYAN = "#00D4FF"
ACCENT_PURPLE = "#7C3AED"
COLOR_BENIGN = "#10B981"
COLOR_MALWARE = "#EF4444"
COLOR_WARNING = "#F59E0B"
TEXT_PRIMARY = "#F9FAFB"
TEXT_SECONDARY = "#9CA3AF"
BORDER_COLOR = "#1F2937"
GRID_COLOR = "#1F2937"

MODEL_COLORS = {
    "XGBoost": "#00D4FF",
    "Random Forest": "#7C3AED",
    "LightGBM": "#10B981",
    "CatBoost": "#F59E0B",
    "Hybrid (IF+GB)": "#EF4444",
}

DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=BG_COLOR,
    plot_bgcolor=CARD_COLOR,
    font=dict(family="Inter, sans-serif", color=TEXT_PRIMARY, size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
)


def _apply_dark(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply unified dark theme to any figure."""
    fig.update_layout(**DARK_LAYOUT, title=dict(text=title, font=dict(size=16, color=TEXT_PRIMARY)))
    return fig


# ─── ROC Curves ───────────────────────────────────────────────────────────────

def plot_roc_curves(roc_data: Dict[str, Dict]) -> go.Figure:
    """
    Plot ROC curves for multiple models on one figure.

    Args:
        roc_data: {model_name: {fpr: [...], tpr: [...], auc: float}}
    """
    fig = go.Figure()
    # Diagonal (random)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color=TEXT_SECONDARY, width=1),
        name="Random Classifier", showlegend=True,
    ))
    for model_name, data in roc_data.items():
        color = MODEL_COLORS.get(model_name, ACCENT_CYAN)
        auc = data.get("auc", 0)
        fig.add_trace(go.Scatter(
            x=data["fpr"], y=data["tpr"], mode="lines",
            name=f"{model_name} (AUC={auc:.4f})",
            line=dict(color=color, width=2.5),
            hovertemplate=f"<b>{model_name}</b><br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        **DARK_LAYOUT,
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate (Recall)",
        legend=dict(bgcolor=CARD_COLOR, bordercolor=BORDER_COLOR),
    )
    return fig


# ─── Precision-Recall Curves ──────────────────────────────────────────────────

def plot_pr_curves(pr_data: Dict[str, Dict]) -> go.Figure:
    """Plot PR curves for multiple models."""
    fig = go.Figure()
    for model_name, data in pr_data.items():
        color = MODEL_COLORS.get(model_name, ACCENT_CYAN)
        auc = data.get("pr_auc", 0)
        fig.add_trace(go.Scatter(
            x=data["recall"], y=data["precision"], mode="lines",
            name=f"{model_name} (PR-AUC={auc:.4f})",
            line=dict(color=color, width=2.5),
            hovertemplate=f"<b>{model_name}</b><br>Recall: %{{x:.3f}}<br>Precision: %{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        **DARK_LAYOUT,
        title="Precision-Recall Curves — All Models",
        xaxis_title="Recall",
        yaxis_title="Precision",
        legend=dict(bgcolor=CARD_COLOR, bordercolor=BORDER_COLOR),
    )
    return fig


# ─── Metrics Grouped Bar Chart ────────────────────────────────────────────────

def plot_metrics_bar(metrics_df: pd.DataFrame, selected_metrics: Optional[List[str]] = None) -> go.Figure:
    """Grouped bar chart of all models across key metrics."""
    if selected_metrics is None:
        selected_metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc", "mcc"]

    fig = go.Figure()
    for model_name in metrics_df.index:
        color = MODEL_COLORS.get(model_name, ACCENT_CYAN)
        values = [metrics_df.loc[model_name, m] if m in metrics_df.columns else 0
                  for m in selected_metrics]
        fig.add_trace(go.Bar(
            name=model_name,
            x=selected_metrics,
            y=values,
            marker_color=color,
            hovertemplate=f"<b>{model_name}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>",
        ))
    fig.update_layout(
        **DARK_LAYOUT,
        title="Model Performance Comparison",
        barmode="group",
        xaxis_tickangle=-30,
        legend=dict(bgcolor=CARD_COLOR, bordercolor=BORDER_COLOR),
        yaxis=dict(range=[0, 1.05], gridcolor=GRID_COLOR),
    )
    return fig


# ─── Radar Chart ──────────────────────────────────────────────────────────────

def plot_radar_chart(metrics_df: pd.DataFrame) -> go.Figure:
    """Spider/radar chart comparing models across 6 dimensions."""
    dims = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "mcc"]
    dims_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "MCC"]

    fig = go.Figure()
    for model_name in metrics_df.index:
        color = MODEL_COLORS.get(model_name, ACCENT_CYAN)
        vals = [metrics_df.loc[model_name, d] if d in metrics_df.columns else 0 for d in dims]
        vals_closed = vals + [vals[0]]  # close the radar
        dims_closed = dims_labels + [dims_labels[0]]

        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=dims_closed,
            fill="toself",
            name=model_name,
            line_color=color,
            fillcolor=color + "33",  # 20% opacity
        ))

    fig.update_layout(
        **DARK_LAYOUT,
        title="Model Performance Radar",
        polar=dict(
            bgcolor=CARD_COLOR,
            radialaxis=dict(visible=True, range=[0, 1], color=TEXT_SECONDARY, gridcolor=GRID_COLOR),
            angularaxis=dict(color=TEXT_PRIMARY, gridcolor=GRID_COLOR),
        ),
        legend=dict(bgcolor=CARD_COLOR, bordercolor=BORDER_COLOR),
    )
    return fig


# ─── Confusion Matrix ─────────────────────────────────────────────────────────

def plot_confusion_matrix(tp: int, tn: int, fp: int, fn: int, model_name: str = "") -> go.Figure:
    """Heatmap confusion matrix."""
    z = [[tn, fp], [fn, tp]]
    annotations = [
        [f"TN<br>{tn:,}", f"FP<br>{fp:,}"],
        [f"FN<br>{fn:,}", f"TP<br>{tp:,}"],
    ]
    fig = go.Figure(go.Heatmap(
        z=z,
        x=["Predicted Benign", "Predicted Malware"],
        y=["Actual Benign", "Actual Malware"],
        colorscale=[[0, CARD_COLOR], [0.5, ACCENT_PURPLE], [1, ACCENT_CYAN]],
        showscale=False,
        text=annotations,
        texttemplate="%{text}",
        hovertemplate="Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        title=f"Confusion Matrix — {model_name}",
        font=dict(size=14),
    )
    return fig


# ─── SHAP Summary (Beeswarm approximation) ────────────────────────────────────

def plot_shap_summary(shap_data: Dict[str, Any]) -> go.Figure:
    """
    Approximate beeswarm SHAP summary plot using scatter.
    """
    shap_vals = np.array(shap_data["shap_values"])        # (n, k)
    feat_vals = np.array(shap_data["feature_values"])     # (n, k)
    feat_names = shap_data["feature_names"]               # (k,)
    importance = shap_data["feature_importance"]          # (k,)

    n, k = shap_vals.shape
    fig = go.Figure()

    # Add jitter for beeswarm effect
    rng = np.random.default_rng(42)

    for i in range(k - 1, -1, -1):  # plot from bottom to top
        feat_name = feat_names[i]
        sv = shap_vals[:, i]
        fv = feat_vals[:, i]

        # Normalize feature values for color
        fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-8)
        colors = [f"rgb({int(255*v)}, {int(100 + 50*(1-v))}, {int(255*(1-v))})" for v in fv_norm]

        jitter = rng.normal(0, 0.1, size=n)
        fig.add_trace(go.Scatter(
            x=sv,
            y=[feat_name] * n + jitter,  # type: ignore
            mode="markers",
            marker=dict(color=colors, size=3, opacity=0.6),
            name=feat_name,
            showlegend=False,
            hovertemplate=f"<b>{feat_name}</b><br>SHAP: %{{x:.4f}}<extra></extra>",
        ))

    fig.add_vline(x=0, line_color=TEXT_SECONDARY, line_dash="dot")
    fig.update_layout(
        **DARK_LAYOUT,
        title="SHAP Feature Importance (Summary)",
        xaxis_title="SHAP Value (impact on prediction)",
        height=max(400, k * 25),
    )
    return fig


# ─── SHAP Waterfall ───────────────────────────────────────────────────────────

def plot_shap_waterfall(waterfall_data: Dict[str, Any], prediction: float = 0.5) -> go.Figure:
    """Waterfall chart for a single sample's SHAP breakdown."""
    base = waterfall_data["base_value"]
    shap_vals = waterfall_data["shap_values"]
    feat_names = waterfall_data["feature_names"]
    feat_vals = waterfall_data["feature_values"]

    # Sort by |SHAP|
    order = np.argsort(np.abs(shap_vals))[::-1]
    sv = np.array(shap_vals)[order]
    fn = [feat_names[i] for i in order]
    fv = [feat_vals[i] for i in order]

    colors = [COLOR_MALWARE if v > 0 else COLOR_BENIGN for v in sv]
    labels = [f"{n}<br>={v:.3f}" for n, v in zip(fn, fv)]

    fig = go.Figure(go.Waterfall(
        orientation="h",
        measure=["relative"] * len(sv) + ["total"],
        x=list(sv) + [base],
        y=labels + ["Base Value"],
        connector=dict(line=dict(color=BORDER_COLOR)),
        increasing=dict(marker=dict(color=COLOR_MALWARE)),
        decreasing=dict(marker=dict(color=COLOR_BENIGN)),
        totals=dict(marker=dict(color=ACCENT_PURPLE)),
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        title=f"SHAP Waterfall — Predicted P(malware)={prediction:.3f}",
        xaxis_title="SHAP Value",
        height=max(400, len(sv) * 35),
    )
    return fig


# ─── Feature Importance Bar ───────────────────────────────────────────────────

def plot_feature_importance(importance_df: pd.DataFrame, title: str = "Feature Importance") -> go.Figure:
    """Horizontal bar chart of feature importances."""
    df = importance_df.head(20).sort_values(
        importance_df.columns[-1] if "importance" in importance_df.columns else importance_df.columns[0]
    )
    fig = go.Figure(go.Bar(
        x=df.iloc[:, -1],
        y=df.iloc[:, 0] if "feature" in df.columns else df.index,
        orientation="h",
        marker=dict(
            color=df.iloc[:, -1],
            colorscale=[[0, ACCENT_PURPLE], [1, ACCENT_CYAN]],
            showscale=False,
        ),
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        title=title,
        xaxis_title="Importance",
        height=500,
    )
    return fig


# ─── Distribution Plots ───────────────────────────────────────────────────────

def plot_feature_distributions(df: pd.DataFrame, features: List[str], n_cols: int = 3) -> go.Figure:
    """Violin plots of selected features split by class."""
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=features)

    benign = df[df["label"] == 0]
    malware_df = df[df["label"] == 1]

    for i, feat in enumerate(features):
        row, col = divmod(i, n_cols)
        row += 1; col += 1
        fig.add_trace(go.Violin(
            x=[0] * len(benign), y=benign[feat],
            name="Benign", legendgroup="benign", showlegend=(i == 0),
            line_color=COLOR_BENIGN, fillcolor=COLOR_BENIGN + "55",
            side="negative", meanline_visible=True,
        ), row=row, col=col)
        fig.add_trace(go.Violin(
            x=[0] * len(malware_df), y=malware_df[feat],
            name="Malware", legendgroup="malware", showlegend=(i == 0),
            line_color=COLOR_MALWARE, fillcolor=COLOR_MALWARE + "55",
            side="positive", meanline_visible=True,
        ), row=row, col=col)

    fig.update_layout(
        **DARK_LAYOUT,
        title="Feature Distributions by Class",
        height=n_rows * 250,
        violingap=0,
        violinmode="overlay",
    )
    return fig


# ─── Correlation Heatmap ──────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, max_features: int = 25) -> go.Figure:
    """Correlation heatmap for top features."""
    numeric = df.select_dtypes(include=[np.number])
    if len(numeric.columns) > max_features:
        # Select top correlated with label
        corr_label = numeric.corr()["label"].abs().drop("label").sort_values(ascending=False)
        top_cols = corr_label.head(max_features - 1).index.tolist() + ["label"]
        numeric = numeric[top_cols]

    corr = numeric.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[[0, COLOR_MALWARE], [0.5, CARD_COLOR], [1, ACCENT_CYAN]],
        zmid=0,
        hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        title="Feature Correlation Matrix",
        height=600,
        xaxis_tickangle=-45,
    )
    return fig


# ─── Class Distribution Donut ─────────────────────────────────────────────────

def plot_class_distribution(n_benign: int, n_malware: int) -> go.Figure:
    """Donut chart for class distribution."""
    fig = go.Figure(go.Pie(
        labels=["Benign", "Malware"],
        values=[n_benign, n_malware],
        hole=0.6,
        marker=dict(colors=[COLOR_BENIGN, COLOR_MALWARE],
                    line=dict(color=BG_COLOR, width=3)),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        title="Class Distribution",
        annotations=[dict(
            text=f"{n_benign + n_malware:,}<br>total",
            x=0.5, y=0.5, font_size=16, showarrow=False,
            font_color=TEXT_PRIMARY,
        )],
    )
    return fig


# ─── Optuna History ───────────────────────────────────────────────────────────

def plot_optuna_history(history: Dict[str, Any]) -> go.Figure:
    """Plot Optuna optimization history."""
    trial_nums = history["trial_numbers"]
    values = history["values"]
    best_values = history["best_values"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trial_nums, y=values, mode="markers",
        name="Trial AUC", marker=dict(color=ACCENT_PURPLE, size=5, opacity=0.6),
    ))
    fig.add_trace(go.Scatter(
        x=trial_nums, y=best_values, mode="lines",
        name="Best AUC", line=dict(color=ACCENT_CYAN, width=2.5),
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        title=f"Optuna Hyperparameter Optimization ({len(trial_nums)} trials)",
        xaxis_title="Trial Number",
        yaxis_title="ROC-AUC Score",
        legend=dict(bgcolor=CARD_COLOR),
    )
    return fig


# ─── Interaction Strength Heatmap ─────────────────────────────────────────────

def plot_interaction_heatmap(interaction_matrix: np.ndarray, feature_names: List[str]) -> go.Figure:
    """Heatmap showing pairwise feature interaction strengths."""
    fig = go.Figure(go.Heatmap(
        z=interaction_matrix,
        x=feature_names,
        y=feature_names,
        colorscale=[[0, BG_COLOR], [0.5, ACCENT_PURPLE], [1, ACCENT_CYAN]],
        hovertemplate="%{y} × %{x}: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        title="Feature Interaction Strength Matrix",
        height=600,
        xaxis_tickangle=-45,
    )
    return fig


# ─── t-SNE / UMAP Scatter ─────────────────────────────────────────────────────

def plot_embedding_scatter(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str = "Feature Space Visualization",
) -> go.Figure:
    """2D scatter plot of t-SNE or UMAP embeddings."""
    benign_mask = labels == 0
    malware_mask = labels == 1

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=embeddings[benign_mask, 0], y=embeddings[benign_mask, 1],
        mode="markers", name="Benign",
        marker=dict(color=COLOR_BENIGN, size=3, opacity=0.5),
    ))
    fig.add_trace(go.Scatter(
        x=embeddings[malware_mask, 0], y=embeddings[malware_mask, 1],
        mode="markers", name="Malware",
        marker=dict(color=COLOR_MALWARE, size=4, opacity=0.7),
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        title=title,
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        legend=dict(bgcolor=CARD_COLOR),
    )
    return fig


# ─── Gauge (Confidence / Threat Level) ───────────────────────────────────────

def plot_confidence_gauge(probability: float, title: str = "Threat Confidence") -> go.Figure:
    """Gauge chart for malware probability."""
    color = COLOR_MALWARE if probability >= 0.5 else COLOR_BENIGN
    label = "MALWARE" if probability >= 0.5 else "BENIGN"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        number=dict(suffix="%", font=dict(color=color, size=36)),
        title=dict(text=f"{title}<br><span style='font-size:20px;color:{color}'>{label}</span>",
                   font=dict(color=TEXT_PRIMARY, size=14)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=TEXT_SECONDARY),
            bar=dict(color=color),
            bgcolor=CARD_COLOR,
            bordercolor=BORDER_COLOR,
            steps=[
                dict(range=[0, 30], color=COLOR_BENIGN + "44"),
                dict(range=[30, 70], color=COLOR_WARNING + "44"),
                dict(range=[70, 100], color=COLOR_MALWARE + "44"),
            ],
            threshold=dict(line=dict(color=TEXT_PRIMARY, width=3), thickness=0.75, value=50),
        ),
    ))
    fig.update_layout(paper_bgcolor=BG_COLOR, font=dict(color=TEXT_PRIMARY), height=300)
    return fig
