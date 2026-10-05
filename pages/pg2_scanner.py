"""Page 2 - Live Malware Scanner with SHAP explanations"""
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from pathlib import Path

from utils.preprocessing import ALL_FEATURES, FEATURE_GROUPS
from data.generate_dataset import generate_benign_sample, generate_malware_sample, generate_polymorphic_sample

SAVED = Path(__file__).parent.parent / "models" / "saved"


@st.cache_resource
def _load_models():
    files = {
        "XGBoost": "xgboost_model.pkl",
        "Random Forest": "random_forest_model.pkl",
        "LightGBM": "lightgbm_model.pkl",
        "CatBoost": "catboost_model.pkl",
        "Hybrid (IF+GB)": "hybrid_model.pkl",
    }
    models = {}
    for name, fname in files.items():
        p = SAVED / fname
        if p.exists():
            try:
                models[name] = joblib.load(str(p))
            except Exception:
                pass
    return models


@st.cache_resource
def _load_preprocessor():
    p = SAVED / "preprocessor.pkl"
    return joblib.load(str(p)) if p.exists() else None


@st.cache_resource
def _load_shap_explainer():
    p = SAVED / "shap_explainer.pkl"
    return joblib.load(str(p)) if p.exists() else None


def _predict(sample_dict, models, preprocessor):
    X_raw = np.array([[float(sample_dict.get(f, 0)) for f in ALL_FEATURES]])
    results = {}
    for name, model in models.items():
        try:
            if name == "Hybrid (IF+GB)":
                proba = float(model.predict_proba(X_raw)[0, 1])
            elif preprocessor is not None:
                X_s = preprocessor.transform(pd.DataFrame(X_raw, columns=ALL_FEATURES))
                proba = float(model.predict_proba(X_s)[0, 1])
            else:
                proba = float(model.predict_proba(X_raw)[0, 1])
            results[name] = proba
        except Exception as e:
            results[name] = -1.0
    return results


def _compute_shap(sample_dict, explainer):
    """Compute SHAP values for a single sample using the saved XGBoost explainer."""
    try:
        X = np.array([float(sample_dict.get(f, 0)) for f in ALL_FEATURES])
        if hasattr(explainer, "get_waterfall_data"):
            wdata = explainer.get_waterfall_data(X, max_display=12)
            return {
                "features": wdata["feature_names"],
                "values": wdata["shap_values"],
                "feature_vals": wdata["feature_values"],
            }
        elif hasattr(explainer, "explainer") and hasattr(explainer.explainer, "shap_values"):
            shap_vals = explainer.explainer.shap_values(X.reshape(1, -1))
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            sv = shap_vals[0]
            order = np.argsort(np.abs(sv))[::-1][:12]
            return {
                "features": [ALL_FEATURES[i] for i in order],
                "values": sv[order].tolist(),
                "feature_vals": X[order].tolist(),
            }
        return None
    except Exception:
        return None


def render():
    st.markdown("# Scanner")
    st.markdown("Load a sample or enter custom feature values, then run analysis with all 5 models.")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    models = _load_models()
    preprocessor = _load_preprocessor()
    shap_explainer = _load_shap_explainer()

    if not models:
        st.warning("Models not trained yet. Run `python models/train_all_models.py` first.")
        return

    # Session state
    if "sample" not in st.session_state:
        st.session_state.sample = generate_benign_sample()
    if "results" not in st.session_state:
        st.session_state.results = None

    # Preset buttons
    st.markdown("### Load preset sample")
    st.caption("These presets use the same feature distributions as the training data generator.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("Benign sample", use_container_width=True):
            st.session_state.sample = generate_benign_sample()
            st.session_state.results = None
            st.rerun()
    with c2:
        if st.button("Malware sample", use_container_width=True):
            st.session_state.sample = generate_malware_sample()
            st.session_state.results = None
            st.rerun()
    with c3:
        if st.button("Polymorphic sample", use_container_width=True):
            st.session_state.sample = generate_polymorphic_sample()
            st.session_state.results = None
            st.rerun()
    with c4:
        uploaded = st.file_uploader("Upload CSV (one row)", type=["csv"], label_visibility="collapsed")
        if uploaded:
            try:
                df_up = pd.read_csv(uploaded)
                missing = set(ALL_FEATURES) - set(df_up.columns)
                if missing:
                    st.error(f"CSV missing features: {sorted(missing)[:5]}...")
                elif len(df_up) == 0:
                    st.error("CSV file is empty.")
                else:
                    st.session_state.sample = {c: float(df_up.iloc[0][c]) for c in ALL_FEATURES if c in df_up.columns}
                    st.session_state.results = None
                    st.rerun()
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")

    # Feature editor
    sample = dict(st.session_state.sample)

    with st.expander("Edit feature values", expanded=False):
        for group, feats in FEATURE_GROUPS.items():
            st.markdown(f"**{group}**")
            cols = st.columns(min(4, len(feats)))
            for j, feat in enumerate(feats):
                with cols[j % len(cols)]:
                    val = sample.get(feat, 0)
                    if feat in {"has_debug", "has_signature", "uses_crypto_api",
                                "uses_network_api", "uses_process_api", "uses_registry_api"}:
                        sample[feat] = int(st.checkbox(feat, value=bool(val), key=f"e_{feat}"))
                    elif feat in {"polymorphic_score", "obfuscation_level", "code_mutation_rate",
                                  "c2_similarity_score", "memory_allocation_pattern"}:
                        sample[feat] = st.slider(feat, 0.0, 1.0, float(val), 0.01, key=f"e_{feat}")
                    elif feat in {"avg_section_entropy", "max_section_entropy", "packet_entropy"}:
                        sample[feat] = st.slider(feat, 0.0, 8.0, float(val), 0.1, key=f"e_{feat}")
                    else:
                        sample[feat] = st.number_input(feat, value=float(val), key=f"e_{feat}", format="%g")

    st.session_state.sample = sample
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Scan button
    if st.button("Run analysis", type="primary", use_container_width=True):
        with st.spinner("Running models..."):
            st.session_state.results = _predict(sample, models, preprocessor)

    # Results
    if st.session_state.results:
        results = st.session_state.results
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        xgb_p = results.get("XGBoost", 0.5)
        is_mal = xgb_p >= 0.5

        # Verdict banner
        if is_mal:
            st.markdown(f"""
            <div style="background:#3b0a0a; border:1px solid #dc2626; border-radius:8px; padding:1rem 1.5rem; text-align:center;">
              <div style="font-size:1.3rem; font-weight:700; color:#f87171;">Malware detected</div>
              <div style="color:#fca5a5; font-size:0.85rem; margin-top:0.3rem;">XGBoost malware probability: {xgb_p:.1%}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:#052e16; border:1px solid #16a34a; border-radius:8px; padding:1rem 1.5rem; text-align:center;">
              <div style="font-size:1.3rem; font-weight:700; color:#4ade80;">Benign</div>
              <div style="color:#86efac; font-size:0.85rem; margin-top:0.3rem;">XGBoost benign probability: {1-xgb_p:.1%}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # All models
        st.markdown("### Model predictions")
        cols = st.columns(len(results))
        for col, (name, proba) in zip(cols, results.items()):
            with col:
                if proba < 0:
                    lbl, clr = "Error", "#6b7280"
                    display = "err"
                elif proba >= 0.5:
                    lbl, clr = "Malware", "#f87171"
                    display = f"{proba:.3f}"
                else:
                    lbl, clr = "Benign", "#4ade80"
                    display = f"{proba:.3f}"
                st.markdown(f"""
                <div class="stat-card" style="border-top:3px solid {clr}">
                  <div style="font-size:0.75rem; color:#9ca3af; margin-bottom:0.3rem;">{name}</div>
                  <div style="font-size:1.4rem; font-weight:700; color:{clr}; font-family:'JetBrains Mono';">{display}</div>
                  <div style="font-size:0.72rem; color:{clr}; margin-top:0.2rem;">{lbl}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # SHAP explanation
        st.markdown("### Why this prediction was made")
        if shap_explainer is not None:
            shap_result = _compute_shap(sample, shap_explainer)
            if shap_result is not None:
                st.caption("SHAP values show each feature's contribution to the XGBoost prediction. "
                           "Positive values push toward malware; negative toward benign.")
                features = shap_result["features"]
                sv = shap_result["values"]
                fv = shap_result["feature_vals"]

                colors = ["#ef4444" if v > 0 else "#3b82f6" for v in sv]
                fig = go.Figure(go.Bar(
                    y=features,
                    x=sv,
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:.4f}" for v in sv],
                    textposition="outside",
                    customdata=[[round(f, 4)] for f in fv],
                    hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<br>Feature value: %{customdata[0]}<extra></extra>",
                ))
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0f1117",
                    plot_bgcolor="#1e2433",
                    font=dict(family="Inter, sans-serif", color="#e5e7eb", size=11),
                    height=420,
                    margin=dict(l=160, r=80, t=30, b=40),
                    xaxis=dict(title="SHAP value (impact on malware probability)", gridcolor="#2d3748",
                               zeroline=True, zerolinecolor="#4b5563", zerolinewidth=1.5),
                    yaxis=dict(gridcolor="#2d3748"),
                )
                st.plotly_chart(fig)

                # Table
                rows = [{"Feature": f, "Feature value": round(fv_, 4),
                         "SHAP value": round(sv_, 4),
                         "Direction": "toward malware" if sv_ > 0 else "toward benign"}
                        for f, fv_, sv_ in zip(features, fv, sv)]
                st.dataframe(pd.DataFrame(rows), hide_index=True)
            else:
                st.info("SHAP computation failed for this sample.")
        else:
            # Fallback: show key feature values with risk context
            st.caption("SHAP explainer not available. Showing key feature values for context.")
            risk_feats = ["polymorphic_score", "obfuscation_level", "code_mutation_rate",
                          "avg_section_entropy", "max_section_entropy", "c2_similarity_score",
                          "packet_entropy", "connection_attempts", "num_suspicious_imports"]
            feat_rows = [{"Feature": f, "Value": round(float(sample.get(f, 0)), 4)}
                         for f in risk_feats if f in sample]
            st.dataframe(pd.DataFrame(feat_rows), hide_index=True)
