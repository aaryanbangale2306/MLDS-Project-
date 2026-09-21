"""Page 4 — Live Malware Scanner"""
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from pages import (
    gradient_divider, section_header, models_not_trained_warning,
    CYAN, PURPLE, BENIGN_COLOR, MALWARE_COLOR, BG, CARD, TEXT_SEC, SAVED_DIR,
)
from utils.preprocessing import ALL_FEATURES, FEATURE_GROUPS
from utils.visualizations import plot_confidence_gauge
from data.generate_dataset import generate_benign_sample, generate_malware_sample, generate_polymorphic_sample


@st.cache_resource
def load_all_models():
    """Load all 5 trained models."""
    model_files = {
        "XGBoost": SAVED_DIR / "xgboost_model.pkl",
        "Random Forest": SAVED_DIR / "random_forest_model.pkl",
        "LightGBM": SAVED_DIR / "lightgbm_model.pkl",
        "CatBoost": SAVED_DIR / "catboost_model.pkl",
        "Hybrid (IF+GB)": SAVED_DIR / "hybrid_model.pkl",
    }
    models = {}
    for name, path in model_files.items():
        if path.exists():
            try:
                models[name] = joblib.load(str(path))
            except Exception as e:
                st.warning(f"Could not load {name}: {e}")
    return models


@st.cache_resource
def load_preprocessor():
    path = SAVED_DIR / "preprocessor.pkl"
    return joblib.load(str(path)) if path.exists() else None


@st.cache_resource
def load_shap_explainer():
    path = SAVED_DIR / "shap_explainer.pkl"
    return joblib.load(str(path)) if path.exists() else None


def _sample_to_array(sample_dict: dict) -> np.ndarray:
    """Convert feature dict to numpy array in correct feature order."""
    return np.array([[float(sample_dict.get(f, 0)) for f in ALL_FEATURES]])


def _run_all_models(sample_dict: dict, models: dict, preprocessor) -> dict:
    """Run all models on a single sample, return probabilities."""
    X_raw = _sample_to_array(sample_dict)
    results = {}
    for name, model in models.items():
        try:
            if name == "Hybrid (IF+GB)":
                proba = model.predict_proba(X_raw)[0, 1]
            elif preprocessor is not None:
                X_scaled = preprocessor.transform(pd.DataFrame(X_raw, columns=ALL_FEATURES))
                proba = model.predict_proba(X_scaled)[0, 1]
            else:
                proba = model.predict_proba(X_raw)[0, 1]
            results[name] = float(proba)
        except Exception as e:
            results[name] = -1.0  # error sentinel
    return results


def _feature_risk_by_group(shap_values_dict: dict) -> dict:
    """Sum SHAP values by feature group."""
    group_risks = {}
    for group, features in FEATURE_GROUPS.items():
        total = sum(abs(shap_values_dict.get(f, 0)) for f in features)
        group_risks[group] = total
    total_all = sum(group_risks.values()) + 1e-8
    return {g: v / total_all for g, v in group_risks.items()}


def render():
    st.markdown('<h1 class="gradient-text" style="font-size:2rem; font-weight:800;">🧪 Live Malware Scanner</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#9CA3AF;">Upload a CSV file or enter feature values manually to scan for malware in real-time using all 5 trained models.</p>', unsafe_allow_html=True)
    gradient_divider()

    models = load_all_models()
    preprocessor = load_preprocessor()
    shap_explainer = load_shap_explainer()

    if not models:
        models_not_trained_warning()
        return

    # ── Initialize session state ──────────────────────────────────────────────
    if "scan_sample" not in st.session_state:
        st.session_state.scan_sample = generate_benign_sample()
    if "scan_results" not in st.session_state:
        st.session_state.scan_results = None

    # ── Sample Presets ────────────────────────────────────────────────────────
    section_header("📋 Load Sample Preset")
    col_b, col_m, col_p = st.columns(3)
    with col_b:
        if st.button("✅ Load Benign Sample", use_container_width=True):
            st.session_state.scan_sample = generate_benign_sample()
            st.session_state.scan_results = None
            st.rerun()
    with col_m:
        if st.button("⚠️ Load Malware Sample", use_container_width=True):
            st.session_state.scan_sample = generate_malware_sample()
            st.session_state.scan_results = None
            st.rerun()
    with col_p:
        if st.button("🧬 Load Polymorphic Sample", use_container_width=True):
            st.session_state.scan_sample = generate_polymorphic_sample()
            st.session_state.scan_results = None
            st.rerun()

    gradient_divider()

    # ── Input Method ──────────────────────────────────────────────────────────
    input_method = st.radio("Input method:", ["Manual Feature Entry", "Upload CSV"], horizontal=True)

    sample_dict = dict(st.session_state.scan_sample)

    if input_method == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV (must have the correct feature columns):", type=["csv"])
        if uploaded:
            try:
                df_upload = pd.read_csv(uploaded)
                if len(df_upload) > 0:
                    sample_dict = {col: float(df_upload.iloc[0][col]) for col in ALL_FEATURES if col in df_upload.columns}
                    st.success(f"Loaded sample from CSV ({len(df_upload)} rows found, using row 1)")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    else:
        # Manual sliders/inputs by group
        with st.expander("⚙️ Feature Entry (expand to edit)", expanded=False):
            group_colors_map = {
                "PE Header": CYAN, "Section Analysis": PURPLE,
                "Import Features": BENIGN_COLOR, "String Features": "#F59E0B",
                "Behavioral": MALWARE_COLOR, "Network": "#EC4899",
            }
            for group, features in FEATURE_GROUPS.items():
                color = group_colors_map.get(group, CYAN)
                st.markdown(f'<div style="color:{color}; font-weight:700; margin:0.5rem 0 0.3rem 0; font-size:0.85rem;">{group}</div>', unsafe_allow_html=True)
                cols = st.columns(min(3, len(features)))
                for j, feat in enumerate(features):
                    with cols[j % 3]:
                        current_val = sample_dict.get(feat, 0)
                        if feat in {"has_debug", "has_signature", "uses_crypto_api",
                                    "uses_network_api", "uses_process_api", "uses_registry_api"}:
                            sample_dict[feat] = int(st.checkbox(feat, value=bool(current_val), key=f"ck_{feat}"))
                        elif feat in {"polymorphic_score", "obfuscation_level", "code_mutation_rate",
                                      "c2_similarity_score", "memory_allocation_pattern"}:
                            sample_dict[feat] = st.slider(feat, 0.0, 1.0, float(current_val), 0.01, key=f"sl_{feat}")
                        elif feat in {"avg_section_entropy", "max_section_entropy", "packet_entropy"}:
                            sample_dict[feat] = st.slider(feat, 0.0, 8.0, float(current_val), 0.1, key=f"sl_{feat}")
                        else:
                            sample_dict[feat] = st.number_input(feat, value=float(current_val), key=f"ni_{feat}", format="%g")

    st.session_state.scan_sample = sample_dict

    gradient_divider()

    # ── Scan Button ───────────────────────────────────────────────────────────
    col_scan, col_spacer = st.columns([1, 3])
    with col_scan:
        scan_clicked = st.button("🔍 SCAN FOR MALWARE", use_container_width=True)

    if scan_clicked:
        with st.spinner("🔬 Scanning... Analyzing features across all models..."):
            import time
            time.sleep(0.5)  # Small delay for UX
            results = _run_all_models(sample_dict, models, preprocessor)
            st.session_state.scan_results = results

    # ── Results Panel ─────────────────────────────────────────────────────────
    if st.session_state.scan_results:
        results = st.session_state.scan_results
        gradient_divider()
        section_header("🎯 Scan Results")

        # Primary result from XGBoost
        xgb_proba = results.get("XGBoost", 0.5)
        is_malware = xgb_proba >= 0.5
        verdict_color = MALWARE_COLOR if is_malware else BENIGN_COLOR
        verdict_label = "⚠️ MALWARE DETECTED" if is_malware else "✅ BENIGN FILE"
        badge_class = "threat-badge-malware" if is_malware else "threat-badge-benign"

        col_verdict, col_gauge = st.columns([1, 1])
        with col_verdict:
            st.markdown(f"""
            <div style="margin-top:1rem;">
                <div class="{badge_class}" style="font-size:1.5rem; padding:1.5rem;">
                    {verdict_label}
                </div>
                <div style="background:{CARD}; border:1px solid #1F2937; border-radius:10px;
                             padding:1rem; margin-top:0.8rem;">
                    <div style="color:#9CA3AF; font-size:0.75rem; margin-bottom:0.5rem;">XGBoost Confidence</div>
                    <div style="background:#1F2937; border-radius:6px; height:12px; overflow:hidden;">
                        <div style="width:{xgb_proba*100:.1f}%; background:linear-gradient(90deg,{BENIGN_COLOR},{MALWARE_COLOR});
                                     height:100%; border-radius:6px; transition:width 0.5s;"></div>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-top:0.3rem;">
                        <span style="color:{BENIGN_COLOR}; font-size:0.75rem;">Benign</span>
                        <span style="color:{verdict_color}; font-weight:700; font-family:JetBrains Mono;">{xgb_proba:.3f}</span>
                        <span style="color:{MALWARE_COLOR}; font-size:0.75rem;">Malware</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_gauge:
            fig_gauge = plot_confidence_gauge(xgb_proba, "XGBoost Malware Probability")
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── All Models Comparison ─────────────────────────────────────────────
        section_header("🤖 All Models Verdict")
        model_colors_map = {
            "XGBoost": CYAN, "Random Forest": PURPLE, "LightGBM": BENIGN_COLOR,
            "CatBoost": "#F59E0B", "Hybrid (IF+GB)": "#EF4444",
        }
        cols = st.columns(len(results))
        for col, (model_name, proba) in zip(cols, results.items()):
            with col:
                if proba < 0:
                    verdict = "ERROR"
                    v_color = TEXT_SEC
                    bar_val = 0
                else:
                    verdict = "MALWARE" if proba >= 0.5 else "BENIGN"
                    v_color = MALWARE_COLOR if proba >= 0.5 else BENIGN_COLOR
                    bar_val = proba
                mcolor = model_colors_map.get(model_name, CYAN)
                st.markdown(f"""
                <div style="background:{CARD}; border:1px solid {mcolor}44; border-top:3px solid {mcolor};
                             border-radius:10px; padding:1rem; text-align:center;">
                    <div style="color:{mcolor}; font-size:0.75rem; font-weight:700; margin-bottom:0.5rem;">{model_name}</div>
                    <div style="color:{v_color}; font-weight:800; font-size:1rem; margin-bottom:0.5rem;">{verdict}</div>
                    <div style="font-family:JetBrains Mono; font-size:1.3rem; color:{v_color};">{proba:.3f}</div>
                    <div style="background:#1F2937; border-radius:4px; height:6px; margin-top:0.5rem; overflow:hidden;">
                        <div style="width:{bar_val*100:.0f}%; background:{v_color}; height:100%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        gradient_divider()

        # ── Risk by Feature Group ─────────────────────────────────────────────
        section_header("🏷️ Risk Breakdown by Feature Group")
        if shap_explainer and hasattr(shap_explainer, 'explainer') and shap_explainer.explainer:
            try:
                X_raw = _sample_to_array(sample_dict)
                if preprocessor:
                    X_scaled = preprocessor.transform(pd.DataFrame(X_raw, columns=ALL_FEATURES))
                else:
                    X_scaled = X_raw
                wd = shap_explainer.get_waterfall_data(X_scaled[0])
                shap_by_feat = dict(zip(wd["feature_names"], wd["shap_values"]))
                group_risks = _feature_risk_by_group(shap_by_feat)

                cols = st.columns(len(group_risks))
                grp_colors = [CYAN, PURPLE, BENIGN_COLOR, "#F59E0B", MALWARE_COLOR, "#EC4899"]
                for i, (col, (group, risk)) in enumerate(zip(cols, group_risks.items())):
                    with col:
                        g_color = grp_colors[i % len(grp_colors)]
                        risk_level = "HIGH" if risk > 0.25 else ("MEDIUM" if risk > 0.1 else "LOW")
                        level_color = MALWARE_COLOR if risk > 0.25 else ("#F59E0B" if risk > 0.1 else BENIGN_COLOR)
                        st.markdown(f"""
                        <div style="background:{CARD}; border:1px solid {g_color}44; border-radius:10px;
                                     padding:0.8rem; text-align:center;">
                            <div style="color:{g_color}; font-size:0.72rem; font-weight:700;">{group}</div>
                            <div style="color:{level_color}; font-weight:800; font-size:0.85rem; margin:0.3rem 0;">{risk_level}</div>
                            <div style="background:#1F2937; border-radius:4px; height:6px; overflow:hidden; margin-top:0.3rem;">
                                <div style="width:{risk*100:.0f}%; background:{g_color}; height:100%;"></div>
                            </div>
                            <div style="color:{g_color}; font-family:JetBrains Mono; font-size:0.75rem; margin-top:0.2rem;">{risk:.1%}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # SHAP waterfall for individual explanation
                from utils.visualizations import plot_shap_waterfall
                fig_wf = plot_shap_waterfall(wd, prediction=xgb_proba)
                st.plotly_chart(fig_wf, use_container_width=True)
            except Exception as e:
                st.info(f"SHAP explanation not available: {e}")
        else:
            # Show feature values as fallback
            st.info("SHAP explainer not available. Showing top feature values.")
            feat_df = pd.DataFrame([
                {"Feature": f, "Value": v, "Group": [g for g, fs in FEATURE_GROUPS.items() if f in fs][0] if any(f in fs for g, fs in FEATURE_GROUPS.items()) else "Other"}
                for f, v in sample_dict.items()
                if f in ALL_FEATURES
            ])
            st.dataframe(feat_df, use_container_width=True)
