"""Page 2 — Live Threat Scanner with Random 33-Feature Sample Generator"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
from pathlib import Path

from utils.preprocessing import ALL_FEATURES, FEATURE_GROUPS
from data.generate_dataset import generate_benign_sample, generate_malware_sample, generate_polymorphic_sample

SAVED = Path(__file__).parent.parent / "models" / "saved"
DATA_DIR = Path(__file__).parent.parent / "data"

BINARY_COLS = {
    "has_debug", "has_signature", "uses_crypto_api",
    "uses_network_api", "uses_process_api", "uses_registry_api"
}

INT_COLS = {
    "file_size", "num_sections", "entry_point_offset", "image_base",
    "dll_characteristics", "num_executable_sections", "num_writable_sections",
    "num_imports", "num_suspicious_imports", "num_unique_dlls",
    "num_urls", "num_ips", "num_registry_keys", "num_file_paths",
    "num_printable_strings", "connection_attempts"
}


@st.cache_data
def _load_empirical_dataset():
    p = DATA_DIR / "synthetic_malware_data.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


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


def generate_random_synthetic_sample(seed=None):
    """
    Generate a valid 33-feature synthetic sample informed by the empirical dataset.
    Preserves valid types (binary, integers, bounded floats) and correlation consistency.
    """
    rng = np.random.default_rng(seed)
    df = _load_empirical_dataset()

    if df is not None and len(df) > 0:
        idx = int(rng.integers(0, len(df)))
        base_row = df.iloc[idx].to_dict()
    else:
        # Fallback to generator
        base_row = generate_benign_sample() if rng.random() > 0.5 else generate_malware_sample()

    sample = {}
    for f in ALL_FEATURES:
        val = base_row.get(f, 0.0)
        if f in BINARY_COLS:
            sample[f] = int(val if rng.random() > 0.08 else 1 - val)
        elif f in INT_COLS:
            jitter = rng.normal(0, max(1.0, abs(val) * 0.05))
            new_val = int(round(val + jitter))
            if f in {"num_sections", "num_executable_sections"}:
                new_val = max(1, new_val)
            elif f not in {"entry_point_offset", "image_base", "dll_characteristics", "num_registry_keys", "num_printable_strings"}:
                new_val = max(0, new_val)
            sample[f] = new_val
        else:
            jitter = rng.normal(0, max(0.01, abs(val) * 0.05))
            new_val = float(val + jitter)
            if "entropy" in f:
                new_val = float(np.clip(new_val, 0.0, 8.0))
            elif f in {"polymorphic_score", "obfuscation_level", "code_mutation_rate", "c2_similarity_score", "memory_allocation_pattern"}:
                new_val = float(np.clip(new_val, 0.0, 1.0))
            elif f == "api_call_frequency":
                new_val = float(np.clip(new_val, 1.0, 500.0))
            sample[f] = round(new_val, 4)

    # Consistency rule: max_section_entropy >= avg_section_entropy
    if sample.get("max_section_entropy", 0) < sample.get("avg_section_entropy", 0):
        sample["max_section_entropy"] = round(min(8.0, sample["avg_section_entropy"] + float(rng.uniform(0.1, 0.5))), 4)

    return sample


def _validate_sample(sample_dict):
    """Safety validation for 33 input features."""
    if len(sample_dict) < len(ALL_FEATURES):
        missing = set(ALL_FEATURES) - set(sample_dict.keys())
        return False, f"Missing features: {list(missing)[:5]}"

    for f in ALL_FEATURES:
        if f not in sample_dict:
            return False, f"Missing feature: {f}"
        val = sample_dict[f]
        if val is None or np.isnan(val) or np.isinf(val):
            return False, f"Invalid value for {f}: {val}"

    return True, "OK"


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
        except Exception:
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
    st.markdown("# Threat Scanner")
    st.markdown("Dynamic synthetic feature testing with multi-model consensus and real-time SHAP explanation.")

    # ── State Initialization ──────────────────────────────────────────────────
    if "sample_counter" not in st.session_state:
        st.session_state.sample_counter = 1
    if "sample_id" not in st.session_state:
        st.session_state.sample_id = "Random Sample #001"
    if "sample" not in st.session_state:
        st.session_state.sample = generate_random_synthetic_sample(seed=42)
    if "previous_sample" not in st.session_state:
        st.session_state.previous_sample = None
    if "results" not in st.session_state:
        st.session_state.results = None
    if "shap_data" not in st.session_state:
        st.session_state.shap_data = None

    models = _load_models()
    preprocessor = _load_preprocessor()
    shap_explainer = _load_shap_explainer()

    if not models:
        st.warning("Models not trained yet. Run `python models/train_all_models.py` first.")
        return

    # ── Top Metadata Bar ──────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns([1.2, 1, 1, 2])
    with m1:
        st.markdown(f'<div class="stat-card"><div class="stat-val" style="font-size:1.2rem; color:#60a5fa;">{st.session_state.sample_id}</div><div class="stat-lbl">Active Sample</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="stat-card"><div class="stat-val" style="font-size:1.2rem;">33</div><div class="stat-lbl">Feature Count</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="stat-card"><div class="stat-val" style="font-size:1.2rem; color:#34d399;">Synthetic</div><div class="stat-lbl">Input Domain</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div style="background:#161b27; border:1px solid #1f2937; border-radius:8px; padding:0.6rem 0.9rem; font-size:0.75rem; color:#9ca3af; height:100%;">
          <strong>Dynamic Feature Testing:</strong> Generate controlled 33-feature vectors to observe how models respond to mutating signatures.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Main Generator & Analysis Actions ─────────────────────────────────────
    st.markdown("### 🎲 Dynamic Sample Generation & Control")
    st.caption("Generate a new synthetic 33-feature sample or enter custom values, then run inference across all 5 models.")

    c_gen, c_reset, c_analyze, c_seed = st.columns([1.2, 1.2, 1.2, 1])

    with c_seed:
        seed_input = st.number_input("Seed (optional)", value=None, step=1, min_value=0, placeholder="Random", help="Optional integer seed for reproducible generation")

    with c_gen:
        if st.button("🎲 Generate New Sample", type="secondary", use_container_width=True):
            st.session_state.previous_sample = dict(st.session_state.sample)
            st.session_state.sample_counter += 1
            st.session_state.sample_id = f"Random Sample #{st.session_state.sample_counter:03d}"
            effective_seed = int(seed_input) if seed_input is not None else None
            st.session_state.sample = generate_random_synthetic_sample(seed=effective_seed)
            st.session_state.results = None
            st.session_state.shap_data = None
            st.rerun()

    with c_reset:
        if st.button("🔄 Reset / New Sample", use_container_width=True):
            st.session_state.previous_sample = dict(st.session_state.sample)
            st.session_state.sample_counter += 1
            st.session_state.sample_id = f"Random Sample #{st.session_state.sample_counter:03d}"
            st.session_state.sample = generate_random_synthetic_sample(seed=None)
            st.session_state.results = None
            st.session_state.shap_data = None
            st.rerun()

    with c_analyze:
        if st.button("🔍 Analyze Sample", type="primary", use_container_width=True):
            is_valid, err_msg = _validate_sample(st.session_state.sample)
            if not is_valid:
                st.error(f"Sample Validation Error: {err_msg}")
            else:
                with st.spinner("Executing models and computing SHAP attributions..."):
                    st.session_state.results = _predict(st.session_state.sample, models, preprocessor)
                    if shap_explainer is not None:
                        st.session_state.shap_data = _compute_shap(st.session_state.sample, shap_explainer)

    # ── Difference from Previous Sample ───────────────────────────────────────
    if st.session_state.previous_sample is not None:
        prev = st.session_state.previous_sample
        curr = st.session_state.sample
        changed_feats = [f for f in ALL_FEATURES if prev.get(f) != curr.get(f)]
        st.markdown(f"""
        <div style="background:#131824; border:1px solid #2d3748; border-radius:6px; padding:0.4rem 0.8rem; margin-top:0.4rem; font-size:0.78rem; color:#9ca3af;">
          <span style="color:#60a5fa;">● Transition:</span> Previous Sample → <strong>{st.session_state.sample_id}</strong> &nbsp;|&nbsp;
          <strong>{len(changed_feats)} / 33</strong> features modified &nbsp;|&nbsp;
          <span style="color:#6b7280;">Ready for analysis</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Secondary Presets (Expander) ──────────────────────────────────────────
    with st.expander("📁 Classic Presets & CSV Upload (Optional)", expanded=False):
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            if st.button("Preset: Benign", use_container_width=True):
                st.session_state.previous_sample = dict(st.session_state.sample)
                st.session_state.sample = generate_benign_sample()
                st.session_state.sample_id = "Preset: Benign"
                st.session_state.results = None
                st.session_state.shap_data = None
                st.rerun()
        with p2:
            if st.button("Preset: Malware", use_container_width=True):
                st.session_state.previous_sample = dict(st.session_state.sample)
                st.session_state.sample = generate_malware_sample()
                st.session_state.sample_id = "Preset: Malware"
                st.session_state.results = None
                st.session_state.shap_data = None
                st.rerun()
        with p3:
            if st.button("Preset: Polymorphic", use_container_width=True):
                st.session_state.previous_sample = dict(st.session_state.sample)
                st.session_state.sample = generate_polymorphic_sample()
                st.session_state.sample_id = "Preset: Polymorphic"
                st.session_state.results = None
                st.session_state.shap_data = None
                st.rerun()
        with p4:
            uploaded = st.file_uploader("Upload CSV (1 row, 33 features)", type=["csv"], label_visibility="collapsed")
            if uploaded:
                try:
                    df_up = pd.read_csv(uploaded)
                    missing = set(ALL_FEATURES) - set(df_up.columns)
                    if missing:
                        st.error(f"Missing features in CSV: {list(missing)[:5]}")
                    elif len(df_up) > 0:
                        st.session_state.previous_sample = dict(st.session_state.sample)
                        st.session_state.sample = {c: float(df_up.iloc[0][c]) for c in ALL_FEATURES if c in df_up.columns}
                        st.session_state.sample_id = f"Uploaded CSV ({uploaded.name})"
                        st.session_state.results = None
                        st.session_state.shap_data = None
                        st.rerun()
                except Exception as e:
                    st.error(f"CSV read error: {e}")

    # ── Feature Inspection & Manual Editor ────────────────────────────────────
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### 📋 33 Feature Values (Inspect / Edit)")
    st.caption("Review all 33 generated values grouped by cyber domain. You can adjust any parameter before analyzing.")

    sample = dict(st.session_state.sample)

    for group, feats in FEATURE_GROUPS.items():
        with st.expander(f"**{group}** ({len(feats)} features)", expanded=False):
            cols = st.columns(min(4, len(feats)))
            for j, feat in enumerate(feats):
                with cols[j % len(cols)]:
                    val = sample.get(feat, 0)
                    if feat in BINARY_COLS:
                        sample[feat] = int(st.checkbox(feat, value=bool(val), key=f"e_{feat}_{st.session_state.sample_id}"))
                    elif feat in {"polymorphic_score", "obfuscation_level", "code_mutation_rate",
                                  "c2_similarity_score", "memory_allocation_pattern"}:
                        sample[feat] = st.slider(feat, 0.0, 1.0, float(val), 0.01, key=f"e_{feat}_{st.session_state.sample_id}")
                    elif feat in {"avg_section_entropy", "max_section_entropy", "packet_entropy"}:
                        sample[feat] = st.slider(feat, 0.0, 8.0, float(val), 0.1, key=f"e_{feat}_{st.session_state.sample_id}")
                    else:
                        sample[feat] = st.number_input(feat, value=float(val), key=f"e_{feat}_{st.session_state.sample_id}", format="%g")

    st.session_state.sample = sample

    # ── Analysis & Predictions Section ────────────────────────────────────────
    if st.session_state.results:
        results = st.session_state.results
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown(f"### 🛡️ PolyShield Analysis: {st.session_state.sample_id}")

        xgb_p = results.get("XGBoost", 0.5)
        valid_probas = [p for p in results.values() if p >= 0]
        malware_votes = sum(1 for p in valid_probas if p >= 0.5)
        total_votes = len(valid_probas)
        consensus_text = f"{malware_votes} / {total_votes} models"

        risk_score = int(round(xgb_p * 100))
        if risk_score >= 70:
            risk_level = "HIGH RISK"
            risk_color = "#f87171"
            risk_bg = "#3b0a0a"
            risk_border = "#dc2626"
        elif risk_score >= 35:
            risk_level = "MEDIUM RISK"
            risk_color = "#fbbf24"
            risk_bg = "#2d1f00"
            risk_border = "#d97706"
        else:
            risk_level = "LOW RISK"
            risk_color = "#4ade80"
            risk_bg = "#052e16"
            risk_border = "#16a34a"

        # ── PolyShield Analysis Summary Card ──────────────────────────────────
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            verdict = "MALICIOUS" if xgb_p >= 0.5 else "BENIGN"
            conf = xgb_p if xgb_p >= 0.5 else (1.0 - xgb_p)
            st.markdown(f"""
            <div class="stat-card" style="border:1px solid {risk_border}; background:{risk_bg};">
              <div style="font-size:1.3rem; font-weight:700; color:{risk_color};">{verdict}</div>
              <div class="stat-lbl" style="color:{risk_color};">Confidence: {conf:.1%}</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="stat-card">
              <div class="stat-val" style="color:{risk_color};">{risk_score} / 100</div>
              <div class="stat-lbl">Risk Score ({risk_level})</div>
            </div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
            <div class="stat-card">
              <div class="stat-val" style="color:#93c5fd;">{consensus_text}</div>
              <div class="stat-lbl">Model Consensus</div>
            </div>""", unsafe_allow_html=True)
        with r4:
            st.markdown(f"""
            <div class="stat-card">
              <div class="stat-val" style="font-size:1.1rem; color:#d1d5db;">{st.session_state.sample_id}</div>
              <div class="stat-lbl">Analyzed Vector (33 feats)</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # ── Individual Model Cards ────────────────────────────────────────────
        st.markdown("#### Individual Model Scores")
        cols = st.columns(len(results))
        for col, (name, proba) in zip(cols, results.items()):
            with col:
                if proba < 0:
                    lbl, clr = "Error", "#6b7280"
                    dsp = "ERR"
                elif proba >= 0.5:
                    lbl, clr = "Malicious", "#f87171"
                    dsp = f"{proba:.3f}"
                else:
                    lbl, clr = "Benign", "#4ade80"
                    dsp = f"{proba:.3f}"
                st.markdown(f"""
                <div class="stat-card" style="border-top:3px solid {clr};">
                  <div style="font-size:0.75rem; color:#9ca3af; margin-bottom:0.3rem;">{name}</div>
                  <div style="font-size:1.3rem; font-weight:700; color:{clr}; font-family:'JetBrains Mono';">{dsp}</div>
                  <div style="font-size:0.72rem; color:{clr}; margin-top:0.2rem;">{lbl}</div>
                </div>""", unsafe_allow_html=True)

        # ── SHAP Attribution Section ──────────────────────────────────────────
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("### 🔍 SHAP Feature Attribution (Why this prediction was made)")

        if st.session_state.shap_data:
            shap_res = st.session_state.shap_data
            st.caption("Exact Shapley feature contributions to the XGBoost decision. Red bars indicate features pushing toward Malicious; blue bars push toward Benign.")

            feats = shap_res["features"]
            svs = shap_res["values"]
            fvs = shap_res["feature_vals"]

            bar_colors = ["#ef4444" if v > 0 else "#3b82f6" for v in svs]
            fig = go.Figure(go.Bar(
                y=feats,
                x=svs,
                orientation="h",
                marker_color=bar_colors,
                text=[f"{v:+.4f}" for v in svs],
                textposition="outside",
                customdata=[[round(f, 4)] for f in fvs],
                hovertemplate="<b>%{y}</b><br>SHAP value: %{x:+.4f}<br>Input value: %{customdata[0]}<extra></extra>",
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0f1117",
                plot_bgcolor="#1e2433",
                font=dict(family="Inter, sans-serif", color="#e5e7eb", size=11),
                height=420,
                margin=dict(l=170, r=90, t=30, b=40),
                xaxis=dict(title="SHAP Impact on Log-Odds", gridcolor="#2d3748", zeroline=True, zerolinecolor="#6b7280"),
                yaxis=dict(gridcolor="#2d3748"),
            )
            st.plotly_chart(fig)

            # Attribution breakdown table
            table_rows = [
                {
                    "Feature Name": f,
                    "Input Value": round(float(fv), 4),
                    "SHAP Impact": f"{sv:+.4f}",
                    "Influence Direction": "Pushes Malicious (Risk ↑)" if sv > 0 else "Pushes Benign (Clean ↓)"
                }
                for f, fv, sv in zip(feats, fvs, svs)
            ]
            st.dataframe(pd.DataFrame(table_rows), hide_index=True)
        else:
            st.info("SHAP attribution could not be calculated for this sample.")

        st.caption("Note: These are synthetic feature values for model demonstration and do not represent executable malware binaries.")
