"""
Streamlit UI — HEA Fatigue Life Prediction
===========================================
Interactive dashboard for end-users to:
  1. Input alloy composition + stress
  2. View predicted survival curve
  3. Read reliability statement
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import joblib

# Ensure project root is on the path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from hea_fatigue.feature_engineering.feature_engineer import (
    parse_composition,
    compute_VEC,
    compute_delta_S_mix,
    compute_delta_H_mix,
)

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HEA Fatigue Life Predictor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    .main { background: #0d1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e, #16213e);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #f0f6fc; }
    .metric-label { font-size: 0.85rem; color: #8b949e; margin-top: 4px; }
    .reliability-box {
        background: linear-gradient(135deg, #0d2137, #0f3052);
        border-left: 4px solid #00b4d8;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
        font-size: 1.05rem;
        color: #cdd9e5;
    }
    .stButton>button {
        background: linear-gradient(90deg, #0077b6, #00b4d8);
        color: white; border: none; border-radius: 8px;
        padding: 10px 28px; font-weight: 600; font-size: 1rem;
    }
    h1 { color: #f0f6fc !important; }
    h2, h3 { color: #cdd9e5 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading trained model…")
def load_rsf():
    path = ROOT / "hea_fatigue" / "models" / "saved" / "rsf_model.pkl"
    if not path.exists():
        return None
    return joblib.load(str(path))


@st.cache_resource(show_spinner="Loading preprocessor…")
def load_preprocessor():
    path = ROOT / "hea_fatigue" / "models" / "saved" / "preprocessor.pkl"
    if not path.exists():
        return None
    return joblib.load(str(path))


rsf_model = load_rsf()
preprocessor = load_preprocessor()


# ─── Helper ───────────────────────────────────────────────────────────────────
def predict_survival(rsf, feature_row: dict, times: np.ndarray) -> np.ndarray:
    """Run RSF inference with correct preprocessing (same scaler as training)."""
    feat_cols = rsf.feature_cols
    # Build single-row DataFrame
    row_df = pd.DataFrame([{c: feature_row.get(c, 0.0) for c in feat_cols}], columns=feat_cols)
    # Apply trained preprocessor if available
    if preprocessor is not None:
        try:
            row_df = preprocessor.transform(row_df)[feat_cols]
        except Exception:
            pass  # fallback: use raw values
    x = row_df.values
    sf_df = rsf.predict_survival_function(x, times=times)
    return sf_df.iloc[0].values


def build_feature_row(
    stress: float, mat_type: str, composition: str,
    ys: float, uts: float, E: float,
) -> dict:
    fractions = parse_composition(composition) if composition.strip() else None
    return {
        "stress_amplitude":   stress,
        "is_MPEA":            1 if mat_type == "MPEA" else 0,
        "yield_strength_MPa": ys,
        "uts_MPa":            uts,
        "youngs_modulus_GPa": E,
        "elongation_pct":     30 if mat_type == "MPEA" else 2,
        "load_ratio":         -1.0,
        "frequency_Hz":       10.0,
        "test_temperature_C": 25.0,
        "is_air":             1,
        "VEC":          compute_VEC(fractions) if fractions else (7.0 if mat_type == "MPEA" else 6.0),
        "delta_S_mix":  compute_delta_S_mix(fractions) if fractions else (13.4 if mat_type == "MPEA" else 9.0),
        "delta_H_mix":  compute_delta_H_mix(fractions) if fractions else (-2.5 if mat_type == "MPEA" else -15.0),
        "n_elements":   float(len(fractions)) if fractions else (5.0 if mat_type == "MPEA" else 3.0),
    }


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/atom.png", width=64)
    st.title("⚙️ HEA Fatigue Predictor")
    st.markdown("---")

    st.subheader("🧪 Alloy Composition")
    composition = st.text_input(
        "Composition string",
        value="CrMnFeCoNi",
        help="e.g. 'CrMnFeCoNi' (equal) or 'Zr55.0-Cu30.0-Al10.0-Ni5.0' (at%)",
    )
    material_type = st.selectbox("Material Type", ["MPEA", "MG"])

    st.subheader("⚡ Loading Conditions")
    stress = st.slider("Stress Amplitude (MPa)", min_value=50, max_value=1000, value=300, step=10)

    st.subheader("🔩 Mechanical Properties")
    with st.expander("Optional overrides", expanded=False):
        ys  = st.number_input("Yield Strength (MPa)", value=400 if material_type == "MPEA" else 1000, min_value=10)
        uts = st.number_input("UTS (MPa)", value=600 if material_type == "MPEA" else 1500, min_value=10)
        E   = st.number_input("Young's Modulus (GPa)", value=200 if material_type == "MPEA" else 90, min_value=10)

    run_btn = st.button("🔮 Predict", use_container_width=True)

st.title("ML-Guided Probabilistic Fatigue Life Prediction")
st.markdown("**Metastable High Entropy Alloys (HEAs) — Survival Analysis System**")
st.markdown("---")


# ─── Display composition info ─────────────────────────────────────────────────
fractions = parse_composition(composition)
if fractions:
    vec_val  = compute_VEC(fractions)
    ds_val   = compute_delta_S_mix(fractions)
    dh_val   = compute_delta_H_mix(fractions)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Elements", len(fractions))
    c2.metric("VEC", f"{vec_val:.2f} e⁻/atom")
    c3.metric("ΔSmix", f"{ds_val:.2f} J/mol·K")
    c4.metric("ΔHmix", f"{dh_val:.2f} kJ/mol")
    st.caption(f"Parsed composition: {', '.join(f'{el}: {ci*100:.1f} at%' for el, ci in fractions.items())}")
else:
    st.info("Enter a composition string in the sidebar (e.g. 'CrMnFeCoNi' or 'Zr55.0-Cu30.0-Al10.0-Ni5.0').")

st.markdown("---")

# ─── Prediction ───────────────────────────────────────────────────────────────
if run_btn or True:  # auto-run on load for demo
    if rsf_model is None:
        st.error(
            "⚠️ Trained model not found. Please run `python main.py` first to train the models.",
            icon="🚨",
        )
    else:
        # Use log10 time axis (matches RSF training) for interpolation
        log_times = np.linspace(3, 8, 120)   # 10^3 to 10^8 in log10 space
        raw_times = 10.0 ** log_times          # actual cycle counts for display

        feature_row = build_feature_row(stress, material_type, composition, ys, uts, E)

        with st.spinner("Computing survival probabilities…"):
            sf = predict_survival(rsf_model, feature_row, log_times)

        # ── UTS guard — if stress >= UTS, material fails immediately ────────
        if stress >= uts:
            st.error(
                f"⚠️ **Applied stress ({stress} MPa) ≥ UTS ({uts} MPa).** "
                "The material fails immediately — survival probability is **0%** at all cycle counts.",
                icon="🚨",
            )
            sf = np.zeros_like(sf)

        # ── Metrics ─────────────────────────────────────────────────────────
        prob_1e5 = float(np.interp(5.0, log_times, sf))   # log10(1e5) = 5
        prob_1e6 = float(np.interp(6.0, log_times, sf))   # log10(1e6) = 6
        prob_1e7 = float(np.interp(7.0, log_times, sf))   # log10(1e7) = 7

        # Median life = where S crosses 0.5 (undefined when sf is all zeros)
        median_idx = np.searchsorted(-sf, -0.50)
        median_log = log_times[min(median_idx, len(log_times) - 1)]
        median_life = 10.0 ** median_log
        median_life_str = "< 10³" if stress >= uts else f"{median_life:.2e}"

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("S(10⁵ cycles)", f"{prob_1e5*100:.1f}%")
        col2.metric("S(10⁶ cycles)", f"{prob_1e6*100:.1f}%")
        col3.metric("S(10⁷ cycles)", f"{prob_1e7*100:.1f}%")
        col4.metric("Median Life (N₅₀)", median_life_str)

        # ── Reliability statement ────────────────────────────────────────────
        pct_at_stress = int(prob_1e6 * 100)
        if stress < uts:
            st.markdown(f"""
            <div class='reliability-box'>
            📊 <b>Reliability Statement:</b><br>
            At <b>{stress} MPa</b> stress amplitude, a <b>{material_type}</b> alloy ({composition})
            has a <b>{pct_at_stress}%</b> probability of surviving 10⁶ cycles.
            The predicted median fatigue life is <b>{median_life:.2e}</b> cycles.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='reliability-box'>
            📊 <b>Reliability Statement:</b><br>
            At <b>{stress} MPa</b> stress amplitude — which exceeds the UTS of <b>{uts} MPa</b> —
            the <b>{material_type}</b> alloy ({composition}) has a <b>0%</b> survival probability
            at any cycle count. Reduce the applied stress below UTS.
            </div>
            """, unsafe_allow_html=True)

        # ── Survival curve plot ──────────────────────────────────────────────
        st.subheader("📈 Survival Curve S(N)")
        fig_sf = go.Figure()
        fig_sf.add_trace(go.Scatter(
            x=raw_times, y=sf * 100,
            mode="lines",
            name="S(N)",
            line=dict(color="#00b4d8", width=3),
            fill="tozeroy",
            fillcolor="rgba(0,180,216,0.1)",
        ))
        if stress < uts:
            fig_sf.add_vline(
                x=median_life, line_dash="dash", line_color="#f77f00",
                annotation_text=f"N₅₀ = {median_life:.1e}", annotation_position="top right",
            )
            fig_sf.add_hline(y=50, line_dash="dot", line_color="#fcbf49", line_width=1)

        fig_sf.update_layout(
            title=f"Survival Probability — {material_type} @ {stress} MPa",
            xaxis=dict(title="Cycles to Failure (N)", type="log"),
            yaxis=dict(title="Survival Probability S(N) (%)", range=[0, 105]),
            template="plotly_dark",
            height=420,
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_sf, use_container_width=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "ML-Guided Probabilistic Fatigue Life Prediction · "
    "Random Survival Forest · FatigueData-CMA2022 · "
    "Metastable High Entropy Alloys"
)
