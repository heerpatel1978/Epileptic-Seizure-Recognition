import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# ---------- PAGE SETUP (THEME + SIDEBAR) ----------
st.set_page_config(
    page_title="Seizure Detection",
    layout="wide",
)

# Global background and text color
page_bg = """
<style>
.stApp, div[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a1f3f 0%, #1a3a52 100%) !important;
    color: #ffffff !important;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background-color: #041f3f;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ============================
# SIDEBAR - About This Tool
# ============================
with st.sidebar:
    st.markdown("""
        <h3 style='color: #1dd1a1; font-weight: 700; margin-bottom: 1rem;'>ℹ About This Tool</h3>
        <div style="
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(29, 209, 161, 0.3);
            border-radius: 12px;
            padding: 1rem;
            backdrop-filter: blur(10px);
        ">
            <p style='color: #ffffff; font-size: 0.92em; margin: 0 0 0.7rem 0; line-height: 1.45;'>
                An AI-powered system that analyzes EEG time-series data (178 data points) to detect epileptic seizure events, assisting in automated neurological screening.
            </p>
            <p style='color: #b0bec5; font-size: 0.8em; margin: 0; line-height: 1.4;'>
                Results are for research and educational purposes only and must not be
                used as a substitute for professional medical diagnosis.
            </p>
        </div>
    """, unsafe_allow_html=True)


# ---------- MAIN TITLE ----------
col_header = st.container()
with col_header:
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 0.5rem;'>🧠 Epileptic Seizure Detection AI</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: #1dd1a1; font-size: 1.1em; margin-top: 0;'>Advanced EEG Analysis System</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: #b0bec5; margin-top: 1rem;'>Upload EEG (178 values) for automated seizure detection</p>",
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

# ---------- MODEL + INPUT ----------
model = tf.keras.models.load_model("resnet_eeg_model.h5")
scaler = joblib.load("scaler_eeg.pkl")

uploaded_file = st.file_uploader("Upload EEG CSV (178 values)", type=["csv"])
input_data = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header=None)
    if df.shape[1] != 178:
        st.error(f"Invalid input shape: {df.shape}. Expected 1 x 178 CSV.")
    else:
        eeg_values = df.values.reshape(1, -1)
        input_data = eeg_values
        st.write("EEG Loaded:", input_data)
else:
    eeg_text = st.text_area("Or paste 178 comma-separated values",placeholder="0.5, 1.2, -0.3, ...")
    if eeg_text:
        try:
            eeg_values = np.array([float(x) for x in eeg_text.split(",")])
            if len(eeg_values) != 178:
                st.error("You must enter exactly 178 values.")
            else:
                input_data = eeg_values.reshape(1, -1)
                st.write("EEG Loaded:", input_data)
        except Exception:
            st.error("Invalid numbers. Check your input.")

# ============================
# PREDICTION + RESULT UI
# ============================
if input_data is not None:
    # 1) Model prediction
    scaled = scaler.transform(input_data)
    scaled = scaled.reshape(1, 178, 1)
    prob = float(model.predict(scaled)[0][0])

    seizure_prob = prob * 100.0
    normal_prob = (1 - prob) * 100.0
    label = "SEIZURE DETECTED" if prob >= 0.5 else "NO SEIZURE DETECTED"

    # 2) Description + colors depending on label
    if prob >= 0.5:
        desc_text = "High probability of seizure activity."
        title_color = "#ff6b6b"      # red
        border_color = "rgba(244, 67, 54, 0.35)"
    else:
        desc_text = "Low probability of seizure activity."
        title_color = "#1dd1a1"      # green
        border_color = "rgba(29, 209, 161, 0.55)"

    # 3) Result card
    st.markdown(f"""
    <div style="
        margin-top: 2rem;
        padding: 1.5rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 18px 45px rgba(0,0,0,0.45);
    ">
        <h2 style="color:#FFFFFF; margin:0 0 0.75rem 0; font-size:1.6rem;">
            🔍 Analysis Results
        </h2>
        <div style="
            margin-top:0.75rem;
            padding:1.2rem 1.4rem;
            border-radius:14px;
            background:rgba(15,25,45,0.85);
            border:1px solid {border_color};
        ">
            <p style="color:{title_color}; font-weight:700; font-size:1.2rem; margin:0 0 0.4rem 0;">
                ⚠ {label}
            </p>
            <p style="color:#ffcdd2; margin:0; font-size:0.95rem;">
                {desc_text}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<h3 style='color:#ffffff; margin-bottom:0.5rem;'>Confidence Scores</h3>",
        unsafe_allow_html=True,
    )

    # 4) Custom progress bars
    def metric_bar(title, value, bar_color):
        st.markdown(f"""
        <div style="
            margin-top:0.9rem;
            padding:1.0rem 1.2rem;
            border-radius:16px;
            background:rgba(255,255,255,0.03);
            border:1px solid rgba(255,255,255,0.10);
        ">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                <span style="color:#ffffff; font-weight:600;">{title}</span>
                <span style="color:#ffffff; font-weight:600;">{value:.2f}%</span>
            </div>
            <div style="
                width:100%;
                height:9px;
                border-radius:999px;
                background:rgba(255,255,255,0.06);
                overflow:hidden;
            ">
                <div style="
                    width:{value}%;
                    height:100%;
                    background:{bar_color};
                    box-shadow:0 0 12px {bar_color};
                "></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    metric_bar("Seizure Activity", seizure_prob, "#ff6b81")
    metric_bar("Normal EEG", normal_prob, "#00d2d3")

# ---------- FOOTER ----------
st.markdown("<div style='height: 3rem;'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; padding-top: 2rem; border-top: 1px solid rgba(255, 255, 255, 0.1);'>
    <p style='color: #90a4ae; font-size: 0.85em; margin: 0;'>Seizure Detection AI | Powered by Advanced Neural Networks</p>
    <p style='color: #37474f; font-size: 0.8em; margin: 0.5rem 0 0 0;'>For informational purposes only. Not a substitute for professional medical advice.</p>
</div>
""", unsafe_allow_html=True)
