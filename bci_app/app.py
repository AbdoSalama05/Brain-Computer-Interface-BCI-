

import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import welch

# --------------------------------------------------------------------------
# Config / paths
# --------------------------------------------------------------------------
MODEL_PATH = "lda_band_power_model.joblib"
SCALER_PATH = "lda_band_power_scaler.joblib"
CONFIG_PATH = "lda_band_power_config.json"

# The training epochs were 2 seconds at 250Hz = 500 samples. The saved config
# doesn't store this directly, so it's fixed here to match the notebook.
EXPECTED_N_SAMPLES = 500

st.set_page_config(page_title="BCI Motor Imagery Demo", page_icon="🧠", layout="centered")


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_resource
def load_model_scaler_config():
    if not os.path.exists(CONFIG_PATH) or not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None, None

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler, config


def load_trial(uploaded_file, channel_order):
    """Read an uploaded CSV/XLSX and return a DataFrame with the required channel columns."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    missing = [c for c in channel_order if c not in df.columns]
    if missing:
        raise ValueError(f"Uploaded file is missing required channel column(s): {missing}")

    return df[channel_order]


def fit_to_expected_length(values, n_samples):
    """Trim/pad a (samples, channels) array to n_samples rows, same as training epochs."""
    if len(values) >= n_samples:
        return values[:n_samples], False
    pad = np.zeros((n_samples - len(values), values.shape[1]), dtype=np.float32)
    return np.vstack([values, pad]), True


def band_power(signal, fs, low, high, nperseg):
    f, pxx = welch(signal, fs=fs, nperseg=min(nperseg, len(signal)))
    mask = (f >= low) & (f <= high)
    return np.trapezoid(pxx[mask], f[mask])


def extract_features(trial_df, config):
    """
    Reproduces the exact training pipeline:
    C3/C4 mu+beta band power -> log1p -> scaler.transform
    """
    values = trial_df.values.astype(np.float32)
    values, was_padded = fit_to_expected_length(values, EXPECTED_N_SAMPLES)

    c3 = values[:, config["c3_index"]]
    c4 = values[:, config["c4_index"]]

    fs = config["fs"]
    nperseg = config["welch_nperseg"]
    mu_lo, mu_hi = config["bands"]["mu"]
    beta_lo, beta_hi = config["bands"]["beta"]

    c3_mu = band_power(c3, fs, mu_lo, mu_hi, nperseg)
    c4_mu = band_power(c4, fs, mu_lo, mu_hi, nperseg)
    c3_beta = band_power(c3, fs, beta_lo, beta_hi, nperseg)
    c4_beta = band_power(c4, fs, beta_lo, beta_hi, nperseg)

    # order must match config["feature_order"]: ["C3_mu", "C4_mu", "C3_beta", "C4_beta"]
    raw_features = np.array([[c3_mu, c4_mu, c3_beta, c4_beta]], dtype=np.float64)
    log_features = np.log1p(raw_features)
    return log_features, was_padded


def make_example_trial(channel_order, n_samples):
    """Random synthetic trial, purely so reviewers can click through the UI without a file."""
    rng = np.random.default_rng()
    data = rng.normal(0, 1, size=(n_samples, len(channel_order)))
    return pd.DataFrame(data, columns=channel_order)


def parse_manual_channel(text):
    """Parse a comma/space/newline-separated string of numbers into a float array."""
    if not text or not text.strip():
        return None
    cleaned = text.replace(",", " ").replace("\n", " ").replace("\t", " ")
    parts = [p for p in cleaned.split(" ") if p.strip() != ""]
    return np.array([float(p) for p in parts], dtype=np.float32)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🧠 BCI Motor Imagery Classifier")
st.caption("Band Power + LDA demo, classifies imagined LEFT vs RIGHT hand movement from EEG")

model, scaler, config = load_model_scaler_config()

if model is None:
    st.error(
        f"Couldn't find `{MODEL_PATH}`, `{SCALER_PATH}`, and/or `{CONFIG_PATH}` in the app folder. "
        "Copy those three files (produced at the end of the classical ML notebook) into the same "
        "directory as app.py, then restart the app."
    )
    st.stop()

channel_order = config["channel_order"]
label_map = config["label_map"]
inv_label_map = {v: k for k, v in label_map.items()}

split_method = config.get("split_method", "unknown")
test_accuracy = config.get("test_accuracy", float("nan"))
st.caption(f"Split method used to select this model: {split_method}. Test accuracy: {test_accuracy:.3f}")

st.divider()

# --------------------------------------------------------------------------
# Session state: holds the currently loaded trial, so nothing is shown
# until the user actually uploads a file, clicks the example button, or
# submits manual values.
# --------------------------------------------------------------------------
if "trial_df" not in st.session_state:
    st.session_state.trial_df = None
    st.session_state.source_label = None

# --------------------------------------------------------------------------
# Preview and prediction, shown above the input controls
# --------------------------------------------------------------------------
if st.session_state.trial_df is not None:
    st.subheader("Trial preview")
    st.caption(st.session_state.source_label)
    st.line_chart(st.session_state.trial_df.reset_index(drop=True))

    with st.spinner("Running inference..."):
        features, was_padded = extract_features(st.session_state.trial_df, config)
        scaled_features = scaler.transform(features)
        probs = model.predict_proba(scaled_features)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = inv_label_map[pred_idx]
        confidence = probs[pred_idx]

    if was_padded:
        st.caption(f"Note: input was shorter than {EXPECTED_N_SAMPLES} samples and was zero-padded.")

    st.subheader("Prediction")
    c1, c2 = st.columns(2)
    c1.metric("Predicted class", pred_label.upper())
    c2.metric("Confidence", f"{confidence * 100:.1f}%")

    prob_df = pd.DataFrame(
        {"Class": [inv_label_map[i].upper() for i in range(len(probs))], "Probability": probs}
    ).set_index("Class")
    st.bar_chart(prob_df)
else:
    st.info("Upload a trial file or click 'Try example trial' below to see a prediction.")

st.divider()

# --------------------------------------------------------------------------
# Input controls, shown below the preview
# --------------------------------------------------------------------------
input_mode = st.radio(
    "Input method",
    ["Upload CSV/XLSX", "Enter channels manually"],
    horizontal=True,
)

if input_mode == "Upload CSV/XLSX":
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            f"Upload one trial (CSV or XLSX) with columns: {', '.join(channel_order)}",
            type=["csv", "xlsx", "xls"],
        )
    with col2:
        st.write("")
        st.write("")
        use_example = st.button("Try example trial", use_container_width=True)

    if uploaded_file is not None:
        try:
            st.session_state.trial_df = load_trial(uploaded_file, channel_order)
            st.session_state.source_label = f"Uploaded file: {uploaded_file.name}"
            st.rerun()
        except Exception as e:
            st.error(str(e))

    elif use_example:
        st.session_state.trial_df = make_example_trial(channel_order, EXPECTED_N_SAMPLES)
        st.session_state.source_label = "Synthetic example trial (random noise, for UI demo only)"
        st.rerun()

else:  # Enter channels manually
    st.caption(
        f"Paste up to {EXPECTED_N_SAMPLES} values per channel, separated by commas, spaces, or new lines. "
        "Shorter series are zero-padded; longer ones are trimmed."
    )
    manual_values = {}

    fill_example = st.button("Fill with example values")

    for ch in channel_order:
        default_text = ""
        if fill_example:
            rng = np.random.default_rng()
            default_text = ", ".join(f"{v:.3f}" for v in rng.normal(0, 1, size=EXPECTED_N_SAMPLES))
        manual_values[ch] = st.text_area(f"{ch} values", value=default_text, height=80, key=f"manual_{ch}")

    if st.button("Run prediction on manual input", type="primary"):
        try:
            parsed = {}
            for ch in channel_order:
                arr = parse_manual_channel(manual_values[ch])
                if arr is None:
                    raise ValueError(f"Channel {ch} is empty. Paste some values first.")
                parsed[ch] = arr

            max_len = max(len(v) for v in parsed.values())
            for ch in channel_order:
                v = parsed[ch]
                if len(v) < max_len:
                    parsed[ch] = np.pad(v, (0, max_len - len(v)))

            st.session_state.trial_df = pd.DataFrame(parsed)[channel_order]
            st.session_state.source_label = "Manually entered channel values"
            st.rerun()
        except Exception as e:
            st.error(f"Couldn't parse manual input: {e}")

st.divider()
with st.expander("About this model"):
    st.write(
        f"""
        Model: {config.get('model_type', 'LinearDiscriminantAnalysis')}

        Features: {', '.join(config.get('feature_order', []))}

        Channels expected: {', '.join(channel_order)}

        Sampling rate: {config.get('fs')} Hz

        Classes: {list(label_map.keys())}
        """
    )