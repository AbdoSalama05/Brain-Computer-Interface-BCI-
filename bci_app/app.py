"""
BCI Motor Imagery Classification — Streamlit Demo
--------------------------------------------------
Loads the trained EEGNet model + config produced at the end of Modeling.ipynb
and lets a user upload one EEG trial (CSV/XLSX with FZ, C3, CZ, C4 columns)
to get a LEFT / RIGHT prediction with a confidence score.

Expected files (place next to this script, or update the paths below):
  - eegnet_model.pth     (torch.save(model.state_dict(), ...) from the notebook)
  - eegnet_config.json   (architecture + channel_order + label_map)

Run with:
    streamlit run app.py
"""

import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# Config / paths
# --------------------------------------------------------------------------
MODEL_PATH = "eegnet_model.pth"
CONFIG_PATH = "eegnet_config.json"

st.set_page_config(page_title="BCI Motor Imagery Demo", page_icon="🧠", layout="centered")


# --------------------------------------------------------------------------
# Model definition (must match the architecture used in Modeling.ipynb)
# --------------------------------------------------------------------------
class EEGNet(nn.Module):
    def __init__(self, n_channels=4, n_samples=500, n_classes=2, dropout=0.5):
        super(EEGNet, self).__init__()

        F1 = 8
        D = 2
        F2 = F1 * D
        kernel_length = 64

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            out = self.block2(self.block1(dummy))
            flat_size = out.view(1, -1).shape[1]

        self.classifier = nn.Linear(flat_size, n_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_resource
def load_model_and_config():
    if not os.path.exists(CONFIG_PATH) or not os.path.exists(MODEL_PATH):
        return None, None

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    model = EEGNet(
        n_channels=config["n_channels"],
        n_samples=config["n_samples"],
        n_classes=config["n_classes"],
        dropout=config["dropout"],
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model, config


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


def preprocess_trial(df, n_samples):
    """
    Trim/pad to n_samples and z-score each channel using this trial's own
    mean/std (no session-level scaler is available at inference time, since
    that requires the training sessions' fitted statistics).
    """
    values = df.values.astype(np.float32)

    if len(values) >= n_samples:
        values = values[:n_samples]
    else:
        pad = np.zeros((n_samples - len(values), values.shape[1]), dtype=np.float32)
        values = np.vstack([values, pad])

    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    values = (values - mean) / std

    # (n_samples, n_channels) -> (1, 1, n_channels, n_samples)
    reshaped = values.T[np.newaxis, np.newaxis, :, :]
    return torch.tensor(reshaped, dtype=torch.float32)


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
st.caption("EEGNet demo — classifies imagined LEFT vs RIGHT hand movement from EEG")

model, config = load_model_and_config()

if model is None:
    st.error(
        f"Couldn't find `{MODEL_PATH}` and/or `{CONFIG_PATH}` in the app folder. "
        "Copy those two files (produced at the end of Modeling.ipynb) into the same "
        "directory as app.py, then restart the app."
    )
    st.stop()

channel_order = config["channel_order"]
n_samples = config["n_samples"]
label_map = config["label_map"]
inv_label_map = {v: k for k, v in label_map.items()}

st.divider()

input_mode = st.radio(
    "Input method",
    ["Upload CSV/XLSX", "Enter channels manually"],
    horizontal=True,
)

trial_df = None
source_label = None

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
            trial_df = load_trial(uploaded_file, channel_order)
            source_label = f"Uploaded file: {uploaded_file.name}"
        except Exception as e:
            st.error(str(e))

    elif use_example:
        trial_df = make_example_trial(channel_order, n_samples)
        source_label = "Synthetic example trial (random noise, for UI demo only)"

else:  # Enter channels manually
    st.caption(
        f"Paste {n_samples} values per channel, separated by commas, spaces, or new lines. "
        "Shorter series are zero-padded; longer ones are trimmed."
    )
    manual_values = {}
    manual_error = None

    fill_example = st.button("Fill with example values")

    for ch in channel_order:
        default_text = ""
        if fill_example:
            rng = np.random.default_rng()
            default_text = ", ".join(f"{v:.3f}" for v in rng.normal(0, 1, size=n_samples))
        manual_values[ch] = st.text_area(f"{ch} values", value=default_text, height=80, key=f"manual_{ch}")

    if st.button("Run prediction on manual input", type="primary"):
        try:
            parsed = {}
            for ch in channel_order:
                arr = parse_manual_channel(manual_values[ch])
                if arr is None:
                    raise ValueError(f"Channel {ch} is empty — paste some values first.")
                parsed[ch] = arr

            max_len = max(len(v) for v in parsed.values())
            for ch in channel_order:
                v = parsed[ch]
                if len(v) < max_len:
                    parsed[ch] = np.pad(v, (0, max_len - len(v)))

            trial_df = pd.DataFrame(parsed)[channel_order]
            source_label = "Manually entered channel values"
        except Exception as e:
            st.error(f"Couldn't parse manual input: {e}")

if trial_df is not None:
    st.subheader("Trial preview")
    st.caption(source_label)
    st.line_chart(trial_df.reset_index(drop=True))

    with st.spinner("Running inference..."):
        x = preprocess_trial(trial_df, n_samples)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_label = inv_label_map[pred_idx]
        confidence = probs[pred_idx]

    st.subheader("Prediction")
    c1, c2 = st.columns(2)
    c1.metric("Predicted class", pred_label.upper())
    c2.metric("Confidence", f"{confidence * 100:.1f}%")

    prob_df = pd.DataFrame(
        {"Class": [inv_label_map[i].upper() for i in range(len(probs))], "Probability": probs}
    ).set_index("Class")
    st.bar_chart(prob_df)

    st.info(
        "Remember: with this model, confidence scores do not reflect real predictive "
        "accuracy (see the warning above). Use this output to demo the pipeline UI, "
        "not to make claims about EEG-decoded intent.",
        icon="ℹ️",
    )
else:
    st.info("Upload a trial file or click 'Try example trial' to see a prediction.")

st.divider()
with st.expander("About this model"):
    st.write(
        f"""
        - Architecture: EEGNet ({config['n_channels']} channels, {config['n_samples']} samples/epoch)
        - Channels expected: {', '.join(channel_order)}
        - Classes: {list(label_map.keys())}
        """
    )