import io
import os
import pickle
import tempfile
from pathlib import Path
import numpy as np
import streamlit as st
import yaml

from vaanishield.boundary.localization import find_codeswitch_points, CodeSwitchPoint
from vaanishield.boundary.features import extract_boundary_features, aggregate_boundary_features
from vaanishield.speaker_null.snap_projection import SpeakerNullingProjector
from vaanishield.fusion.classifier import FusionClassifier
from vaanishield.fusion.explain import explain_prediction
from vaanishield.utils.audio_io import load_audio

st.set_page_config(page_title="VaaniShield Detection Interface", layout="centered")

# Strict, utilitarian, plain-text styling:
# Neutral charcoal/slate palette, monospace/system fonts, zero gradients, zero rounded bubbly cards, zero drop shadows.
st.markdown(
    """
    <style>
    /* Reset background and fonts to functional neutral slate */
    .stApp {
        background-color: #121518;
        color: #d1d5db;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    header, footer {
        visibility: hidden;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #f3f4f6;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    /* Simple borders, rectangular edges, no rounded pill containers */
    div[data-testid="stMetricValue"] {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        color: #f9fafb;
    }
    div[data-testid="stMetricLabel"] {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        color: #9ca3af;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
    }
    .data-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 1rem;
        font-size: 0.85rem;
    }
    .data-table th, .data-table td {
        border: 1px solid #2d3748;
        padding: 6px 10px;
        text-align: left;
    }
    .data-table th {
        background-color: #1a202c;
        color: #a0aec0;
    }
    .status-spoof {
        color: #e53e3e;
        font-weight: bold;
    }
    .status-bonafide {
        color: #38a169;
        font-weight: bold;
    }
    button, input, select {
        border-radius: 0px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("VaaniShield: Acoustic Boundary Seam Detection")
st.text("Telephony Deepfake Voice Detection for Code-Switched Hinglish Speech")
st.write("---")

CONFIG_PATH = "configs/default.yaml"
CHECKPOINT_PATH = "checkpoints/fusion_checkpoint.pkl"


@st.cache_resource
def load_pipeline():
    if not os.path.exists(CHECKPOINT_PATH):
        return None, None, None

    with open(CHECKPOINT_PATH, "rb") as f:
        data = pickle.load(f)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return data["model"], data["projector"], cfg


model, projector, cfg = load_pipeline()

if model is None:
    st.error(f"Checkpoint not found at {CHECKPOINT_PATH}. Run python scripts/train.py first.")
    st.stop()

# Input Configuration
st.subheader("Input Audio & Reference Text")

sample_options = {
    "Manual Upload": None,
    "Sample 1: Authentic Human (OpenSLR 103 spk_1119)": (
        "data/raw/spk_1119_1119_020.wav",
        "Mera laptop start nahi ho raha, can you check the battery?"
    ),
    "Sample 2: Cloned Speech (Coqui XTTS v2)": (
        "data/processed/train_clones/spk_1119_utt_000_xtts.wav",
        "Mera laptop start nahi ho raha, can you check the battery?"
    ),
    "Sample 3: Cloned Speech + Telephony Codec Compression": (
        "data/processed/augmented/spk_1119_utt_000_xtts_codec.wav",
        "Mera laptop start nahi ho raha, can you check the battery?"
    )
}

selected_sample = st.selectbox("Select Preset or Upload File", list(sample_options.keys()))

uploaded_file = None
transcript_input = ""
audio_source_path = None

if selected_sample == "Manual Upload":
    uploaded_file = st.file_uploader("Upload audio file (.wav)", type=["wav"])
    transcript_input = st.text_input(
        "Transcript (Optional - leave blank for automatic alignment)",
        value=""
    )
    if not transcript_input.strip():
        transcript_input = "Mera laptop start nahi ho raha, can you check the battery?"
else:
    preset_path, preset_transcript = sample_options[selected_sample]
    audio_source_path = preset_path
    transcript_input = preset_transcript
    st.text(f"File: {preset_path}")
    st.text(f"Text: {preset_transcript}")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        audio_source_path = tmp_file.name

if audio_source_path and os.path.exists(audio_source_path):
    st.audio(audio_source_path)
    
    if st.button("Run Acoustic Seam Analysis"):
        try:
            audio, sr = load_audio(audio_source_path)
            duration = len(audio) / sr
            tokens = [t.strip() for t in transcript_input.split() if t.strip()]

            points = find_codeswitch_points(
                audio_path=audio_source_path,
                transcript_tokens=tokens,
                window_ms_before=300,
                window_ms_after=300,
            )

            b_feats = extract_boundary_features(audio_source_path, points)
            agg_boundary_vector = aggregate_boundary_features(b_feats, aggregation="mean_and_max")

            raw_emb = projector.embed(audio_source_path)
            nulled_emb = projector.project(raw_emb)

            probs, ci_low, ci_high = model.predict_with_ci(
                agg_boundary_vector.reshape(1, -1),
                nulled_emb.reshape(1, -1),
                n_bootstrap=500,
                ci=0.95,
            )

            prob_val = float(probs[0])
            ci_low_val = float(ci_low[0])
            ci_high_val = float(ci_high[0])

            fused_vec = np.concatenate([agg_boundary_vector, nulled_emb])
            explanations = explain_prediction(fused_vec, model._calibrator, points)

            st.write("---")
            st.subheader("Results")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Spoof Probability", f"{prob_val * 100:.1f}%")
            with c2:
                st.metric("95% CI", f"[{ci_low_val * 100:.1f}%, {ci_high_val * 100:.1f}%]")
            with c3:
                classification = "SPOOF / CLONED" if prob_val >= 0.5 else "BONA-FIDE"
                st.metric("Classification", classification)

            st.text(f"Audio Duration: {duration:.2f}s | Sample Rate: {sr} Hz | Boundaries: {len(points)}")

            st.write("---")
            st.subheader("Acoustic Boundary Analysis Table")
            if explanations:
                table_html = """
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Boundary Index</th>
                            <th>Timestamp (s)</th>
                            <th>Acoustic Metric</th>
                            <th>Attribution Score</th>
                            <th>Acoustic Observation</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                for exp in explanations:
                    table_html += f"""
                        <tr>
                            <td>{exp['boundary_index']}</td>
                            <td>{exp['timestamp_sec']:.2f}</td>
                            <td>{exp['primary_acoustic_cue']}</td>
                            <td>{exp['contribution_score']:.4f}</td>
                            <td>{exp['description']}</td>
                        </tr>
                    """
                table_html += "</tbody></table>"
                st.markdown(table_html, unsafe_allow_html=True)
            else:
                st.text("No intra-sentence code-switch points identified.")

        finally:
            if uploaded_file is not None and os.path.exists(audio_source_path):
                os.remove(audio_source_path)

