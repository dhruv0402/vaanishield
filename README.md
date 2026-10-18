# VaaniShield

**Code-Switch-Aware Deepfake Voice Detection for Financial Fraud Calls in Indian Multilingual Speech**

VaaniShield detects synthetic and voice-cloned segments in Indian code-switched (e.g. Hinglish)
telephony speech by scoring acoustic discontinuities at the exact points where a speaker
switches languages mid-sentence, fused with an orthogonal speaker-nulled global artifact stream
so the model generalizes across unseen callers and accents.

---

## 1. Why this project exists

Voice-cloning fraud (fake "relative in distress", urgent loan verification, or bank impersonation) is an accelerating threat in India. Existing deepfake audio detectors are typically trained on monolingual English or Chinese datasets and treat the entire utterance as a single uniform clip.

In India, conversational speech is overwhelmingly **code-switched** (e.g., Hindi ↔ English, Kannada ↔ English, Tamil ↔ English). Voice-cloning engines (such as autoregressive token synthesizers and cross-lingual neural vocoders) often struggle with acoustic prosody transitions across phonetic inventories, leaving measurable acoustic "seams" (pitch jumps, energy steps, and formant phase resets).

VaaniShield evaluates this hypothesis: **Do cross-lingual boundary seam features, when combined with an orthogonal speaker-nulled global artifact stream, improve deepfake voice detection under telephony channel compression?**

---

## 2. Architecture & Pipeline

```
                    ┌──────────────────────────────────────────────┐
                    │ Inbound Telephony Audio (8 kHz / 16 kHz)     │
                    └──────────────────────┬───────────────────────┘
                                           │
                    ┌──────────────────────┴───────────────────────┐
                    │  Acoustic VAD & Speech Duration Alignment    │
                    └───────┬──────────────────────────────┬───────┘
                            │                              │
         ┌──────────────────▼────────────────┐  ┌──────────▼──────────────────────────┐
         │ Boundary Stream (Local Seams)     │  │ Global Stream (Speaker-Invariant)   │
         │ - Code-switch point localization  │  │ - Multi-scale CMN modulation & MFCCs │
         │ - FastText LID + acoustic VAD     │  │ - SNAP Speaker-Nulling Projection   │
         │ - F0 reset & pitch jumps (st)     │  │ - Orthogonal complement of speaker  │
         │ - Energy step & Formant deltas    │  │   centroid covariance subspace      │
         └──────────────────┬────────────────┘  └──────────┬──────────────────────────┘
                            │                              │
                            └──────────────┬───────────────┘
                                           │
                    ┌──────────────────────▼───────────────────────┐
                    │ Multi-Stream L2 Shrinkage Fusion (C=0.01)    │
                    │ - Calibrated Platt scaling (Sigmoid)         │
                    │ - Operational Tiers: PASS / STEP-UP / BLOCK  │
                    │ - 95% Bootstrap Confidence Intervals         │
                    └──────────────────────┬───────────────────────┘
                                           │
                    ┌──────────────────────▼───────────────────────┐
                    │ Telemetry & Privacy-Compliant Explanations   │
                    │ - Spoof Probability & 95% Confidence Interval│
                    │ - Acoustic Seam Attribution (Zero Transcript)│
                    │ - DPDPA 2023 Non-Semantic Compliance         │
                    └──────────────────────────────────────────────┘
```

---

## 3. Benchmark Results (Strict Single Source of Truth)

All metrics are evaluated across **4 strictly speaker-disjoint splits** (Train: 200, Val: 50, Test: 100, Held-Out Speaker: 50; N=400 total utterances):

| Model Configuration | Validation EER | In-Domain Test EER | 95% Bootstrap CI | Held-Out Speaker EER | Test Precision / Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline A (Generic Acoustic + LR) | 0.3200 | 0.1900 | [0.1200, 0.2800] | 0.3200 | 72.1% / 88.0% |
| Baseline B (Indian Accent Mel + LFCC) | 0.2000 | 0.1700 | [0.1000, 0.2400] | 0.1600 | 76.9% / 80.0% |
| **VaaniShield Fused Model** | **0.1000** | **0.1200** | **[0.0800, 0.1600]** | **0.0600** | **81.8% / 90.0%** |

### Three-Way Ablation Analysis (Scientific Finding)

Ablation evaluating individual and combined stream contributions across test and held-out sets:

| Stream / Variant | Test Set EER | Held-Out Speaker EER | Core Finding & Architectural Role |
| :--- | :---: | :---: | :--- |
| **Boundary-Local Stream Only** | 0.3600 | 0.3400 | Boundary cues alone degrade under heavy telephony codecs (PSTN/G.711, AMR-WB). |
| **Speaker-Nulled Global Stream Only** | 0.1600 | 0.0800 | Orthogonal projection removes speaker identity and carries the bulk of vocoder artifact detection. |
| **VaaniShield Fused Model** | **0.1200** | **0.0600** | Fusing boundary transition dynamics with speaker-nulled residuals yields lowest overall EER. |

> **Scientific Finding**: Intra-utterance boundary cues alone are vulnerable to narrowband loss (36% EER); however, fusing them with orthogonal speaker-nulled global embeddings provides the lowest error rate and highest generalization on unseen speakers.

---

## 4. Repository Structure

```
vaanishield/
├── configs/                   # YAML configs: feature params, model hyperparams, paths
├── data/
│   ├── raw/                   # Source recordings (bona-fide OpenSLR + neural clones)
│   ├── processed/             # Processed master_manifest.jsonl and augmented audio
│   ├── consent_forms/         # Speaker registry and open corpus licensing records
│   └── scripts/               # Manifest generation, noise augmentation, alignment
├── vaanishield/                # Core Python package
│   ├── boundary/               # Code-switch localization + boundary-local feature extraction
│   ├── speaker_null/            # SNAP speaker-nulling orthogonal subspace projection
│   ├── fusion/                  # Calibrated fusion classifier + acoustic explainability
│   ├── baselines/               # Baseline A (generic acoustic) and Baseline B (Indian-accent LFCC)
│   ├── evaluation/              # Metrics calculation, bootstrap CIs, ablation harness
│   └── utils/                   # Audio I/O, logging, signal processing
├── demo/                      # FastAPI research demo server and streaming WebSocket
├── web/                       # React / Vite web dashboard (desktop-optimized)
├── tests/                     # 13 pytest unit and integration tests
└── scripts/                   # CLI tools: verify.py, train.py, evaluate.py, run_ablation.py
```

---

## 5. Quickstart & Execution

### 1. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Test Suite & Acceptance Gate
```bash
# Run unit and integration tests
pytest tests/ -v

# Run 8-point strict acceptance gate
python scripts/verify.py

# Validate manifest integrity and corpus licenses
python scripts/validate_dataset.py
```

### 3. Launch Research Demo Server & Web Dashboard
```bash
# Start backend API (FastAPI) on port 8000
python -m uvicorn demo.api.server:app --port 8000

# In a separate terminal, launch the web interface
cd web
npm install
npm run dev
```
Open `http://localhost:5173` in your browser to inspect test audio samples, record audio via microphone, and view acoustic seam explanations.

---

## 6. Privacy & Ethics (DPDPA 2023 Compliance)

1. **Lawful Basis & Corpus Provenance**: The speech dataset is constructed from open public academic corpora: OpenSLR 103 (Hindi, public research license), OpenSLR 65 (Tamil, CC BY-SA 4.0), and OpenSLR 79 (Kannada, CC BY-SA 4.0). Clones were generated using Coqui XTTS v2 for scientific benchmarking.
2. **Zero Transcript Storage**: In compliance with India's Digital Personal Data Protection Act (DPDPA 2023), VaaniShield performs strictly non-semantic acoustic analysis. Conversational speech content is never transcribed, stored, or leaked by the explainability layer.
3. **Research Prototype**: This repository is an academic research prototype designed to study cross-lingual synthetic speech boundaries. It does not contain live carrier wiretapping capabilities.
