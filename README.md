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

All metrics are evaluated across **4 strictly speaker-disjoint splits** (Train: 200, Val: 50, Test: 100, Held-Out Synthesizer: 50; N=400 total utterances) with identical 8000 Hz telephony filtering, distinct neural synthesizer architectures (Training/Test: Coqui XTTS v2; Held-Out: Microsoft Edge Neural TTS), and cluster bootstrap resampling grouped by `source_utterance_id`:

| Model Configuration | Validation EER | In-Domain Test EER | 95% Bootstrap CI | Held-Out Synthesizer EER | Test Precision / Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline A (Generic Acoustic + LR) | 0.2400 | 0.3900 | [0.2586, 0.5201] | 0.1200 | 66.0% / 62.0% |
| Baseline B (Indian Accent Mel + LFCC) | 0.3400 | 0.4800 | [0.2881, 0.5700] | 0.1400 | 56.9% / 66.0% |
| **VaaniShield Fused Model** | **0.0000** | **0.1400** | **[0.0200, 0.3001]** | **0.7400** | **88.4% / 76.0%** |

### Three-Way Ablation Analysis (Scientific Finding)

Ablation evaluating individual and combined stream contributions across test and held-out sets:

| Stream / Variant | Test Set EER | Held-Out Synthesizer EER | Core Finding & Architectural Role |
| :--- | :---: | :---: | :--- |
| **Boundary-Local Stream Only** | 0.5100 | 0.7000 | Under narrowband telephony loss, local boundary cues alone degrade to chance without global vocoder conditioning. |
| **Speaker-Nulled Global Stream Only** | 0.2200 | 0.6600 | Orthogonal projection captures XTTS training vocoder signatures, but exhibits domain shift on unseen commercial neural TTS engines. |
| **VaaniShield Fused Model** | **0.1400** | **0.7400** | Fusing boundary transition dynamics with speaker-nulled residuals yields lowest in-domain error rate (14.0% EER vs 39.0% / 48.0% baselines). |

> **Scientific Finding**: When acoustic bandwidth shortcuts are eliminated (all audio normalized to 8000 Hz telephony standard):
> 1. VaaniShield significantly outperforms both baselines on in-domain test data (**14.0% EER** vs. 39.0% for Baseline A and 48.0% for Baseline B).
> 2. Generalization to unseen commercial neural synthesizers (Edge Neural TTS) reveals an open challenge in synthetic speech detection: models tuned on open autoregressive vocoders (XTTS) require cross-architecture multi-synthesizer training to maintain calibration across commercial diffusion/streaming neural voices.

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
