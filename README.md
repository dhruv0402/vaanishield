# VaaniShield

**Code-Switch-Aware Deepfake Voice Detection for Financial Fraud Calls in Indian Multilingual Speech**

VaaniShield detects likely voice-cloned segments in Indian code-switched (Hinglish-style)
phone speech by scoring acoustic discontinuity at the exact points where a speaker
switches languages mid-sentence, fused with a speaker-invariant artifact signal so the
model generalizes across unseen callers.

This repository is a **scaffold**, not a finished implementation. It is meant to be built
out module by module — see `ANTIGRAVITY_PROMPT.md` for the full build specification, and
`docs/` for the PRD, TRD, and project proposal this scaffold is derived from.

> **Note on Experimental Results**: Prior preliminary outputs have been moved to `results_SYNTHETIC_SMOKE_TEST/` as they were generated during mock/synthetic pipeline validation. All authoritative results in `results/` are produced strictly from real consented audio recordings and authentic neural synthesizers passing the acceptance gate (`scripts/verify.py`).
>
> **Python Environment**: Python 3.10 or 3.11 is required for neural TTS voice cloning dependencies (`coqui-tts` / `TTS`). FastText language model (`lid.176.bin`) and Montreal Forced Aligner setup instructions are documented in `data/scripts/README.md`.

## Why this project exists (short version)

Voice-cloning fraud (fake "relative in trouble" or fake bank-official calls) is a fast-growing
threat in India. Existing Indian-accent deepfake detectors and audio+text fusion vishing
detectors already exist elsewhere — see `docs/TRD.md` §Related Work for what's already been
done, and why this project deliberately does not repeat it. This project's specific,
verified-novel contribution is: real Indian speech routinely code-switches mid-sentence
(English ↔ a regional language), and voice-cloning systems — trained mostly on monolingual
or cleanly-separated multilingual data — are hypothesized to leave a detectable acoustic
"seam" at those switch points. VaaniShield is built to test that hypothesis rigorously
against existing baselines, not just to build "another" deepfake detector.

## Repository Structure

```
vaanishield/
├── ANTIGRAVITY_PROMPT.md      # Master build prompt/spec — read this first
├── docs/                      # PRD, TRD, proposal (source-of-truth for scope & requirements)
├── configs/                   # YAML configs: feature params, model hyperparams, paths
├── data/
│   ├── raw/                   # Untouched source recordings (bona-fide + cloned) — gitignored
│   ├── processed/             # Aligned, annotated, feature-extracted data — gitignored
│   ├── consent_forms/         # Consent form template for source speakers — NOT gitignored
│   └── scripts/               # Dataset construction: script generation, cloning, alignment, augmentation
├── vaanishield/                # Main Python package
│   ├── boundary/               # Code-switch localization + boundary-local feature extraction
│   ├── speaker_null/            # Speaker-nulling (SNAP-style) projection on SSL embeddings
│   ├── fusion/                  # Fusion classifier + SHAP-based explainability
│   ├── baselines/               # Baseline A (generic) and Baseline B (published Indian-accent model)
│   ├── evaluation/              # EER, bootstrap CIs, ablation harness
│   └── utils/                   # Audio I/O, logging, shared helpers
├── demo/                      # Streamlit demo app
├── notebooks/                 # Exploration notebooks (not for production logic)
├── tests/                     # Unit tests, one file per module
└── scripts/                   # CLI entry points: train.py, evaluate.py, run_ablation.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data & Ethics — Read Before Collecting Anything

No dataset of code-switched Indian speech paired with cloned versions exists publicly, so
this project builds one. **Every source speaker must sign the consent form in
`data/consent_forms/` before any recording or cloning happens.** See
`docs/PRIVACY_AND_ETHICS.md` for the full constraints (DPDPA compliance, no real fraud-call
audio, no scraping voices from social media, license terms of the cloning tools used). These
constraints are non-negotiable and are also embedded in `ANTIGRAVITY_PROMPT.md` so an AI
coding agent building this out respects them by default.

## Status

Scaffold only — see `ANTIGRAVITY_PROMPT.md` for the phased build plan.
