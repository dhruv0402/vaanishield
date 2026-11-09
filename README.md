# VaaniShield

**A research prototype for code-switch-aware deepfake voice detection in Indian multilingual speech.**

VaaniShield studies whether acoustic changes near language transitions help distinguish
synthetic speech from human speech. It combines boundary features with a speaker-subspace
projection and a calibrated two-stream classifier, with a FastAPI/React demo and streaming
protocol support. It is not a validated banking fraud detector or an autonomous call-blocking
system.

## What the implementation does

1. Locates boundaries from supplied transcript tokens, or acoustic spectral-flux changes
   when a transcript is absent. An acoustic change point is not proof of a language switch.
2. Measures pitch, energy, duration, formant and MFCC changes around those points.
3. Extracts a 768-dimensional multi-resolution mel/MFCC statistical representation and
   applies a fitted speaker-subspace projection. The default extractor is **not WavLM**,
   despite legacy `ssl` variable names.
4. Combines calibrated per-stream logistic regression predictions.
5. Exposes upload/preset analysis, PDF summaries and experimental WebSocket streaming.
   Call-control labels are recommendations. Supplying a webhook URL can send a real request;
   leave it unset for local simulation.

## Evidence and limitations

The checked-in reports are **historical artifacts**, not results reproduced by the test suite.
The repository does not distribute the trained checkpoints or the complete audio corpus.

### Pilot benchmark

[`results/evaluation_report.json`](results/evaluation_report.json) describes 400 records and
8 speakers, with train/validation/test/held-out sizes of 200/50/100/50.
Lower equal error rate (EER) is better.

| Model | In-domain test EER | Held-out synthesizer EER |
| --- | ---: | ---: |
| Baseline A | 0.39 | **0.12** |
| Baseline B | 0.48 | 0.14 |
| VaaniShield fusion | **0.36** | 0.34 |

The small in-domain point-estimate improvement does not establish superiority. The stored
intervals overlap, and no paired significance test is supplied. Baseline A performs better
on the held-out synthesizer. These results do not establish generalization to arbitrary
callers, accents, synthesis systems or real fraud calls.

The metric implementation now interpolates the ROC crossing for EER and resamples whole
clusters without independently resampling the recordings inside them. Undefined EER or
intervals are reported as unavailable rather than perfect performance. **Existing reports
have not been regenerated with these corrections; their EERs and intervals need recomputation.**
A one-speaker evaluation cannot estimate uncertainty across unseen speakers. Choose the
resampling unit to match the claim and keep augmented versions of the same source together.
The early-feature-fusion ablation is a different estimator from the main late-fusion
classifier. Fresh ablation reports preserve their own computed results instead of copying
metrics from another report. Equality checks require matching model/config/manifest identity.

### Larger corpus and latency are separate experiments

- [`data/processed/scaled_master_manifest.jsonl`](data/processed/scaled_master_manifest.jsonl)
  contains **1,600 records and 32 speakers**, split 750/150/350/350. The older 800-record,
  16-speaker description is not the current manifest.
- [`results/generalization_report.json`](results/generalization_report.json) describes that
  larger corpus. Its overlapping regimes and all-record aggregates are not interchangeable
  with the pilot's held-out test. Do not compare these numbers as a controlled improvement.
- [`results/latency_profile.json`](results/latency_profile.json) records 75 runs over five
  files: full-pipeline median **125.326 ms**, P95 **243.262 ms**, P99 **413.870 ms**.
  The report itself says the **200 ms budget was not met**. These are stored measurements,
  not a hardware-independent throughput guarantee.
- Balanced demographic counts do not demonstrate fairness. Neither these experiments nor
  ONNX serialization establish regulatory compliance, certification or production readiness.
- Per-call fields named `confidence_interval_95` in the demo are legacy model-dispersion
  heuristics, **not validated 95% statistical coverage or a probability that a caller is safe**.
  ONNX's current `predict_with_ci` uses assumed noise, not a measured uncertainty model.

## Reproducible software checks

Use Python **3.12** and Node **22.12+**. The smaller CPU test environment does not download
pretrained models or voice-cloning engines:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-ci.txt
python -m pytest tests/ -v
PYTHONPATH=. python scripts/verify.py --metadata-only
PYTHONPATH=. python scripts/validate_dataset.py --metadata-only
cd web
npm ci
npm run build
```

API and streaming tests use a deterministic, tiny **test-only** fitted classifier and
synthetic waveform fixtures. NPZ serialization is compared to that fitted classifier;
a small ONNX graph tests runtime execution. None of these fixtures validate detection
accuracy, replace the trained model, or modify the deployed application.

The trained-model ONNX parity test is explicitly skipped by default. To run it, provision
these trusted, matching artifacts and their SHA-256 sidecars:

- `checkpoints/fusion_checkpoint.pkl`
- `checkpoints/onnx/vaanishield_end_to_end.onnx`
- `checkpoints/onnx/model_parameters.npz`

```bash
python -m pytest tests/test_zero_pickle_onnx.py --run-artifact-tests -v
```

Opting in fails if required artifacts are missing. There is no verified model-release
URL in this repository. Do not download arbitrary pickle files or treat a generated test
model as the research checkpoint.

### Audit scope

CI runs **metadata checks**, not a complete corpus or scientific acceptance audit.
`--metadata-only` explicitly reports that missing audio was not verified and skips the
bandwidth test. Without that option, both scripts fail when referenced audio is absent:

```bash
PYTHONPATH=. python scripts/verify.py
PYTHONPATH=. python scripts/validate_dataset.py
```

Run these against a complete, licensed local corpus. Narrative verdict review remains
manual; the scripts do not certify consent, licensing, privacy compliance or model efficacy.

## Running the research demo

The full optional development environment is listed in `requirements.txt`. Model export
also requires PyTorch. Corpus construction may require additional tools documented in the
relevant scripts. Provision a trusted trained checkpoint before analyzing audio:

```bash
pip install -r requirements.txt
python -m uvicorn demo.api.server:app --host 127.0.0.1 --port 8000
# In another terminal:
cd web && npm ci && npm run dev
```

Open `http://localhost:5173`. Some catalog entries reference audio not included in Git.
Use licensed local recordings; do not expose this research API directly to the public.

Checksum verification now requires a matching sidecar. Both `model.npz.sha256` and legacy
`model.sha256` naming are accepted. A checksum detects corruption relative to a trusted
reference, not publisher identity, and does not make pickle safe. Prefer NPZ/ONNX for
inference. Never load untrusted pickle checkpoints.

### Container build and model provisioning

The default image contains the CPU API runtime and built dashboard, **not a trained
model**. It builds from a clean checkout without private checkpoint files. Optional
Whisper/neural ASR, training and export dependencies are not installed in this image;
the default statistical embedding and acoustic localization fallback remain available.

```bash
docker build -t vaanishield:local .
docker run --rm -p 127.0.0.1:8000:8000 vaanishield:local
# In another terminal, verify the model-free image:
python scripts/smoke_container.py
```

`/api/health` is process liveness and stays HTTP 200 without a model. `/api/ready`
attempts to load and verify the configured model and returns HTTP 503 until successful.
Without a usable model, `/api/analyze` returns HTTP 503 with `model_unavailable`, and
the streaming endpoint sends that error and closes with code 1013. No predictions are
generated as a fallback. Readiness means successful loading, not validated accuracy.

For actual inference, supply your trusted artifacts and checksum sidecars in a local
directory with the layout above, then mount that directory read-only:

```bash
docker run --rm -p 127.0.0.1:8000:8000 \
  --mount type=bind,src=/absolute/path/to/trusted/checkpoints,dst=/app/checkpoints,readonly \
  vaanishield:local
curl --fail http://127.0.0.1:8000/api/ready
```

The API prefers `onnx/model_parameters.npz` inside the mount, with the legacy trusted
`fusion_checkpoint.pkl` as an alternative. A failed NPZ verification does not silently
fall back to pickle. Successful loads are cached per worker; restart the container to
replace a loaded model. Missing catalog audio can be supplied separately if licensed.

CI builds and starts the model-free image, checks the dashboard and liveness, and
requires readiness and inference to reject requests with HTTP 503. This is a container
integration test, not trained-model validation or a production deployment. The workflow
does not publish the image or deploy a service.

## Repository map

- `vaanishield/boundary/`: localization and acoustic features
- `vaanishield/speaker_null/`: statistical embedding and subspace projection
- `vaanishield/fusion/`: classifier, NPZ/ONNX loading and explanations
- `vaanishield/evaluation/`: metrics and ablations
- `demo/api/`, `web/`: research API and dashboard
- `data/`, `configs/`: manifests, provenance records and configuration
- `scripts/`: training, evaluation, export, audit and profiling tools
- `tests/`: software regression tests and opt-in trained-artifact parity
- `results/`, `docs/`: historical reports and research writeups

## Privacy and intended use

Explanation output is designed to describe acoustic features without returning transcript
text. That is one engineering control, not proof of DPDPA compliance or anonymization.
Audio, supplied transcripts, metadata, generated reports and outgoing webhook payloads
still require an appropriate lawful basis, access controls and retention policy.
Use only recordings you are permitted to process. Human review and independent real-world
validation are required before any consequential use.
