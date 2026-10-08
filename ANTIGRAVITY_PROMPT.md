# Master Build Prompt — VaaniShield

**Paste this entire file into Antigravity as your project brief.** It is written to be
self-contained: everything the agent needs to build this repository out correctly —
context, constraints, module specs, build order, and what "done" looks like — is here.
The repo already contains a scaffold (folder structure, stub modules with docstrings and
`TODO(agent)` markers, configs, and docs) — your job is to fill it in, not restructure it
unless a section below explicitly says otherwise.

---

## 0. How to work through this prompt

Work phase by phase (Section 4). Within a phase, read the relevant module docstrings in
the scaffold first (they contain the detailed spec for that specific function) — this
prompt gives you the *why* and the *how it fits together*; the docstrings give you the
*exact interface to implement*. Do not skip straight to writing model code before the
dataset exists — Phase 1 (data) is a hard prerequisite for everything after it.

After each phase, check it against the "Definition of Done" in Section 9 before moving
on. If something in a docstring's `TODO(agent)` conflicts with this prompt, this prompt
wins — the docstrings were written first and this file is the source of truth.

---

## 1. Project Context

**What this is:** VaaniShield detects likely AI-cloned ("deepfake") voices in Indian
phone calls, specifically targeting the financial-fraud scenario where a scammer clones
a relative's or bank official's voice to extract money or credentials.

**Why it's structured the way it is (read this before writing any model code):** A
literature and open-source review (documented in
`docs/VaaniShield_Abstract_ProblemStatement.pdf`) found that "Indian-accent deepfake
audio detection" and "audio+text fusion vishing detection" are already populated
research/engineering areas — including a nearly identical open-source real-time
on-device system (SentinelEdge). Rebuilding that would not be a novel contribution.
**The one verified-unaddressed gap is this project's entire reason to exist:** none of
the surveyed systems account for intra-sentence code-switching (e.g. Hindi-English
"Hinglish" mixing), which is how real Indian phone conversations — including the exact
register fraud calls imitate — actually sound. The hypothesis under test is that voice
clones of code-switched speech leave a detectable acoustic "seam" right at the point
where the language switches, because cloning/TTS systems are trained mostly on
monolingual or cleanly-separated multilingual data.

**What this means for you as the implementer:** the boundary-localization and
boundary-feature modules (`vaanishield/boundary/`) are the heart of the project and
deserve the most care and the most testing. The baselines (`vaanishield/baselines/`) are
not busywork — they are what makes the novelty claim checkable rather than asserted, and
skipping or under-implementing them undermines the entire point of the project. Do not
treat them as lower priority just because they're "someone else's idea."

**What this is not:** a production fraud-detection product. Per `docs/PRD.md`
(Out of Scope), real-time streaming inference, on-device quantization, and federated
learning are explicitly excluded from this build — they are established engineering
(again, see SentinelEdge) and not this project's claimed contribution. Do not add them
unless the human explicitly asks later. A simple Streamlit demo that processes an
uploaded clip in a few seconds is sufficient.

---

## 2. Non-Negotiable Ethical & Legal Constraints

Read `docs/PRIVACY_AND_ETHICS.md` in full before writing any dataset-construction code.
The critical points, restated here because they must shape the code itself, not just be
documented:

1. **Never write code that scrapes voice audio from social media, YouTube, or any public
   source to use as a cloning source.** `data/scripts/clone_voices.py` must only accept
   audio from a `consented_speakers.csv`-style manifest — build that check into the
   script itself, not just as a comment.
2. **Never use or fetch real fraud-call recordings.** All scam-register content in the
   dataset must come from `data/scripts/generate_codeswitch_scripts.py` (scripted
   sentences), never scraped real scam calls.
3. **The explainability layer (`vaanishield/fusion/explain.py`) must never surface
   transcript content** — only descriptions of acoustic patterns. Enforce this with an
   actual test (`tests/test_fusion.py::test_explanation_does_not_contain_transcript_text`),
   not just a docstring promise.
4. **Do not add live-call interception capability.** The demo (`demo/streamlit_app.py`)
   accepts an uploaded file only, never a live call tap.
5. **Signed consent forms and raw/processed audio stay out of version control** — the
   `.gitignore` already reflects this; do not remove those entries when adding new data
   paths.

If a task seems to require violating one of these (e.g. "just grab some sample audio
online to test with quickly"), stop and use synthetic/self-recorded test audio instead,
or flag it to the human rather than working around the constraint.

---

## 3. Repository Structure (already scaffolded)

```
vaanishield/
├── ANTIGRAVITY_PROMPT.md        <- this file
├── README.md
├── requirements.txt
├── docs/                        <- PRD.md*, TRD.md*, proposal PDF, PRIVACY_AND_ETHICS.md
│                                    (*docs are PDFs; treat them as read-only source-of-truth)
├── configs/default.yaml         <- all tunable parameters — read from here, don't hardcode
├── data/
│   ├── raw/, processed/         <- gitignored; audio lives here at runtime
│   ├── consent_forms/           <- template + where signed forms go (gitignored subfolder)
│   └── scripts/                 <- dataset construction (Phase 1)
├── vaanishield/                  <- main package (Phases 2-4)
│   ├── boundary/                 <- Phase 2: THE core novel signal
│   ├── speaker_null/              <- Phase 3: generalization mechanism
│   ├── fusion/                    <- Phase 3: combines the two + explainability
│   ├── baselines/                 <- Phase 2 (parallel): what we're beating
│   ├── evaluation/                <- Phase 4: metrics, ablation
│   └── utils/                     <- shared helpers, build as needed
├── demo/streamlit_app.py         <- Phase 5
├── tests/                        <- write alongside each module, not at the end
└── scripts/                      <- train.py, evaluate.py, run_ablation.py — CLI glue
```

Every stub file already has a docstring with a `TODO(agent)` describing exactly what to
implement and why. Read the docstring in the file you're about to edit before writing
code into it.

---

## 4. Build Order / Phases

Follow this order — later phases depend on earlier ones producing real data/outputs, not
just interfaces.

### Phase 1 — Dataset Construction (`data/scripts/`)
1. `generate_codeswitch_scripts.py` — build the sentence bank (control + fraud-register,
   both needed to avoid confounding urgency with cloning).
2. Human review of generated sentences for naturalness (flag this as a manual step, do
   not skip it silently).
3. Recording protocol + `data/consent_forms/` sign-off for source speakers.
4. `clone_voices.py` — generate spoofed counterparts via XTTS + Bark (training set) and
   OpenVoice (held-out generalization set only — keep physically separate directories).
5. `align_codeswitch_points.py` — batch-run boundary localization over the raw set.
6. `augment_noise.py` — apply channel degradation to both classes equally.

### Phase 2 — Core Detection Signal + Baselines (parallel work)
- `vaanishield/boundary/localization.py` and `features.py` — the novel signal. Implement
  each feature extractor in `features.py` with a unit test against synthetic/controlled
  audio before moving on (see `tests/test_features.py`).
- `vaanishield/baselines/baseline_a_generic.py` and `baseline_b_indian_accent.py` — build
  these fully, in parallel with the boundary module, not after. They need to exist before
  Phase 4 evaluation can mean anything.

### Phase 3 — Speaker-Nulling + Fusion
- `vaanishield/speaker_null/snap_projection.py` — implement after reading the SNAP
  reference (see docs). Validate that projection actually reduces speaker-identifiability
  (e.g. a simple speaker-classification probe should perform near-chance on the
  projected embeddings — consider adding this as a diagnostic, not just an assumption).
- `vaanishield/fusion/classifier.py` and `explain.py` — combine both streams; keep the
  classifier simple (logistic regression or small MLP) so ablation stays interpretable.

### Phase 4 — Evaluation
- `vaanishield/evaluation/metrics.py` — EER, bootstrap CIs, breakdown-by-condition.
- `vaanishield/evaluation/ablation.py` — the three-way ablation (boundary-only,
  speaker-null-only, fused) plus the held-out-synthesizer generalization check.
- `scripts/evaluate.py` and `scripts/run_ablation.py` — wire it all together and produce
  a results report. This report is the deliverable that backs the novelty claim — treat
  its correctness as seriously as the model code itself.

### Phase 5 — Demo
- `demo/streamlit_app.py` — upload → pipeline → risk score + CI + per-boundary-point
  explanation. Keep it simple; this is a proof-of-concept UI, not a product.

---

## 5. Coding Standards

- **Read config, don't hardcode.** Every tunable parameter (window sizes, model choice,
  which cloning systems are held out, etc.) already has a home in
  `configs/default.yaml` — read from it, and add new parameters there rather than as
  magic numbers in code.
- **Type hints + docstrings on every public function.** The stubs already model this
  style — match it.
- **Log, don't print**, in anything under `data/scripts/` or `vaanishield/` (use
  `vaanishield.utils.logging_utils.get_logger`) — reproducibility of dataset construction
  matters here.
- **Write the test alongside the module**, not in a final "testing phase" — the test
  stubs in `tests/` are placed next to what they test and marked `@pytest.mark.skip`
  with a reason; remove the skip as you implement the corresponding function.
- **Keep `baselines/` self-contained** — no importing from `boundary/` or
  `speaker_null/` — so they remain a clean, independent point of comparison.
- **No large/complex model in `fusion/classifier.py`** without flagging it to the human
  first — interpretability of the fusion step for ablation purposes is a stated
  requirement, not a nice-to-have.

---

## 6. What NOT to Build (scope guardrails, from `docs/PRD.md`)

Do not implement, even if it seems like a natural next step:
- Real-time/streaming inference or low-latency optimization.
- On-device quantization (TFLite/ONNX export) or mobile app packaging.
- Federated learning / cross-device model updates.
- Coverage of every Indian language pair or every voice-cloning system on the market —
  Hindi-English is required; a second pair is optional if time allows.
- Any live-call / telecom-side integration.

If asked to build one of these later, treat it as a deliberate scope change, not an
extension of the current plan.

---

## 7. Definition of Done (per phase)

- **Phase 1 done when:** a processed dataset exists with bona-fide + cloned
  code-switched audio (training synthesizers + a held-out synthesizer kept separate),
  annotated with code-switch timestamps, in both clean and noise-augmented versions —
  and every source speaker has a signed consent form on file.
- **Phase 2 done when:** `extract_boundary_features` returns real (non-`NotImplementedError`)
  fixed-length vectors on real data, each feature extractor has a passing unit test, and
  both baselines can be fit and produce `predict_proba` outputs on the processed dataset.
- **Phase 3 done when:** the speaker-nulling projection measurably reduces
  speaker-identifiability (diagnostic probe near chance), and the fusion classifier
  produces calibrated probabilities with confidence intervals; the explainability test
  confirming no transcript leakage passes.
- **Phase 4 done when:** `scripts/evaluate.py` produces a report comparing the fusion
  model against both baselines with EER/precision/recall/CIs broken down by language
  pair and noise condition, plus the three-way ablation and the held-out-synthesizer
  generalization result — this is the artifact that proves (or honestly disproves) the
  project's central hypothesis.
- **Phase 5 done when:** a human can upload a short call clip to the Streamlit demo and
  see a risk score, CI, and per-boundary-point explanation with no transcript content
  shown.

If Phase 4's results don't support the boundary-artifact hypothesis, that is still a
valid, reportable outcome for an academic submission — do not quietly adjust the
evaluation methodology to force a positive result. Report what the data shows.
