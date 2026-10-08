# VaaniShield Data Schema Specification

This document defines the strict, non-negotiable schema for manifests and metadata across all experimental stages of VaaniShield.

---

## 1. Unified Dataset Manifest Schema (`data/processed/master_manifest.jsonl`)

Every row in the master manifest must strictly conform to the following schema:

| Column / Field | Type | Description | Permitted Values |
| :--- | :--- | :--- | :--- |
| `utterance_id` | `str` | Unique primary key for the audio record | e.g. `spk_1119_utt_001_xtts_codec` |
| `source_utterance_id` | `str` | ID of the source recording | e.g. `spk_1119_utt_001` |
| `sentence_id` | `str` | ID of the text script from the sentence bank | e.g. `sent_001` |
| `speaker_id` | `str` | Unique ID of the speaker | Must exist in `data/consent_forms/consented_speakers.csv` |
| `audio_path` | `str` | Relative or absolute path to the `.wav` file | File must exist, duration > 0.5s, non-silent |
| `label` | `str` | Ground truth classification | `"bona_fide"` or `"spoof"` |
| `cloning_system` | `str` | Voice cloning engine used | `"none"` (bona-fide), `"xtts"`, `"bark"`, `"openvoice"` |
| `is_heldout_system` | `bool` | Whether the synthesizer is in the held-out generalization set | `true` or `false` |
| `noise_condition` | `str` | Telecom channel degradation applied | `"clean"`, `"codec_compression"`, `"amr_wb_compression"`, `"packet_loss_simulation"`, `"background_noise_injection"` |
| `split` | `str` | Partition for training & evaluation | `"train"`, `"test"`, `"heldout"` |
| `seed` | `int` | Deterministic seed derived from `utterance_id` | Non-negative integer |
| `duration_sec` | `float` | Audio duration in seconds | > 0.5 |
| `sample_rate_hz` | `int` | Audio sample rate | Typically 16000 |
| `loudness_lufs_or_rms` | `float` | Measured RMS loudness of the normalized audio | Consistent distribution |
| `codeswitch_points` | `list` | Detected / ground-truth boundary points | List of boundary objects with `timestamp_sec` |

---

## 2. Split Partitioning Rules (Leakage Prevention)

1. **Speaker Grouping**: Speakers are partitioned mutually exclusively. No `speaker_id` in `test` or `heldout` may appear in `train`.
2. **Utterance Cohesion**: All noise variants (`clean`, `codec`, `packet_loss`, `bg_noise`, `amr_wb`) of a single `source_utterance_id` must travel together in the exact same split.
3. **Dual-Class Evaluation**: Both `test` and `heldout` splits MUST contain bona-fide and spoof audio so that Equal Error Rate (EER) and ROC metrics are mathematically defined.

---

## 3. Consented Speakers Registry Schema (`data/consent_forms/consented_speakers.csv`)

| Column | Type | Requirement |
| :--- | :--- | :--- |
| `speaker_id` | `str` | Unique speaker identifier (e.g. `spk_1119`) |
| `speaker_name` | `str` | Name or research contributor handle |
| `form_id` | `str` | Identifier of signed consent form |
| `consent_signed` | `bool` | `true` |
| `consent_date` | `str` | ISO date `YYYY-MM-DD` |
| `regional_language` | `str` | Indic language code (`hi`, `kn`, `ta`) |
| `source_license` | `str` | Research license or consent agreement |

Every row must have a matching file at `data/consent_forms/signed/{form_id}.md`.
