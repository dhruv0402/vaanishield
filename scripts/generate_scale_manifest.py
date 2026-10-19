#!/usr/bin/env python3
"""
Production Scale Manifest Generator for Indian Dialectal Anti-Spoofing.

Generates the stratified corpus metadata specification and manifest template
for scaling from the VaaniShield 400-clip pilot to the 50,000-clip enterprise
banking standard (2,000 speakers, 28 states, 12 Indic language pairs).
"""
import json
from pathlib import Path


INDIC_LANGUAGE_SPEC = [
    {"pair": "hi-en", "name": "Hinglish (Hindi-English)", "region": "North India (Delhi, UP, MP, Bihar, Rajasthan)"},
    {"pair": "ta-en", "name": "Tanglish (Tamil-English)", "region": "South India (Tamil Nadu, Puducherry)"},
    {"pair": "kn-en", "name": "Kanglish (Kannada-English)", "region": "South India (Karnataka)"},
    {"pair": "te-en", "name": "Tenglish (Telugu-English)", "region": "South India (Andhra Pradesh, Telangana)"},
    {"pair": "bn-en", "name": "Benglish (Bengali-English)", "region": "East India (West Bengal, Tripura)"},
    {"pair": "mr-en", "name": "Marathilish (Marathi-English)", "region": "West India (Maharashtra, Goa)"},
    {"pair": "gu-en", "name": "Gujlish (Gujarati-English)", "region": "West India (Gujarat)"},
    {"pair": "ml-en", "name": "Manglish (Malayalam-English)", "region": "South India (Kerala)"},
    {"pair": "pa-en", "name": "Punglish (Punjabi-English)", "region": "North India (Punjab, Chandigarh, Haryana)"},
    {"pair": "or-en", "name": "Odialish (Odia-English)", "region": "East India (Odisha)"},
    {"pair": "as-en", "name": "Assamese-English", "region": "Northeast India (Assam)"},
    {"pair": "ur-en", "name": "Urdu-English", "region": "North/Central/South India (Telangana, UP, J&K)"},
]

TELEPHONY_CONDITIONS = [
    "clean_pcm",
    "g711_ulaw_telephony_8khz",
    "amr_wb_telephony_16khz",
    "opus_voip_variable",
    "acoustic_background_traffic",
    "acoustic_background_callcenter",
    "packet_loss_burst_5pct",
]


def generate_production_corpus_schema(output_path: str = "configs/production_corpus_schema.json"):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    schema = {
        "schema_version": "2.0.0-enterprise",
        "title": "VaaniShield Enterprise Banking Corpus Specification",
        "description": "Standardized schema for 50,000+ utterances across 2,000+ Indian speakers",
        "target_metrics": {
            "total_speakers": 2000,
            "speakers_per_language_pair": 166,
            "total_utterances": 50000,
            "utterances_per_speaker_mean": 25,
            "splits": {
                "train": {"speakers": 1200, "pct_speakers": 60, "utterances": 30000},
                "val": {"speakers": 200, "pct_speakers": 10, "utterances": 5000},
                "test_disjoint": {"speakers": 300, "pct_speakers": 15, "utterances": 7500},
                "heldout_synthesizers": {"speakers": 300, "pct_speakers": 15, "utterances": 7500},
            },
        },
        "supported_language_pairs": INDIC_LANGUAGE_SPEC,
        "telephony_environmental_conditions": TELEPHONY_CONDITIONS,
        "cloning_architectures": {
            "training_and_validation": [
                "Coqui XTTS v2",
                "Bark Neural Audio",
                "VITS Zero-Shot",
            ],
            "held_out_evaluation_only": [
                "Microsoft Edge Neural Multilingual",
                "ElevenLabs Voice Cloning v3",
                "OpenAI Voice Engine",
                "Cartesia Sonic Fast",
            ],
        },
        "privacy_compliance": {
            "framework": "Digital Personal Data Protection Act (DPDPA 2023) India",
            "biometric_retention_policy": "Episodic transient acoustic features only; zero raw transcript logging",
            "sha256_audit_trail": "Mandatory tamper-evident cryptographic sidecars for all utterances",
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    print(f"Production corpus schema generated at {path}")
    return schema


if __name__ == "__main__":
    generate_production_corpus_schema()
