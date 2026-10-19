import os
import pytest
from vaanishield.reporting.certificate import generate_forensic_certificate_pdf, compute_sha256

def test_generate_pdf_certificate():
    mock_analysis = {
        "fusion_score": 0.88,
        "calibrated_p_spoof": 0.942,
        "tier": "CRITICAL",
        "decision": "BLOCK_AND_ALERT",
        "ci_lower": 0.89,
        "ci_upper": 0.98,
        "audio_sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        "boundary_analysis": {
            "seams_detected": 2,
            "seam_timestamps": [1.45, 3.20]
        },
        "language_transitions": [
            {
                "timestamp_seconds": 1.45,
                "from_language": "hi",
                "to_language": "en",
                "cue": "Phase discontinuity & vocoder mismatch",
                "anomaly_score": 0.88
            }
        ]
    }

    pdf_bytes = generate_forensic_certificate_pdf(mock_analysis)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    # PDF header signature check
    assert pdf_bytes.startswith(b"%PDF")

def test_compute_sha256(tmp_path):
    test_file = tmp_path / "sample.txt"
    test_file.write_text("VaaniShield Test Audio Content")
    sha = compute_sha256(str(test_file))
    assert len(sha) == 64
    assert sha != "UNKNOWN_HASH_FILE_NOT_FOUND"
