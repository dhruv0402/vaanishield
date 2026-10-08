"""
Unit tests for vaanishield.boundary.localization.

TODO(agent): once implemented, test at minimum:
    - A synthetic transcript with a known code-switch point is detected at the right
      approximate timestamp.
    - A purely monolingual utterance produces zero CodeSwitchPoints.
    - Windows are clipped correctly at utterance start/end (no out-of-bounds slicing).
"""

import numpy as np
import pytest
from pathlib import Path
from vaanishield.boundary.localization import (
    tag_token_languages,
    align_tokens_to_audio,
    find_codeswitch_points,
    CodeSwitchPoint,
)
from vaanishield.utils.audio_io import save_audio


@pytest.fixture
def sample_wav(tmp_path):
    wav_path = tmp_path / "sample.wav"
    sr = 16000
    duration = 2.0
    audio = np.zeros(int(sr * duration), dtype=np.float32)
    save_audio(str(wav_path), audio, sr)
    return str(wav_path)


def test_detects_known_codeswitch_point(sample_wav):
    # Hindi ("Mera") followed by English ("laptop")
    tokens = ["mera", "laptop"]
    tags = tag_token_languages(tokens)
    assert tags == ["hi", "en"]

    points = find_codeswitch_points(
        audio_path=sample_wav,
        transcript_tokens=tokens,
        window_ms_before=300,
        window_ms_after=300,
    )

    assert len(points) == 1
    pt = points[0]
    assert pt.from_lang == "hi"
    assert pt.to_lang == "en"
    assert 0.0 < pt.timestamp_sec < 2.0
    assert pt.window_start_sec >= 0.0
    assert pt.window_end_sec <= 2.0


def test_monolingual_utterance_has_no_switch_points(sample_wav):
    # Pure English
    en_tokens = ["this", "is", "a", "clean", "meeting", "call"]
    tags_en = tag_token_languages(en_tokens)
    assert all(t == "en" for t in tags_en)

    points_en = find_codeswitch_points(
        audio_path=sample_wav,
        transcript_tokens=en_tokens,
    )
    assert len(points_en) == 0

    # Pure Hindi
    hi_tokens = ["mera", "ghar", "bohot", "door", "hai"]
    tags_hi = tag_token_languages(hi_tokens)
    assert all(t == "hi" for t in tags_hi)

    points_hi = find_codeswitch_points(
        audio_path=sample_wav,
        transcript_tokens=hi_tokens,
    )
    assert len(points_hi) == 0


def test_window_clipping_at_audio_edges(sample_wav):
    # Switches near start and end of utterance
    tokens = ["mera", "phone", "break", "hua"]
    points = find_codeswitch_points(
        audio_path=sample_wav,
        transcript_tokens=tokens,
        window_ms_before=1500,  # Deliberately large window to test clipping
        window_ms_after=1500,
    )

    assert len(points) >= 1
    for pt in points:
        assert pt.window_start_sec >= 0.0
        assert pt.window_end_sec <= 2.0
        assert pt.window_start_sec <= pt.timestamp_sec <= pt.window_end_sec

