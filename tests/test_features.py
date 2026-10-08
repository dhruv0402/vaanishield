"""
Unit tests for vaanishield.boundary.features.

TODO(agent): test each feature extractor on synthetic/controlled audio (e.g. a pure
tone with a deliberate pitch jump should produce a large f0_reset_magnitude), and test
that extract_boundary_features returns a fixed-length vector regardless of how many
boundaries were detected (per the aggregation policy in configs/default.yaml).
"""

import numpy as np
import pytest
from vaanishield.boundary.features import (
    extract_f0_reset_magnitude,
    extract_energy_discontinuity,
    extract_phone_duration_ratio,
    extract_formant_transition_smoothness,
    extract_mfcc_delta_magnitude,
    aggregate_boundary_features,
    BoundaryFeatureVector,
    FEATURE_NAMES,
)
from vaanishield.boundary.localization import CodeSwitchPoint


def test_f0_reset_detects_synthetic_pitch_jump():
    sr = 16000
    duration = 0.4
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    mid = len(t) // 2

    # Synthetic tone with smooth continuous pitch
    f0_smooth = 150.0
    smooth_tone = np.sin(2 * np.pi * f0_smooth * t).astype(np.float32)
    jump_smooth = extract_f0_reset_magnitude(smooth_tone, sr)

    # Synthetic tone with sharp pitch jump (150Hz -> 230Hz) at midpoint
    jump_tone = np.zeros_like(t)
    jump_tone[:mid] = np.sin(2 * np.pi * 150.0 * t[:mid])
    jump_tone[mid:] = np.sin(2 * np.pi * 230.0 * t[mid:])
    jump_discontinuous = extract_f0_reset_magnitude(jump_tone.astype(np.float32), sr)

    assert jump_discontinuous > jump_smooth
    assert jump_discontinuous > 0.3


def test_energy_discontinuity_detects_amplitude_step():
    sr = 16000
    duration = 0.4
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    mid = len(t) // 2

    # Smooth constant amplitude tone
    const_tone = np.sin(2 * np.pi * 200.0 * t).astype(np.float32)
    jump_smooth = extract_energy_discontinuity(const_tone, sr)

    # Abrupt 10x amplitude jump at midpoint
    step_tone = const_tone.copy()
    step_tone[mid:] *= 10.0
    jump_discontinuous = extract_energy_discontinuity(step_tone, sr)

    assert jump_discontinuous > jump_smooth
    assert jump_discontinuous > 1.5


def test_formant_and_mfcc_delta_extractors():
    sr = 16000
    duration = 0.4
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = np.sin(2 * np.pi * 220.0 * t).astype(np.float32)

    smoothness = extract_formant_transition_smoothness(tone, sr)
    assert 0.0 <= smoothness <= 1.0

    mfcc_delta = extract_mfcc_delta_magnitude(tone, sr)
    assert mfcc_delta >= 0.0

    dur_ratio = extract_phone_duration_ratio(tone, sr)
    assert dur_ratio >= 0.0


def test_feature_vector_is_fixed_length():
    sr = 16000
    # Case 1: Empty boundaries (e.g. monolingual utterance)
    vec_empty = aggregate_boundary_features([], aggregation="mean_and_max")
    assert isinstance(vec_empty, np.ndarray)
    assert vec_empty.shape == (10,)
    assert np.all(vec_empty == 0.0)

    # Case 2: 1 boundary
    pt1 = CodeSwitchPoint(1.0, "hi", "en", 0.7, 1.3)
    feat1 = BoundaryFeatureVector(pt1, {k: 1.0 for k in FEATURE_NAMES})
    vec_1 = aggregate_boundary_features([feat1], aggregation="mean_and_max")
    assert vec_1.shape == (10,)

    # Case 3: 3 boundaries -> must still produce exact same 10-dim vector
    pt2 = CodeSwitchPoint(2.0, "en", "hi", 1.7, 2.3)
    pt3 = CodeSwitchPoint(3.0, "hi", "en", 2.7, 3.3)
    feat2 = BoundaryFeatureVector(pt2, {k: 2.0 for k in FEATURE_NAMES})
    feat3 = BoundaryFeatureVector(pt3, {k: 3.0 for k in FEATURE_NAMES})
    vec_3 = aggregate_boundary_features([feat1, feat2, feat3], aggregation="mean_and_max")
    assert vec_3.shape == (10,)
    # Mean of [1, 2, 3] = 2.0, Max = 3.0
    assert np.allclose(vec_3[:5], 2.0)
    assert np.allclose(vec_3[5:], 3.0)

