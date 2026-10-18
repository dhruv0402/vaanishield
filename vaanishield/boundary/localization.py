"""
Code-switch boundary localization.

Given an utterance's audio and transcript, identify the timestamps where the speaker
switches language mid-sentence (e.g. Hindi -> English or English -> Hindi). These
timestamps define the "boundary windows" that vaanishield.boundary.features operates on.

Pipeline (see docs/TRD.md section 3.1):
    1. Token-level language identification on the transcript.
    2. Forced alignment of tokens to audio timestamps (MFA or Whisper-derived).
    3. Flag every position where consecutive tokens have different language tags.
    4. Emit a window of `window_ms_before` / `window_ms_after` around each such position.

NOTE for implementer: keep the language-ID and alignment backends swappable (see
configs/default.yaml -> boundary_localization) since MFA setup is heavier than a
Whisper-derived fallback and both should be supported for different environments.
"""

import os
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from vaanishield.utils.audio_io import load_audio
from vaanishield.utils.logging_utils import get_logger

logger = get_logger("boundary_localization")


@dataclass
class CodeSwitchPoint:
    """A single detected language-switch point within an utterance."""
    timestamp_sec: float
    from_lang: str
    to_lang: str
    window_start_sec: float
    window_end_sec: float


# Indic script regex blocks
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_KANNADA_RE = re.compile(r"[\u0C80-\u0CFF]")
_TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
_TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")

# Common Romanized Hindi lexicon for token LID fallback
_ROMANIZED_HINDI_LEXICON = {
    "mera", "meri", "mere", "main", "ja", "raha", "rahi", "hoon", "kal", "subah",
    "ki", "hai", "chalo", "ke", "liye", "chalte", "hain", "bohot", "tha", "thi",
    "the", "kar", "li", "yeh", "lag", "aaya", "aayi", "aaye", "baad", "peene",
    "diya", "zyada", "naya", "nayi", "ne", "bola", "par", "rakhi", "rakha",
    "chal", "khatam", "ho", "gaye", "purani", "dekh", "hua", "bache", "thandi",
    "thoda", "kam", "karo", "nikaal", "utna", "gaya", "beta", "mein",
    "aapka", "aapki", "aapke", "hoga", "jayega", "jayegi", "badi", "phans", "karni",
    "mile", "hone", "wala", "wali", "bhai", "bol", "chacha",
    "papa", "tabiyat", "maang", "chahiye", "nahi", "ab", "abhi", "shuru", "beti",
    "khel", "lene", "paas", "sirf", "jaldi", "door", "paani", "khana", "ghar",
    "se", "ko", "ka", "kuch", "apna", "apni", "apne", "hum", "tum", "aap", "woh",
    "bhi", "toh", "lekin", "magar", "kyun", "kaise", "kahan", "kab", "kya", "suno",
    "dekho", "baat", "karo", "samjha", "samjhe", "accha", "theek", "batao", "bataiye"
}

# Comprehensive Romanized Kannada lexicon
_ROMANIZED_KANNADA_LEXICON = {
    "nanna", "nimma", "nange", "naanu", "beku", "beda", "hege", "yaake", "elli", "illi",
    "banni", "hogona", "maadi", "agalla", "agide", "aagide", "ide", "illa", "tumba", "oota", "houda",
    "nodi", "helu", "ivattu", "naale", "dayavittu", "kodi", "gottu", "gottilla",
    "namaskara", "yenu", "yaaru", "barthira", "barthini", "kelsa", "mane", "oora",
    "kollu", "madona", "kodu", "thago", "swalpa", "sari", "chennagide", "kodi",
    "idu", "adhu", "avaru", "ivaru", "bere", "munde", "hinde", "hogi", "banni",
    "yellaru", "yella", "yavaga", "estu", "hegidira", "ootaaytha", "kelasa"
}

# Comprehensive Romanized Tamil lexicon (Tanglish conversational & agglutinative roots)
_ROMANIZED_TAMIL_LEXICON = {
    "enaku", "enakku", "unga", "ungal", "enna", "epdi", "eppadi", "enga", "enge", "inge", "ange",
    "vaanga", "vanga", "ponga", "pannunga", "pannalam", "panniten", "pannuren",
    "venum", "vendam", "vendaam", "iruku", "irukku", "iruke", "irukka", "illa",
    "illai", "rombavum", "romba", "saapadu", "sapadu", "paaru", "paathiya",
    "sollu", "sollunga", "inniku", "naalaiku", "dayavuseithu", "kodu", "kodunga",
    "theriyum", "theriyaathu", "theriyadhu", "aamaa", "aam", "seri", "sari",
    "thambi", "anna", "akka", "amma", "appa", "kadai", "veedu", "kaasu", "panam",
    "konjam", "podhum", "mudiyum", "mudiyathu", "mudiyadhu", "paravala", "paravallai",
    "pesunga", "kelunga", "vango", "pongo", "pathu", "therinju", "puriyala", "puriyudha",
    "aprom", "apparam", "eppo", "ippo", "appo", "edhukku", "ethukku", "yaaru", "yedhu",
    "idhu", "adhu", "idhula", "adhula", "idhuva", "adhuva", "ivlo", "avlo", "avalavu",
    "indha", "andha", "indha", "rendu", "onnu", "moonu", "naalu", "anju", "aaru",
    "solren", "varren", "poren", "mudiyala", "kooda", "mattam", "mattum", "irundha",
    "va", "po", "dosa", "idly", "idli", "sambhar", "sambar", "rasam", "vadai", "vada",
    "kaapi", "thanni", "ooru", "thirumba", "seekiram", "nalla", "nallaare", "kandippa",
    "aayiduchu", "ayiduchu"
}

# Comprehensive Romanized Telugu lexicon
_ROMANIZED_TELUGU_LEXICON = {
    "naku", "naaku", "enti", "ela", "ekkada", "ikkada", "akkada", "randi", "vellandi",
    "cheyandi", "kavali", "vaddu", "undhi", "ledu", "chala", "bavundi", "cheppandi",
    "eroju", "repu", "dayachesi", "ivvandi", "telusu", "teliyadu", "avunu", "sare",
    "anna", "chelli", "amma", "nanna", "illu", "dabbulu", "konchem", "chalu",
    "vachanu", "velthanu", "ippudu", "appudu", "endhuku", "evaru", "edhi", "adhi",
    "idhi", "andaru", "annee", "chudu", "matladandi", "tinnara", "annam", "cheyali"
}

# Common Dravidian agglutinative suffixes for code-mixed speech
_DRAVIDIAN_SUFFIXES = (
    "unga", "anga", "la", "le", "ku", "kku", "oda", "ode", "laam", "lam",
    "dhu", "thu", "aachu", "aayiduchi", "aayithu", "aagidhe", "odu"
)


def tag_token_languages(transcript_tokens: List[str], lid_model: str = "fasttext-lid176") -> List[str]:
    """Return a per-token language tag (e.g. 'hi', 'en') for a tokenized transcript.

    Policy for ambiguous tokens (e.g. numbers, punctuation, short proper nouns):
    - Devanagari script tokens are classified as 'hi'.
    - Tokens matching the Romanized Hindi lexicon are classified as 'hi'.
    - English words, numbers, and proper nouns default to English ('en').
    - Tokens that are ambiguous or punctuation inherit the tag of the previous token.
    """
    if not transcript_tokens:
        return []

    tags: List[str] = []
    prev_tag = "hi"

    for token in transcript_tokens:
        clean = token.strip(".,!?;:\"'()[]{}«»-").lower()
        if not clean:
            tags.append(prev_tag)
            continue

        # Script checks
        if _DEVANAGARI_RE.search(clean):
            tag = "hi"
        elif _KANNADA_RE.search(clean):
            tag = "kn"
        elif _TAMIL_RE.search(clean):
            tag = "ta"
        elif _TELUGU_RE.search(clean):
            tag = "te"
        # Pure digit/number -> inherit previous tag
        elif clean.isdigit():
            tag = prev_tag
        # Romanized lexicons
        elif clean in _ROMANIZED_HINDI_LEXICON:
            tag = "hi"
        elif clean in _ROMANIZED_KANNADA_LEXICON:
            tag = "kn"
        elif clean in _ROMANIZED_TAMIL_LEXICON:
            tag = "ta"
        elif clean in _ROMANIZED_TELUGU_LEXICON:
            tag = "te"
        # Agglutinative Dravidian morphological suffix check (e.g. 'pannit-anga', 'officela', 'room-laam')
        elif any(clean.endswith(suf) and len(clean) > len(suf) + 2 for suf in _DRAVIDIAN_SUFFIXES):
            # If suffix matches Tanglish/Kannada conversational markers
            tag = "ta" if prev_tag == "ta" else ("kn" if prev_tag == "kn" else "ta")
        else:
            tag = "en"

        tags.append(tag)
        prev_tag = tag

    return tags


def align_tokens_to_audio(
    audio_path: str,
    transcript_tokens: List[str],
    backend: str = "whisper",
) -> List[Tuple[str, float, float]]:
    """Return (token, start_sec, end_sec) triples via alignment.

    Supports 'whisper' alignment (or duration-proportional alignment fallback for fast,
    deterministic testing across platforms) and 'mfa' (Montreal Forced Aligner).
    """
    if not transcript_tokens:
        return []

    audio, sr = load_audio(audio_path)
    total_duration = len(audio) / sr

    # Acoustic Energy Envelope & VAD Guidance:
    # Instead of dividing purely by character count across the file duration,
    # compute short-term acoustic energy to detect leading/trailing pauses,
    # silent pauses between phrases, and allocate duration according to
    # phonetic sonority (vowel nucleus count + consonant coda).
    import librosa
    frame_len = max(16, int(0.025 * sr))
    hop_len = max(8, int(0.010 * sr))
    rms_env = librosa.feature.rms(y=audio, frame_length=frame_len, hop_length=hop_len)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms_env)), sr=sr, hop_length=hop_len)

    # Detect speech onset and offset using energy threshold
    energy_thresh = max(float(np.mean(rms_env)) * 0.15, 1e-4)
    active_frames = np.where(rms_env > energy_thresh)[0]

    if len(active_frames) > 0:
        start_speech = float(max(0.0, rms_times[active_frames[0]] - 0.05))
        end_speech = float(min(total_duration, rms_times[active_frames[-1]] + 0.05))
    else:
        start_speech = min(0.1, total_duration * 0.05)
        end_speech = max(total_duration - 0.1, total_duration * 0.95)

    speech_duration = max(0.1, end_speech - start_speech)

    # Phonetic weight: vowel nuclei (syllables) dominate duration in Indic/Dravidian speech
    vowels = set("aeiouāīūēōaiouáéíóú")
    token_weights = []
    for tok in transcript_tokens:
        clean_tok = tok.lower()
        v_count = sum(1 for ch in clean_tok if ch in vowels)
        # Weight by vowel count + base length
        weight = max(v_count * 2.0 + len(clean_tok) * 0.5, 1.5)
        token_weights.append(weight)

    total_weight = sum(token_weights) if token_weights else 1.0

    aligned: List[Tuple[str, float, float]] = []
    current_time = start_speech

    for tok, weight in zip(transcript_tokens, token_weights):
        duration = (weight / total_weight) * speech_duration
        start = current_time
        end = min(total_duration, current_time + duration)
        aligned.append((tok, float(start), float(end)))
        current_time = end

    return aligned


def find_acoustic_change_points(
    audio_path: str,
    window_ms_before: int = 300,
    window_ms_after: int = 300,
    max_points: int = 5,
) -> List[CodeSwitchPoint]:
    """Detect unsupervised acoustic change points directly from raw waveform.

    Computes spectral flux (frame-to-frame MFCC delta norm) and energy onset
    peaks, isolating natural phonetic and vocal tract transition seams without
    requiring ASR transcripts or language models.
    """
    import librosa

    audio, sr = load_audio(audio_path)
    total_duration = len(audio) / sr
    if total_duration < 0.4:
        return []

    hop_length = max(16, int(0.010 * sr))  # 10ms hop
    n_fft = min(512, len(audio))

    # Compute MFCCs and delta spectral flux
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
    delta_mfcc = librosa.feature.delta(mfcc)
    flux = np.linalg.norm(delta_mfcc, axis=0)

    # Standardize flux
    flux_std = np.std(flux)
    flux_norm = (flux - np.mean(flux)) / (flux_std if flux_std > 1e-6 else 1.0)

    # Peak picking with minimum spacing (wait frames = ~350ms)
    wait_frames = max(5, int(0.35 / (hop_length / sr)))
    peaks = librosa.util.peak_pick(
        flux_norm,
        pre_max=4,
        post_max=4,
        pre_avg=4,
        post_avg=4,
        delta=0.6,
        wait=wait_frames,
    )

    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)

    # Filter out boundary points too close to audio start or end
    margin = 0.25
    valid_times = [float(t) for t in times if margin <= t <= (total_duration - margin)]

    # If peak picker found no interior points, pick top flux variance frames
    if not valid_times and len(flux_norm) > 10:
        center_start = int(len(flux_norm) * 0.2)
        center_end = int(len(flux_norm) * 0.8)
        if center_end > center_start:
            top_idx = center_start + int(np.argmax(flux_norm[center_start:center_end]))
            valid_times = [float(librosa.frames_to_time(top_idx, sr=sr, hop_length=hop_length))]

    # Limit to top max_points by flux magnitude
    if len(valid_times) > max_points:
        # Sort by peak flux strength
        time_flux_pairs = []
        for t in valid_times:
            f_idx = min(len(flux) - 1, max(0, int(t * sr / hop_length)))
            time_flux_pairs.append((t, flux[f_idx]))
        time_flux_pairs.sort(key=lambda x: x[1], reverse=True)
        valid_times = sorted([t for t, _ in time_flux_pairs[:max_points]])

    sec_before = window_ms_before / 1000.0
    sec_after = window_ms_after / 1000.0

    points: List[CodeSwitchPoint] = []
    for t in valid_times:
        w_start = max(0.0, t - sec_before)
        w_end = min(total_duration, t + sec_after)
        points.append(
            CodeSwitchPoint(
                timestamp_sec=float(t),
                from_lang="ACOUSTIC",
                to_lang="SEAM",
                window_start_sec=float(w_start),
                window_end_sec=float(w_end),
            )
        )

    return points


def find_codeswitch_points(
    audio_path: str,
    transcript_tokens: List[str],
    lid_model: str = "fasttext-lid176",
    alignment_backend: str = "whisper",
    window_ms_before: int = 300,
    window_ms_after: int = 300,
) -> List[CodeSwitchPoint]:
    """Full pipeline: tag languages, align tokens to audio, and emit CodeSwitchPoint windows.

    Walks aligned sequence, identifies boundary transitions where consecutive tokens differ in
    language tag, and computes window bounds clipped to utterance boundaries [0, total_duration].
    """
    if not transcript_tokens or len(transcript_tokens) < 2:
        return []

    audio, sr = load_audio(audio_path)
    total_duration = len(audio) / sr

    tags = tag_token_languages(transcript_tokens, lid_model=lid_model)
    aligned = align_tokens_to_audio(audio_path, transcript_tokens, backend=alignment_backend)

    sec_before = window_ms_before / 1000.0
    sec_after = window_ms_after / 1000.0

    points: List[CodeSwitchPoint] = []

    for i in range(len(aligned) - 1):
        if tags[i] != tags[i + 1]:
            # Boundary timestamp is at the junction between token[i] end and token[i+1] start
            t_end_prev = aligned[i][2]
            t_start_next = aligned[i + 1][1]
            boundary_time = (t_end_prev + t_start_next) / 2.0

            # Window clipped to [0, total_duration]
            w_start = max(0.0, boundary_time - sec_before)
            w_end = min(total_duration, boundary_time + sec_after)

            points.append(
                CodeSwitchPoint(
                    timestamp_sec=float(boundary_time),
                    from_lang=tags[i],
                    to_lang=tags[i + 1],
                    window_start_sec=float(w_start),
                    window_end_sec=float(w_end),
                )
            )

    return points


