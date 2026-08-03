"""Feature extraction for the sustained-vowel screening model.

The previous implementation invented approximations for RPDE, DFA, spread1,
spread2 and PPE.  Those values had different units from the Oxford/UCI
columns and pushed ordinary phone recordings dozens of standard deviations
outside the training data.  Inference now uses only directly reproducible
Praat measurements and returns recording-quality metadata separately.
"""

from __future__ import annotations

import numpy as np
import parselmouth
from parselmouth.praat import call

from feature_schema import FEATURE_ORDER
from praat_extractor import extract_praat_voice_report


def extract_features_from_wav(wav_path: str) -> dict[str, float]:
    """Return exactly the versioned model features, in their training units."""
    extracted = extract_praat_voice_report(wav_path)
    return {name: float(extracted[name]) for name in FEATURE_ORDER}


def extract_recording_quality(wav_path: str) -> dict[str, float]:
    """Measure capture usability without interpreting the patient's health."""
    sound = parselmouth.Sound(wav_path)
    if sound.get_number_of_channels() > 1:
        sound = call(sound, "Extract one channel", 1)

    samples = np.asarray(sound.values, dtype=float).reshape(-1)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    clipped = (
        float(np.mean(np.abs(samples) >= 0.999)) if samples.size else 0.0
    )

    pitch = sound.to_pitch_cc(
        time_step=0.01,
        pitch_floor=75,
        pitch_ceiling=600,
        silence_threshold=0.03,
        voicing_threshold=0.45,
        octave_cost=0.01,
        octave_jump_cost=0.35,
        voiced_unvoiced_cost=0.14,
        very_accurate=False,
    )
    frequencies = pitch.selected_array["frequency"]
    voiced_fraction = (
        float(np.mean(frequencies > 0)) if frequencies.size else 0.0
    )

    return {
        "duration_seconds": float(sound.duration),
        "voiced_fraction": voiced_fraction,
        "peak_amplitude": peak,
        "clipped_fraction": clipped,
    }
