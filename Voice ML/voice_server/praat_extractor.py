# praat_extractor.py

import os
import re

import parselmouth
from parselmouth import PraatError

# Import the “call” helper directly
from parselmouth.praat import call

from feature_schema import PRAAT_PERCENT_FEATURES

# Regexes for everything the GUI “Voice report…” spits out:
# praat_extractor.py (just the VOICE_REPORT_REGEXES part)
VOICE_REPORT_REGEXES = {
    "MDVP:Fo(Hz)":        r"Mean pitch:\s*([\d\.]+)\s*Hz",
    "MDVP:Fhi(Hz)":       r"Maximum pitch:\s*([\d\.]+)\s*Hz",
    "MDVP:Flo(Hz)":       r"Minimum pitch:\s*([\d\.]+)\s*Hz",
    "MDVP:Stddev(Hz)":    r"Standard deviation:\s*([\d\.]+)\s*Hz",
    "MDVP:Median(Hz)":    r"Median pitch:\s*([\d\.]+)\s*Hz",
    "MDVP:Jitter(%)":     r"Jitter \(local\):\s*([\d\.]+)%",
    "MDVP:Jitter(Abs)":   r"Jitter \(local, absolute\):\s*([\d\.E\-]+)\s*seconds",
    "MDVP:RAP":           r"Jitter \(rap\):\s*([\d\.]+)%",
    "MDVP:PPQ":           r"Jitter \(ppq5\):\s*([\d\.]+)%",
    "Jitter:DDP":         r"Jitter \(ddp\):\s*([\d\.]+)%",
    "MDVP:Shimmer":       r"Shimmer \(local\):\s*([\d\.]+)%",
    "MDVP:Shimmer(dB)":   r"Shimmer \(local, dB\):\s*([\d\.]+)\s*dB",
    "Shimmer:APQ3":       r"Shimmer \(apq3\):\s*([\d\.]+)%",
    "Shimmer:APQ5":       r"Shimmer \(apq5\):\s*([\d\.]+)%",
    "MDVP:APQ":           r"Shimmer \(apq11\):\s*([\d\.]+)%",  # if your model used MDVP:APQ for apq11
    "Shimmer:DDA":        r"Shimmer \(dda\):\s*([\d\.]+)%",
    "NHR":                r"Mean noise-to-harmonics ratio:\s*([\d\.]+)",
    "HNR":                r"Mean harmonics-to-noise ratio:\s*([\d\.]+)\s*dB",
}


def parse_voice_report(report: str, source_name: str = "recording") -> dict:
    """Parse a Praat report, failing closed if its format has changed."""
    feats = {"file": source_name}
    missing = []
    for name, rx in VOICE_REPORT_REGEXES.items():
        match = re.search(rx, report)
        if match is None:
            missing.append(name)
            continue
        value = float(match.group(1))
        if name in PRAAT_PERCENT_FEATURES:
            value /= 100.0
        feats[name] = value

    if missing:
        raise RuntimeError(
            "Praat Voice report did not contain required fields: "
            + ", ".join(missing)
        )
    return feats


def extract_praat_voice_report(wav_path: str) -> dict:
    snd = parselmouth.Sound(wav_path)

    # 1) EXACT same Pitch as GUI Voice report (cross-correlation)
    pitch = snd.to_pitch_cc(
        time_step=0.01,          # 10 ms
        pitch_floor=75,          # Hz
        pitch_ceiling=600,       # Hz
        silence_threshold=0.03,  # default Voice report
        voicing_threshold=0.45,  # default Voice report
        octave_cost=0.01,
        octave_jump_cost=0.35,
        voiced_unvoiced_cost=0.14,
        very_accurate=False      # matches GUI
    )

    # 2) Build PointProcess from SOUND (not from pitch!)
    pulses = call(snd, "To PointProcess (periodic, cc)", 75, 600)

    # 3) Now run Voice report… on all three objects
    try:
        report = call(
            [snd, pitch, pulses], "Voice report...",
            0.0, 0.0,    # start/end = whole sound
            75, 600,    # floor, ceiling
            1.3, 1.6,   # jitter / shimmer windows
            0.03, 0.45  # silence / voicing thresh.
        )
    except PraatError as e:
        raise RuntimeError(f"Praat Voice report failed: {e!r}")

    # 4) Parse every required field. Missing values are an extraction failure,
    # never zero: a fabricated zero could create a confident-looking score.
    return parse_voice_report(report, os.path.basename(wav_path))
