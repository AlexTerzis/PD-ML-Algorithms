"""Versioned feature contract shared by training and inference.

Only features that Praat can reproduce directly from a sustained vowel are
kept here.  The UCI nonlinear features (RPDE, DFA, spread1, spread2 and PPE)
are intentionally excluded: the public dataset contains their final values,
but not the original recordings and implementation needed to reproduce them
faithfully for a new phone recording.
"""

MODEL_VERSION = "praat16_subject_grouped_v2"

FEATURE_ORDER = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
]

# Praat's Voice report prints these values with a percent sign.  The UCI CSV
# stores the corresponding quantities as fractions (0.521% -> 0.00521).
PRAAT_PERCENT_FEATURES = {
    "MDVP:Jitter(%)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
}

# Input-domain guardrails.  A model trained on 31 people must abstain when a
# new recording is unlike the reference data instead of extrapolating an
# extreme, confidence-looking score.
OUTLIER_Z = 3.0
SEVERE_OUTLIER_Z = 6.0
MAX_OUTLIER_FEATURES = 2

MIN_DURATION_SECONDS = 4.0
MIN_VOICED_FRACTION = 0.55
MAX_CLIPPED_FRACTION = 0.01
