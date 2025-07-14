# feature_extraction.py

import numpy as np
import parselmouth
from parselmouth.praat import call
import antropy

from praat_extractor import extract_praat_voice_report
import os
os.environ["NUMBA_DISABLE_JIT"] = "1"


def _rpde(periods, emb_dim=4, tol=0.12):
    """Recurrence Period Density Entropy"""
    # require enough points
    if len(periods) < emb_dim * 2:
        return 0.0
    # build embedding
    N = len(periods)
    emb = np.stack([periods[i:N-(emb_dim-1)+i] for i in range(emb_dim)], axis=1)
    rec_times = []
    for i in range(len(emb)):
        # distances from point i
        dists = np.linalg.norm(emb - emb[i], axis=1)
        # find first return beyond i
        idx = np.where((dists < tol) & (np.arange(len(dists)) > i))[0]
        if idx.size:
            rec_times.append(idx[0] - i)
    if not rec_times:
        return 0.0
    counts, _ = np.histogram(rec_times, bins=np.arange(1, max(rec_times) + 2), density=True)
    # if only one bin, entropy zero
    if len(counts) <= 1:
        return 0.0
    p = counts[counts > 0]
    H = -np.sum(p * np.log(p))
    return H / np.log(len(counts))


def _dfa(signal, Ls=np.arange(50, 201, 10)):
    """Detrended Fluctuation Analysis"""
    N = len(signal)
    if N < max(Ls):
        return 0.0
    # integrate
    y = np.cumsum(signal - np.mean(signal))
    Fs = []
    for L in Ls:
        nseg = N // L
        segs = y[:nseg*L].reshape(nseg, L)
        flucs = []
        x = np.arange(L)
        xx = x - x.mean()
        denom = np.sum(xx * xx)
        for seg in segs:
            a = np.dot(xx, seg - seg.mean()) / denom
            trend = a*x + (seg.mean() - a*x.mean())
            flucs.append(np.sqrt(np.mean((seg - trend)**2)))
        Fs.append(np.mean(flucs))
    logL = np.log(Ls)
    logF = np.log(Fs)
    alpha = np.cov(logL, logF, bias=True)[0,1] / np.var(logL)
    # normalize to (0,1)
    return 1 / (1 + np.exp(-alpha))


def extract_features_from_wav(wav_path: str) -> dict:
    # 1) base Praat report fields
    feats = extract_praat_voice_report(wav_path)

    # 2) load sound
    snd = parselmouth.Sound(wav_path)
    if snd.get_number_of_channels() > 1:
        snd = call(snd, "Extract one channel", 1)

    # 3) pitch via cross-correlation
    pitch = snd.to_pitch_cc(
        time_step=0.01,
        pitch_floor=75,
        pitch_ceiling=600,
        silence_threshold=0.03,
        voicing_threshold=0.45,
        octave_cost=0.01,
        octave_jump_cost=0.35,
        voiced_unvoiced_cost=0.14,
        very_accurate=False
    )
    freqs = pitch.selected_array['frequency']
    voiced = freqs[freqs > 0]

    # 4) nonlinear measures
    if voiced.size:
        periods = 1.0 / voiced
        feats['RPDE']    = _rpde(periods)
        feats['DFA']     = _dfa(voiced)
        diffs = voiced[2:] - voiced[:-2] if voiced.size > 2 else np.array([])
        feats['spread1'] = float(np.mean(diffs)) if diffs.size else 0.0
        feats['spread2'] = float(np.var(diffs))  if diffs.size else 0.0
        feats['PPE']     = float(antropy.sample_entropy(voiced)) if voiced.size > 3 else 0.0
    else:
        for k in ('RPDE','DFA','spread1','spread2','PPE'):
            feats[k] = 0.0

    return feats
