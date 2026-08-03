from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from feature_extraction import (
    extract_features_from_wav,
    extract_recording_quality,
)
from feature_schema import (
    FEATURE_ORDER,
    MAX_CLIPPED_FRACTION,
    MAX_OUTLIER_FEATURES,
    MIN_DURATION_SECONDS,
    MIN_VOICED_FRACTION,
    MODEL_VERSION,
    OUTLIER_Z,
    SEVERE_OUTLIER_Z,
)


HERE = Path(__file__).resolve().parent
MODEL = joblib.load(HERE / "voice_gbm.joblib")
METADATA = json.loads(
    (HERE / "voice_model_metadata.json").read_text(encoding="utf-8")
)

if METADATA["model_version"] != MODEL_VERSION:
    raise RuntimeError("Voice model and feature schema versions do not match")
if METADATA["feature_order"] != FEATURE_ORDER:
    raise RuntimeError("Voice model and feature order do not match")

TRAIN_MEAN = METADATA["training"]["feature_mean"]
TRAIN_STD = METADATA["training"]["feature_std"]
DECISION_THRESHOLD = METADATA["validation"]["decision_threshold"]
VALIDATED_FOR_CLINICAL_USE = bool(
    METADATA.get("validated_for_clinical_use", False)
)

app = FastAPI(title="Sustained-vowel research comparison service")


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _input_assessment(
    features: dict[str, float], quality: dict[str, float]
) -> dict:
    warnings: list[str] = []

    if quality["duration_seconds"] < MIN_DURATION_SECONDS:
        warnings.append("recording_too_short")
    if quality["voiced_fraction"] < MIN_VOICED_FRACTION:
        warnings.append("insufficient_sustained_voice")
    if quality["clipped_fraction"] > MAX_CLIPPED_FRACTION:
        warnings.append("audio_clipping_detected")

    z_scores: dict[str, float] = {}
    for name in FEATURE_ORDER:
        value = features[name]
        std = TRAIN_STD[name]
        z_scores[name] = 0.0 if std <= 0 else (value - TRAIN_MEAN[name]) / std

    outliers = [name for name, z in z_scores.items() if abs(z) >= OUTLIER_Z]
    severe = [
        name for name, z in z_scores.items() if abs(z) >= SEVERE_OUTLIER_Z
    ]

    if warnings:
        status = "capture_quality_failed"
    elif severe or len(outliers) > MAX_OUTLIER_FEATURES:
        status = "outside_reference_range"
        warnings.append("recording_does_not_match_training_distribution")
    else:
        status = "ok"

    return {
        "status": status,
        "usable": status == "ok",
        "warnings": warnings,
        "outlier_features": outliers,
        "severe_outlier_features": severe,
        "z_scores": z_scores,
        **quality,
    }


def _prediction_assessment(
    features: dict[str, float], quality: dict[str, float]
) -> dict:
    """Combine capture checks with the model's deployment authorization."""
    assessment = _input_assessment(features, quality)
    if assessment["usable"] and not VALIDATED_FOR_CLINICAL_USE:
        assessment["status"] = "research_model_not_validated"
        assessment["usable"] = False
        assessment["warnings"].append(
            "model_not_validated_for_phone_recordings"
        )
    return assessment


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename or "voice.wav").suffix or ".wav"
    temp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(await file.read())
            temp_path = temp.name

        features = extract_features_from_wav(temp_path)
        quality = extract_recording_quality(temp_path)
        if any(not _finite(value) for value in features.values()):
            raise ValueError("Feature extraction produced a non-finite value")

        assessment = _prediction_assessment(features, quality)
        score: float | None = None
        if assessment["usable"]:
            row = pd.DataFrame(
                [[features[name] for name in FEATURE_ORDER]],
                columns=FEATURE_ORDER,
            )
            score = float(MODEL.predict_proba(row)[0, 1])

        return JSONResponse(
            {
                "model_version": MODEL_VERSION,
                "validated_for_clinical_use": VALIDATED_FOR_CLINICAL_USE,
                "scoring_enabled": VALIDATED_FOR_CLINICAL_USE,
                "result_status": assessment["status"],
                "is_usable": assessment["usable"],
                "screening_score": score,
                "decision_threshold": DECISION_THRESHOLD,
                "above_research_threshold": (
                    score >= DECISION_THRESHOLD if score is not None else None
                ),
                "interpretation": "research_voice_pattern_score_not_diagnosis",
                "input_quality": assessment,
                "features": features,
            }
        )
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/model-info")
async def model_info():
    """Expose the feature contract and honest validation summary."""
    return {
        **METADATA,
        "scoring_enabled": VALIDATED_FOR_CLINICAL_USE,
        "scoring_policy": "requires_independent_phone_validation",
    }


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Voice pattern research service</title></head>
<body>
  <h1>Sustained-vowel research comparison</h1>
  <p>This service does not diagnose Parkinson's disease.</p>
  <form id="uploadForm">
    <input type="file" id="wavFile" accept=".wav" required>
    <button type="submit">Analyse recording</button>
  </form>
  <pre id="result"></pre>
  <script>
    document.getElementById('uploadForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const data = new FormData();
      data.append('file', document.getElementById('wavFile').files[0]);
      const response = await fetch('/predict', {method: 'POST', body: data});
      document.getElementById('result').textContent =
        JSON.stringify(await response.json(), null, 2);
    });
  </script>
</body>
</html>
"""
