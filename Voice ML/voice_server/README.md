# Sustained-vowel research service

This FastAPI service compares reproducible Praat measurements from a sustained
`Aaaa` recording with the Oxford/UCI Parkinson voice dataset. Its output is a
research comparison score, not a probability of Parkinson's disease or a
diagnosis.

The model uses 16 Praat features whose units can be reproduced at inference.
Evaluation leaves one speaker out at a time so recordings from one person do
not leak into both training and validation. Recordings that fail capture checks
or fall outside the reference feature range receive an inconclusive response
with no score. The current metadata also sets `validated_for_clinical_use` to
`false`, so even an in-range recording cannot receive a patient-facing score.
Scoring should only be enabled after independent validation on representative
phone recordings.

## Retrain

From `voice_server`, using the repository virtual environment:

```powershell
& '..\venv\Scripts\python.exe' train_model.py
```

This regenerates `voice_gbm.joblib` and `voice_model_metadata.json` together.
Do not deploy one without the other.

## Test and run locally

```powershell
& '..\venv\Scripts\python.exe' -m unittest -v test_voice_pipeline.py
& '..\venv\Scripts\python.exe' -m uvicorn main:app --reload
```

Use `POST /predict` with a WAV file. `GET /model-info` exposes the exact feature
contract, validation results, threshold, and limitations. A usable prediction
has `result_status: "ok"`; all other statuses have a null `screening_score`.

The Flutter client currently calls the hosted Render service, so these files
must be redeployed there before the mobile app can use this model version.
