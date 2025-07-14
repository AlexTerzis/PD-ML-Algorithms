# main.py

import os
import tempfile
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from feature_extraction import extract_features_from_wav

app = FastAPI()

# Load scaler & model
scaler = joblib.load("voice_scaler.joblib")
model  = joblib.load("voice_gbm.joblib")

# The exact feature order your model expects:
FEATURE_ORDER = [
    'MDVP:Fo(Hz)',
    'MDVP:Fhi(Hz)',
    'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)',
    'MDVP:Jitter(Abs)',
    'MDVP:RAP',
    'MDVP:PPQ',
    'Jitter:DDP',
    'MDVP:Shimmer',
    'MDVP:Shimmer(dB)',
    'Shimmer:APQ3',
    'Shimmer:APQ5',
    'MDVP:APQ',
    'Shimmer:DDA',
    'NHR',
    'HNR',
    'RPDE',
    'DFA',
    'spread1',
    'spread2',
    'PPE'
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1) Save the uploaded WAV to a temp file
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # 2) Extract features
    feats = extract_features_from_wav(tmp_path)
    os.remove(tmp_path)

    # 3) Build DataFrame in the correct order
    row = [feats.get(f, 0.0) for f in FEATURE_ORDER]
    df  = pd.DataFrame([row], columns=FEATURE_ORDER)

    # 4) Scale & predict
    # Debug prints (in your console) to confirm alignment:
    print(f"⏺️ Scaler was trained on {scaler.n_features_in_} features:")
    print("   ", list(scaler.feature_names_in_))

    Xs = scaler.transform(df)  # pass DataFrame so feature names match

    cls  = int(model.predict(Xs)[0])
    prob = float(model.predict_proba(Xs)[0, 1])

    # 5) Return JSON response
    return JSONResponse({
        "predicted_class": cls,
        "probability_PD": prob,
        "features": { f: float(feats[f]) for f in FEATURE_ORDER }
    })


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Voice PD Predictor</title>
</head>
<body>
<h1>Upload a WAV for Parkinson’s prediction</h1>
<form id="uploadForm">
    <input type="file" id="wavFile" accept=".wav" required />
    <button type="submit">Send to Server</button>
</form>
<pre id="result" style="background:#f0f0f0; padding:1em;"></pre>

<script>
    const form = document.getElementById("uploadForm");
    const resultEl = document.getElementById("result");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const fileInput = document.getElementById("wavFile");
        if (!fileInput.files.length) {
            alert("Please select a .wav file first");
            return;
        }
        const formData = new FormData();
        formData.append("file", fileInput.files[0]);

        resultEl.textContent = "⏳ Uploading and predicting…";

        try {
            const resp = await fetch("/predict", {
                method: "POST",
                body: formData
            });
            if (!resp.ok) {
                throw new Error(`Server returned ${resp.status}`);
            }
            const json = await resp.json();
            resultEl.textContent = JSON.stringify(json, null, 2);
        } catch (err) {
            resultEl.textContent = "❌ Error: " + err;
        }
    });
</script>
</body>
</html>
"""
