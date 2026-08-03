"""Train the versioned sustained-vowel research classifier.

Evaluation is performed by leaving one person out at a time.  Multiple
recordings from the same speaker therefore never appear in both train and
test, avoiding the identity leakage in the former row-level split.

The resulting score is not a diagnostic probability.  The server exposes it
as a research comparison score and refuses to score out-of-distribution
recordings.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from feature_schema import FEATURE_ORDER, MODEL_VERSION


HERE = Path(__file__).resolve().parent
DATASET = HERE.parent / "Data" / "parkinsons.data"
MODEL_PATH = HERE / "voice_gbm.joblib"
METADATA_PATH = HERE / "voice_model_metadata.json"


def subject_id(recording_name: str) -> str:
    """Remove only the final recording suffix, retaining the subject id."""
    return re.sub(r"_\d+$", "", recording_name)


def build_model():
    # Strong regularisation and balanced classes are preferable to a flexible
    # boosted tree on only a few dozen independent people.
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.25,
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        ),
    )


def main() -> None:
    data = pd.read_csv(DATASET)
    x = data[FEATURE_ORDER]
    y = data["status"].astype(int)
    groups = data["name"].map(subject_id)

    probabilities = cross_val_predict(
        build_model(),
        x,
        y,
        groups=groups,
        cv=LeaveOneGroupOut(),
        method="predict_proba",
    )[:, 1]
    predicted = (probabilities >= 0.5).astype(int)

    threshold_candidates = np.linspace(0.1, 0.9, 161)
    decision_threshold = float(
        max(
            threshold_candidates,
            key=lambda threshold: balanced_accuracy_score(
                y, probabilities >= threshold
            ),
        )
    )
    threshold_predicted = (probabilities >= decision_threshold).astype(int)

    # Also report one result per speaker id. The deployed endpoint receives a
    # single recording, so recording-level metrics remain above, but repeated
    # samples must not make the dataset look larger than it is.
    speaker_validation = pd.DataFrame(
        {"speaker_id": groups, "status": y, "score": probabilities}
    ).groupby("speaker_id", as_index=False).agg(
        status=("status", "first"), score=("score", "mean")
    )
    speaker_y = speaker_validation["status"]
    speaker_scores = speaker_validation["score"]
    speaker_predicted = speaker_scores >= decision_threshold

    validation = {
        "method": "leave_one_subject_out",
        "unit": "recording",
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "confusion_matrix": confusion_matrix(y, predicted).tolist(),
        "decision_threshold": decision_threshold,
        "balanced_accuracy_at_decision_threshold": float(
            balanced_accuracy_score(y, threshold_predicted)
        ),
        "confusion_matrix_at_decision_threshold": confusion_matrix(
            y, threshold_predicted
        ).tolist(),
        "speaker_id_level": {
            "speaker_ids": int(len(speaker_validation)),
            "aggregation": "mean_out_of_fold_score_per_speaker_id",
            "accuracy_at_decision_threshold": float(
                accuracy_score(speaker_y, speaker_predicted)
            ),
            "balanced_accuracy_at_decision_threshold": float(
                balanced_accuracy_score(speaker_y, speaker_predicted)
            ),
            "roc_auc": float(roc_auc_score(speaker_y, speaker_scores)),
            "confusion_matrix_at_decision_threshold": confusion_matrix(
                speaker_y, speaker_predicted
            ).tolist(),
        },
    }

    model = build_model()
    model.fit(x, y)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "model_version": MODEL_VERSION,
        "validated_for_clinical_use": False,
        "feature_order": FEATURE_ORDER,
        "training": {
            "dataset": "UCI Parkinsons (id=174)",
            "recordings": int(len(data)),
            "subject_ids": int(groups.nunique()),
            "pd_recordings": int(y.sum()),
            "healthy_recordings": int((1 - y).sum()),
            "feature_mean": {
                name: float(x[name].mean()) for name in FEATURE_ORDER
            },
            "feature_std": {
                name: float(x[name].std(ddof=0)) for name in FEATURE_ORDER
            },
        },
        "validation": validation,
        "limitations": [
            "Research comparison score; not a probability or diagnosis.",
            "Small reference cohort with repeated recordings per speaker.",
            "External smartphone validation has not been performed.",
        ],
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(validation, indent=2))
    print(f"Saved {MODEL_PATH.name} and {METADATA_PATH.name}")


if __name__ == "__main__":
    main()
