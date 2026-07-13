"""
Train the voice check model.

The model looks at the numbers in audio_features.csv and learns to tell
whose voice a clip belongs to (james, kevin, or sheilla).

It also learns to say "I don't know this voice". When the model is not sure
enough about any known person, we treat the clip as an unknown/unauthorized
voice and block it. This is what protects the system from a stranger.

We measure how good the model is with three numbers:
  - Accuracy : how often it picks the right person
  - F1 score : a balanced score that is fair when classes are small
  - Loss     : how far off its confidence is (lower is better)

Run it:
    python src/audio/train_voiceprint.py

The trained model is saved in models/voiceprint_model.joblib so the app can use it.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
from sklearn.model_selection import train_test_split

FEATURES_FILE = Path("data/audio/features/audio_features.csv")
MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "voiceprint_model.joblib"

# label columns are not features
LABEL_COLS = ["file", "speaker", "phrase", "source"]

# if the model's best guess is less confident than this, we treat the
# voice as unknown and block it
CONFIDENCE_THRESHOLD = 0.5


def load_data():
    """Read the features file and split it into the numbers (X) and the answer (y)."""
    df = pd.read_csv(FEATURES_FILE)
    feature_cols = [c for c in df.columns if c not in LABEL_COLS]
    X = df[feature_cols].values
    y = df["speaker"].values
    return X, y, feature_cols


def main():
    X, y, feature_cols = load_data()
    print(f"Loaded {len(X)} clips from {len(set(y))} speakers: {sorted(set(y))}")

    # keep some clips aside to test the model on voices it trained on,
    # split evenly so every speaker shows up in both parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Training on {len(X_train)} clips, testing on {len(X_test)} clips.\n")

    # the model itself
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # test it
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    loss = log_loss(y_test, y_proba, labels=model.classes_)

    print("How the model did on the test clips:")
    print(f"  Accuracy : {accuracy:.3f}")
    print(f"  F1 score : {f1:.3f}")
    print(f"  Loss     : {loss:.3f}\n")

    print("Detailed report per speaker:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # retrain on all the clips so the saved model is as strong as possible
    final_model = RandomForestClassifier(n_estimators=200, random_state=42)
    final_model.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "feature_cols": feature_cols,
            "speakers": sorted(set(y)),
            "threshold": CONFIDENCE_THRESHOLD,
        },
        MODEL_FILE,
    )
    print(f"Saved the trained model to {MODEL_FILE}")


if __name__ == "__main__":
    main()
