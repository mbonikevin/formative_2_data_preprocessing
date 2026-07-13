"""
train the voice check model

the model looks at the numbers in audio_features.csv and learns whose voice
a clip belongs to (james, kevin or sheilla)

it also needs to say "i dont know this voice" when a stranger shows up, so we
add a second check that measures how far a clip sits from the real voices we
have seen, a stranger sits much further away and gets blocked

we measure the model with three numbers:
  accuracy, how often it picks the right person
  f1 score, a balanced score that is fair when the classes are small
  loss, how far off its confidence is, lower is better

run it with:
    python src/audio/train_voiceprint.py

the trained model gets saved in models/voiceprint_model.joblib so the app can use it
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

features_file = Path("data/audio/features/audio_features.csv")
model_dir = Path("models")
model_file = model_dir / "voiceprint_model.joblib"

# these columns are labels, not features
label_cols = ["file", "speaker", "phrase", "source"]

# if the model's best guess is less sure than this, we treat the voice as unknown
confidence_threshold = 0.5


def load_data():
    # read the features file and split it into the numbers and the answer
    df = pd.read_csv(features_file)
    feature_cols = [c for c in df.columns if c not in label_cols]
    X = df[feature_cols].values
    y = df["speaker"].values
    return X, y, feature_cols


def main():
    X, y, feature_cols = load_data()
    print(f"loaded {len(X)} clips from {len(set(y))} speakers: {sorted(set(y))}")

    # keep some clips aside to test on, split evenly so every speaker shows up in both parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"training on {len(X_train)} clips, testing on {len(X_test)} clips\n")

    # the model itself
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # test it
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    loss = log_loss(y_test, y_proba, labels=model.classes_)

    print("how the model did on the test clips:")
    print(f"accuracy is {accuracy:.3f}")
    print(f"f1 score is {f1:.3f}")
    print(f"loss is {loss:.3f}\n")

    print("report per speaker:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # retrain on all the clips so the saved model is as strong as we can make it
    final_model = RandomForestClassifier(n_estimators=200, random_state=42)
    final_model.fit(X, y)

    # second check, learn what a real voice looks like so we can spot strangers.
    # we put all the features on the same scale, then for each real clip measure
    # how far its closest neighbour is, a stranger will sit further out
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    neighbours = NearestNeighbors(n_neighbors=2).fit(X_scaled)
    dists, _ = neighbours.kneighbors(X_scaled)
    nearest = dists[:, 1]  # column 0 is the point itself so we use column 1

    # anything further than this from a real voice is treated as a stranger,
    # we take the biggest gap between real clips and leave a bit of room
    distance_cutoff = float(nearest.max() * 1.5)
    print(f"\nstranger distance cutoff set to {distance_cutoff:.2f}")

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "feature_cols": feature_cols,
            "speakers": sorted(set(y)),
            "threshold": confidence_threshold,
            "scaler": scaler,
            "neighbours": neighbours,
            "distance_cutoff": distance_cutoff,
        },
        model_file,
    )
    print(f"saved the trained model to {model_file}")


if __name__ == "__main__":
    main()
