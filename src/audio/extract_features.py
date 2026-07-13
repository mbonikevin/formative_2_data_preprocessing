"""
Turn every recording into a row of numbers and save them in audio_features.csv.

For each clip we measure:
  - MFCCs        : 13 numbers that describe the shape of the voice
  - spectral rolloff : where most of the sound energy sits (high or low pitch)
  - energy (RMS) : how strong the sound is on average
  - zero crossing rate : how noisy or clear the sound is

We use both the original recordings and the augmented copies, so the model
has more examples to learn from.

Each row also gets two labels taken from the file name:
  - speaker : who is talking (james, kevin, sheilla)
  - phrase  : which phrase they said (approve or confirm)

Run it:
    python src/audio/extract_features.py
"""

from pathlib import Path

import librosa
import numpy as np
import pandas as pd

RAW_DIR = Path("data/audio/raw")
AUG_DIR = Path("data/audio/augmented")
FEATURES_DIR = Path("data/audio/features")
OUT_FILE = FEATURES_DIR / "audio_features.csv"

SAMPLE_RATE = 16000
N_MFCC = 13


def read_labels(name):
    """Pull the speaker and phrase out of a file name like 'kevin_approve_pitch'."""
    parts = name.split("_")
    speaker = parts[0]
    phrase = parts[1]
    return speaker, phrase


def clip_features(path):
    """Measure all the numbers for one recording and return them as a dict."""
    samples, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    row = {}

    # MFCCs: take the average of each of the 13 numbers over the whole clip
    mfcc = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        row[f"mfcc_{i + 1}"] = float(np.mean(mfcc[i]))

    # spectral rolloff: average pitch level where the energy is concentrated
    rolloff = librosa.feature.spectral_rolloff(y=samples, sr=sr)
    row["spectral_rolloff"] = float(np.mean(rolloff))

    # energy: how loud the clip is on average
    rms = librosa.feature.rms(y=samples)
    row["energy"] = float(np.mean(rms))

    # zero crossing rate: how noisy the sound is
    zcr = librosa.feature.zero_crossing_rate(y=samples)
    row["zero_crossing_rate"] = float(np.mean(zcr))

    return row


def main():
    clips = sorted(RAW_DIR.glob("*.wav")) + sorted(AUG_DIR.glob("*.wav"))
    if not clips:
        print("No wav files found. Run the earlier steps first.")
        return

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in clips:
        speaker, phrase = read_labels(path.stem)
        row = clip_features(path)
        row["speaker"] = speaker
        row["phrase"] = phrase
        row["source"] = "raw" if path.parent == RAW_DIR else "augmented"
        row["file"] = path.name
        rows.append(row)
        print(f"  measured {path.name}")

    df = pd.DataFrame(rows)

    # put the label columns first so the file is easy to read
    label_cols = ["file", "speaker", "phrase", "source"]
    feature_cols = [c for c in df.columns if c not in label_cols]
    df = df[label_cols + feature_cols]

    df.to_csv(OUT_FILE, index=False)
    print(f"\nDone. Saved {len(df)} rows and {len(feature_cols)} features to {OUT_FILE}")


if __name__ == "__main__":
    main()
