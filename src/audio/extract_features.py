"""
turn every recording into a row of numbers and save them in audio_features.csv

for each clip we measure a few things:
  mfccs, 13 numbers that describe the shape of the voice
  spectral rolloff, where most of the sound energy sits
  energy (rms), how strong the sound is on average
  zero crossing rate, how noisy or clear the sound is

we use both the original recordings and the augmented copies so the model
has more examples to learn from

each row also gets two labels taken from the file name, the speaker (who is
talking) and the phrase (approve or confirm)

run it with:
    python src/audio/extract_features.py
"""

from pathlib import Path

import librosa
import numpy as np
import pandas as pd

raw_dir = Path("data/audio/raw")
aug_dir = Path("data/audio/augmented")
features_dir = Path("data/audio/features")
out_file = features_dir / "audio_features.csv"

sample_rate = 16000
n_mfcc = 13


def read_labels(name):
    # pull the speaker and phrase out of a name like kevin_approve_pitch
    parts = name.split("_")
    speaker = parts[0]
    phrase = parts[1]
    return speaker, phrase


def clip_features(path):
    # measure all the numbers for one recording and give back a dict
    samples, sr = librosa.load(path, sr=sample_rate, mono=True)

    row = {}

    # mfccs, take the average of each of the 13 numbers over the whole clip
    mfcc = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=n_mfcc)
    for i in range(n_mfcc):
        row[f"mfcc_{i + 1}"] = float(np.mean(mfcc[i]))

    # spectral rolloff, roughly the pitch level where the energy sits
    rolloff = librosa.feature.spectral_rolloff(y=samples, sr=sr)
    row["spectral_rolloff"] = float(np.mean(rolloff))

    # energy, how loud the clip is on average
    rms = librosa.feature.rms(y=samples)
    row["energy"] = float(np.mean(rms))

    # zero crossing rate, how noisy the sound is
    zcr = librosa.feature.zero_crossing_rate(y=samples)
    row["zero_crossing_rate"] = float(np.mean(zcr))

    return row


def main():
    clips = sorted(raw_dir.glob("*.wav")) + sorted(aug_dir.glob("*.wav"))
    if not clips:
        print("no wav files found, run the earlier steps first")
        return

    features_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in clips:
        speaker, phrase = read_labels(path.stem)
        row = clip_features(path)
        row["speaker"] = speaker
        row["phrase"] = phrase
        row["source"] = "raw" if path.parent == raw_dir else "augmented"
        row["file"] = path.name
        rows.append(row)
        print(f"  measured {path.name}")

    df = pd.DataFrame(rows)

    # put the label columns first so the file is easier to read
    label_cols = ["file", "speaker", "phrase", "source"]
    feature_cols = [c for c in df.columns if c not in label_cols]
    df = df[label_cols + feature_cols]

    df.to_csv(out_file, index=False)
    print(f"\ndone, saved {len(df)} rows and {len(feature_cols)} features to {out_file}")


if __name__ == "__main__":
    main()
