"""
Make extra copies of each recording so the model has more to learn from.

For every recording we create three new versions:
  1. pitch shift   - makes the voice a little higher
  2. time stretch  - makes the speech a little faster
  3. added noise   - mixes in some quiet background noise

Run it:
    python src/audio/augment.py

The new files are saved in data/audio/augmented/ with a tag added to the name,
for example kevin_approve_pitch.wav, kevin_approve_stretch.wav, kevin_approve_noise.wav
"""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

RAW_DIR = Path("data/audio/raw")
AUG_DIR = Path("data/audio/augmented")

SAMPLE_RATE = 16000


def pitch_shift(samples, sr):
    """Raise the voice by 2 half steps so it sounds a bit higher."""
    return librosa.effects.pitch_shift(samples, sr=sr, n_steps=2)


def time_stretch(samples):
    """Speed the speech up to 1.15x (a value above 1 makes it faster)."""
    return librosa.effects.time_stretch(samples, rate=1.15)


def add_noise(samples):
    """Mix in a small amount of random background noise."""
    noise = np.random.normal(0, 0.005, samples.shape)
    return samples + noise


def augment_clip(path):
    """Create the three versions for one recording and save them."""
    samples, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    name = path.stem

    versions = {
        "pitch": pitch_shift(samples, sr),
        "stretch": time_stretch(samples),
        "noise": add_noise(samples),
    }

    saved = []
    for tag, new_samples in versions.items():
        out = AUG_DIR / f"{name}_{tag}.wav"
        sf.write(out, new_samples, sr)
        saved.append(out.name)
    return saved


def main():
    clips = sorted(RAW_DIR.glob("*.wav"))
    if not clips:
        print(f"No wav files found in {RAW_DIR}. Add the recordings first.")
        return

    AUG_DIR.mkdir(parents=True, exist_ok=True)

    # keep the results the same every time we run it
    np.random.seed(42)

    print(f"Found {len(clips)} recordings. Making 3 versions of each.")
    total = 0
    for path in clips:
        saved = augment_clip(path)
        for name in saved:
            print(f"  saved {name}")
        total += len(saved)

    print(f"\nDone. Made {total} new files in data/audio/augmented/")


if __name__ == "__main__":
    main()
