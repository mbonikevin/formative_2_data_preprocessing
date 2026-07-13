"""
make extra copies of each recording so the model has more to learn from

for every recording we make three new versions:
  pitch shift, makes the voice a bit higher
  time stretch, makes the speech a bit faster
  added noise, mixes in some quiet background noise

run it with:
    python src/audio/augment.py

the new files land in data/audio/augmented/ with a tag added to the name,
like kevin_approve_pitch.wav, kevin_approve_stretch.wav and so on
"""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

raw_dir = Path("data/audio/raw")
aug_dir = Path("data/audio/augmented")

sample_rate = 16000


def pitch_shift(samples, sr):
    # raise the voice by 2 half steps so it sounds a little higher
    return librosa.effects.pitch_shift(samples, sr=sr, n_steps=2)


def time_stretch(samples):
    # speed the speech up to 1.15x, a value above 1 makes it faster
    return librosa.effects.time_stretch(samples, rate=1.15)


def add_noise(samples):
    # mix in a small amount of random background noise
    noise = np.random.normal(0, 0.005, samples.shape)
    return samples + noise


def augment_clip(path):
    # make the three versions for one recording and save them
    samples, sr = librosa.load(path, sr=sample_rate, mono=True)
    name = path.stem

    versions = {
        "pitch": pitch_shift(samples, sr),
        "stretch": time_stretch(samples),
        "noise": add_noise(samples),
    }

    saved = []
    for tag, new_samples in versions.items():
        out = aug_dir / f"{name}_{tag}.wav"
        sf.write(out, new_samples, sr)
        saved.append(out.name)
    return saved


def main():
    clips = sorted(raw_dir.glob("*.wav"))
    if not clips:
        print(f"no wav files in {raw_dir}, add the recordings first")
        return

    aug_dir.mkdir(parents=True, exist_ok=True)

    # fix the randomness so we get the same results each run
    np.random.seed(42)

    print(f"found {len(clips)} recordings, making 3 versions of each")
    total = 0
    for path in clips:
        saved = augment_clip(path)
        for name in saved:
            print(f"  saved {name}")
        total += len(saved)

    print(f"\ndone, made {total} new files in data/audio/augmented/")


if __name__ == "__main__":
    main()
