"""
Load every voice recording and show it as a waveform and a spectrogram.

A waveform shows how loud the sound is over time.
A spectrogram shows which pitches (frequencies) are present over time.

Run it:
    python src/audio/load_display.py

The pictures are saved in data/audio/plots/ so we can put them in the report.
"""

from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# folders
RAW_DIR = Path("data/audio/raw")
PLOTS_DIR = Path("data/audio/plots")

# every recording is loaded at this sample rate so they all match
SAMPLE_RATE = 16000


def load_clip(path):
    """Read one wav file and return the sound samples and the sample rate."""
    samples, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return samples, sr


def show_clip(path, save=True):
    """Draw the waveform and spectrogram for one recording."""
    samples, sr = load_clip(path)
    name = path.stem  # file name without .wav

    fig, (ax_wave, ax_spec) = plt.subplots(2, 1, figsize=(10, 6))

    # top picture: the waveform
    librosa.display.waveshow(samples, sr=sr, ax=ax_wave)
    ax_wave.set_title(f"{name} - waveform")
    ax_wave.set_xlabel("time (seconds)")
    ax_wave.set_ylabel("loudness")

    # bottom picture: the spectrogram (in decibels so it is easy to read)
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(samples)), ref=np.max)
    img = librosa.display.specshow(spec, sr=sr, x_axis="time", y_axis="hz", ax=ax_spec)
    ax_spec.set_title(f"{name} - spectrogram")
    ax_spec.set_ylabel("pitch (Hz)")
    fig.colorbar(img, ax=ax_spec, format="%+2.0f dB")

    fig.tight_layout()

    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        out = PLOTS_DIR / f"{name}.png"
        fig.savefig(out, dpi=120)
        print(f"saved {out}")

    return fig


def main():
    clips = sorted(RAW_DIR.glob("*.wav"))
    if not clips:
        print(f"No wav files found in {RAW_DIR}. Add the recordings first.")
        return

    print(f"Found {len(clips)} recordings.")
    for path in clips:
        samples, sr = load_clip(path)
        seconds = len(samples) / sr
        print(f"  {path.name}  ({seconds:.1f}s)")
        show_clip(path)

    print("\nDone. The pictures are in data/audio/plots/")


if __name__ == "__main__":
    main()
