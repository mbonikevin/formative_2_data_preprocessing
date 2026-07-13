"""
load every voice recording and show it as a waveform and a spectrogram

a waveform shows how loud the sound is over time, and a spectrogram shows
which pitches show up over time

run it with:
    python src/audio/load_display.py

the pictures get saved in data/audio/plots/ so we can drop them in the report
"""

from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

raw_dir = Path("data/audio/raw")
plots_dir = Path("data/audio/plots")

# load everything at the same sample rate so the clips match
sample_rate = 16000


def load_clip(path):
    # read one wav file and give back the samples plus the sample rate
    samples, sr = librosa.load(path, sr=sample_rate, mono=True)
    return samples, sr


def show_clip(path, save=True):
    # draw the waveform and the spectrogram for one recording
    samples, sr = load_clip(path)
    name = path.stem  # file name without the .wav

    fig, (ax_wave, ax_spec) = plt.subplots(2, 1, figsize=(10, 6))

    # top picture is the waveform
    librosa.display.waveshow(samples, sr=sr, ax=ax_wave)
    ax_wave.set_title(f"{name} - waveform")
    ax_wave.set_xlabel("time (seconds)")
    ax_wave.set_ylabel("loudness")

    # bottom picture is the spectrogram, in decibels so its easier to read
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(samples)), ref=np.max)
    img = librosa.display.specshow(spec, sr=sr, x_axis="time", y_axis="hz", ax=ax_spec)
    ax_spec.set_title(f"{name} - spectrogram")
    ax_spec.set_ylabel("pitch (Hz)")
    fig.colorbar(img, ax=ax_spec, format="%+2.0f dB")

    fig.tight_layout()

    if save:
        plots_dir.mkdir(parents=True, exist_ok=True)
        out = plots_dir / f"{name}.png"
        fig.savefig(out, dpi=120)
        print(f"saved {out}")

    return fig


def main():
    clips = sorted(raw_dir.glob("*.wav"))
    if not clips:
        print(f"no wav files in {raw_dir}, add the recordings first")
        return

    print(f"found {len(clips)} recordings")
    for path in clips:
        samples, sr = load_clip(path)
        seconds = len(samples) / sr
        print(f"  {path.name} ({seconds:.1f}s)")
        show_clip(path)

    print("\ndone, the pictures are in data/audio/plots/")


if __name__ == "__main__":
    main()
