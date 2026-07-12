# Multimodal Data Preprocessing and Verification Pipeline

This project checks who a user is before showing them product recommendations.
A user has to pass two checks, one after the other:

```
Face check > Voice check > Product recommendations
   (fail)       (fail)
Access Denied  Access Denied
```

If the face check fails, the user is stopped.
If the voice check fails, the user is also stopped.
Only after both checks pass do they see any recommendations.

## Who does what

- Member 1 works on the face part: face photos and the face recognition model.
- Member 2 works on the voice part: recordings and the voice check model. This is the track in this repo.
- Member 3 works on the data and the final app that puts everything together.

## Folders

```
data/
  raw/ (the original customer CSV files)
  processed/ (cleaned and merged data, Member 3)
  audio/
    raw/ (the voice recordings from each member)
    augmented/ (edited copies: pitch, speed, and noise)
    features/ (audio_features.csv, the numbers taken from the audio)
src/
  audio/ (the scripts for the voice track)
notebooks/ (charts, tests, and model training)
```

## What the voice part does

1. Each member records two phrases: "Yes, approve" and "Confirm transaction".
2. Show each recording as a waveform and a spectrogram.
3. Make extra copies by changing the pitch, the speed, and adding noise.
4. Take useful numbers from the audio and save them in audio_features.csv.
5. Train a model that can tell a known voice from an unknown one.
6. Test it with a fake voice to make sure it blocks the wrong person.

## Recording your audio

Each member records two short clips, one for each phrase, and saves them as WAV files.
Name them so we know who recorded what, for example kevin_approve_01.wav and kevin_confirm_01.wav.
Put the finished recordings in data/audio/raw/.
