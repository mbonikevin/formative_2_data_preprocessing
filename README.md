# data preprocessing

this project checks who a user is before showing them product recommendations.
a user has to pass two checks, one after the other:

```
face check > voice check > product recommendations
   (fail)       (fail)
access denied  access denied
```

if the face check fails the user is stopped, and if the voice check fails they
are stopped too, only after both checks pass do they see any recommendations

## who does what

- member 1 does the face part, the face photos and the face recognition model
- member 2 does the voice part, the recordings and the voice check model, thats the track in this repo
- member 3 does the data and the final app that ties everything together

## folders

```
data/
  raw/ (the original customer csv files)
  processed/ (cleaned and merged data, member 3)
  audio/
    raw/ (the voice recordings from each member)
    augmented/ (edited copies: pitch, speed and noise)
    features/ (audio_features.csv, the numbers taken from the audio)
    plots/ (waveform and spectrogram pictures)
src/
  audio/ (the scripts for the voice track)
notebooks/ (charts, tests and model training)
```

## setup

you need python and the libraries in requirements.txt, the easiest way:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## the voice part, step by step

each script does one job and they run in this order:

- load_display.py, shows each recording as a waveform and a spectrogram
- augment.py, makes extra copies by changing pitch, speed and adding noise
- extract_features.py, pulls the numbers out of every clip into audio_features.csv
- train_voiceprint.py, trains the voice model and measures accuracy, f1 and loss
- verify.py, takes one clip and says pass or deny, and blocks a fake stranger

so a full run looks like:

```
python src/audio/load_display.py
python src/audio/augment.py
python src/audio/extract_features.py
python src/audio/train_voiceprint.py
python src/audio/verify.py
```

the notebook in notebooks/audio_track.ipynb walks through all of this with the
charts and the explanation

## recording your audio

each member records two short clips, one for each phrase, saved as wav files,
name them so we know who recorded what, like kevin_approve_01.wav and
kevin_confirm_01.wav, then put them in data/audio/raw/
