# Data Preprocessing

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