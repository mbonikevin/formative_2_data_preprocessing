# Multimodal User Identity & Product Recommendation System

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![librosa](https://img.shields.io/badge/librosa-4B0082?style=for-the-badge&logo=soundcloud&logoColor=white)
![SoundFile](https://img.shields.io/badge/SoundFile-FF6F00?style=for-the-badge&logo=audiomack&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![scikit-image](https://img.shields.io/badge/scikit--image-FFB13B?style=for-the-badge&logo=python&logoColor=white)

A multimodal machine-learning pipeline that authenticates a user with **face
recognition** and **voiceprint verification** before it will show them a
**personalised product recommendation**. It is a sequential system with a
fail-fast security gate: if any check fails, access is denied and the pipeline
stops.

---

## System Flow

```
   Input face image
         |
         v
  [ 1. FACE CHECK ]  --- fail --->  ACCESS DENIED
         |
       pass
         v
   Input voice clip
         |
         v
  [ 2. VOICE CHECK ] --- fail --->  ACCESS DENIED
         |
       pass
         v
  [ 3. PRODUCT RECOMMENDATION ]
         |
         v
   Recommended product shown
```

1. A user submits a face image. It is matched against the known members with the
   facial recognition model.
2. If the face passes, the user submits a voice clip. The voiceprint model
   confirms the speaker and blocks strangers.
3. Only when both checks pass does the product recommendation model predict the
   product the customer is most likely to buy.

---

## Team

| Member | Focus Area |
| --- | --- |
| **Mukunzi Ndahiro James** | Data merge & feature engineering, product recommendation model, command-line simulation app |
| **Sheilla Keza Ruvugabigwi** | Image data collection, augmentation & facial recognition model |
| **Kevin Kaneza Mbonimpaye** | Audio data collection, augmentation & voiceprint verification model |

Every member also contributed their own facial images (neutral, smile,
surprised) and voice recordings (approve, confirm).

---

## Repository Structure

```
data/
  raw/                 original customer CSV files (social profiles + transactions)
  processed/           cleaned and merged dataset
  images/
    raw/               each member's face photos (neutral, smile, surprised) + unauthorized
    extracted/         grids, augmentation demos, unauthorized attempt
    image_features.csv HOG + colour histogram features per image
  audio/
    raw/               the voice recordings from each member
    augmented/         edited copies (pitch, speed, noise)
    features/          audio_features.csv (numbers taken from the audio)
    plots/             waveform and spectrogram pictures
    unauthorized/      the fake stranger clip for the deny test
src/
  audio/               scripts for the voice track
models/                saved trained models (face, voice, product)
notebooks/
  data_merge.ipynb         merges the two CSV files and builds the product model
  image_face_track.ipynb   the face photos and facial recognition model
  audio_track.ipynb        the voice recordings and voice verification model
simulation.py          the command-line app that ties all three models together
requirements.txt
```

---

## The Three Models

| Model | Algorithm | Features | Saved As |
| --- | --- | --- | --- |
| Facial Recognition | Random Forest | HOG + colour histograms (128x128) | `models/facial_recognition_model.pkl` |
| Voiceprint Verification | Random Forest + nearest-neighbour distance | 13 MFCCs, spectral roll-off, energy, zero-crossing rate | `models/voiceprint_model.joblib` |
| Product Recommendation | Random Forest | merged tabular + engineered features | `models/product_recommendation_model.pkl` |

Each model is evaluated with **Accuracy**, **F1-Score** and **Log Loss** in its
notebook.

---

## Setup

You need Python 3.9+ and the libraries in `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running the Command-Line Simulation

The final app is `simulation.py`. It takes a face image, a voice clip and the
claimed member identity, then runs all three steps.

Authorized attempt (passes every check and shows a recommendation):

```bash
python3 simulation.py \
  --face data/images/raw/kevin_neutral.jpg \
  --voice data/audio/raw/kevin_approve.wav \
  --member Kevin
```

Unauthorized attempt (denied at the face step):

```bash
python3 simulation.py \
  --face "data/images/raw/unauthorized attempt.jpg" \
  --voice data/audio/raw/kevin_approve.wav \
  --member Kevin
```

Arguments:

- `--face` path to a face image
- `--voice` path to a voice recording
- `--member` claimed identity (`James`, `Kevin` or `Sheilla`)

---

## The Voice Track, Step by Step

Each script does one job and they run in this order:

- `load_display.py` shows each recording as a waveform and a spectrogram
- `augment.py` makes extra copies by changing pitch, speed and adding noise
- `extract_features.py` pulls the numbers out of every clip into `audio_features.csv`
- `train_voiceprint.py` trains the voice model and measures accuracy, F1 and loss
- `verify.py` takes one clip and says pass or deny, and blocks a fake stranger

A full run:

```bash
python src/audio/load_display.py
python src/audio/augment.py
python src/audio/extract_features.py
python src/audio/train_voiceprint.py
python src/audio/verify.py
```

The notebook `notebooks/audio_track.ipynb` walks through all of this with the
charts and explanations.

---

## Datasets

- `customer_social_profiles.csv` and `customer_transactions.csv` are merged on
  the customer ID into `data/processed/merged_dataset.csv`.
- Feature engineering adds purchase month, sentiment score, and a big-purchase
  flag, then one-hot encodes the main social platform.
- The product recommendation model predicts the product category a customer is
  most likely to buy.

---

## Recording Your Audio

Each member records two short clips, one per phrase ("Yes, approve" and "Confirm
transaction"), saved as WAV files named so we know who recorded what, e.g.
`kevin_approve.wav` and `kevin_confirm.wav`, then placed in `data/audio/raw/`.

---

## Known Limitations

- The voiceprint reliably blocks inputs that are **not a human voice** (noise,
  tones, silence). With only a few clips per person it cannot yet reliably block
  an **unknown human** voice; more recordings per member and a tuned threshold
  fix this without code changes.
