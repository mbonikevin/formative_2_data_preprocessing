#!/usr/bin/env python3
"""
User identity and product recommendation system.

Flow:
1. Face verification (Random Forest on HOG + colour features) -> access granted or denied
2. Voice verification (Random Forest voiceprint) -> transaction approved or denied
3. Product recommendation (Random Forest) -> product predicted

Usage:
    python simulation.py --face data/images/extracted/james_face.jpg \
                         --voice data/audio/raw/james_approve.wav \
                         --member James

    python simulation.py --face data/images/extracted/unauthorized_attempt.png \
                         --voice data/audio/unauthorized/stranger_attempt.wav \
                         --member James
"""

import os
import sys
import time
import argparse
import warnings
import importlib.util
warnings.filterwarnings('ignore')

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import joblib
from skimage.feature import hog

_verify_path = Path(__file__).resolve().parent / 'src' / 'audio' / 'verify.py'
sys.path.append(str(_verify_path.parent))
_verify_spec = importlib.util.spec_from_file_location('verify', _verify_path)
_verify = importlib.util.module_from_spec(_verify_spec)
_verify_spec.loader.exec_module(_verify)
verify_voice = _verify.verify_voice

MODELS_DIR = 'models'
MERGED_PATH = 'data/processed/merged_dataset.csv'

FACE_MODEL_PATH = os.path.join(MODELS_DIR, 'facial_recognition_model.pkl')
FACE_ENCODER_PATH = os.path.join(MODELS_DIR, 'facial_recognition_label_encoder.pkl')
PRODUCT_MODEL_PATH = os.path.join(MODELS_DIR, 'product_recommendation_model.pkl')

FACE_THRESHOLD = 0.6


def status(message):
    print(f"    {message}")


def loading(message, duration=1.2):
    print(f"\n  {message}", end='', flush=True)
    for _ in range(3):
        time.sleep(duration / 3)
        print('.', end='', flush=True)
    print()


def canonical_member(name):
    base = name.split('_')[-1]
    return base.capitalize()


def load_models():
    try:
        face_model = joblib.load(FACE_MODEL_PATH)
        face_encoder = joblib.load(FACE_ENCODER_PATH)
        product_bundle = joblib.load(PRODUCT_MODEL_PATH)
        return face_model, face_encoder, product_bundle
    except FileNotFoundError as e:
        print(f"\n  Model file not found: {e}")
        print("  Make sure all models are in the 'models/' folder.")
        sys.exit(1)


def extract_face_features(img, size=(128, 128)):
    resized = cv2.resize(img, size)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    hog_features = hog(
        gray, orientations=9, pixels_per_cell=(16, 16),
        cells_per_block=(2, 2), block_norm="L2-Hys"
    )

    hist_features = []
    for ch in range(3):
        hist = cv2.calcHist([resized], [ch], None, [32], [0, 256])
        hist_features.extend(hist.flatten())
    hist_features = np.array(hist_features)
    hist_features = hist_features / (np.sum(hist_features) + 1e-6)

    return np.concatenate([hog_features, hist_features]).reshape(1, -1)


def verify_face(probe_path, claimed_member, face_model, face_encoder):
    if not os.path.exists(probe_path):
        print(f"  Probe image not found: {probe_path}")
        return False, 'unknown', 0.0

    img = cv2.imread(probe_path)
    if img is None:
        print(f"  Could not read image: {probe_path}")
        return False, 'unknown', 0.0

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    features = extract_face_features(img)

    proba = face_model.predict_proba(features)[0]
    best_idx = int(np.argmax(proba))
    confidence = round(float(proba[best_idx]) * 100, 2)
    predicted_member = face_encoder.inverse_transform([best_idx])[0]

    verified = confidence / 100 >= FACE_THRESHOLD and predicted_member == claimed_member
    return verified, predicted_member, confidence


def verify_speaker(audio_path, claimed_member):
    if not os.path.exists(audio_path):
        print(f"  Audio file not found: {audio_path}")
        return False, 'unknown', 0.0

    try:
        result = verify_voice(audio_path)
    except Exception as e:
        print(f"  Voice verification error: {e}")
        return False, 'unknown', 0.0

    predicted_member = result['speaker'].capitalize()
    confidence = round(result['confidence'] * 100, 2)
    verified = result['granted'] and predicted_member == claimed_member
    return verified, predicted_member, confidence


def get_product_recommendation(product_bundle):
    model = product_bundle['model']
    feature_names = product_bundle['feature_names']

    if not os.path.exists(MERGED_PATH):
        print(f"  Merged dataset not found: {MERGED_PATH}")
        return 'Unknown', 0.0

    try:
        df = pd.read_csv(MERGED_PATH)

        drop_cols = ['customer_id_legacy', 'transaction_id', 'customer_id',
                     'purchase_date', 'main_sentiment', 'product_category']
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        X = pd.get_dummies(X, columns=['main_platform'], prefix='platform')
        X = X.reindex(columns=feature_names, fill_value=0)

        X_sample = X.median().to_frame().T[feature_names]

        pred = model.predict(X_sample)[0]
        confidence = round(float(model.predict_proba(X_sample).max()) * 100, 2)
        return pred, confidence
    except Exception as e:
        print(f"  Recommendation error: {e}")
        return 'Unknown', 0.0


def run_simulation(face_path, voice_path, member):
    claimed_member = canonical_member(member)
    face_model, face_encoder, product_bundle = load_models()

    print("\n  USER IDENTITY & PRODUCT RECOMMENDATION SYSTEM")
    print(f"  Claimed Identity : {claimed_member}")
    print(f"  Face Input       : {face_path}")
    print(f"  Voice Input      : {voice_path}")

    print("\n  [ STEP 1 ] FACIAL VERIFICATION")
    loading("  Running facial recognition")

    face_verified, face_member, face_confidence = verify_face(
        face_path, claimed_member, face_model, face_encoder
    )

    if face_verified:
        status(f"Face VERIFIED  |  Predicted: {face_member}  |  Confidence: {face_confidence}%")
    else:
        status(f"Face NOT verified  |  Predicted: {face_member}  |  Confidence: {face_confidence}%")
        print("\n  ACCESS DENIED - Face verification failed.")
        print("      The system has rejected this access attempt.")
        return

    print("\n  [ STEP 2 ] VOICE VERIFICATION")
    loading("  Analysing voiceprint")

    voice_verified, voice_member, voice_confidence = verify_speaker(voice_path, claimed_member)

    if voice_verified:
        status(f"Voice VERIFIED  |  Predicted: {voice_member}  |  Confidence: {voice_confidence}%")
    else:
        status(f"Voice NOT verified  |  Predicted: {voice_member}  |  Confidence: {voice_confidence}%")
        print("\n  ACCESS DENIED - Voice verification failed.")
        print("      Both face and voice must be verified to proceed.")
        return

    print("\n  [ STEP 3 ] PRODUCT RECOMMENDATION")
    loading("  Generating personalised recommendation")

    product, prod_confidence = get_product_recommendation(product_bundle)

    status(f"Recommended Product : {product}")
    status(f"Model Confidence    : {prod_confidence}%")

    print("\n  TRANSACTION COMPLETE")
    print(f"\n  Welcome, {claimed_member}!")
    print(f"  Based on your profile, we recommend: {product}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='User Identity & Product Recommendation System Simulation'
    )
    parser.add_argument('--face', required=True,
                        help='Path to face image (e.g. data/images/extracted/james_face.jpg)')
    parser.add_argument('--voice', required=True,
                        help='Path to voice recording (e.g. data/audio/raw/james_approve.wav)')
    parser.add_argument('--member', required=True,
                        help='Claimed member identity (e.g. James, Kevin or Sheilla)')

    args = parser.parse_args()
    run_simulation(args.face, args.voice, args.member)
