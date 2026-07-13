"""
check a single voice clip and decide, known voice (pass) or unknown voice (deny)

how it works:
  measure the same numbers we used for training
  ask the model who it thinks is talking and how sure it is
  if the model is sure enough and the clip looks like a real voice we pass,
  otherwise we treat it as a stranger and deny

this file can be used two ways

as a command:
    python src/audio/verify.py                     runs a small demo
    python src/audio/verify.py path/to/clip.wav    checks one clip

as a function, which is what the final app uses:
    from verify import verify_voice
    result = verify_voice("some_clip.wav")
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf

# reuse the exact same measuring code from the feature step
sys.path.append(str(Path(__file__).resolve().parent))
from extract_features import clip_features

model_file = Path("models/voiceprint_model.joblib")
unknown_dir = Path("data/audio/unauthorized")
sample_rate = 16000


def load_model():
    # load the trained model and the settings saved with it
    if not model_file.exists():
        raise FileNotFoundError(
            f"no model at {model_file}, run train_voiceprint.py first"
        )
    return joblib.load(model_file)


def verify_voice(clip_path, bundle=None):
    # look at one clip and give back the decision as a dict with granted,
    # speaker and confidence
    if bundle is None:
        bundle = load_model()

    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    threshold = bundle["threshold"]
    scaler = bundle["scaler"]
    neighbours = bundle["neighbours"]
    distance_cutoff = bundle["distance_cutoff"]

    # measure the clip and line the numbers up in the same order as training
    row = clip_features(Path(clip_path))
    features = np.array([[row[c] for c in feature_cols]])

    # ask the model who it thinks is talking and how sure it is
    proba = model.predict_proba(features)[0]
    best = int(np.argmax(proba))
    speaker = model.classes_[best]
    confidence = float(proba[best])

    # second check, how far is this clip from a real voice we have seen
    features_scaled = scaler.transform(features)
    dist, _ = neighbours.kneighbors(features_scaled)
    distance = float(dist[0][0])
    looks_like_a_real_voice = distance <= distance_cutoff

    # to pass, the model has to be sure and the clip has to look like a real voice
    granted = confidence >= threshold and looks_like_a_real_voice

    return {
        "granted": granted,
        "speaker": speaker if granted else "unknown",
        "confidence": confidence,
        "distance": distance,
    }


def make_unauthorized_clip():
    # make a fake stranger clip for the demo, its a buzzer sound (two steady
    # tones) not a human voice, so the numbers look nothing like real speech
    # and the system should block it, it stands in for any input thats not a
    # recognised voice
    unknown_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(sample_rate * 3) / sample_rate  # 3 seconds
    buzzer = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 620 * t)
    out = unknown_dir / "stranger_attempt.wav"
    sf.write(out, buzzer.astype("float32"), sample_rate)
    return out


def print_result(clip_path, result):
    print(f"\nclip: {clip_path}")
    print(f"confidence {result['confidence']:.2f}")
    if result["granted"]:
        print(f"access granted, recognised as {result['speaker']}")
    else:
        print("access denied, voice not recognised")


def demo():
    # show a known voice passing and a stranger getting blocked
    bundle = load_model()

    print("voice check demo")
    print("----------------")

    # a real member should pass
    known_clip = Path("data/audio/raw/kevin_approve.wav")
    if known_clip.exists():
        result = verify_voice(known_clip, bundle)
        print("\na known member tries to log in:")
        print_result(known_clip, result)

    # a stranger should be blocked
    stranger_clip = make_unauthorized_clip()
    result = verify_voice(stranger_clip, bundle)
    print("\nan unknown voice tries to log in:")
    print_result(stranger_clip, result)


def main():
    if len(sys.argv) > 1:
        clip_path = sys.argv[1]
        result = verify_voice(clip_path)
        print_result(clip_path, result)
    else:
        demo()


if __name__ == "__main__":
    main()
