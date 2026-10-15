"""Emotion detection helpers used by the Streamlit mood-based flow."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple
from pathlib import Path

import numpy as np

try:
    import cv2

    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False

try:
    from tensorflow.keras.models import load_model

    _HAS_TF = True
except Exception:
    load_model = None
    _HAS_TF = False


EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

EMOTION_TO_MOOD = {
    "Happy": "happy",
    "Sad": "sad",
    "Angry": "angry",
    "Disgust": "angry",
    "Fear": "anxious",
    "Surprise": "excited",
    "Neutral": "calm",
}

MOODS = ["happy", "sad", "angry", "anxious", "excited", "calm"]
_MOOD_WEIGHTS = {
    # Rebalanced to avoid over-predicting angry/happy in real webcam conditions.
    "Angry": {"angry": 0.8, "anxious": 0.25},
    "Disgust": {"angry": 0.35, "anxious": 0.25, "sad": 0.1},
    "Fear": {"anxious": 1.0, "sad": 0.35},
    "Happy": {"happy": 0.9, "excited": 0.25},
    "Sad": {"sad": 1.0, "calm": 0.1},
    "Surprise": {"excited": 0.9, "anxious": 0.3, "happy": 0.1},
    "Neutral": {"calm": 1.0},
}

if _HAS_CV2:
    def _load_cascade():
        cascade_paths = []
        cv2_data = getattr(cv2, "data", None)
        if cv2_data and getattr(cv2_data, "haarcascades", None):
            cascade_paths.append(
                str(Path(cv2_data.haarcascades) / "haarcascade_frontalface_default.xml")
            )
        cv2_file = getattr(cv2, "__file__", "")
        if cv2_file:
            cascade_paths.append(
                str(
                    Path(cv2_file).resolve().parent
                    / "data"
                    / "haarcascade_frontalface_default.xml"
                )
            )
            cascade_paths.append(
                str(
                    Path(cv2_file).resolve().parent
                    / "haarcascades"
                    / "haarcascade_frontalface_default.xml"
                )
            )
        cascade_paths.append("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")
        cascade_paths.append("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml")

        for p in cascade_paths:
            if p and Path(p).exists():
                cc = cv2.CascadeClassifier(p)
                if not cc.empty():
                    return cc
        return None

    FACE_CASCADE = _load_cascade()
else:
    FACE_CASCADE = None

_MODEL: Optional[Any] = None


def _ensure_runtime_deps() -> None:
    if not _HAS_CV2 or FACE_CASCADE is None:
        raise RuntimeError(
            "OpenCV face cascade not available. Install/reinstall `opencv-python` to enable mood detection."
        )
    if not _HAS_TF or load_model is None:
        raise RuntimeError(
            "TensorFlow is not installed. Install `tensorflow` to enable mood detection."
        )


def _preprocess_fer_gray(face_gray: np.ndarray) -> np.ndarray:
    """FER-style preprocessing with contrast enhancement for webcam frames."""
    norm = cv2.normalize(face_gray, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(norm)
    x = enhanced.astype("float32") / 255.0
    x = (x - 0.5) * 2.0
    return x


def _model_input_spec(model) -> Tuple[int, int, int, bool]:
    """Return (height, width, channels, channels_first)."""
    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list) and input_shape:
        input_shape = input_shape[0]

    # safe defaults for FER mini-XCEPTION
    height, width, channels, channels_first = 48, 48, 1, False
    if isinstance(input_shape, tuple):
        if len(input_shape) == 4:
            # (None, H, W, C) or (None, C, H, W)
            if input_shape[1] in (1, 3) and input_shape[2] and input_shape[3]:
                channels_first = True
                channels = int(input_shape[1])
                height = int(input_shape[2])
                width = int(input_shape[3])
            else:
                if input_shape[1] is not None:
                    height = int(input_shape[1])
                if input_shape[2] is not None:
                    width = int(input_shape[2])
                if input_shape[3] is not None:
                    channels = int(input_shape[3])
        elif len(input_shape) == 3:
            if input_shape[1] is not None:
                height = int(input_shape[1])
            if input_shape[2] is not None:
                width = int(input_shape[2])
    return height, width, channels, channels_first


def _load_emotion_model(model_path: Optional[str] = None):
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    _ensure_runtime_deps()
    # Resolve model path with fallback candidates so stale env vars do not break runtime.
    env_path = os.getenv("EMOTION_MODEL_PATH", "").strip()
    candidates = []
    if model_path and str(model_path).strip():
        candidates.append(str(model_path).strip())
    if env_path and env_path != "/absolute/path/to/your/model.h5":
        candidates.append(env_path)
    repo_root = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            str(repo_root / "models" / "emotion_model.h5"),
            str(repo_root / "src" / "fer2013_mini_XCEPTION.102-0.66.hdf5"),
            "models/emotion_model.h5",
        ]
    )

    path = next((p for p in candidates if os.path.exists(p)), "")
    if not path:
        raise FileNotFoundError(
            "Emotion model not found. Checked EMOTION_MODEL_PATH and default locations: "
            "models/emotion_model.h5, src/fer2013_mini_XCEPTION.102-0.66.hdf5."
        )
    # Inference-only usage: avoid deserializing legacy optimizer configs.
    _MODEL = load_model(path, compile=False)
    return _MODEL


def _extract_face(frame: np.ndarray, model) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
    )
    if len(faces) == 0:
        return None

    # Use largest face for stability when multiple faces are present.
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    # Add padding so eyebrows/mouth are kept, which helps expression detection.
    pad_w = int(w * 0.20)
    pad_h = int(h * 0.20)
    img_h, img_w = gray.shape[:2]
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_w, x + w + pad_w)
    y2 = min(img_h, y + h + pad_h)
    face_gray = gray[y1:y2, x1:x2]

    target_h, target_w, channels, channels_first = _model_input_spec(model)
    face_gray = cv2.resize(
        face_gray, (int(target_w), int(target_h)), interpolation=cv2.INTER_AREA
    )
    face_gray = _preprocess_fer_gray(face_gray)

    if channels == 1:
        if channels_first:
            face = face_gray.reshape(1, 1, target_h, target_w)
        else:
            face = face_gray.reshape(1, target_h, target_w, 1)
    else:
        face = np.repeat(face_gray[..., None], 3, axis=2)
        if channels_first:
            face = np.transpose(face, (2, 0, 1)).reshape(1, channels, target_h, target_w)
        else:
            face = face.reshape(1, target_h, target_w, channels)
    return face.astype("float32")


def _predict_probs(face: np.ndarray, model) -> np.ndarray:
    probs = np.asarray(model.predict(face, verbose=0)[0], dtype="float32").reshape(-1)
    if probs.size != len(EMOTIONS):
        # keep current behavior for unknown models
        fixed = np.zeros(len(EMOTIONS), dtype="float32")
        n = min(len(fixed), probs.size)
        fixed[:n] = probs[:n]
        probs = fixed
    probs = np.clip(probs, 0.0, 1.0)
    total = float(probs.sum())
    if total > 0:
        probs = probs / total
    return probs


def _mood_scores(probs: np.ndarray) -> Dict[str, float]:
    """Build debug mood scores directly from emotion probabilities."""
    scores = {m: 0.0 for m in MOODS}
    for i, emotion in enumerate(EMOTIONS):
        p = float(probs[i])
        if emotion == "Disgust":
            scores["angry"] += p * 0.80
            scores["sad"] += p * 0.20
            continue
        if emotion == "Fear":
            scores["anxious"] += p * 0.75
            scores["sad"] += p * 0.25
            continue
        if emotion == "Surprise":
            scores["excited"] += p * 0.70
            scores["happy"] += p * 0.30
            continue
        mood = EMOTION_TO_MOOD.get(emotion, "calm")
        scores[mood] += p
    return scores


def _pick_emotion(probs: np.ndarray) -> Tuple[str, float]:
    """Emotion-first selector with tie-breaks to avoid calm/angry lock-in."""
    p = np.asarray(probs, dtype="float32").reshape(-1)
    if p.size != len(EMOTIONS):
        idx = int(np.argmax(p))
        return EMOTIONS[idx], float(p[idx])

    order = np.argsort(p)[::-1]
    top_idx = int(order[0])
    second_idx = int(order[1]) if p.size > 1 else top_idx
    top_emotion = EMOTIONS[top_idx]
    top_conf = float(p[top_idx])
    second_emotion = EMOTIONS[second_idx]
    second_conf = float(p[second_idx])

    neutral_p = float(p[EMOTIONS.index("Neutral")])
    angry_p = float(p[EMOTIONS.index("Angry")])
    disgust_p = float(p[EMOTIONS.index("Disgust")])
    sad_p = float(p[EMOTIONS.index("Sad")])

    # If neutral is only slightly ahead, promote a meaningful non-neutral class.
    if top_emotion == "Neutral" and top_conf < 0.86:
        non_neutral = [
            (EMOTIONS[i], float(p[i]))
            for i in range(len(EMOTIONS))
            if EMOTIONS[i] != "Neutral"
        ]
        non_neutral.sort(key=lambda x: x[1], reverse=True)
        if non_neutral and non_neutral[0][1] >= 0.06:
            best_name, best_score = non_neutral[0]
            # Avoid defaulting to angry when sad/fear/happy are very close.
            if best_name in {"Angry", "Disgust"}:
                for cand_name, cand_score in non_neutral[1:]:
                    if cand_name in {"Sad", "Fear", "Happy"} and (best_score - cand_score) <= 0.05:
                        return cand_name, cand_score
            return best_name, best_score

    # If angry is weak and sadness/neutral are mixed, avoid forcing angry.
    if top_emotion in {"Angry", "Disgust"} and max(angry_p, disgust_p) < 0.55:
        if sad_p >= 0.14:
            return "Sad", sad_p
        if neutral_p >= 0.32 and second_conf > 0.10:
            return second_emotion, second_conf

    return top_emotion, top_conf


def _pick_mood_from_probs(probs: np.ndarray) -> Tuple[str, float, str]:
    emotion, emo_conf = _pick_emotion(probs)
    mood = EMOTION_TO_MOOD.get(emotion, "calm")
    return mood, float(emo_conf), emotion


def detect_emotion_details(frame: np.ndarray, model=None) -> Optional[Dict[str, Any]]:
    """Return emotion details for the top detected face.

    Returns:
    - None if no face is detected.
    - dict with keys: emotion, probs, confidence.
    """
    _ensure_runtime_deps()
    if frame is None:
        return None

    model = model or _load_emotion_model()
    face = _extract_face(frame, model=model)
    if face is None:
        return None

    probs = _predict_probs(face, model=model)
    mood, mood_conf, emotion = _pick_mood_from_probs(probs)
    face_img = face[0]
    if face_img.ndim == 2 or (face_img.ndim == 3 and face_img.shape[-1] == 1):
        view = face_img if face_img.ndim == 2 else face_img[..., 0]
        view = ((view / 2.0) + 0.5) * 255.0
        face_preview = np.repeat(view[..., None], 3, axis=2)
    elif face_img.ndim == 3 and face_img.shape[0] in (1, 3) and face_img.shape[-1] != 3:
        # channels-first preview
        view = np.transpose(face_img, (1, 2, 0))
        face_preview = np.clip(view * 255.0, 0, 255)
    elif face_img.ndim == 3 and face_img.shape[-1] == 1:
        face_preview = np.repeat(face_img, 3, axis=2)
    else:
        face_preview = np.clip(face_img * 255.0, 0, 255)

    mood_scores = _mood_scores(probs)
    ranked_moods = sorted(mood_scores.items(), key=lambda x: x[1], reverse=True)
    return {
        "emotion": emotion,
        "mood": mood,
        "probs": probs.tolist(),
        "confidence": float(np.max(probs)),
        "mood_confidence": mood_conf,
        "mood_scores": {k: float(v) for k, v in mood_scores.items()},
        "top_moods": [(m, float(s)) for m, s in ranked_moods[:3]],
        "face_preview": np.clip(face_preview, 0, 255).astype("uint8"),
    }


def detect_mood_from_frame(frame: np.ndarray, model=None) -> Optional[str]:
    """Map detected emotion to app mood. Returns None when no face is detected."""
    details = detect_emotion_details(frame, model=model)
    if not details:
        return None
    return details["mood"]
