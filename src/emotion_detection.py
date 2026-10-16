"""Face emotion detection used by the Streamlit mood-based flow.

Pipeline: YuNet finds the face, then HSEmotion (EfficientNet-B2 trained on
AffectNet, 7 classes) classifies the expression. Both models are ONNX files in
`models/` and run on CPU through onnxruntime.

Model sources:
- YuNet: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- HSEmotion: https://github.com/HSE-asavchenko/face-emotion-recognition
  (code Apache-2.0; weights trained on AffectNet, which is licensed for
  non-commercial research use)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2

    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False

try:
    import onnxruntime as ort

    _HAS_ORT = True
except Exception:
    ort = None
    _HAS_ORT = False


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
FACE_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
EMOTION_MODEL_PATH = MODELS_DIR / "hsemotion_enet_b2_7.onnx"

# Output order of the HSEmotion 7-class model
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

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

_EMOTION_INPUT_SIZE = 260  # EfficientNet-B2
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype="float32")
_FACE_SCORE_THRESHOLD = 0.7

_FACE_DETECTOR: Optional[Any] = None
_EMOTION_SESSION: Optional[Any] = None


def _ensure_runtime_deps() -> None:
    if not _HAS_CV2 or not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError(
            "OpenCV with FaceDetectorYN is required. Install `opencv-python>=4.10,<5`."
        )
    if not _HAS_ORT:
        raise RuntimeError("onnxruntime is not installed. Install `onnxruntime`.")
    for p in (FACE_MODEL_PATH, EMOTION_MODEL_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Model file missing: {p}")


def _load_face_detector():
    global _FACE_DETECTOR
    if _FACE_DETECTOR is None:
        _FACE_DETECTOR = cv2.FaceDetectorYN.create(
            str(FACE_MODEL_PATH), "", (320, 320), _FACE_SCORE_THRESHOLD
        )
    return _FACE_DETECTOR


def _load_emotion_model():
    global _EMOTION_SESSION
    if _EMOTION_SESSION is None:
        _EMOTION_SESSION = ort.InferenceSession(
            str(EMOTION_MODEL_PATH), providers=["CPUExecutionProvider"]
        )
    return _EMOTION_SESSION


def _extract_face(frame: np.ndarray) -> Optional[np.ndarray]:
    """Return the largest face in a BGR frame as an RGB crop, or None."""
    detector = _load_face_detector()
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None or len(faces) == 0:
        return None

    x, y, bw, bh = faces[int(np.argmax(faces[:, 2] * faces[:, 3]))][:4].astype(int)
    crop = frame[max(0, y) : min(h, y + bh), max(0, x) : min(w, x + bw)]
    if crop.size == 0:
        return None
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def _predict_probs(face_rgb: np.ndarray) -> np.ndarray:
    x = cv2.resize(face_rgb, (_EMOTION_INPUT_SIZE, _EMOTION_INPUT_SIZE)).astype("float32") / 255.0
    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    x = x.transpose(2, 0, 1)[np.newaxis].astype("float32")
    logits = _load_emotion_model().run(None, {"input": x})[0][0]
    probs = np.exp(logits - np.max(logits))
    return probs / probs.sum()


def _mood_scores(probs: np.ndarray) -> Dict[str, float]:
    scores = {m: 0.0 for m in MOODS}
    for emotion, p in zip(EMOTIONS, probs):
        scores[EMOTION_TO_MOOD[emotion]] += float(p)
    return scores


def detect_emotion_details(frame: np.ndarray) -> Optional[Dict[str, Any]]:
    """Return emotion details for the largest face in a BGR frame.

    Returns None if no face is detected, otherwise a dict with keys: emotion,
    mood, probs, confidence, mood_confidence, mood_scores, top_moods,
    face_preview (RGB uint8 crop).
    """
    _ensure_runtime_deps()
    if frame is None:
        return None

    face = _extract_face(frame)
    if face is None:
        return None

    probs = _predict_probs(face)
    idx = int(np.argmax(probs))
    emotion = EMOTIONS[idx]
    mood_scores = _mood_scores(probs)
    ranked: List[Tuple[str, float]] = sorted(mood_scores.items(), key=lambda x: x[1], reverse=True)
    return {
        "emotion": emotion,
        "mood": EMOTION_TO_MOOD[emotion],
        "probs": probs.tolist(),
        "confidence": float(probs[idx]),
        "mood_confidence": mood_scores[EMOTION_TO_MOOD[emotion]],
        "mood_scores": mood_scores,
        "top_moods": ranked[:3],
        "face_preview": face,
    }


def detect_emotion_from_bytes(image_bytes: bytes) -> Optional[Dict[str, Any]]:
    """Decode encoded image bytes (JPEG/PNG) and return JSON-safe emotion details.

    Returns None when the image has no detectable face. Raises ValueError when
    the bytes can't be decoded as an image.
    """
    _ensure_runtime_deps()
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image bytes")
    details = detect_emotion_details(frame)
    if not details:
        return None
    # face_preview is a numpy array, which FastAPI can't serialize
    details.pop("face_preview", None)
    return details


def detect_mood_from_frame(frame: np.ndarray) -> Optional[str]:
    """Map detected emotion to app mood. Returns None when no face is detected."""
    details = detect_emotion_details(frame)
    if not details:
        return None
    return details["mood"]


def capture_backend_frame(camera_index: int = 0, warmup_frames: int = 20) -> Optional[np.ndarray]:
    """Grab one frame from the local webcam, after letting auto-exposure settle."""
    if not _HAS_CV2:
        raise RuntimeError("OpenCV is not installed; backend camera capture is unavailable.")

    # On Windows the default MSMF backend takes ~5s to open a webcam; DirectShow ~0.3s
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap or not cap.isOpened():
        raise RuntimeError("Cannot access camera device for backend face scan.")

    frame = None
    try:
        for _ in range(max(1, warmup_frames)):
            ok, frm = cap.read()
            if ok:
                frame = frm
    finally:
        cap.release()

    if frame is None:
        raise RuntimeError("Camera frame capture failed.")
    return frame


def detect_face_mood_backend() -> Optional[Dict[str, Any]]:
    """Capture a webcam frame and return emotion details, or None if no face is found."""
    details = detect_emotion_details(capture_backend_frame())
    if details:
        details["source"] = "face"
    return details
