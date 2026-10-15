from __future__ import annotations

import io
import importlib.util
import re
import threading
import wave
from typing import Any, Dict, Optional

import numpy as np

try:
    import cv2

    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False

from src.emotion_detection import MOODS, detect_emotion_details


_MOOD_LIST = list(MOODS)


def _empty_scores() -> Dict[str, float]:
    return {m: 0.0 for m in _MOOD_LIST}


def capture_backend_frame(camera_index: int = 0, warmup_frames: int = 8) -> Optional[np.ndarray]:
    if not _HAS_CV2:
        raise RuntimeError("OpenCV is not installed; backend camera capture is unavailable.")

    cap = cv2.VideoCapture(camera_index)
    if not cap or not cap.isOpened():
        raise RuntimeError("Cannot access camera device for backend face scan.")

    frame = None
    try:
        for _ in range(max(1, warmup_frames)):
            ok, frm = cap.read()
            if ok:
                frame = frm
        if frame is None:
            ok, frm = cap.read()
            if ok:
                frame = frm
    finally:
        cap.release()

    if frame is None:
        raise RuntimeError("Camera frame capture failed.")
    return frame


def detect_face_mood_backend() -> Optional[Dict[str, Any]]:
    frame = capture_backend_frame()
    return detect_emotion_details(frame)


def _decode_wav(audio_bytes: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        arr = np.frombuffer(raw, dtype=np.int16).astype("float32") / 32768.0
    elif sampwidth == 1:
        arr = (np.frombuffer(raw, dtype=np.uint8).astype("float32") - 128.0) / 128.0
    else:
        raise RuntimeError("Unsupported audio format for mood analysis.")

    if n_channels > 1:
        arr = arr.reshape(-1, n_channels).mean(axis=1)
    return np.clip(arr, -1.0, 1.0)


def analyze_voice_mood_from_wav_bytes(audio_bytes: bytes) -> Dict[str, Any]:
    signal = _decode_wav(audio_bytes)
    if signal.size < 200:
        raise RuntimeError("Voice sample too short for analysis.")

    rms = float(np.sqrt(np.mean(np.square(signal))))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(signal).astype(np.int8)))))
    peak = float(np.max(np.abs(signal)))

    scores = _empty_scores()
    # Simple rule-based voice sentiment proxy.
    scores["excited"] += min(1.0, rms * 7.0) + min(0.8, zcr * 12.0)
    scores["angry"] += min(1.0, rms * 8.5) + min(1.0, peak * 0.8)
    scores["happy"] += min(1.0, rms * 6.0) + min(0.5, zcr * 6.0)
    scores["anxious"] += min(1.0, zcr * 14.0) + min(0.5, rms * 4.0)
    scores["sad"] += max(0.0, 0.8 - rms * 8.0) + max(0.0, 0.25 - zcr * 4.0)
    scores["calm"] += max(0.0, 0.9 - rms * 9.0) + max(0.0, 0.22 - zcr * 4.0)

    total = sum(max(0.0, v) for v in scores.values())
    if total <= 0:
        probs = {m: (1.0 / len(_MOOD_LIST)) for m in _MOOD_LIST}
    else:
        probs = {k: max(0.0, v) / total for k, v in scores.items()}

    ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    mood, conf = ranked[0]
    return {
        "source": "voice",
        "mood": mood,
        "confidence": float(conf),
        "top_moods": [(m, float(s)) for m, s in ranked[:3]],
        "features": {"rms": rms, "zcr": zcr, "peak": peak},
        "mood_scores": probs,
    }


def capture_voice_wav_bytes_backend(duration_sec: float = 3.0, sample_rate: int = 16000) -> bytes:
    try:
        import sounddevice as sd
    except Exception as exc:
        raise RuntimeError(
            "Backend voice capture requires `sounddevice` package and a microphone on server."
        ) from exc

    frames = int(max(1.0, duration_sec) * sample_rate)
    recording = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    mono = np.asarray(recording).reshape(-1)

    # Build wav bytes in-memory to reuse the same analyzer.
    buf = io.BytesIO()
    pcm = np.clip(mono, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def capture_voice_mood_backend(duration_sec: float = 3.0, sample_rate: int = 16000) -> Dict[str, Any]:
    return analyze_voice_mood_from_wav_bytes(
        capture_voice_wav_bytes_backend(duration_sec=duration_sec, sample_rate=sample_rate)
    )


def transcribe_movie_title_from_wav_bytes(audio_bytes: bytes) -> Optional[str]:
    """Best-effort speech-to-text for spoken movie-name requests."""
    out = transcribe_movie_title_with_debug(audio_bytes)
    return out.get("text")


def _clean_transcript(text: str) -> str:
    t = str(text or "").strip()
    if not t:
        return ""
    t = re.sub(r"[^\w\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    prefixes = (
        "recommend me ",
        "play ",
        "show ",
        "search ",
        "find ",
        "movie ",
        "watch ",
        "recommend ",
    )
    low = t.lower()
    for p in prefixes:
        if low.startswith(p):
            t = t[len(p) :].strip()
            low = t.lower()
    return t


def transcribe_movie_title_with_debug(audio_bytes: bytes) -> Dict[str, Any]:
    """Speech-to-text with engine fallback and debug metadata."""
    try:
        import speech_recognition as sr
    except Exception as exc:
        return {"text": None, "engine": None, "error": f"speech_recognition_missing: {exc}"}

    rec = sr.Recognizer()
    errors = []

    def _format_engine_error(engine: str, exc: Exception) -> str:
        msg = str(exc or "").strip()
        if not msg:
            msg = exc.__class__.__name__
        return f"{engine}:{msg}"
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as src:
            audio_data = rec.record(src)
    except Exception as exc:
        return {"text": None, "engine": None, "error": f"audio_decode_failed: {exc}"}

    # Primary cloud recognizer.
    try:
        text = rec.recognize_google(audio_data, language="en-US")
        cleaned = _clean_transcript(str(text or ""))
        if cleaned:
            return {"text": cleaned, "engine": "google", "error": None}
    except Exception as exc:
        errors.append(_format_engine_error("google", exc))

    # Optional offline fallback if pocketsphinx is installed.
    if importlib.util.find_spec("pocketsphinx") is not None:
        try:
            text = rec.recognize_sphinx(audio_data)
            cleaned = _clean_transcript(str(text or ""))
            if cleaned:
                return {"text": cleaned, "engine": "sphinx", "error": None}
        except Exception as exc:
            errors.append(_format_engine_error("sphinx", exc))
    else:
        errors.append("sphinx:PocketSphinx not installed")

    return {
        "text": None,
        "engine": None,
        "error": "; ".join(errors) if errors else "transcription_failed",
    }


def listen_for_movie_title_backend(duration_sec: float = 3.0) -> Dict[str, Any]:
    """Capture microphone audio and transcribe a spoken movie title."""
    voice_bytes = capture_voice_wav_bytes_backend(duration_sec=duration_sec)
    out = transcribe_movie_title_with_debug(voice_bytes)
    out["duration_sec"] = float(duration_sec)
    return out


def fuse_mood_signals(
    face_details: Optional[Dict[str, Any]],
    voice_details: Optional[Dict[str, Any]],
    face_weight: float = 0.70,
    voice_weight: float = 0.30,
) -> Optional[Dict[str, Any]]:
    if not face_details and not voice_details:
        return None

    scores = _empty_scores()
    used_sources = []

    if face_details:
        used_sources.append("face")
        face_scores = face_details.get("mood_scores") or {face_details.get("mood"): 1.0}
        for mood, val in face_scores.items():
            if mood in scores:
                scores[mood] += float(val) * face_weight

    if voice_details:
        used_sources.append("voice")
        voice_scores = voice_details.get("mood_scores") or {voice_details.get("mood"): 1.0}
        for mood, val in voice_scores.items():
            if mood in scores:
                scores[mood] += float(val) * voice_weight

    total = sum(max(0.0, x) for x in scores.values())
    if total > 0:
        scores = {k: max(0.0, v) / total for k, v in scores.items()}

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    mood, conf = ranked[0]
    return {
        "source": "+".join(used_sources),
        "mood": mood,
        "confidence": float(conf),
        "top_moods": [(m, float(s)) for m, s in ranked[:3]],
        "mood_scores": scores,
    }


def detect_multimodal_mood_backend(
    voice_duration_sec: float = 3.0,
    use_parallel_threads: bool = False,
) -> Optional[Dict[str, Any]]:
    """Capture face and voice together, then fuse into one mood signal.

    Voice capture is optional; if it fails, face-only detection is used.
    """
    face_details: Optional[Dict[str, Any]] = None
    voice_details: Optional[Dict[str, Any]] = None
    spoken_text: Optional[str] = None

    face_error: Optional[Exception] = None
    voice_error: Optional[Exception] = None
    transcript_error: Optional[str] = None

    def _run_face() -> None:
        nonlocal face_details, face_error
        try:
            face_details = detect_face_mood_backend()
        except Exception as exc:  # keep best-effort behavior
            face_error = exc

    def _run_voice() -> None:
        nonlocal voice_details, voice_error, spoken_text, transcript_error
        try:
            voice_bytes = capture_voice_wav_bytes_backend(duration_sec=voice_duration_sec)
            voice_details = analyze_voice_mood_from_wav_bytes(voice_bytes)
            stt = transcribe_movie_title_with_debug(voice_bytes)
            spoken_text = stt.get("text")
            transcript_error = stt.get("error")
        except Exception as exc:  # keep best-effort behavior
            voice_error = exc

    if use_parallel_threads:
        t_face = threading.Thread(target=_run_face, daemon=True)
        t_voice = threading.Thread(target=_run_voice, daemon=True)
        t_face.start()
        t_voice.start()
        t_face.join()
        t_voice.join()
    else:
        # Sequential execution is slower but typically more stable with
        # OpenCV/TensorFlow audio-video device stacks in constrained runtimes.
        _run_voice()
        _run_face()

    fused = fuse_mood_signals(face_details, voice_details)
    if fused is None and face_error and voice_error:
        raise RuntimeError("Face and voice backend detection both failed.")
    if fused is not None and spoken_text:
        fused["spoken_text"] = spoken_text
    if fused is not None and transcript_error:
        fused["voice_transcript_error"] = transcript_error
    if fused is not None and voice_error:
        fused["voice_error"] = str(voice_error)
    return fused
