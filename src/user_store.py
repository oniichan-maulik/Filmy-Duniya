from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _store_path() -> Path:
    return _repo_root() / "data" / "user_learning.json"


def _events_path() -> Path:
    return _repo_root() / "data" / "feedback_events.jsonl"


def _default_profile() -> Dict[str, Any]:
    return {
        "version": 1,
        "feedback_events": 0,
        "mood_counts": {},
        "mood_genre_weights": {},
        "title_preferences": {},
    }


def load_user_profile() -> Dict[str, Any]:
    path = _store_path()
    if not path.exists():
        return _default_profile()
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _default_profile()
        merged = _default_profile()
        merged.update(data)
        return merged
    except Exception:
        return _default_profile()


def save_user_profile(profile: Dict[str, Any]) -> None:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)


def _append_feedback_event(event: Dict[str, Any]) -> None:
    p = _events_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _parse_genres(movie: Dict[str, Any]) -> list[str]:
    raw = str(movie.get("genres", "") or "")
    if not raw.strip():
        return []
    normalized = raw.replace("/", "|").replace(",", "|")
    out = []
    for g in normalized.split("|"):
        gg = g.strip().lower()
        if gg:
            out.append(gg)
    return out


def _title_key(movie: Dict[str, Any]) -> str:
    return str(movie.get("title", "") or "").strip().lower()


def _ensure_nested_dict(d: Dict[str, Any], key: str) -> Dict[str, float]:
    if key not in d or not isinstance(d.get(key), dict):
        d[key] = {}
    return d[key]


def record_feedback(
    mood: Optional[str],
    movie: Dict[str, Any],
    rating: Optional[float] = None,
    favorite: bool = False,
) -> None:
    profile = load_user_profile()
    mood_key = str(mood or "").strip().lower()
    title_key = _title_key(movie)

    if mood_key:
        mood_counts = profile.setdefault("mood_counts", {})
        mood_counts[mood_key] = int(mood_counts.get(mood_key, 0)) + 1

    signal = 0.0
    if rating is not None:
        try:
            r = float(rating)
            signal += (r - 2.5) / 2.5
        except (TypeError, ValueError):
            pass
    if favorite:
        signal += 0.6

    if title_key and signal != 0.0:
        title_weights = profile.setdefault("title_preferences", {})
        prev = float(title_weights.get(title_key, 0.0))
        title_weights[title_key] = prev + signal

    if mood_key:
        mood_genres = _ensure_nested_dict(profile.setdefault("mood_genre_weights", {}), mood_key)
        for genre in _parse_genres(movie):
            prev = float(mood_genres.get(genre, 0.0))
            mood_genres[genre] = prev + signal

    profile["feedback_events"] = int(profile.get("feedback_events", 0)) + 1
    save_user_profile(profile)
    event_count = int(profile.get("feedback_events", 0))

    base_score = movie.get("score", movie.get("rating", 0.0))
    try:
        base_score = float(base_score)
    except (TypeError, ValueError):
        base_score = 0.0

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mood": mood_key or None,
        "rating": float(rating) if rating is not None else None,
        "favorite": bool(favorite),
        "signal": float(signal),
        "movie": {
            "title": str(movie.get("title", "") or ""),
            "genres": str(movie.get("genres", "") or ""),
            "director": str(movie.get("director", "") or ""),
            "media_type": str(movie.get("media_type", "") or "movie"),
            "rating": movie.get("rating"),
            "popularity": movie.get("popularity"),
            "base_score": base_score,
        },
    }
    _append_feedback_event(event)

    # Keep training incremental without blocking every feedback action.
    # Train slightly more frequently to keep the personalization model fresh.
    if event_count > 0 and event_count % 3 == 0:
        try:
            from src.personalization_model import train_personalization_model

            train_personalization_model(min_events=5)
        except Exception:
            pass


def _learning_score(movie: Dict[str, Any], mood: Optional[str], profile: Dict[str, Any]) -> float:
    score = 0.0
    title_key = _title_key(movie)
    if title_key:
        score += float(profile.get("title_preferences", {}).get(title_key, 0.0)) * 0.25

    mood_key = str(mood or "").strip().lower()
    if mood_key:
        mood_map = profile.get("mood_genre_weights", {}).get(mood_key, {})
        for genre in _parse_genres(movie):
            score += float(mood_map.get(genre, 0.0)) * 0.10
    return score


def rerank_results_for_learning(results: list[Dict[str, Any]], mood: Optional[str]) -> list[Dict[str, Any]]:
    if not results:
        return results
    profile = load_user_profile()
    scored = []
    for movie in results:
        base = movie.get("score", movie.get("rating", 0.0))
        try:
            base_score = float(base)
        except (TypeError, ValueError):
            base_score = 0.0
        rule_boost = _learning_score(movie, mood, profile)
        model_boost = 0.0
        try:
            from src.personalization_model import predict_personalization_boost

            model_boost = predict_personalization_boost(movie, mood)
        except Exception:
            model_boost = 0.0
        model_total_boost = max(-1.2, min(1.2, float(model_boost)))
        boost = float(rule_boost) + model_total_boost
        enriched = dict(movie)
        enriched["rule_boost"] = float(rule_boost)
        enriched["model_boost"] = float(model_boost)
        enriched["model_total_boost"] = float(model_total_boost)
        enriched["learning_boost"] = float(boost)
        enriched["score"] = base_score + boost
        scored.append(enriched)
    scored.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return scored


def learning_summary() -> Dict[str, Any]:
    profile = load_user_profile()
    return {
        "feedback_events": int(profile.get("feedback_events", 0)),
        "mood_counts": profile.get("mood_counts", {}),
    }


def learning_progress(target_events: int = 200) -> Dict[str, Any]:
    summary = learning_summary()
    events = int(summary.get("feedback_events", 0))
    target = max(1, int(target_events))
    pct = min(100.0, (events / target) * 100.0)
    if pct < 20:
        stage = "cold-start"
    elif pct < 60:
        stage = "learning"
    elif pct < 90:
        stage = "adapting"
    else:
        stage = "well-trained"
    return {
        "events": events,
        "target_events": target,
        "progress_pct": float(pct),
        "stage": stage,
    }


def train_personalization_now(min_events: int = 10) -> Dict[str, Any]:
    linear_out: Dict[str, Any]
    try:
        from src.personalization_model import train_personalization_model

        linear_out = train_personalization_model(min_events=max(5, int(min_events)))
    except Exception as exc:
        linear_out = {"trained": False, "error": str(exc)}

    return {
        "trained": bool(linear_out.get("trained")),
        "linear": linear_out,
    }


def personalization_status() -> Dict[str, Any]:
    linear_status: Dict[str, Any]
    linear_err: Optional[str] = None

    try:
        from src.personalization_model import personalization_model_status

        linear_status = personalization_model_status()
    except Exception as exc:
        linear_err = str(exc)
        linear_status = {"available": False, "error": linear_err}

    out: Dict[str, Any] = {
        "available": bool(linear_status.get("available")),
        "linear": linear_status,
    }
    if linear_err:
        out["errors"] = {"linear": linear_err}
    return out
