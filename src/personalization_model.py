from __future__ import annotations

import argparse
import csv
import json
import pickle
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDRegressor


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def feedback_events_path() -> Path:
    return _repo_root() / "data" / "feedback_events.jsonl"


def model_path() -> Path:
    return _repo_root() / "data" / "personalization_model.pkl"


def _legacy_profile_path() -> Path:
    return _repo_root() / "data" / "user_learning.json"


def _movies_csv_path() -> Path:
    return _repo_root() / "data" / "movies.csv"


def _parse_genres(raw: Any) -> List[str]:
    text = str(raw or "").strip().lower()
    if not text:
        return []
    out: List[str] = []
    for part in text.replace(",", "|").replace("/", "|").split("|"):
        g = part.strip()
        if g:
            out.append(g)
    return out


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_feedback_events(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    p = path or feedback_events_path()
    if not p.exists():
        return _load_legacy_feedback_events()
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    if rows:
        return rows
    return _load_legacy_feedback_events()


def _movie_lookup() -> Dict[str, Dict[str, Any]]:
    path = _movies_csv_path()
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                title = str(row.get("title", "") or "").strip().lower()
                if not title:
                    continue
                out[title] = {
                    "title": row.get("title", ""),
                    "genres": row.get("genres", ""),
                    "director": row.get("director", ""),
                    "media_type": row.get("media_type", "movie"),
                    "rating": row.get("rating"),
                    "popularity": row.get("popularity"),
                    "base_score": row.get("rating", 0.0),
                }
    except Exception:
        return {}
    return out


def _load_legacy_feedback_events() -> List[Dict[str, Any]]:
    p = _legacy_profile_path()
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8") as f:
            profile = json.load(f)
    except Exception:
        return []
    if not isinstance(profile, dict):
        return []

    title_prefs = profile.get("title_preferences", {}) or {}
    mood_genres = profile.get("mood_genre_weights", {}) or {}
    lookup = _movie_lookup()
    events: List[Dict[str, Any]] = []

    if isinstance(title_prefs, dict):
        for title_key, signal in title_prefs.items():
            tkey = str(title_key or "").strip().lower()
            movie = lookup.get(
                tkey,
                {
                    "title": str(title_key or ""),
                    "genres": "",
                    "director": "",
                    "media_type": "movie",
                    "rating": 0.0,
                    "popularity": 0.0,
                    "base_score": 0.0,
                },
            )
            events.append(
                {
                    "source": "legacy_profile",
                    "mood": None,
                    "signal": _safe_float(signal, 0.0),
                    "movie": movie,
                }
            )

    if isinstance(mood_genres, dict):
        for mood, gmap in mood_genres.items():
            if not isinstance(gmap, dict):
                continue
            for genre, signal in gmap.items():
                events.append(
                    {
                        "source": "legacy_profile",
                        "mood": str(mood or "").strip().lower() or None,
                        "signal": _safe_float(signal, 0.0),
                        "movie": {
                            "title": f"__legacy_genre_{mood}_{genre}",
                            "genres": str(genre or ""),
                            "director": "",
                            "media_type": "movie",
                            "rating": 0.0,
                            "popularity": 0.0,
                            "base_score": 0.0,
                        },
                    }
                )
    return events
    return rows


def _feature_dict(movie: Dict[str, Any], mood: Optional[str]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    mood_key = str(mood or "").strip().lower()
    if mood_key:
        feats[f"mood={mood_key}"] = 1.0

    media_type = str(movie.get("media_type", "") or "").strip().lower()
    if media_type:
        feats[f"media_type={media_type}"] = 1.0

    for genre in _parse_genres(movie.get("genres", "")):
        feats[f"genre={genre}"] = 1.0

    director = str(movie.get("director", "") or "").strip().lower()
    if director:
        feats[f"director={director}"] = 1.0

    feats["base_score"] = _safe_float(movie.get("base_score", movie.get("score", 0.0)))
    feats["rating"] = _safe_float(movie.get("rating", 0.0))
    feats["popularity"] = _safe_float(movie.get("popularity", 0.0))
    return feats


def _training_rows(events: Iterable[Dict[str, Any]]) -> tuple[List[Dict[str, float]], List[float]]:
    x_rows: List[Dict[str, float]] = []
    y_rows: List[float] = []
    for e in events:
        signal = _safe_float(e.get("signal", 0.0))
        if abs(signal) < 0.01:
            continue
        x_rows.append(_feature_dict(e.get("movie", {}) or {}, e.get("mood")))
        y_rows.append(float(np.clip(signal, -2.0, 2.0)))
    return x_rows, y_rows


def train_personalization_model(min_events: int = 10) -> Dict[str, Any]:
    events = load_feedback_events()
    x_rows, y_rows = _training_rows(events)
    if len(x_rows) < int(min_events):
        return {
            "trained": False,
            "reason": "not_enough_labeled_feedback",
            "total_events": len(events),
            "labeled_events": len(x_rows),
            "min_events": int(min_events),
        }

    has_positive = any(y > 0 for y in y_rows)
    has_negative = any(y < 0 for y in y_rows)

    vec = DictVectorizer(sparse=True)
    x = vec.fit_transform(x_rows)
    y = np.asarray(y_rows, dtype=np.float32)

    model = SGDRegressor(
        loss="huber",
        alpha=1e-4,
        penalty="l2",
        random_state=42,
        max_iter=3000,
        tol=1e-3,
    )
    model.fit(x, y)

    preds = model.predict(x)
    mae = float(np.mean(np.abs(preds - y)))

    artifact = {
        "version": 1,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "samples": int(len(y_rows)),
        "features": int(len(vec.feature_names_)),
        "mae_train": mae,
        "vectorizer": vec,
        "model": model,
    }
    p = model_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(artifact, f)
    _load_model_cached.cache_clear()

    return {
        "trained": True,
        "path": str(p),
        "samples": artifact["samples"],
        "features": artifact["features"],
        "mae_train": artifact["mae_train"],
        "trained_at": artifact["trained_at"],
        "feedback_distribution": {
            "has_positive": bool(has_positive),
            "has_negative": bool(has_negative),
        },
    }


@lru_cache(maxsize=1)
def _load_model_cached(path_str: str) -> Optional[Dict[str, Any]]:
    p = Path(path_str)
    if not p.exists():
        return None
    try:
        with p.open("rb") as f:
            obj = pickle.load(f)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if "vectorizer" not in obj or "model" not in obj:
        return None
    return obj


def load_model_artifact() -> Optional[Dict[str, Any]]:
    return _load_model_cached(str(model_path()))


def predict_personalization_boost(
    movie: Dict[str, Any],
    mood: Optional[str],
    max_abs_boost: float = 0.75,
) -> float:
    artifact = load_model_artifact()
    if not artifact:
        return 0.0
    vec = artifact.get("vectorizer")
    model = artifact.get("model")
    if vec is None or model is None:
        return 0.0
    try:
        x = vec.transform([_feature_dict(movie, mood)])
        pred = float(model.predict(x)[0])
    except Exception:
        return 0.0
    return float(np.clip(pred, -abs(max_abs_boost), abs(max_abs_boost)))


def personalization_model_status() -> Dict[str, Any]:
    p = model_path()
    artifact = load_model_artifact()
    if artifact is None:
        events = load_feedback_events()
        labeled, y = _training_rows(events)
        return {
            "available": False,
            "model_path": str(p),
            "events_total": len(events),
            "events_labeled": len(y),
        }
    return {
        "available": True,
        "model_path": str(p),
        "trained_at": artifact.get("trained_at"),
        "samples": int(artifact.get("samples", 0)),
        "features": int(artifact.get("features", 0)),
        "mae_train": float(artifact.get("mae_train", 0.0)),
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description="Train or inspect personalization model")
    parser.add_argument("--status", action="store_true", help="Print current model status")
    parser.add_argument("--train", action="store_true", help="Train model from feedback logs")
    parser.add_argument("--min-events", type=int, default=10, help="Minimum labeled events")
    args = parser.parse_args()

    if args.status or not args.train:
        print(json.dumps(personalization_model_status(), indent=2))
    if args.train:
        print(json.dumps(train_personalization_model(min_events=args.min_events), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
