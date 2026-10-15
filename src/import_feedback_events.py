from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_feedback_events_path() -> Path:
    return _repo_root() / "data" / "feedback_events.jsonl"


def _first_value(row: Dict[str, Any], keys: tuple[str, ...]) -> Optional[str]:
    for key in keys:
        if key in row:
            val = row.get(key)
            if val is None:
                continue
            text = str(val).strip()
            if text != "":
                return text
    return None


def _as_float(value: Optional[str], default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_optional_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t", "on"}


def _normalize_media_type(value: Optional[str], default_media_type: str = "movie") -> str:
    text = str(value or "").strip().lower()
    if text in {"movie", "tv", "series", "show"}:
        if text in {"series", "show"}:
            return "tv"
        return text
    return default_media_type


def _derive_signal(row: Dict[str, Any]) -> float:
    explicit_signal = _first_value(row, ("signal", "target", "label"))
    if explicit_signal is not None:
        return max(-2.0, min(2.0, _as_float(explicit_signal, 0.0)))

    signal = 0.0
    rating = _first_value(row, ("rating", "user_rating", "stars"))
    if rating is not None:
        r = _as_float(rating, 0.0)
        # Auto-handle both 0..5 and 0..10 style ratings.
        if r > 5.0:
            signal += (r - 5.0) / 5.0
        else:
            signal += (r - 2.5) / 2.5

    if _as_bool(_first_value(row, ("favorite", "favourite", "is_favorite"))):
        signal += 0.6
    if _as_bool(_first_value(row, ("liked", "is_liked", "thumbs_up"))):
        signal += 0.7
    if _as_bool(_first_value(row, ("disliked", "is_disliked", "thumbs_down"))):
        signal -= 0.8

    completion = _first_value(row, ("completion", "watch_pct", "watch_percent", "watched_pct"))
    if completion is not None:
        c = _as_float(completion, 0.0)
        if c > 1.0:
            c = c / 100.0
        if c >= 0.9:
            signal += 0.4
        elif c <= 0.2:
            signal -= 0.3

    return max(-2.0, min(2.0, signal))


def row_to_feedback_event(
    row: Dict[str, Any],
    default_media_type: str = "movie",
) -> Optional[Dict[str, Any]]:
    title = _first_value(row, ("title", "movie_title", "name", "content_title"))
    if not title:
        return None

    mood_raw = _first_value(row, ("mood", "emotion"))
    mood = str(mood_raw).strip().lower() if mood_raw else None
    if mood == "":
        mood = None

    rating_text = _first_value(row, ("rating", "user_rating", "stars"))
    rating = _as_optional_float(rating_text)
    popularity = _as_optional_float(_first_value(row, ("popularity", "views", "imdb_votes")))
    base_score = _as_float(_first_value(row, ("base_score", "score", "reco_score", "rank_score")), 0.0)
    if base_score == 0.0 and rating is not None:
        base_score = float(rating)

    signal = _derive_signal(row)
    timestamp = _first_value(row, ("timestamp", "time", "event_time", "created_at"))
    if not timestamp:
        timestamp = datetime.now(timezone.utc).isoformat()

    favorite = _as_bool(_first_value(row, ("favorite", "favourite", "is_favorite")))
    genres = _first_value(row, ("genres", "genre")) or ""
    director = _first_value(row, ("director", "creator")) or ""
    media_type = _normalize_media_type(
        _first_value(row, ("media_type", "type", "content_type")),
        default_media_type=default_media_type,
    )

    return {
        "timestamp": timestamp,
        "mood": mood,
        "rating": rating,
        "favorite": favorite,
        "signal": float(signal),
        "movie": {
            "title": str(title),
            "genres": str(genres),
            "director": str(director),
            "media_type": media_type,
            "rating": rating,
            "popularity": popularity,
            "base_score": float(base_score),
        },
        "source": "csv_import",
    }


def import_feedback_csv(
    csv_path: Path,
    out_path: Path,
    overwrite: bool = False,
    min_abs_signal: float = 0.0,
    default_media_type: str = "movie",
) -> Dict[str, Any]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "a"

    total_rows = 0
    imported = 0
    skipped_missing_title = 0
    skipped_low_signal = 0

    with csv_path.open("r", encoding="utf-8", newline="") as in_f, out_path.open(
        mode, encoding="utf-8"
    ) as out_f:
        reader = csv.DictReader(in_f)
        for row in reader:
            total_rows += 1
            event = row_to_feedback_event(row, default_media_type=default_media_type)
            if event is None:
                skipped_missing_title += 1
                continue
            signal = abs(float(event.get("signal", 0.0)))
            if signal < float(min_abs_signal):
                skipped_low_signal += 1
                continue
            out_f.write(json.dumps(event, ensure_ascii=False) + "\n")
            imported += 1

    return {
        "csv_path": str(csv_path),
        "out_path": str(out_path),
        "overwrite": bool(overwrite),
        "rows_total": int(total_rows),
        "rows_imported": int(imported),
        "rows_skipped_missing_title": int(skipped_missing_title),
        "rows_skipped_low_signal": int(skipped_low_signal),
        "min_abs_signal": float(min_abs_signal),
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description="Import bulk feedback CSV into feedback_events.jsonl")
    parser.add_argument("--csv", required=True, help="Input CSV path containing interaction events")
    parser.add_argument(
        "--out",
        default=str(default_feedback_events_path()),
        help="Output JSONL path (default: data/feedback_events.jsonl)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file instead of appending")
    parser.add_argument(
        "--min-abs-signal",
        type=float,
        default=0.0,
        help="Skip rows with |signal| below this threshold",
    )
    parser.add_argument(
        "--default-media-type",
        default="movie",
        choices=("movie", "tv"),
        help="Fallback media_type when missing in CSV",
    )
    args = parser.parse_args()

    summary = import_feedback_csv(
        csv_path=Path(args.csv),
        out_path=Path(args.out),
        overwrite=bool(args.overwrite),
        min_abs_signal=float(args.min_abs_signal),
        default_media_type=str(args.default_media_type),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
