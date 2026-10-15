from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def default_custom_emotions_path() -> Path:
    return _repo_root() / "data" / "custom_emotions.jsonl"

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
MOODS = ["happy", "sad", "angry", "anxious", "excited", "calm"]

EMOTION_TO_MOOD = {
    "Happy": "happy",
    "Sad": "sad",
    "Angry": "angry",
    "Disgust": "angry",
    "Fear": "anxious",
    "Surprise": "excited",
    "Neutral": "calm",
}

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

def _as_float(value: Optional[str], default: float = 0.5) -> float:
    """Convert string to float with default."""
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

def _normalize_mood(mood_raw: Optional[str]) -> Optional[str]:
    if mood_raw is None:
        return None
    mood = str(mood_raw).strip().lower()
    if mood in MOODS:
        return mood
    # Fuzzy match to closest emotion
    for emo in EMOTIONS:
        if emo.lower() in mood or mood in emo.lower():
            return EMOTION_TO_MOOD.get(emo)
    return None

def _normalize_color_hex(color_raw: Optional[str]) -> str:
    if color_raw is None:
        return "#808080"  # Gray default
    color = str(color_raw).strip().lower()
    if color.startswith("#"):
        return color[:7]  # Truncate to #RRGGBB
    return f"#{color[:6]}"

def row_to_custom_emotion(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    name = _first_value(row, ("emotion_name", "emotion", "name", "label"))
    if not name:
        return None

    mood_raw = _first_value(row, ("mood", "mood_category", "category"))
    mood = _normalize_mood(mood_raw)

    intensity = _as_optional_float(_first_value(row, ("intensity", "strength", "level")))
    if intensity is None:
        intensity = 0.5  # Default medium

    description = _first_value(row, ("description", "desc", "notes")) or ""

    color_hex = _normalize_color_hex(_first_value(row, ("color", "color_hex", "hex")))

    user_id = _first_value(row, ("user_id", "user", "owner")) or "default"

    timestamp = _first_value(row, ("timestamp", "time", "created_at"))
    if not timestamp:
        timestamp = datetime.now(timezone.utc).isoformat()

    # Metadata
    tags = _first_value(row, ("tags", "keywords")) or ""
    tags = [t.strip() for t in tags.split(",") if t.strip()]

    return {
        "timestamp": timestamp,
        "emotion_name": str(name),
        "mood": mood,
        "intensity": float(intensity),
        "description": str(description),
        "color_hex": color_hex,
        "user_id": str(user_id),
        "tags": tags,
        "source": "csv_import",
    }

def import_custom_emotions_csv(
    csv_path: Path,
    out_path: Path,
    overwrite: bool = False,
    min_intensity: float = 0.0,
) -> Dict[str, Any]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "a"

    total_rows = 0
    imported = 0
    skipped_missing_name = 0
    skipped_low_intensity = 0

    with csv_path.open("r", encoding="utf-8", newline="") as in_f, out_path.open(
        mode, encoding="utf-8"
    ) as out_f:
        reader = csv.DictReader(in_f)
        for row in reader:
            total_rows += 1
            event = row_to_custom_emotion(row)
            if event is None:
                skipped_missing_name += 1
                continue
            intensity = float(event.get("intensity", 0.0))
            if intensity < float(min_intensity):
                skipped_low_intensity += 1
                continue
            out_f.write(json.dumps(event, ensure_ascii=False) + "\n")
            imported += 1

    return {
        "csv_path": str(csv_path),
        "out_path": str(out_path),
        "overwrite": bool(overwrite),
        "rows_total": int(total_rows),
        "rows_imported": int(imported),
        "rows_skipped_missing_name": int(skipped_missing_name),
        "rows_skipped_low_intensity": int(skipped_low_intensity),
        "min_intensity": float(min_intensity),
    }

def _main() -> int:
    parser = argparse.ArgumentParser(description="Import custom emotions CSV into custom_emotions.jsonl")
    parser.add_argument("--csv", required=True, help="Input CSV path with custom emotion data")
    parser.add_argument(
        "--out",
        default=str(default_custom_emotions_path()),
        help="Output JSONL path (default: data/custom_emotions.jsonl)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file instead of appending")
    parser.add_argument(
        "--min-intensity",
        type=float,
        default=0.0,
        help="Skip rows with intensity below this threshold",
    )
    args = parser.parse_args()

    summary = import_custom_emotions_csv(
        csv_path=Path(args.csv),
        out_path=Path(args.out),
        overwrite=bool(args.overwrite),
        min_intensity=float(args.min_intensity),
    )
    print(json.dumps(summary, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(_main())

