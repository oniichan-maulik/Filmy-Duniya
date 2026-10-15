from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np


EMOTION_ID_TO_LABEL = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}
EMOTION_LABELS = list(EMOTION_ID_TO_LABEL.values())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_dest_layout(dest_root: Path) -> None:
    for split in ("train", "test"):
        for label in EMOTION_LABELS:
            (dest_root / split / label).mkdir(parents=True, exist_ok=True)


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        # Avoid rewriting same file repeatedly on reruns.
        if _sha1_file(src) == _sha1_file(dst):
            return
    shutil.copy2(src, dst)


def _iter_image_files(root: Path) -> Iterable[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _find_case_insensitive_dir(parent: Path, name_lower: str) -> Path | None:
    if not parent.exists():
        return None
    for p in parent.iterdir():
        if p.is_dir() and p.name.lower() == name_lower:
            return p
    return None


def import_from_split_dirs(
    source_root: Path,
    dest_root: Path,
    limit_per_class: int = 0,
) -> Dict[str, int]:
    _ensure_dest_layout(dest_root)
    counts: Dict[str, int] = {}

    for split in ("train", "test"):
        split_src = _find_case_insensitive_dir(source_root, split)
        if split_src is None:
            raise FileNotFoundError(
                f"Missing '{split}' directory in source root: {source_root}"
            )

        for label in EMOTION_LABELS:
            src_label_dir = _find_case_insensitive_dir(split_src, label)
            if src_label_dir is None:
                # Allow sparse classes; skip gracefully.
                counts[f"{split}/{label}"] = 0
                continue
            dst_label_dir = dest_root / split / label

            n = 0
            for img in _iter_image_files(src_label_dir):
                if limit_per_class > 0 and n >= limit_per_class:
                    break
                dst = dst_label_dir / img.name
                if dst.exists():
                    stem = img.stem
                    suffix = img.suffix
                    dst = dst_label_dir / f"{stem}_{n}{suffix}"
                _copy_file(img, dst)
                n += 1
            counts[f"{split}/{label}"] = n
    return counts


def _usage_to_split(usage_raw: str) -> str | None:
    usage = (usage_raw or "").strip().lower()
    if usage == "training":
        return "train"
    if usage in {"publictest", "privatetest"}:
        return "test"
    return None


def import_from_fer_csv(
    csv_path: Path,
    dest_root: Path,
    limit_per_class: int = 0,
) -> Dict[str, int]:
    _ensure_dest_layout(dest_root)
    counts = {f"{split}/{label}": 0 for split in ("train", "test") for label in EMOTION_LABELS}

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"emotion", "pixels", "Usage"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(
                f"CSV must contain columns {sorted(required)}. Got {reader.fieldnames}"
            )

        for i, row in enumerate(reader):
            split = _usage_to_split(str(row.get("Usage", "")))
            if split is None:
                continue
            try:
                emotion_id = int(str(row["emotion"]).strip())
            except (TypeError, ValueError):
                continue
            label = EMOTION_ID_TO_LABEL.get(emotion_id)
            if label is None:
                continue

            key = f"{split}/{label}"
            if limit_per_class > 0 and counts[key] >= limit_per_class:
                continue

            pixels_text = str(row["pixels"]).strip()
            if not pixels_text:
                continue
            parts = pixels_text.split()
            if len(parts) != 48 * 48:
                continue
            try:
                arr = np.asarray(parts, dtype=np.uint8).reshape(48, 48)
            except Exception:
                continue

            out_name = f"{split}_{label}_{i:06d}.png"
            out_path = dest_root / split / label / out_name
            ok = cv2.imwrite(str(out_path), arr)
            if ok:
                counts[key] += 1
    return counts


def _main() -> int:
    root = _repo_root()
    parser = argparse.ArgumentParser(
        description="Import FER emotion dataset into data/fer2013/{train,test}/{class} layout."
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=root / "data" / "fer2013",
        help="Destination root (default: data/fer2013).",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Source root with split folders train/test and class subfolders.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="FER CSV path (columns: emotion,pixels,Usage).",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=0,
        help="Optional max samples per class per split (0 means no limit).",
    )
    args = parser.parse_args()

    if bool(args.source_root) == bool(args.csv):
        print("Error: provide exactly one of --source-root or --csv.")
        return 2

    if args.source_root:
        if not args.source_root.exists():
            print(f"Error: source root not found: {args.source_root}")
            return 2
        counts = import_from_split_dirs(
            source_root=args.source_root,
            dest_root=args.dest_root,
            limit_per_class=max(0, int(args.limit_per_class)),
        )
    else:
        assert args.csv is not None
        if not args.csv.exists():
            print(f"Error: CSV not found: {args.csv}")
            return 2
        counts = import_from_fer_csv(
            csv_path=args.csv,
            dest_root=args.dest_root,
            limit_per_class=max(0, int(args.limit_per_class)),
        )

    summary = {"dest_root": str(args.dest_root), "counts": counts}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
