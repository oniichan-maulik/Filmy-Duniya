"""Sentence embeddings of each title's "what it's about" text, cached on disk.

Each text is cached under a hash of itself, so the model (and PyTorch, which
takes several seconds to load) only runs for titles it hasn't seen. With a warm
cache, nothing is loaded at all. Any failure returns None, and callers fall
back to TF-IDF alone.
"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.config import BASE_DIR, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

CACHE_DIR = BASE_DIR / "data" / "cache"


def embedding_texts(df: pd.DataFrame) -> list[str]:
    """Title, genres and plot: the mix that scored best in `python -m src.evaluate`."""
    def col(name: str) -> pd.Series:
        return df[name].fillna("").astype(str) if name in df.columns else pd.Series([""] * len(df), index=df.index)

    return (col("title") + ". " + col("genres").str.replace("|", ", ", regex=False) + ". " + col("description")).tolist()


def _cache_path(model_name: str) -> Path:
    return CACHE_DIR / f"embeddings_{re.sub(r'[^A-Za-z0-9_.-]', '_', model_name)}.npz"


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _sentence_transformer_encoder(model_name: str) -> Callable[[list[str]], np.ndarray]:
    from sentence_transformers import SentenceTransformer  # heavy: imported only when needed

    model = SentenceTransformer(model_name)
    return lambda texts: model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)


def embed_texts(
    texts: list[str],
    model_name: str = EMBEDDING_MODEL,
    encoder: Optional[Callable[[list[str]], np.ndarray]] = None,
    cache_path: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """Unit-length embeddings, one row per text, or None if they can't be made."""
    if not texts:
        return None
    path = cache_path or _cache_path(model_name)
    cached: dict[str, np.ndarray] = {}
    try:
        if path.exists():
            with np.load(path) as data:
                cached = dict(zip(data["keys"].tolist(), data["vectors"]))
    except Exception:  # noqa: BLE001 - a broken cache is rebuilt, not fatal
        logger.warning("Ignoring unreadable embedding cache %s", path)
        cached = {}

    keys = [_hash(t) for t in texts]
    missing = list(dict.fromkeys(k for k in keys if k not in cached))
    if missing:
        text_of = dict(zip(keys, texts))
        try:
            encode = encoder or _sentence_transformer_encoder(model_name)
            vectors = np.asarray(encode([text_of[k] for k in missing]), dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 - no model, no network, no torch: fall back
            logger.warning("Embeddings unavailable (%s); using TF-IDF only", exc)
            return None
        cached.update(zip(missing, vectors))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, keys=np.array(list(cached)), vectors=np.stack(list(cached.values())))
            tmp.replace(path)
        except OSError as exc:
            logger.warning("Couldn't save the embedding cache: %s", exc)
    out = np.stack([cached[k] for k in keys]).astype(np.float32)
    # unit length, whatever the encoder returned, so dot products are cosines
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1.0, norms)
