import numpy as np
import pandas as pd

from src.embeddings import embed_texts, embedding_texts


class FakeEncoder:
    """Stands in for the sentence-transformers model; counts what it was asked to encode."""

    def __init__(self):
        self.seen: list[str] = []

    def __call__(self, texts):
        self.seen += texts
        return np.array([[len(t), 1.0, 0.0] for t in texts])


def test_embeddings_are_unit_length_and_cached_per_text(tmp_path):
    cache = tmp_path / "emb.npz"
    enc = FakeEncoder()
    first = embed_texts(["a", "bb"], encoder=enc, cache_path=cache)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)
    assert enc.seen == ["a", "bb"]

    # a later call only encodes the new text, and keeps the caller's order
    again = embed_texts(["bb", "ccc", "a"], encoder=enc, cache_path=cache)
    assert enc.seen == ["a", "bb", "ccc"]
    assert np.allclose(again[0], first[1]) and np.allclose(again[2], first[0])


def test_warm_cache_never_loads_the_model(tmp_path):
    cache = tmp_path / "emb.npz"
    embed_texts(["a"], encoder=FakeEncoder(), cache_path=cache)

    def no_model(texts):
        raise AssertionError("the model should not be needed")

    assert embed_texts(["a"], encoder=no_model, cache_path=cache) is not None


def test_failures_fall_back_to_none(tmp_path):
    def broken(texts):
        raise RuntimeError("no network")

    assert embed_texts(["a"], encoder=broken, cache_path=tmp_path / "emb.npz") is None
    assert embed_texts([], encoder=FakeEncoder(), cache_path=tmp_path / "emb.npz") is None
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not a cache")
    assert embed_texts(["a"], encoder=FakeEncoder(), cache_path=bad) is not None  # rebuilt


def test_embedding_text_has_title_genres_and_plot():
    df = pd.DataFrame([{"title": "Up", "genres": "Animation|Family", "description": "A house flies."}])
    assert embedding_texts(df) == ["Up. Animation, Family. A house flies."]
