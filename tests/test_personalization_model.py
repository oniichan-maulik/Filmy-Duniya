import json

import pytest

pytest.importorskip("sklearn")

from src import personalization_model as pm
from src import user_store


def test_train_personalization_model_from_feedback(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_repo_root", lambda: tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    events_path = data_dir / "feedback_events.jsonl"

    events = [
        {
            "mood": "happy",
            "signal": 1.0,
            "movie": {
                "title": "A",
                "genres": "Comedy|Romance",
                "director": "Dir1",
                "media_type": "movie",
                "rating": 8.0,
                "popularity": 30.0,
                "base_score": 0.7,
            },
        },
        {
            "mood": "sad",
            "signal": -1.0,
            "movie": {
                "title": "B",
                "genres": "Horror|Thriller",
                "director": "Dir2",
                "media_type": "movie",
                "rating": 3.0,
                "popularity": 5.0,
                "base_score": 0.2,
            },
        },
        {
            "mood": "happy",
            "signal": 0.8,
            "movie": {
                "title": "C",
                "genres": "Comedy",
                "director": "Dir1",
                "media_type": "movie",
                "rating": 7.0,
                "popularity": 20.0,
                "base_score": 0.5,
            },
        },
    ]
    with events_path.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")

    out = pm.train_personalization_model(min_events=2)
    assert out["trained"] is True
    assert (data_dir / "personalization_model.pkl").exists()

    boost = pm.predict_personalization_boost(
        {"genres": "Comedy", "director": "Dir1", "rating": 7.5, "popularity": 18.0, "score": 0.4},
        "happy",
    )
    assert isinstance(boost, float)


def test_record_feedback_writes_profile_and_event(tmp_path, monkeypatch):
    monkeypatch.setattr(user_store, "_repo_root", lambda: tmp_path)
    movie = {
        "title": "Demo Movie",
        "genres": "Comedy|Family",
        "director": "Demo Director",
        "score": 0.6,
        "rating": 7.1,
    }
    user_store.record_feedback(mood="happy", movie=movie, rating=4.5, favorite=False)

    profile = user_store.load_user_profile()
    assert profile["feedback_events"] == 1
    assert "happy" in profile["mood_counts"]

    events_path = tmp_path / "data" / "feedback_events.jsonl"
    assert events_path.exists()
    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

