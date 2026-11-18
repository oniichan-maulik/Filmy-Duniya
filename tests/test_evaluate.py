import pandas as pd
import pytest

from src.evaluate import evaluate, with_collections

DF = pd.DataFrame({
    "movie_id": ["movie:1", "movie:2", "movie:3", "movie:4", "tv:5"],
    "title": ["A", "B", "C", "D", "E"],
})


def fixed(picks):
    """A recommender that returns preset rows per title."""
    return lambda row, k: picks.get(row, [])[:k]


def test_scores_hits_precision_and_recall_against_in_catalogue_answers():
    truth = {
        "movie:1": ["movie:2", "movie:3", "movie:999"],  # 999 isn't in the catalogue: ignored
        "movie:2": ["movie:4"],
        "movie:3": ["movie:888"],  # no answer in the catalogue: not scored
    }
    summary, per_title = evaluate(DF, fixed({0: [1, 4], 1: [0, 2]}), truth, k=2)
    assert summary["scored"] == 2
    a, b = per_title.set_index("title").loc[["A", "B"]].to_dict("records")
    assert (a["found"], a["hit"], a["precision"], a["recall"]) == (1, True, 0.5, 0.5)
    assert (b["found"], b["hit"], b["precision"], b["recall"]) == (0, False, 0.0, 0.0)
    assert summary["hit_rate"] == 0.5
    assert summary["random_precision"] == pytest.approx((2 / 4 + 1 / 4) / 2)


def test_a_title_recommending_itself_never_counts():
    truth = {"movie:1": ["movie:1", "movie:2"]}
    summary, per_title = evaluate(DF, fixed({0: [0, 2]}), truth, k=2)
    assert per_title.iloc[0]["answers"] == 1 and per_title.iloc[0]["found"] == 0


def test_films_in_the_same_series_count_as_answers():
    truth = {"movie:1": ["movie:4"], "movie:2": []}
    collections = {"movie:1": "77", "movie:2": "77", "movie:3": "", "movie:4": "88"}
    out = with_collections(truth, collections)
    assert out["movie:1"] == ["movie:4", "movie:2"]
    assert out["movie:2"] == ["movie:1"]
    assert "movie:3" not in out or out["movie:3"] == []
    assert truth["movie:1"] == ["movie:4"]  # input left untouched


def test_nothing_scorable():
    summary, per_title = evaluate(DF, fixed({}), {}, k=10)
    assert summary["scored"] == 0 and per_title.empty
