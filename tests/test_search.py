import pandas as pd
import pytest

from src.search import normalize, search_catalogue


@pytest.fixture
def df():
    return pd.DataFrame(
        [
            {"title": "Spider-Man: No Way Home", "genres": "Action|Adventure", "cast": "Tom Holland|Zendaya", "director": "Jon Watts", "rating": 8.0},
            {"title": "Spider-Man", "genres": "Action", "cast": "Tobey Maguire", "director": "Sam Raimi", "rating": 7.3},
            {"title": "Interstellar", "genres": "Science Fiction|Drama", "cast": "Matthew McConaughey", "director": "Christopher Nolan", "rating": 8.6},
            {"title": "Insidious: Out of the Further", "genres": "Horror|Thriller", "cast": "Lin Shaye", "director": "Jacob Chase", "rating": 6.1},
            {"title": "X", "genres": "Horror", "cast": "Mia Goth", "director": "Ti West", "rating": 6.6},
            {"title": "Amélie", "genres": "Comedy|Romance", "cast": "Audrey Tautou", "director": "Jean-Pierre Jeunet", "rating": 7.9},
        ]
    )


def titles(df, query):
    hits, _ = search_catalogue(df, query)
    return hits["title"].tolist()


def test_normalize_drops_case_accents_and_punctuation():
    assert normalize("Spider-Man: No Way Home") == "spider man no way home"
    assert normalize("Amélie") == "amelie"
    assert normalize(None) == ""


@pytest.mark.parametrize("query", ["spiderman", "spider man", "SPIDER-MAN", "spidr man"])
def test_spelling_and_punctuation_variants_find_the_title(df, query):
    assert titles(df, query)[:2] == ["Spider-Man", "Spider-Man: No Way Home"]


def test_exact_title_ranks_before_longer_titles(df):
    assert titles(df, "spider-man")[0] == "Spider-Man"


@pytest.mark.parametrize("query,expected", [("intersteller", "Interstellar"), ("insidius", "Insidious: Out of the Further"), ("amelie", "Amélie")])
def test_typos_and_accents(df, query, expected):
    assert titles(df, query)[0] == expected


def test_people_and_genres_are_searched(df):
    assert titles(df, "zendaya") == ["Spider-Man: No Way Home"]
    assert titles(df, "nolen") == ["Interstellar"]
    assert titles(df, "sci fi") == ["Interstellar"]
    assert set(titles(df, "horror")) == {"Insidious: Out of the Further", "X"}


def test_short_title_is_not_found_inside_a_longer_query(df):
    assert "X" not in titles(df, "xylophone")


def test_no_match_offers_a_suggestion_only_when_close(df):
    hits, suggestion = search_catalogue(df, "interstellr odyssey part two")
    assert hits.empty and suggestion == "Interstellar"
    hits, suggestion = search_catalogue(df, "qqqqqq")
    assert hits.empty and suggestion is None


def test_empty_query_or_catalogue(df):
    assert search_catalogue(df, "   ")[0].empty
    assert search_catalogue(df.iloc[0:0], "spider")[0].empty
