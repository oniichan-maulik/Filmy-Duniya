import numpy as np
import pandas as pd
import pytest

from src.recommender import similar_rows, similarity_index, taste_scores

DF = pd.DataFrame(
    [
        {"title": "Space One", "media_type": "movie", "genres": "Science Fiction|Adventure", "cast": "Ann Lee", "director": "Cara Nolan", "description": "astronauts cross a wormhole"},
        {"title": "Space Two", "media_type": "movie", "genres": "Science Fiction|Drama", "cast": "Ann Lee", "director": "Cara Nolan", "description": "a wormhole mission to save earth"},
        {"title": "Ghost House", "media_type": "movie", "genres": "Horror", "cast": "Bo Park", "director": "Dee Wan", "description": "a family haunted in their new house"},
        {"title": "Ghost House 2", "media_type": "movie", "genres": "Horror|Thriller", "cast": "Bo Park", "director": "Dee Wan", "description": "the haunted house returns"},
        {"title": "Office Laughs", "media_type": "tv", "genres": "Comedy", "cast": "Tom Hale", "director": None, "description": None},
    ]
)


def test_similar_rows_finds_the_sequel_and_never_itself():
    m = similarity_index(DF)
    assert similar_rows(m, 0, top_n=1) == [1]
    assert similar_rows(m, 2, top_n=1) == [3]
    assert 0 not in similar_rows(m, 0)


def test_people_with_a_shared_first_name_are_not_linked():
    df = pd.DataFrame(
        [
            {"title": "A", "genres": "", "cast": "Tom Hanks", "director": "", "description": ""},
            {"title": "B", "genres": "", "cast": "Tom Hardy", "director": "", "description": ""},
            {"title": "C", "genres": "", "cast": "Ann Lee", "director": "", "description": ""},
        ]
    )
    m = similarity_index(df)
    sim = (m @ m.T).toarray()
    # Tom Hardy is no closer to Tom Hanks than Ann Lee is
    assert np.isclose(sim[0, 1], sim[0, 2])


def test_taste_favours_films_like_your_likes_and_away_from_dislikes():
    m = similarity_index(DF)
    taste = taste_scores(m, liked=[0], disliked=[2])
    assert taste[1] > 0 > taste[3]
    assert np.argmax(np.delete(taste, 0)) == 0  # Space Two is the closest unrated film


def test_blend_with_embeddings_is_a_weighted_sum_of_both_similarities():
    tfidf = similarity_index(DF)
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(DF), 8))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    blended = similarity_index(DF, embeddings=emb, tfidf_weight=0.7)
    expected = 0.7 * (tfidf @ tfidf.T).toarray() + 0.3 * emb @ emb.T
    assert np.allclose((blended @ blended.T).toarray(), expected)
    # every row stays unit length, so a title is still most like itself
    assert np.allclose((blended @ blended.T).diagonal(), 1.0)


def test_embeddings_can_move_the_nearest_neighbour():
    # embeddings say Office Laughs is about the same thing as Space One
    emb = np.eye(len(DF))
    emb[4] = emb[0]
    assert similar_rows(similarity_index(DF, embeddings=emb, tfidf_weight=0.3), 0, top_n=1) == [4]


def test_wrong_number_of_embeddings_is_an_error():
    with pytest.raises(ValueError):
        similarity_index(DF, embeddings=np.ones((2, 3)))


def test_no_likes_means_no_taste():
    assert not taste_scores(similarity_index(DF), [], []).any()
    assert similarity_index(DF.iloc[0:0]) is None
