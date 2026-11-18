from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data_loader import load_movies
from src.preprocess import build_soup, clean_text
from src.vectorize import vectorize
from src.similarity import compute_similarity_from_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Mood to genre mapping used by UI/API mood flows. Names must be TMDB genres:
# matching is a case-insensitive substring test, so "Action" also matches the
# TV genre "Action & Adventure" and "Sci-Fi" matches "Sci-Fi & Fantasy".
MOOD_TO_GENRES = {
    "happy": ["Comedy", "Romance", "Family", "Music"],
    "sad": ["Drama", "Romance", "Family"],
    "angry": ["Action", "Thriller", "Crime", "War"],
    "anxious": ["Animation", "Comedy", "Family", "Kids"],
    "excited": ["Adventure", "Science Fiction", "Sci-Fi", "Fantasy"],
    "calm": ["Documentary", "History", "Music", "Animation", "Family"],
}


def _person_tokens(cell) -> str:
    # one token per person, so two films sharing only the first name "tom" stay apart
    if not isinstance(cell, str):
        return ""
    return " ".join("".join(ch for ch in name.lower() if ch.isalnum()) for name in cell.split("|"))


# Share of the similarity that comes from TF-IDF (genres, people, plot words);
# the rest comes from sentence embeddings (what the plot means). 0.7 scored best
# in `python -m src.evaluate`: hit rate@10 55% -> 65% over TF-IDF alone.
TFIDF_WEIGHT = 0.7


def similarity_index(df: pd.DataFrame, embeddings: Optional[np.ndarray] = None, tfidf_weight: float = TFIDF_WEIGHT):
    """One row per title, aligned with df's positions; row @ row.T is how alike two titles are.

    TF-IDF over genres, people and plot words, optionally blended with unit-length
    `embeddings`. Stacking sqrt(w) * tfidf beside sqrt(1 - w) * embeddings makes a
    single dot product equal w * tfidf_cosine + (1 - w) * embedding_cosine, so
    every caller works the same with or without embeddings.

    Built in memory only; build_index() also caches large .npy files to disk.
    """
    if df.empty:
        return None
    tfidf = _tfidf_rows(df)
    if embeddings is None:
        return tfidf
    if len(embeddings) != tfidf.shape[0]:
        raise ValueError(f"{len(embeddings)} embeddings for {tfidf.shape[0]} titles")
    w = float(tfidf_weight)
    return sparse.hstack(
        [tfidf * np.sqrt(w), sparse.csr_matrix(np.asarray(embeddings, dtype=np.float64) * np.sqrt(1.0 - w))],
        format="csr",
    )


def _tfidf_rows(df: pd.DataFrame):

    def col(name: str) -> pd.Series:
        return df[name] if name in df.columns else pd.Series([""] * len(df), index=df.index)

    genres = col("genres").fillna("").astype(str).str.replace("|", " ", regex=False)
    director = col("director").map(_person_tokens)
    # films sit with films and shows with shows
    kind = "kind" + col("media_type").fillna("movie").astype(str).str.replace(r"\W", "", regex=True)
    soup = (
        # genre and director count twice: they're the strongest hints of a film's feel
        genres + " " + genres + " " + director + " " + director + " " + kind + " " + kind + " "
        + col("cast").map(_person_tokens) + " "
        + col("description").fillna("").astype(str)
    ).map(clean_text)
    return TfidfVectorizer(stop_words="english").fit_transform(soup.tolist())


def similar_rows(matrix, row: int, top_n: int = 12) -> list[int]:
    """Positions of the rows most like `row`, best first, never `row` itself."""
    sims = (matrix @ matrix[row].T).toarray().ravel()
    sims[row] = 0.0
    order = np.argsort(-sims, kind="stable")
    return [int(i) for i in order[:top_n] if sims[i] > 0]


def taste_scores(matrix, liked: list[int], disliked: list[int]) -> np.ndarray:
    """Per row: likeness to the closest liked film, minus half that to the closest disliked one."""
    score = np.zeros(matrix.shape[0])
    if liked:
        score += (matrix @ matrix[liked].T).toarray().max(axis=1)
    if disliked:
        score -= 0.5 * (matrix @ matrix[disliked].T).toarray().max(axis=1)
    return score


def build_index(method="auto"):
    df = load_movies()
    df = build_soup(df)
    vec_res = vectorize(df, method=method)
    # compute similarity matrix
    if vec_res['type'] == 'tfidf':
        sim = compute_similarity_from_matrix(vec_res['matrix'], cache_name="tfidf_sim.npy")
    else:
        sim = compute_similarity_from_matrix(vec_res['embeddings'], cache_name="emb_sim.npy")
    return {"df": df, "vectors": vec_res, "similarity": sim}

def _row_dict(df, i):
    """Row as a dict with NaN replaced by None, so it serialises as JSON null."""
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in df.iloc[i].to_dict().items()}


def recommend_by_title(title, index=None, top_n=10):
    if index is None:
        index = build_index(method="auto")
    df = index["df"]
    sim = index["similarity"]
    # find index
    matches = df[df['title'].str.lower() == title.lower()]
    if matches.empty:
        # fallback: fuzzy search
        from difflib import get_close_matches
        choices = df['title'].tolist()
        close = get_close_matches(title, choices, n=1, cutoff=0.6)
        if not close:
            raise ValueError("No matching title found")
        idx = int(df[df['title'] == close[0]].index[0])
    else:
        idx = int(matches.index[0])
    sim_scores = list(enumerate(sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top = sim_scores[1:top_n+1]
    results = []
    for i, score in top:
        row = _row_dict(df, i)
        row['score'] = float(score)
        results.append(row)
    return results


def recommend_by_genre(genre: str, index=None, top_n=10):
    """Recommend movies that match the given genre string.

    Implementation: find movies whose `genres` field contains the genre (case-insensitive),
    compute a centroid vector in the vector space for those movies, then return the
    top_n movies (from the same genre) ranked by cosine similarity to the centroid.
    """
    if index is None:
        index = build_index(method="auto")
    df = index["df"]
    vectors = index["vectors"]

    # find candidate indices where genre string appears
    mask = df['genres'].fillna('').str.lower().str.contains(genre.lower(), regex=False)
    candidate_idxs = df[mask].index.tolist()
    if not candidate_idxs:
        raise ValueError(f"No movies found matching genre '{genre}'")

    # obtain full vectors (either tfidf matrix or embeddings)
    if vectors['type'] == 'tfidf':
        full = vectors['matrix']
        # scipy sparse or array
        try:
            cand_vecs = full[candidate_idxs]
        except Exception:
            cand_vecs = full.toarray()[candidate_idxs]
    else:
        full = vectors['embeddings']
        cand_vecs = full[candidate_idxs]

    # compute centroid
    try:
        centroid = np.mean(cand_vecs, axis=0)
    except Exception:
        # fallback if sparse
        centroid = np.array(cand_vecs.todense()).mean(axis=0)

    # Ensure centroid is a 1-D numpy array (not a numpy.matrix) so sklearn
    # pairwise functions accept it without error.
    try:
        centroid = np.asarray(centroid).ravel()
    except Exception:
        centroid = np.array(centroid).reshape(-1)

    # compute similarity between centroid and all vectors
    try:
        all_vecs = full
        # if sparse matrix, cosine_similarity handles it; centroid is dense
        sims = cosine_similarity(centroid.reshape(1, -1), all_vecs).flatten()
    except Exception:
        # convert to dense and compute
        all_arr = full.toarray() if hasattr(full, 'toarray') else np.array(full)
        sims = cosine_similarity(centroid.reshape(1, -1), all_arr).flatten()

    # rank candidates by similarity
    ranked = sorted([(i, float(sims[i])) for i in candidate_idxs], key=lambda x: x[1], reverse=True)
    top = ranked[:top_n]
    results = []
    for i, score in top:
        row = _row_dict(df, i)
        row['score'] = float(score)
        results.append(row)
    return results
