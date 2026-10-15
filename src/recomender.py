import numpy as np
import pandas as pd
from src.data_loader import load_movies
from src.preprocess import build_soup
from src.vectorize import vectorize
from src.similarity import compute_similarity_from_matrix, load_similarity
from src.config import USE_EMBEDDINGS
from sklearn.metrics.pairwise import cosine_similarity

# Mood to genre mapping used by UI/API mood flows.
MOOD_TO_GENRES = {
    "happy": ["Comedy", "Romance", "Family"],
    "sad": ["Drama", "Feel-Good"],
    "angry": ["Action", "Thriller"],
    "anxious": ["Animation", "Comedy"],
    "excited": ["Adventure", "Sci-Fi"],
    "calm": ["Documentary", "Slice of Life"],
}


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
        row = df.iloc[i].to_dict()
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
    mask = df['genres'].fillna('').str.lower().str.contains(genre.lower())
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
        row = df.iloc[i].to_dict()
        row['score'] = float(score)
        results.append(row)
    return results


def genres_for_mood(mood: str) -> list[str]:
    if mood is None or not str(mood).strip():
        raise ValueError("Mood must be a non-empty string")
    mood_key = str(mood).strip().lower()
    if mood_key not in MOOD_TO_GENRES:
        raise ValueError(
            f"Unknown mood '{mood}'. Available moods: {', '.join(MOOD_TO_GENRES.keys())}"
        )
    return list(MOOD_TO_GENRES[mood_key])


def recommend_by_mood(mood: str, index=None, top_n=10):
    """Recommend movies using mood -> genres mapping."""
    if index is None:
        index = build_index(method="auto")
    df = index["df"]
    vectors = index["vectors"]
    mapped_genres = genres_for_mood(mood)
    mapped = [g.lower() for g in mapped_genres]

    mask = df["genres"].fillna("").str.lower().apply(lambda x: any(g in x for g in mapped))
    candidate_idxs = df[mask].index.tolist()
    if not candidate_idxs:
        raise ValueError(f"No movies found matching mood '{mood}'")

    if vectors["type"] == "tfidf":
        full = vectors["matrix"]
        try:
            cand_vecs = full[candidate_idxs]
        except Exception:
            cand_vecs = full.toarray()[candidate_idxs]
    else:
        full = vectors["embeddings"]
        cand_vecs = full[candidate_idxs]

    try:
        centroid = np.mean(cand_vecs, axis=0)
    except Exception:
        centroid = np.array(cand_vecs.todense()).mean(axis=0)
    centroid = np.asarray(centroid).ravel()

    try:
        sims = cosine_similarity(centroid.reshape(1, -1), full).flatten()
    except Exception:
        full_arr = full.toarray() if hasattr(full, "toarray") else np.array(full)
        sims = cosine_similarity(centroid.reshape(1, -1), full_arr).flatten()

    ranked = sorted([(i, float(sims[i])) for i in candidate_idxs], key=lambda x: x[1], reverse=True)
    top = ranked[:top_n]
    results = []
    for i, score in top:
        row = df.iloc[i].to_dict()
        row["score"] = float(score)
        results.append(row)
    return results
