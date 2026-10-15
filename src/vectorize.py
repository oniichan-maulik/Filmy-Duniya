import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import USE_EMBEDDINGS, EMBEDDING_MODEL
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

from typing import List, Tuple
from scipy.sparse import csr_matrix

def tfidf_vectorize(texts: List[str], max_features: int = 20000) -> Tuple[TfidfVectorizer, csr_matrix]:
    vec = TfidfVectorizer(stop_words='english', max_features=max_features)
    matrix = vec.fit_transform(texts)
    return vec, matrix

def embedding_vectorize(texts, model_name=EMBEDDING_MODEL):
    # lazy import: sentence-transformers is optional
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return model, embeddings

def vectorize(df, method="auto"):
    texts = df['soup'].tolist()
    if method == "auto":
        method = "embeddings" if USE_EMBEDDINGS else "tfidf"
    if method == "tfidf":
        vec, matrix = tfidf_vectorize(texts)
        # cache
        with open(CACHE_DIR / "tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(vec, f)
        np.save(CACHE_DIR / "tfidf_matrix.npy", matrix.toarray())
        return {"type":"tfidf", "vectorizer": vec, "matrix": matrix}
    elif method == "embeddings":
        model, emb = embedding_vectorize(texts)
        np.save(CACHE_DIR / "embeddings.npy", emb)
        return {"type":"embeddings", "model": model, "embeddings": emb}
    else:
        raise ValueError("Unknown method")
