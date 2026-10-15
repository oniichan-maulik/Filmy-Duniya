import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import pickle

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

def compute_similarity_from_matrix(matrix, method="dense", cache_name="sim_matrix.npy"):
    # matrix: sparse or dense
    if hasattr(matrix, "toarray"):
        arr = matrix.toarray()
    else:
        arr = matrix
    sim = cosine_similarity(arr)
    np.save(CACHE / cache_name, sim)
    return sim

def load_similarity(cache_name="sim_matrix.npy"):
    p = CACHE / cache_name
    if p.exists():
        return np.load(p)
    return None
