"""Offline check of "More like this": how often do our picks agree with TMDB's?

For every title, TMDB's own recommendations that are also in our catalogue,
plus the other films of its series (TMDB "collection"), are treated as the
"right answers". We then score our top-k similar titles against them. TMDB
leans to popular titles, so read the score as a before/after comparison between
versions of the recommender, not as a grade.

    python -m src.evaluate              # first run fetches from TMDB (~1 min), then cached
    python -m src.evaluate --k 10 --worst 15
    python -m src.evaluate --no-series  # TMDB's recommendations only
    python -m src.evaluate --refresh    # re-fetch everything from TMDB
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import pandas as pd
import requests

from src.config import BASE_DIR, TMDB_API_KEY
from src.data_loader import load_movies
from src.recommender import similar_rows, similarity_index

CACHE_PATH = BASE_DIR / "data" / "cache" / "tmdb_recommendations.json"
COLLECTIONS_PATH = BASE_DIR / "data" / "cache" / "tmdb_collections.json"
TMDB_PAGES = 2  # 20 results a page


# --------------------------------------------------------------------------
# Fetching TMDB's answers (cached, so later runs are offline and instant)
# --------------------------------------------------------------------------
_NOT_FOUND = object()


def _get(session: requests.Session, path: str, **params):
    """GET a TMDB path; JSON on success, _NOT_FOUND on 404, None when it failed.

    Errors are never printed: their messages contain the request URL, and so
    the API key.
    """
    for attempt in range(4):
        try:
            resp = session.get(
                f"https://api.themoviedb.org/3/{path}",
                params={"api_key": TMDB_API_KEY, "language": "en-US", **params},
                timeout=10,
            )
        except requests.RequestException:
            time.sleep(1 + attempt)
            continue
        if resp.status_code == 429:  # rate limited: back off and retry
            time.sleep(2 + attempt)
            continue
        if resp.status_code == 404:
            return _NOT_FOUND
        return resp.json() if resp.status_code == 200 else None
    return None


def _split_id(movie_id: str) -> tuple[str, str]:
    kind, _, raw = movie_id.partition(":")
    return (kind, raw) if kind in ("movie", "tv") and raw.isdigit() else ("", "")


def _fetch_recommendations(movie_id: str, session: requests.Session) -> Optional[list[str]]:
    """TMDB's recommendations for "movie:123" / "tv:456", as ids in the same form.

    [] for a title TMDB doesn't know; None when a fetch failed, so it is retried
    next run instead of being cached as "no recommendations".
    """
    kind, raw = _split_id(movie_id)
    if not kind:
        return []
    out: list[str] = []
    for page in range(1, TMDB_PAGES + 1):
        data = _get(session, f"{kind}/{raw}/recommendations", page=page)
        if data is None:
            return None
        if data is _NOT_FOUND:
            return out
        # recommendations of a movie are movies, of a show are shows
        out += [f"{kind}:{r['id']}" for r in data.get("results") or [] if r.get("id") is not None]
        if page >= int(data.get("total_pages") or 1):
            break
    return out


def _fetch_collection(movie_id: str, session: requests.Session):
    """The TMDB collection (film series) a movie belongs to: its id, "" for none, None if the fetch failed."""
    kind, raw = _split_id(movie_id)
    if kind != "movie":  # only films have collections
        return ""
    data = _get(session, f"movie/{raw}")
    if data is None:
        return None
    if data is _NOT_FOUND:
        return ""
    collection = data.get("belongs_to_collection") or {}
    return str(collection.get("id") or "")


def _cached_fetch(path, movie_ids: list[str], fetch, what: str, refresh: bool, workers: int = 8) -> dict:
    cache: dict = {}
    if path.exists() and not refresh:
        cache = json.loads(path.read_text(encoding="utf-8"))
    missing = [m for m in dict.fromkeys(movie_ids) if m and m not in cache]
    if not missing:
        return cache
    if not TMDB_API_KEY:
        raise SystemExit("TMDB_API_KEY is not set (see .env.example); it's needed for the first run.")
    print(f"Fetching TMDB {what} for {len(missing)} titles...")
    with requests.Session() as session, ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda m: fetch(m, session), missing))
    failed = 0
    for movie_id, value in zip(missing, results):
        if value is None:
            failed += 1
        else:
            cache[movie_id] = value
    if failed:
        print(f"  {failed} fetches failed; they'll be retried on the next run.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache), encoding="utf-8")
    return cache


def fetch_tmdb_recommendations(movie_ids: list[str], refresh: bool = False) -> dict[str, list[str]]:
    return _cached_fetch(CACHE_PATH, movie_ids, _fetch_recommendations, "recommendations", refresh)


def fetch_tmdb_collections(movie_ids: list[str], refresh: bool = False) -> dict[str, str]:
    return _cached_fetch(COLLECTIONS_PATH, movie_ids, _fetch_collection, "collections", refresh)


def with_collections(truth: dict[str, list[str]], collections: dict[str, str]) -> dict[str, list[str]]:
    """Add the rest of each film's series to its answers.

    TMDB leaves a film's own series out of its recommendations, so without this
    Harry Potter -> the other Harry Potter films would count as a miss.
    """
    series: dict[str, list[str]] = {}
    for movie_id, collection in collections.items():
        if collection:
            series.setdefault(collection, []).append(movie_id)
    out = {m: list(recs) for m, recs in truth.items()}
    for movie_id, collection in collections.items():
        if collection:
            siblings = [s for s in series[collection] if s != movie_id]
            out[movie_id] = list(dict.fromkeys(out.get(movie_id, []) + siblings))
    return out


# --------------------------------------------------------------------------
# Scoring (pure: no network, easy to test, reusable for any recommender)
# --------------------------------------------------------------------------
def evaluate(
    df: pd.DataFrame,
    recommend: Callable[[int, int], list[int]],
    truth: dict[str, list[str]],
    k: int = 10,
) -> tuple[dict, pd.DataFrame]:
    """Score `recommend(row, k) -> rows` against TMDB's answers that are in the catalogue.

    Per title: hit (any of our k picks is an answer), precision (share of our
    picks that are answers) and recall (share of answers we found, out of at
    most k). Titles with no answer inside the catalogue can't be scored and are
    left out. The random baseline is what k random picks would score on average.
    """
    ids = df["movie_id"].astype(str).tolist()
    row_of = {m: i for i, m in enumerate(ids)}
    rows = []
    for i, movie_id in enumerate(ids):
        answers = {row_of[r] for r in truth.get(movie_id, []) if r in row_of} - {i}
        if not answers:
            continue
        picks = [p for p in recommend(i, k) if p != i][:k]
        found = len(answers.intersection(picks))
        rows.append({
            "title": df["title"].iloc[i],
            "answers": len(answers),
            "found": found,
            "hit": found > 0,
            "precision": found / k,
            "recall": found / min(len(answers), k),
            # k random picks from the other n-1 titles find answers*k/(n-1) on average
            "random_precision": len(answers) / (len(ids) - 1),
            "our_picks": [df["title"].iloc[p] for p in picks[:5]],
        })
    per_title = pd.DataFrame(rows)
    if per_title.empty:
        return {"scored": 0, "catalogue": len(ids)}, per_title
    summary = {
        "catalogue": len(ids),
        "scored": len(per_title),
        "k": k,
        "hit_rate": float(per_title["hit"].mean()),
        "precision": float(per_title["precision"].mean()),
        "recall": float(per_title["recall"].mean()),
        "random_precision": float(per_title["random_precision"].mean()),
    }
    return summary, per_title


def tfidf_recommender(df: pd.DataFrame) -> Callable[[int, int], list[int]]:
    """What the app's "More like this" shelf uses today."""
    matrix = similarity_index(df)
    titles = df["title"].astype(str).to_numpy()

    def recommend(row: int, k: int) -> list[int]:
        # like the app: skip other entries with the very same title
        return [r for r in similar_rows(matrix, row, top_n=k * 2) if titles[r] != titles[row]][:k]

    return recommend


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, default=10, help="how many picks to score per title (default 10)")
    p.add_argument("--worst", type=int, default=10, help="how many of the worst titles to list")
    p.add_argument("--refresh", action="store_true", help="re-fetch everything from TMDB")
    p.add_argument("--no-series", action="store_true", help="don't count a film's own series as right answers")
    args = p.parse_args(argv)

    df = load_movies()
    ids = df["movie_id"].astype(str).tolist()
    truth = fetch_tmdb_recommendations(ids, refresh=args.refresh)
    if not args.no_series:
        truth = with_collections(truth, fetch_tmdb_collections(ids, refresh=args.refresh))
    summary, per_title = evaluate(df, tfidf_recommender(df), truth, k=args.k)
    if not summary.get("scored"):
        print("No title has a TMDB recommendation inside the catalogue, so nothing could be scored.")
        return

    print(f"\nScored {summary['scored']} of {summary['catalogue']} titles "
          f"(the rest have no TMDB recommendation inside our catalogue).")
    print(f"  hit rate@{args.k}   {summary['hit_rate']:6.1%}   titles where any pick matches TMDB")
    print(f"  precision@{args.k}  {summary['precision']:6.1%}   share of our picks TMDB agrees with "
          f"(random picks: {summary['random_precision']:.1%})")
    print(f"  recall@{args.k}     {summary['recall']:6.1%}   share of TMDB's answers we found")
    worst = per_title[~per_title["hit"]].sort_values("answers", ascending=False).head(args.worst)
    if not worst.empty:
        print("\nMisses with the most TMDB answers in the catalogue:")
        for _, r in worst.iterrows():
            print(f"  {r['title'][:38]:38}  {r['answers']:2} answers   ours: {', '.join(r['our_picks'][:3])}")


if __name__ == "__main__":
    main()
