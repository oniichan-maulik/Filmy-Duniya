"""TMDB scraper helpers.

This module provides utilities to fetch popular movies and details from TMDB and
persist them to `data/movies.csv`. It includes a CLI so you can run a one-off
scrape or run periodic updates.
"""

from pathlib import Path
import time
import logging
import argparse
from typing import List, Dict, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import TMDB_API_KEY, DATA_PATH
from src.data__loader import append_movie

TMDB_BASE = "https://api.themoviedb.org/3"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _session_with_retries(total_retries: int = 3, backoff_factor: float = 0.3) -> requests.Session:
    s = requests.Session()
    retries = Retry(total=total_retries, backoff_factor=backoff_factor,
                    status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET", "POST"])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s


def fetch_popular(page: int = 1, content_type: str = "movie", session: Optional[requests.Session] = None) -> Dict:
    if not TMDB_API_KEY or TMDB_API_KEY == "YOUR_TMDB_API_KEY_HERE":
        raise RuntimeError("TMDB_API_KEY is not set. Please set the TMDB_API_KEY environment variable.")
    if content_type not in {"movie", "tv"}:
        raise RuntimeError("content_type must be 'movie' or 'tv'")
    session = session or _session_with_retries()
    url = f"{TMDB_BASE}/{content_type}/popular"
    params = {"api_key": TMDB_API_KEY, "language": "en-US", "page": page}
    r = session.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def fetch_details(item_id: int, content_type: str = "movie", session: Optional[requests.Session] = None) -> Dict:
    if not TMDB_API_KEY or TMDB_API_KEY == "YOUR_TMDB_API_KEY_HERE":
        raise RuntimeError("TMDB_API_KEY is not set. Please set the TMDB_API_KEY environment variable.")
    if content_type not in {"movie", "tv"}:
        raise RuntimeError("content_type must be 'movie' or 'tv'")
    session = session or _session_with_retries()
    url = f"{TMDB_BASE}/{content_type}/{item_id}"
    params = {"api_key": TMDB_API_KEY, "language": "en-US", "append_to_response": "credits"}
    r = session.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def _normalize_detail(d: Dict, content_type: str = "movie") -> Dict:
    genres = [g.get('name') for g in d.get('genres', []) if g.get('name')]
    director = ""
    cast = []
    credits = d.get('credits', {}) or {}
    if content_type == "movie":
        for c in credits.get('crew', []):
            if c.get('job') == 'Director':
                director = c.get('name') or ""
                break
    else:
        created_by = d.get("created_by") or []
        if created_by:
            director = created_by[0].get("name") or ""
        if not director:
            for c in credits.get('crew', []):
                if c.get('job') in {"Executive Producer", "Creator", "Director"}:
                    director = c.get('name') or ""
                    if director:
                        break
    for c in credits.get('cast', [])[:10]:
        if c.get('name'):
            cast.append(c.get('name'))

    raw_id = d.get("id")
    normalized_id = f"{content_type}:{raw_id}" if raw_id is not None else ""
    title = d.get("title") or d.get("name") or d.get("original_name") or ""
    return {
        "movie_id": normalized_id,
        "media_type": content_type,
        "title": title,
        "genres": "|".join(genres),
        "cast": "|".join(cast),
        "director": director or "",
        "description": d.get("overview") or "",
        "rating": d.get("vote_average", 0),
        "popularity": d.get("popularity", 0),
        "poster_path": d.get("poster_path", ""),
    }


def _collect_popular_items(
    n: int,
    content_type: str,
    session: requests.Session,
) -> List[Tuple[str, Dict]]:
    items: List[Tuple[str, Dict]] = []
    page = 1
    while len(items) < n:
        data = fetch_popular(page=page, content_type=content_type, session=session)
        results = data.get("results") or []
        if not results:
            break
        for item in results:
            items.append((content_type, item))
            if len(items) >= n:
                break
        page += 1
        time.sleep(0.25)
    return items


def scrape_top_n_movies(
    n: int = 500,
    out_path: Path = DATA_PATH,
    session: Optional[requests.Session] = None,
    append: bool = False,
    max_per_run: int = 0,
    force: bool = False,
    source: str = "tmdb",
    include_tv: bool = False,
) -> List[Dict]:
    """Scrape the top `n` popular movies and write them to CSV.

    The default `source` is "tmdb". In future this parameter can be used to
    switch scraping sources (IMDb, RottenTomatoes, OMDb) if implemented. For
    now only TMDB is supported.

    If `append` is True, each movie will be appended using `append_movie` which
    safely deduplicates (by movie_id or title). Otherwise the function will
    write all fetched movies to the CSV (overwriting).
    Returns the list of normalized movie dicts.
    """
    if source != "tmdb":
        raise RuntimeError(f"Unsupported source: {source}. Only 'tmdb' is supported currently.")
    session = session or _session_with_retries()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # safety check: if file already has >= n movies and not forced, skip work
    if not force and out_path.exists():
        try:
            import csv
            with open(out_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_count = sum(1 for _ in reader)
            if existing_count >= n:
                logger.info("Existing CSV at %s already has %d entries (>= %d). Skipping (use --force to override)", out_path, existing_count, n)
                return []
        except Exception:
            # if counting fails, continue
            pass

    if include_tv:
        per_type = max(1, n // 2)
        movie_items = _collect_popular_items(per_type, "movie", session)
        tv_items = _collect_popular_items(per_type, "tv", session)
        items = movie_items + tv_items
    else:
        items = _collect_popular_items(n, "movie", session)

    rows = []
    for content_type, item in items:
        try:
            d = fetch_details(item['id'], content_type=content_type, session=session)
            row = _normalize_detail(d, content_type=content_type)
            rows.append(row)
        except Exception as e:
            logger.warning("Failed to fetch details for %s id %s: %s", content_type, item.get('id'), e)
        # be polite with API rate limits
        time.sleep(0.2)

    # persist
    if append:
        # dedupe against existing file to avoid unnecessary API work
        existing_titles = set()
        existing_ids = set()
        if out_path.exists():
            import csv
            with open(out_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    t = (r.get('title') or '').strip()
                    if t:
                        existing_titles.add(t.lower())
                    mid = r.get('movie_id')
                    if mid:
                        existing_ids.add(str(mid))

    # filter rows that are new
        new_rows = []
        for r in rows:
            title = (r.get('title') or '').strip()
            mid = r.get('movie_id')
            if mid and str(mid) in existing_ids:
                continue
            if title and title.lower() in existing_titles:
                continue
            new_rows.append(r)
        # limit per run if requested
        if max_per_run and max_per_run > 0:
            new_rows = new_rows[:max_per_run]

        if not new_rows:
            logger.info("No new movies to append to %s", out_path)
        else:
            try:
                from src.data__loader import append_bulk
                append_bulk(new_rows, path=out_path)
                logger.info("Appended %d new movies to %s", len(new_rows), out_path)
            except Exception as e:
                logger.warning("append_bulk failed, falling back to per-row append: %s", e)
                for r in new_rows:
                    try:
                        append_movie(r, path=out_path)
                    except Exception as e:
                        logger.warning("Failed to append movie %s: %s", r.get('title'), e)
    else:
        # write full CSV (overwrite)
        import csv
        keys = ["movie_id", "media_type", "title", "genres", "cast", "director", "description", "rating", "popularity", "poster_path"]
        with open(out_path, "w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Saved %d movies to %s", len(rows), out_path)

    return rows


def run_periodic(
    n: int = 100,
    interval_seconds: int = 3600,
    append: bool = True,
    max_per_run: int = 0,
    force: bool = False,
    include_tv: bool = False,
):
    """Run scraper periodically every `interval_seconds` seconds.

    This will continuously fetch the top `n` movies and append them to CSV.
    Use Ctrl-C to stop.
    """
    session = _session_with_retries()
    logger.info("Starting periodic scraper: n=%d interval=%ds append=%s", n, interval_seconds, append)
    try:
        while True:
            logger.info("Scraping top %d movies...", n)
            scrape_top_n_movies(
                n=n,
                session=session,
                append=append,
                max_per_run=max_per_run,
                force=force,
                include_tv=include_tv,
            )
            logger.info("Sleeping for %d seconds...", interval_seconds)
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        logger.info("Periodic scraper stopped by user")


def _cli():
    p = argparse.ArgumentParser(description="Scrape popular movies from TMDB and save to CSV")
    p.add_argument("--n", type=int, default=200, help="Total number of movies to fetch")
    p.add_argument("--out", type=str, default=str(DATA_PATH), help="Output CSV path")
    p.add_argument("--append", action="store_true", help="Append to existing CSV (dedupe using movie_id/title)")
    p.add_argument("--max-per-run", type=int, default=0, help="If >0, append at most this many new movies per run")
    p.add_argument("--force", action="store_true", help="Force scraping even if CSV already contains >= n movies")
    p.add_argument("--include-tv", action="store_true", help="Also scrape TV series/shows")
    p.add_argument("--interval", type=int, default=0, help="If >0, run periodically every INTERVAL seconds")
    args = p.parse_args()

    if args.interval and args.interval > 0:
        run_periodic(
            n=args.n,
            interval_seconds=args.interval,
            append=args.append,
            max_per_run=args.max_per_run,
            force=args.force,
            include_tv=args.include_tv,
        )
    else:
        scrape_top_n_movies(
            n=args.n,
            out_path=Path(args.out),
            append=args.append,
            max_per_run=args.max_per_run,
            force=args.force,
            include_tv=args.include_tv,
        )


if __name__ == "__main__":
    _cli()
