"""Filmy Duniya: read the viewer's mood, hand them a short list of films for it."""

import html
import json
import logging
import os
import sys
import time
import zlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Make `src` importable when Streamlit runs from another working directory.
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.config import TMDB_API_KEY as DEFAULT_TMDB_KEY
from src.data_loader import load_movies
from src.emotion_detection import detect_face_mood_backend
from src.recommender import MOOD_TO_GENRES
from src.scraper import scrape_top_n_movies
from src.user_store import record_feedback, rerank_results_for_learning

st.set_page_config(page_title="Filmy Duniya", layout="wide")
logger = logging.getLogger("filmy_duniya.ui")

GRID_SIZE = 18  # three rows of six
GRID_COLUMNS = 6
PLACEHOLDER_POSTER = "https://via.placeholder.com/500x750?text=No+poster"
WATCHLIST_PATH = repo_root / "data" / "watchlist.json"

# What the hero says for each mood: a plain read of the face, then what we picked.
MOOD_COPY = {
    "happy": ("You look happy.", "Comedies and feel-good films to keep it going."),
    "sad": ("You seem a little low.", "Gentle dramas and comfort films."),
    "calm": ("You seem calm.", "Slow, thoughtful films for a quiet evening."),
    "excited": ("You look excited.", "Big adventures, sci-fi and fantasy."),
    "angry": ("You look wound up.", "Action and thrillers to burn it off."),
    "anxious": ("You seem on edge.", "Light animation and easy comedies."),
}

try:
    _secrets_key = str(st.secrets.get("TMDB_API_KEY", "")).strip()
except Exception:
    _secrets_key = ""
# Server config only (.env or Streamlit secrets); never rendered in the page.
TMDB_API_KEY = (_secrets_key or DEFAULT_TMDB_KEY or "").strip() or None


# --------------------------------------------------------------------------
# Styles. Palette "plum-black-pink" and type "funnel", dealt by taste-skill:
# the plum-black of a dark screening room, one marquee-pink accent for actions.
# The mood tints the light falling from the top of the page (--mood).
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Funnel+Display:wght@400..700&family=Funnel+Sans:wght@400..600&display=swap');
    :root {
        --bg: #0d060b;
        --surface: #1b0e17;
        --surface-2: #26141f;
        --ink: #ffffff;
        --muted: #9a8f97;
        --line: rgba(255, 236, 246, 0.10);
        --line-strong: rgba(255, 236, 246, 0.22);
        --accent: #ff488b;
        --accent-hover: #ff6aa2;
        --accent-ink: #0d060b;
        --mood: #ff488b;
        --font-display: "Funnel Display", system-ui, sans-serif;
        --font-text: "Funnel Sans", system-ui, sans-serif;
        --s1: 8px; --s2: 16px; --s3: 24px; --s4: 32px; --s6: 48px; --s8: 64px; --s12: 96px;
        --r-md: 10px; --r-pill: 999px;
        --ease: cubic-bezier(.2, .8, .2, 1);
        color-scheme: dark;
    }
    .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden;
               clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }
    .hero-sub, .hero-source, .empty, .detail-body { text-wrap: pretty; }
    button { touch-action: manipulation; }
    /* each mood lights the room in its own colour (no violet) */
    .stApp:has(.st-key-hero-happy)   { --mood: #ffb547; }
    .stApp:has(.st-key-hero-sad)     { --mood: #4f8fff; }
    .stApp:has(.st-key-hero-calm)    { --mood: #9cc5a1; }
    .stApp:has(.st-key-hero-excited) { --mood: #ff488b; }
    .stApp:has(.st-key-hero-angry)   { --mood: #ff5a3c; }
    .stApp:has(.st-key-hero-anxious) { --mood: #c9e36b; }
    .stApp:has(.st-key-hero-none)    { --mood: #9a8f97; }

    /* Streamlit chrome */
    [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
    [data-testid="stSidebarCollapsedControl"], [data-testid="stHeaderActionElements"] {
        display: none !important;
    }
    header[data-testid="stHeader"] { background: transparent; height: 0; }
    .stApp {
        background:
            radial-gradient(ellipse 70% 42% at 50% -6%, color-mix(in srgb, var(--mood) 20%, transparent), transparent 72%),
            var(--bg);
        background-attachment: fixed;
        color: var(--ink);
        font-family: var(--font-text);
    }
    .stApp p, .stApp button, .stApp input { font-family: var(--font-text); }
    .block-container { max-width: 1240px; padding: var(--s3) var(--s4) var(--s12); }

    /* buttons: one pair. Secondary is a quiet outline, primary is the pink action */
    .stButton button {
        min-height: 44px; border-radius: var(--r-pill); padding: 0 var(--s3);
        background: transparent; color: var(--ink); border: 1px solid var(--line-strong);
        font-weight: 500;
        transition: transform .18s var(--ease), background-color .18s var(--ease), border-color .18s var(--ease);
    }
    .stButton button:hover { border-color: var(--ink); background: var(--surface); color: var(--ink); transform: translateY(-1px); }
    .stButton button:active { transform: translateY(1px) scale(.98); }
    .stButton button:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; box-shadow: none; }
    .stButton button[kind="primary"] { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
    .stButton button[kind="primary"]:hover { background: var(--accent-hover); border-color: var(--accent-hover); color: var(--accent-ink); }
    .stButton button p { font-size: 0.95rem; }

    /* top bar */
    .st-key-topbar { padding-bottom: var(--s2); border-bottom: 1px solid var(--line); }
    .brand { display: flex; align-items: center; gap: 10px; font-family: var(--font-display);
             font-size: 1.4rem; font-weight: 500; letter-spacing: -0.02em; line-height: 44px; white-space: nowrap; }
    @media (max-width: 360px) {
        .brand { font-size: 1.2rem; }
        .st-key-nav-watchlist button { padding: 0 var(--s2); }
    }
    .brand-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--accent);
                 box-shadow: 0 0 14px var(--accent); flex: none; }
    [data-testid="stTextInput"] [data-baseweb="input"] {
        border-radius: var(--r-pill); background: var(--surface); border: 1px solid transparent;
        min-height: 44px; transition: border-color .18s var(--ease);
    }
    [data-testid="stTextInput"] [data-baseweb="input"]:hover { border-color: var(--line-strong); }
    [data-testid="stTextInput"] [data-baseweb="input"]:focus-within { border-color: var(--accent) !important; }
    [data-testid="stTextInput"] input { color: var(--ink); padding-left: var(--s3); font-size: 1rem; }
    [data-testid="stTextInput"] input::placeholder { color: var(--muted); opacity: 1; }
    [data-testid="InputInstructions"] { display: none; }

    /* hero: the mood read, set large */
    [class*="st-key-hero-"] { padding: var(--s12) 0 var(--s6); }
    .hero-line {
        font-family: var(--font-display); font-weight: 400; color: var(--ink);
        font-size: clamp(3rem, 8vw, 6.5rem); line-height: 0.98; letter-spacing: -0.04em;
        margin: 0; padding: 0; max-width: 14ch; text-wrap: balance;
        animation: enter .7s var(--ease) both;
    }
    .hero-sub { color: var(--muted); font-size: 1.25rem; line-height: 1.5; margin: var(--s3) 0 var(--s4);
                max-width: 44ch; animation: enter .7s .08s var(--ease) both; }
    .hero-source { color: var(--muted); font-size: 0.92rem; margin: 0; }
    /* Streamlit styles h1/h2 itself; the display face has to win */
    .stApp .stMarkdown :is(h1, h2).hero-line,
    .stApp .stMarkdown :is(h1, h2).detail-title,
    .stApp .stMarkdown :is(h1, h2).grid-heading {
        font-family: var(--font-display) !important; font-weight: 400 !important; padding: 0 !important;
        font-size: var(--h-size) !important; line-height: var(--h-lh) !important; letter-spacing: var(--h-track) !important;
    }
    .hero-line { --h-size: clamp(3rem, 8vw, 6.5rem); --h-lh: 0.98; --h-track: -0.04em; }
    .detail-title { --h-size: clamp(2.5rem, 5vw, 4.25rem); --h-lh: 1; --h-track: -0.035em; }
    .grid-heading { --h-size: clamp(2rem, 4vw, 3rem); --h-lh: 1.05; --h-track: -0.03em; }
    @keyframes enter { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }

    /* mood pills */
    [data-testid="stButtonGroup"] { gap: var(--s1); flex-wrap: wrap; }
    [data-testid="stButtonGroup"] button {
        min-height: 44px; padding: 0 20px; border-radius: var(--r-pill);
        background: transparent; color: var(--ink); border: 1px solid var(--line-strong);
        transition: transform .18s var(--ease), background-color .18s var(--ease), border-color .18s var(--ease);
    }
    [data-testid="stButtonGroup"] button:hover { border-color: var(--ink); background: var(--surface); transform: translateY(-1px); }
    [data-testid="stButtonGroup"] button:active { transform: translateY(1px) scale(.98); }
    [data-testid="stButtonGroup"] button:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; box-shadow: none; }
    [data-testid="stButtonGroup"] button[kind="pillsActive"] {
        background: var(--mood); border-color: var(--mood); color: var(--accent-ink);
    }
    [data-testid="stButtonGroup"] button p { font-size: 1rem; }
    /* Scan again sits on the right content edge, under Watchlist */
    [class*="st-key-hero-"] [data-testid="stColumn"]:last-child [data-testid="stVerticalBlock"] { align-items: flex-end; }
    .stApp .stMarkdown h1.detail-title { margin: 0 0 var(--s2) !important; }
    .stApp .stMarkdown p.hero-sub { margin: var(--s3) 0 var(--s4) !important; }
    .st-key-topbar [data-testid="stMarkdownContainer"] { margin-bottom: 0; }

    /* poster grid: the whole card is one link, the title button stretched over it */
    .grid-heading { font-family: var(--font-display); font-weight: 400; letter-spacing: -0.03em;
                    font-size: clamp(2rem, 4vw, 3rem); line-height: 1.05; margin: var(--s8) 0 var(--s1); }
    [data-testid="stHorizontalBlock"]:has([class*="st-key-card-"]) { gap: var(--s3); margin-bottom: var(--s4); }
    /* Streamlit sizes columns for a 16px gap; re-size them for ours */
    [data-testid="stHorizontalBlock"]:has([class*="st-key-card-"]) > [data-testid="stColumn"] {
        flex: 1 1 calc(16.666% - 20px) !important; min-width: 0 !important; }
    [class*="st-key-card-"] { position: relative; gap: 10px; }
    [class*="st-key-card-"] [data-testid="stElementContainer"],
    [class*="st-key-card-"] .stButton,
    [class*="st-key-card-"] [data-testid="stTooltipHoverTarget"] { position: static; }
    .poster-frame { overflow: hidden; border-radius: var(--r-md); background: var(--surface);
                    aspect-ratio: 2 / 3; box-shadow: 0 0 0 1px var(--line);
                    transition: transform .25s var(--ease), box-shadow .25s var(--ease); }
    .poster { width: 100%; height: 100%; object-fit: cover; display: block; transition: transform .5s var(--ease); }
    [class*="st-key-card-"]:hover .poster-frame { transform: translateY(-2px); box-shadow: 0 0 0 1px var(--line-strong), 0 18px 40px -18px rgba(0,0,0,.9); }
    [class*="st-key-card-"]:hover .poster { transform: scale(1.03); }
    [class*="st-key-card-"]:has(button:focus-visible) .poster-frame { outline: 2px solid var(--accent); outline-offset: 3px; }
    [class*="st-key-card-"] .stButton button {
        background: none !important; border: 0; padding: 0; min-height: 0; border-radius: 0;
        justify-content: flex-start; text-align: left; transform: none !important; color: var(--ink);
    }
    [class*="st-key-card-"] .stButton button::after { content: ""; position: absolute; inset: 0; cursor: pointer; }
    [class*="st-key-card-"] .stButton button:focus-visible { outline: none; }
    [class*="st-key-card-"] .stButton button p { font-size: 1rem; font-weight: 500; line-height: 1.3; text-align: left; }
    .card-meta { color: var(--muted); font-size: 0.875rem; margin-top: -6px; }

    /* film page */
    .st-key-film { padding-top: var(--s6); }
    .film-poster { border-radius: var(--r-md); overflow: hidden; aspect-ratio: 2 / 3; background: var(--surface);
                   box-shadow: 0 0 0 1px var(--line), 0 40px 80px -40px color-mix(in srgb, var(--accent) 45%, transparent); }
    .film-poster img { width: 100%; height: 100%; object-fit: cover; display: block; }
    .detail-title { font-family: var(--font-display); font-weight: 400; letter-spacing: -0.035em;
                    font-size: clamp(2.5rem, 5vw, 4.25rem); line-height: 1; margin: 0 0 var(--s2); text-wrap: balance; }
    .detail-meta { color: var(--muted); font-size: 1rem; margin: 0 0 var(--s3); }
    .detail-body { max-width: 60ch; font-size: 1.125rem; line-height: 1.65; color: rgba(255,255,255,.88); margin: 0 0 var(--s2); }
    .st-key-film-actions [data-testid="stHorizontalBlock"] { gap: var(--s1); flex-wrap: wrap; }
    .st-key-film-actions [data-testid="stColumn"] { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }
    [data-testid="stVideo"], .stVideo iframe { border-radius: var(--r-md); overflow: hidden; margin-top: var(--s3); }

    .empty { color: var(--muted); font-size: 1.125rem; padding: var(--s6) 0; max-width: 48ch; }
    [data-testid="stToast"] { background: var(--surface-2); color: var(--ink); border: 1px solid var(--line); border-radius: var(--r-md); }

    /* tablets: three posters a row; phones: two */
    @media (max-width: 1024px) {
        [data-testid="stHorizontalBlock"]:has([class*="st-key-card-"]) { flex-wrap: wrap; }
        [data-testid="stHorizontalBlock"]:has([class*="st-key-card-"]) > [data-testid="stColumn"] {
            flex: 1 1 calc(33.333% - var(--s3)) !important; min-width: calc(33.333% - var(--s3)) !important; }
    }
    @media (max-width: 640px) {
        .block-container { padding: var(--s2) var(--s2) var(--s8); }
        [class*="st-key-hero-"] { padding: var(--s8) 0 var(--s4); }
        [data-testid="stHorizontalBlock"]:has([class*="st-key-card-"]) { gap: var(--s2); }
        [data-testid="stHorizontalBlock"]:has([class*="st-key-card-"]) > [data-testid="stColumn"] {
            flex: 1 1 calc(50% - var(--s2)) !important; min-width: calc(50% - var(--s2)) !important; }
        [data-testid="stHorizontalBlock"]:has([class*="st-key-card-"]) > [data-testid="stColumn"]:not(:has([class*="st-key-card-"])) { display: none; }
        .film-poster { max-width: 280px; }
        /* top bar: brand and Watchlist share a row, search goes full width below */
        .st-key-topbar [data-testid="stHorizontalBlock"] { flex-wrap: wrap; row-gap: var(--s2); }
        .st-key-topbar [data-testid="stColumn"]:nth-child(1) { flex: 1 1 0 !important; min-width: 0 !important; }
        .st-key-topbar [data-testid="stColumn"]:nth-child(3) { flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; }
        .st-key-topbar [data-testid="stColumn"]:nth-child(2) { order: 3; flex: 1 1 100% !important; min-width: 100% !important; }
        [class*="st-key-hero-"] [data-testid="stColumn"]:last-child [data-testid="stVerticalBlock"] { align-items: flex-start; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { transition: none !important; animation: none !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------
@st.cache_data(ttl=60 * 60, show_spinner=False)
def _tmdb_search(title: str) -> Optional[dict]:
    """First TMDB search hit for a title, or None."""
    if not TMDB_API_KEY or not title:
        return None
    try:
        resp = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": TMDB_API_KEY, "query": title, "page": 1},
            timeout=5,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
        return results[0] if results else None
    except Exception:
        return None


@st.cache_data(ttl=60 * 60, show_spinner=False)
def poster_url(title: str, poster_path: Optional[str], size: str = "w500") -> str:
    path = poster_path.strip() if isinstance(poster_path, str) else ""
    if not path:
        hit = _tmdb_search(title)
        path = (hit or {}).get("poster_path") or ""
    if not path:
        return PLACEHOLDER_POSTER
    return path if path.startswith("http") else f"https://image.tmdb.org/t/p/{size}{path}"


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def trailer_url(title: str) -> Optional[str]:
    """A YouTube trailer or teaser from TMDB, if one exists."""
    hit = _tmdb_search(title)
    if not hit or not hit.get("id"):
        return None
    try:
        resp = requests.get(
            f"https://api.themoviedb.org/3/movie/{hit['id']}/videos",
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
            timeout=6,
        )
        resp.raise_for_status()
        for v in resp.json().get("results") or []:
            if v.get("site") == "YouTube" and v.get("type") in {"Trailer", "Teaser"} and v.get("key"):
                return f"https://www.youtube.com/watch?v={v['key']}"
    except Exception:
        return None
    return None


def _genre_mask(df: pd.DataFrame, genres: list[str]) -> pd.Series:
    wanted = [g.lower() for g in genres]
    return df["genres"].fillna("").str.lower().apply(lambda x: any(g in x for g in wanted))


def _pick_varied(df: pd.DataFrame, n: int, seed: int, avoid: set[str]) -> pd.DataFrame:
    """Pick n rows at random, favouring higher ratings and avoiding the last batch.

    The same seed gives the same pick, so the list only changes when the seed
    changes (a new scan or mood), not on every Streamlit rerun.
    """
    if len(df) <= n:
        return df
    rating = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0).clip(0, 10).to_numpy()
    # each rating point makes a title ~2.7x likelier; picks average ~8/10
    weights = np.exp(rating - 10.0)
    shown_last = df["title"].fillna("").astype(str).isin(avoid).to_numpy()
    weights = np.where(shown_last, weights * 0.25, weights)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=n, replace=False, p=weights / weights.sum())
    return df.iloc[idx]


def mood_picks(mood: str) -> list[dict]:
    df = load_movies()
    if df.empty:
        return []
    matches = df[_genre_mask(df, MOOD_TO_GENRES.get(mood, []))]
    if matches.empty:
        return []
    key = (mood, st.session_state["pick_seed"])
    if st.session_state.get("pick_key") != key:
        # new batch: remember what the previous one showed so it is avoided
        st.session_state["prev_titles"] = st.session_state.get("shown_titles", [])
        st.session_state["pick_key"] = key
    picked = _pick_varied(
        matches,
        GRID_SIZE,
        seed=zlib.crc32(f"{key[0]}:{key[1]}".encode()),
        avoid=set(st.session_state.get("prev_titles", [])),
    )
    st.session_state["shown_titles"] = picked["title"].astype(str).tolist()
    return rerank_results_for_learning(picked.to_dict(orient="records"), mood)


def search_titles(query: str) -> list[dict]:
    df = load_movies()
    if df.empty:
        return []
    hits = df[df["title"].fillna("").str.contains(query, case=False, regex=False)].copy()
    hits["rating"] = pd.to_numeric(hits["rating"], errors="coerce")
    return hits.sort_values("rating", ascending=False).head(GRID_SIZE).to_dict(orient="records")


def _run_daily_scraper() -> None:
    """Refresh the catalogue from TMDB at most once every 24 hours."""
    if not TMDB_API_KEY:
        return
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    stamp, lock = data_dir / ".last_daily_scrape_ts", data_dir / ".daily_scrape.lock"
    now = time.time()
    try:
        if lock.exists() and now - lock.stat().st_mtime < 2 * 3600:
            return
        last = float(stamp.read_text(encoding="utf-8").strip() or "0") if stamp.exists() else 0.0
    except Exception:
        return
    if now - last < 86400:
        return
    try:
        lock.write_text(str(now), encoding="utf-8")
        scrape_top_n_movies(n=400, append=True, max_per_run=200, include_tv=True, force=False)
        stamp.write_text(str(now), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - a refresh failure must never take the UI down
        logger.warning("Daily scrape failed, retrying in an hour: %s", exc)
        try:
            stamp.write_text(str(now - 86400 + 3600), encoding="utf-8")
        except OSError:
            pass
    finally:
        lock.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Watchlist: kept on disk so it survives a page reload
# --------------------------------------------------------------------------
def _load_watchlist() -> list[dict]:
    try:
        data = json.loads(WATCHLIST_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_watchlist(items: list[dict]) -> None:
    try:
        WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = WATCHLIST_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(items, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, WATCHLIST_PATH)
    except OSError as exc:
        st.toast(f"Couldn’t save your watchlist: {exc}")


def _text(movie: dict, field: str, default: str = "") -> str:
    """A movie field as clean text. load_movies keeps gaps as NaN, which is truthy and prints "nan"."""
    value = movie.get(field)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    return str(value).strip() or default


def _movie_key(movie: dict) -> str:
    return _text(movie, "movie_id") or _text(movie, "title")


def in_watchlist(movie: dict) -> bool:
    key = _movie_key(movie)
    return any(_movie_key(m) == key for m in st.session_state["watchlist"])


def _feedback(movie: dict, **signal) -> bool:
    """Record a like, dislike or save for the ranker; a failed write is reported, never fatal."""
    try:
        record_feedback(mood=st.session_state.get("mood"), movie=movie, **signal)
        return True
    except OSError as exc:
        logger.warning("Couldn't record feedback: %s", exc)
        st.toast("Couldn’t save that to your taste profile. Check the data folder is writable.")
        return False


def toggle_watchlist(movie: dict) -> None:
    items = st.session_state["watchlist"]
    if in_watchlist(movie):
        items = [m for m in items if _movie_key(m) != _movie_key(movie)]
        st.toast(f"Removed {_text(movie, 'title', 'the film')} from your watchlist")
    else:
        items = items + [movie]
        _feedback(movie, favorite=True)
        st.toast(f"Saved {_text(movie, 'title', 'the film')} to your watchlist")
    st.session_state["watchlist"] = items
    _save_watchlist(items)


# --------------------------------------------------------------------------
# Mood
# --------------------------------------------------------------------------
def scan_mood() -> None:
    """Read the mood from the webcam; on failure, leave a note and keep the old mood."""
    try:
        details = detect_face_mood_backend()
    except Exception as exc:  # noqa: BLE001 - camera, OpenCV or model failures all fall back to the picker
        # surface the reason; failing silently once hid a broken face detector
        logger.warning("Mood scan failed: %s", exc)
        st.session_state["scan_note"] = f"The camera couldn’t be read ({exc}). Pick a mood instead."
        return
    if not details or not details.get("mood"):
        st.session_state["scan_note"] = "No face found. Face the camera in good light and scan again, or pick a mood."
        return
    st.session_state.update(
        mood=details["mood"],
        mood_pick=details["mood"],
        mood_source="camera",
        scan_note=None,
        pick_seed=st.session_state["pick_seed"] + 1,
    )


def _on_mood_pick() -> None:
    picked = st.session_state.get("mood_pick")
    st.session_state["mood"] = picked
    st.session_state["mood_source"] = "picked" if picked else None
    st.session_state["scan_note"] = None


def _on_scan_again() -> None:
    st.session_state["scan_requested"] = True


# --------------------------------------------------------------------------
# Navigation
# --------------------------------------------------------------------------
def go(view: str, movie: Optional[dict] = None) -> None:
    st.session_state["view"] = view
    st.session_state["movie"] = movie


def _clear_search() -> None:
    st.session_state["search"] = ""


# --------------------------------------------------------------------------
# Views
# --------------------------------------------------------------------------
def _rating_text(movie: dict) -> str:
    rating = pd.to_numeric(movie.get("rating"), errors="coerce")
    return f"{float(rating):.1f}" if not pd.isna(rating) and rating > 0 else ""


def poster_grid(movies: list[dict], key_prefix: str, label: Optional[str] = None) -> None:
    # a heading screen readers can jump to where sighted viewers need none
    if label:
        st.markdown(f"<h2 class='sr-only'>{html.escape(label)}</h2>", unsafe_allow_html=True)
    for start in range(0, len(movies), GRID_COLUMNS):
        row = movies[start:start + GRID_COLUMNS]
        cols = st.columns(GRID_COLUMNS)
        for col, movie in zip(cols, row):
            title = _text(movie, "title", "Untitled")
            uid = f"{key_prefix}-{zlib.crc32(_movie_key(movie).encode())}"
            with col, st.container(key=f"card-{uid}"):
                img = poster_url(title, movie.get("poster_path"))
                st.markdown(
                    f"<div class='poster-frame'><img class='poster' src='{html.escape(img, quote=True)}' "
                    f"alt='Poster for {html.escape(title, quote=True)}' loading='lazy'></div>",
                    unsafe_allow_html=True,
                )
                st.button(title, key=f"open-{uid}", on_click=go, args=("movie", movie))
                genre = _text(movie, "genres").split("|")[0]
                rating = _rating_text(movie)
                meta = f"{genre}, {rating}" if genre and rating else genre or rating
                st.markdown(f"<div class='card-meta'>{html.escape(meta)}</div>", unsafe_allow_html=True)


def render_topbar() -> None:
    with st.container(key="topbar"):
        brand, search, watch = st.columns([2, 3, 1.2], vertical_alignment="center")
        with brand:
            st.markdown(
                "<div class='brand'><span class='brand-dot' aria-hidden='true'></span>Filmy Duniya</div>",
                unsafe_allow_html=True,
            )
        with search:
            st.text_input(
                "Search films",
                placeholder="Search by title",
                autocomplete="off",
                label_visibility="collapsed",
                key="search",
                on_change=go,
                args=("home",),
            )
        with watch:
            count = len(st.session_state["watchlist"])
            st.button(
                f"Watchlist ({count})" if count else "Watchlist",
                key="nav-watchlist",
                icon=":material/bookmark:",
                use_container_width=True,
                on_click=go,
                args=("watchlist",),
            )


def render_home() -> None:
    query = str(st.session_state.get("search") or "").strip()
    if query:
        results = search_titles(query)
        st.markdown(f"<h2 class='grid-heading'>Results for “{html.escape(query)}”</h2>", unsafe_allow_html=True)
        st.button("Clear search", key="clear-search", on_click=_clear_search)
        if results:
            poster_grid(results, "search")
        else:
            st.markdown(
                "<p class='empty'>No films with that title in the catalogue. "
                "Check the spelling or try a shorter name.</p>",
                unsafe_allow_html=True,
            )
        return

    mood = st.session_state.get("mood")
    # the key carries the mood so the page can light itself in the mood's colour
    with st.container(key=f"hero-{mood if mood in MOOD_COPY else 'none'}"):
        if mood in MOOD_COPY:
            line, sub = MOOD_COPY[mood]
        else:
            line, sub = "How are you feeling?", "Let the camera read your mood, or pick one yourself."
        st.markdown(f"<h1 class='hero-line'>{line}</h1><p class='hero-sub'>{sub}</p>", unsafe_allow_html=True)

        pills, scan = st.columns([5, 1.2], vertical_alignment="center")
        # Streamlit drops a widget's state while it's off screen (film page, watchlist,
        # search), so re-seed the pills from the mood when the viewer comes back.
        if "mood_pick" not in st.session_state and mood in MOOD_COPY:
            st.session_state["mood_pick"] = mood
        with pills:
            st.pills(
                "Mood",
                options=list(MOOD_TO_GENRES.keys()),
                format_func=str.title,
                key="mood_pick",
                on_change=_on_mood_pick,
                label_visibility="collapsed",
            )
        with scan:
            st.button(
                "Scan again",
                key="scan-again",
                icon=":material/photo_camera:",
                on_click=_on_scan_again,
                help="Read your mood from the webcam again",
            )
        note = st.session_state.get("scan_note")
        if note:
            st.markdown(f"<p class='hero-source'>{html.escape(note)}</p>", unsafe_allow_html=True)
        elif st.session_state.get("mood_source") == "camera":
            st.markdown(
                "<p class='hero-source'>Read from your webcam. Pick another mood if it’s wrong.</p>",
                unsafe_allow_html=True,
            )

    if not mood:
        return
    picks = mood_picks(mood)
    if picks:
        poster_grid(picks, "mood", f"Films for a {mood} mood")
    else:
        st.markdown(
            "<p class='empty'>Nothing in the catalogue matches this mood yet. Try another one.</p>",
            unsafe_allow_html=True,
        )


def render_watchlist() -> None:
    st.button("Back", key="back-watchlist", icon=":material/arrow_back:", on_click=go, args=("home",))
    st.markdown("<h1 class='grid-heading'>Your watchlist</h1>", unsafe_allow_html=True)
    items = st.session_state["watchlist"]
    if items:
        poster_grid(items, "watchlist", "Saved films")
    else:
        st.markdown(
            "<p class='empty'>Nothing saved yet. Open a film and choose Save to watchlist.</p>",
            unsafe_allow_html=True,
        )


def render_movie(movie: dict) -> None:
    title = _text(movie, "title", "Untitled")
    st.button("Back", key="back-movie", icon=":material/arrow_back:", on_click=go, args=("home",))
    left, right = st.container(key="film").columns([1, 2.2], gap="large")
    with left:
        img = poster_url(title, movie.get("poster_path"), size="w780")
        st.markdown(
            f"<div class='film-poster'><img src='{html.escape(img, quote=True)}' "
            f"alt='Poster for {html.escape(title, quote=True)}'></div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(f"<h1 class='detail-title'>{html.escape(title)}</h1>", unsafe_allow_html=True)
        genres = _text(movie, "genres").replace("|", ", ")
        rating = _rating_text(movie)
        meta = f"{genres}. Rated {rating} out of 10" if genres and rating else genres or (f"Rated {rating} out of 10" if rating else "")
        if meta:
            st.markdown(f"<div class='detail-meta'>{html.escape(meta)}</div>", unsafe_allow_html=True)
        about = _text(movie, "description")
        if not about:
            about = str((_tmdb_search(title) or {}).get("overview") or "").strip()
        if about:
            st.markdown(f"<p class='detail-body'>{html.escape(about)}</p>", unsafe_allow_html=True)
        director = _text(movie, "director")
        if director:
            st.markdown(f"<p class='detail-meta'>Directed by {html.escape(director)}</p>", unsafe_allow_html=True)

        save, liked, disliked = st.container(key="film-actions").columns(3)
        with save:
            saved = in_watchlist(movie)
            st.button(
                "Remove from watchlist" if saved else "Save to watchlist",
                key="toggle-watchlist",
                type="primary",
                icon=":material/bookmark_remove:" if saved else ":material/bookmark_add:",
                on_click=toggle_watchlist,
                args=(movie,),
            )
        # Likes and dislikes train the personal ranker behind the mood picks.
        with liked:
            if st.button("I liked it", key="liked", icon=":material/thumb_up:"):
                if _feedback(movie, rating=5.0):
                    st.toast("Got it. You’ll see more like this.")
        with disliked:
            if st.button("Not for me", key="disliked", icon=":material/thumb_down:"):
                if _feedback(movie, rating=1.0):
                    st.toast("Got it. You’ll see less like this.")

        trailer = trailer_url(title)
        if trailer:
            st.video(trailer)


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
_run_daily_scraper()

st.session_state.setdefault("view", "home")
st.session_state.setdefault("pick_seed", int(time.time()))
if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = _load_watchlist()

# Scan once when the session starts, then only when asked. A manual pick
# stays until the viewer scans again.
if st.session_state.pop("scan_requested", False) or "scanned_once" not in st.session_state:
    st.session_state["scanned_once"] = True
    with st.spinner("Reading your mood…"):
        scan_mood()

render_topbar()
view = st.session_state["view"]
if view == "movie" and st.session_state.get("movie"):
    render_movie(st.session_state["movie"])
elif view == "watchlist":
    render_watchlist()
else:
    render_home()
