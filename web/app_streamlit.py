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

from src.config import BLEND_EMBEDDINGS, TMDB_API_KEY as DEFAULT_TMDB_KEY
from src.embeddings import embed_texts, embedding_texts
from src.data_loader import load_movies
from src.emotion_detection import detect_emotion_from_bytes
from src.recommender import MOOD_TO_GENRES, similar_rows, similarity_index, taste_scores
from src.scraper import scrape_top_n_movies
from src.search import search_catalogue
from src.user_store import load_user_profile, record_feedback, rerank_results_for_learning, title_verdict
from mood_scan import mood_scan_button  # web/ is on sys.path: Streamlit adds the script's folder

st.set_page_config(page_title="Filmy Duniya", layout="wide")
logger = logging.getLogger("filmy_duniya.ui")

GRID_SIZE = 18  # three rows of six
GRID_COLUMNS = 6
SHELF_SIZE = 15
GENRE_PAGE_STEP = 24  # four rows of six per "Show more"
# How far likes and dislikes steer the mood picks. Taste is scaled so the
# candidate closest to your likes scores 1. Its draw weight is multiplied by
# exp(PULL * taste) (~20x at PULL 3; variety barely drops in a simulation), and
# the grid order adds ORDER * taste rating points.
TASTE_PULL = 3.0
TASTE_ORDER = 1.5
# Browse genres, each a family of catalogue tags: TV shows use "Action & Adventure"
# and "Sci-Fi & Fantasy", so matching is a case-insensitive substring test.
BROWSE_GENRES = {
    "Action": ["Action"],
    "Adventure": ["Adventure"],
    "Animation": ["Animation"],
    "Comedy": ["Comedy"],
    "Crime": ["Crime"],
    "Drama": ["Drama"],
    "Family": ["Family", "Kids"],
    "Fantasy": ["Fantasy"],
    "Horror": ["Horror"],
    "Mystery": ["Mystery"],
    "Romance": ["Romance"],
    "Sci-Fi": ["Science Fiction", "Sci-Fi"],
    "Thriller": ["Thriller"],
}
HOME_SHELVES = ["Action", "Comedy", "Drama", "Thriller", "Sci-Fi", "Animation"]
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
    .stApp .stMarkdown :is(h1, h2).grid-heading,
    .stApp .stMarkdown h2.shelf-heading {
        font-family: var(--font-display) !important; font-weight: 400 !important; padding: 0 !important;
        font-size: var(--h-size) !important; line-height: var(--h-lh) !important; letter-spacing: var(--h-track) !important;
    }
    .hero-line { --h-size: clamp(3rem, 8vw, 6.5rem); --h-lh: 0.98; --h-track: -0.04em; }
    .detail-title { --h-size: clamp(2.5rem, 5vw, 4.25rem); --h-lh: 1; --h-track: -0.035em; }
    .grid-heading { --h-size: clamp(2rem, 4vw, 3rem); --h-lh: 1.05; --h-track: -0.03em; }
    .shelf-heading { --h-size: clamp(1.5rem, 2.6vw, 2rem); --h-lh: 1.1; --h-track: -0.025em; margin: 0 !important; }
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
    /* Streamlit 1.5x marks the chosen pill kind="pillsActive"; 1.6x uses data-selected */
    [data-testid="stButtonGroup"] button:is([kind="pillsActive"], [data-selected="true"]) {
        /* !important: Streamlit's own selected-pill rule is more specific */
        background: var(--mood) !important; border-color: var(--mood) !important; color: var(--accent-ink) !important;
    }
    [data-testid="stButtonGroup"] button p { font-size: 1rem; }
    /* Scan again sits on the right content edge, under Watchlist */
    [class*="st-key-hero-"] [data-testid="stColumn"]:last-child [data-testid="stVerticalBlock"] { align-items: flex-end; }
    .stApp .stMarkdown h1.detail-title { margin: 0 0 var(--s2) !important; }
    .stApp .stMarkdown h1.grid-heading { margin-bottom: var(--s2) !important; }
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

    /* browse shelves: one row per list that scrolls sideways; the cut-off last
       poster shows there is more. Kept last so it wins over the grid rules above. */
    [class*="st-key-shelf-"] { margin-top: var(--s6); gap: var(--s2); }
    [class*="st-key-shelfhead-"] [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; }
    [class*="st-key-shelfhead-"] [data-testid="stColumn"] { min-width: 0 !important; }
    [class*="st-key-shelfhead-"] [data-testid="stMarkdownContainer"] { margin-bottom: 0; }
    [class*="st-key-shelfhead-"] [data-testid="stColumn"]:last-child { flex: 0 0 auto !important; width: auto !important; }
    [class*="st-key-shelf-"] [data-testid="stHorizontalBlock"]:has([class*="st-key-card-"]) {
        flex-wrap: nowrap !important; overflow-x: auto; overflow-y: hidden;
        gap: var(--s3); padding: 6px 6px var(--s2); margin: 0 -6px;
        scroll-snap-type: x proximity; scroll-padding-inline: 6px; overscroll-behavior-x: contain;
        scrollbar-width: thin; scrollbar-color: var(--line-strong) transparent;
    }
    [class*="st-key-shelf-"] [data-testid="stHorizontalBlock"]:has([class*="st-key-card-"]) > [data-testid="stColumn"] {
        flex: 0 0 160px !important; min-width: 160px !important; width: 160px !important; scroll-snap-align: start;
    }
    .st-key-genre-more { margin-top: var(--s2); }
    @media (max-width: 640px) {
        [class*="st-key-shelf-"] { margin-top: var(--s4); }
        [class*="st-key-shelf-"] [data-testid="stHorizontalBlock"]:has([class*="st-key-card-"]) { gap: var(--s2); }
        [class*="st-key-shelf-"] [data-testid="stHorizontalBlock"]:has([class*="st-key-card-"]) > [data-testid="stColumn"] {
            flex: 0 0 136px !important; min-width: 136px !important; width: 136px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------
class _TmdbUnavailable(Exception):
    """TMDB couldn't be reached. Raised inside cached helpers so the failure isn't cached."""


# st.cache_data caches a None return like any other result but never caches an
# exception. So the cached helpers raise on a failed call and the wrappers below
# turn that into None: a real "no match" is cached, a network blip is retried.
@st.cache_data(ttl=60 * 60, show_spinner=False)
def _tmdb_search_cached(title: str) -> Optional[dict]:
    try:
        resp = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": TMDB_API_KEY, "query": title, "page": 1},
            timeout=5,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
    except (requests.RequestException, ValueError) as exc:
        raise _TmdbUnavailable(type(exc).__name__) from exc  # the message holds the URL, key included
    return results[0] if results else None


def _tmdb_search(title: str) -> Optional[dict]:
    """First TMDB search hit for a title, or None."""
    if not TMDB_API_KEY or not title:
        return None
    try:
        return _tmdb_search_cached(title)
    except _TmdbUnavailable as exc:
        logger.warning("TMDB search failed for %r: %s", title, exc)
        return None


def poster_url(title: str, poster_path: Optional[str], size: str = "w500") -> str:
    path = poster_path.strip() if isinstance(poster_path, str) else ""
    if not path:
        hit = _tmdb_search(title)
        path = (hit or {}).get("poster_path") or ""
    if not path:
        return PLACEHOLDER_POSTER
    return path if path.startswith("http") else f"https://image.tmdb.org/t/p/{size}{path}"


# Card posters show at 136-200 CSS px, so w500 for every one wasted ~2/3 of the bytes.
# The browser picks from these by layout width and screen density.
CARD_SIZES = ("w185", "w342", "w500")
FILM_SIZES = ("w342", "w500", "w780")
# How wide each poster shows, matching the CSS breakpoints below.
GRID_SIZES_ATTR = "(max-width: 640px) 50vw, (max-width: 1024px) 33vw, min(16vw, 200px)"
SHELF_SIZES_ATTR = "(max-width: 640px) 136px, 160px"
FILM_SIZES_ATTR = "(max-width: 640px) 280px, 380px"
EAGER_POSTERS = 6  # about the first visible row; the rest load lazily


def poster_img(title: str, poster_path: Optional[str], sizes: tuple[str, ...], sizes_attr: str,
               cls: str = "", eager: bool = False) -> str:
    """An <img> with a TMDB srcset, so the browser fetches only as many pixels as it shows."""
    src = poster_url(title, poster_path, size=sizes[1])
    attrs = [f"src='{html.escape(src, quote=True)}'"]
    marker = f"/t/p/{sizes[1]}/"
    if marker in src:  # a TMDB path; a full URL or the placeholder has one size only
        srcset = ", ".join(f"{src.replace(marker, f'/t/p/{s}/')} {s[1:]}w" for s in sizes)
        attrs += [f"srcset='{html.escape(srcset, quote=True)}'", f"sizes='{sizes_attr}'"]
    if cls:
        attrs.append(f"class='{cls}'")
    attrs.append(f"alt='Poster for {html.escape(title, quote=True)}'")
    attrs.append("loading='eager' fetchpriority='high'" if eager else "loading='lazy'")
    return f"<img {' '.join(attrs)}>"


# Streamlit re-executes this script on every rerun, so this count starts fresh each time.
_eager_left = [EAGER_POSTERS]


def _take_eager() -> bool:
    """True for the first EAGER_POSTERS cards drawn in this run of the script."""
    if _eager_left[0] > 0:
        _eager_left[0] -= 1
        return True
    return False


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _trailer_url_cached(title: str) -> Optional[str]:
    hit = _tmdb_search_cached(title)  # raises _TmdbUnavailable, so a failure isn't cached
    if not hit or not hit.get("id"):
        return None
    try:
        resp = requests.get(
            f"https://api.themoviedb.org/3/movie/{hit['id']}/videos",
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
            timeout=6,
        )
        resp.raise_for_status()
        videos = resp.json().get("results") or []
    except (requests.RequestException, ValueError) as exc:
        raise _TmdbUnavailable(type(exc).__name__) from exc  # the message holds the URL, key included
    for v in videos:
        if v.get("site") == "YouTube" and v.get("type") in {"Trailer", "Teaser"} and v.get("key"):
            return f"https://www.youtube.com/watch?v={v['key']}"
    return None


def trailer_url(title: str) -> Optional[str]:
    """A YouTube trailer or teaser from TMDB, if one exists."""
    if not TMDB_API_KEY or not title:
        return None
    try:
        return _trailer_url_cached(title)
    except _TmdbUnavailable as exc:
        logger.warning("TMDB trailer lookup failed for %r: %s", title, exc)
        return None


def _genre_mask(df: pd.DataFrame, genres: list[str]) -> pd.Series:
    wanted = [g.lower() for g in genres]
    return df["genres"].fillna("").str.lower().apply(lambda x: any(g in x for g in wanted))


def _pick_varied(
    df: pd.DataFrame, n: int, seed: int, avoid: set[str], taste: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """Pick n rows at random, favouring higher ratings, your taste, and avoiding the last batch.

    The same seed gives the same pick, so the list only changes when the seed
    changes (a new scan or mood), not on every Streamlit rerun.
    """
    if len(df) <= n:
        return df
    rating = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0).clip(0, 10).to_numpy()
    # each rating point makes a title ~2.7x likelier; picks average ~8/10
    weights = np.exp(rating - 10.0)
    if taste is not None:
        weights = weights * np.exp(TASTE_PULL * taste)
    shown_last = df["title"].fillna("").astype(str).isin(avoid).to_numpy()
    weights = np.where(shown_last, weights * 0.25, weights)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=n, replace=False, p=weights / weights.sum())
    return df.iloc[idx]


@st.cache_data(ttl=10 * 60, show_spinner=False)
def catalogue() -> pd.DataFrame:
    """The catalogue, read once per ten minutes instead of on every rerun."""
    df = load_movies()
    for col in ("rating", "popularity"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _top(df: pd.DataFrame, by: str, limit: Optional[int]) -> list[dict]:
    if df.empty or by not in df.columns:
        return []
    out = df.sort_values(by, ascending=False, na_position="last").drop_duplicates("title")
    return (out if limit is None else out.head(limit)).to_dict(orient="records")


def trending(limit: int = SHELF_SIZE) -> list[dict]:
    return _top(catalogue(), "popularity", limit)


def genre_titles(genre: str, by: str = "popularity", limit: Optional[int] = None) -> list[dict]:
    df = catalogue()
    if df.empty or genre not in BROWSE_GENRES:
        return []
    return _top(df[_genre_mask(df, BROWSE_GENRES[genre])], by, limit)


@st.cache_resource(ttl=10 * 60, show_spinner="Getting recommendations ready…")
def similar_space() -> tuple[pd.DataFrame, object]:
    """The catalogue with its similarity vectors; cached together so their rows always line up.

    Embeddings come from a disk cache, so only the very first start (or new
    titles) runs the model. Shared across sessions: read it, never mutate it.
    """
    df = catalogue().reset_index(drop=True)
    embeddings = embed_texts(embedding_texts(df)) if BLEND_EMBEDDINGS and not df.empty else None
    return df, similarity_index(df, embeddings=embeddings)


def _row_of(df: pd.DataFrame, movie: dict) -> Optional[int]:
    movie_id, title = _text(movie, "movie_id"), _text(movie, "title")
    if movie_id and "movie_id" in df.columns:
        hits = np.flatnonzero(df["movie_id"].astype(str).to_numpy() == movie_id)
        if len(hits):
            return int(hits[0])
    hits = np.flatnonzero(df["title"].astype(str).to_numpy() == title)
    return int(hits[0]) if len(hits) else None


def more_like_this(movie: dict, limit: int = SHELF_SIZE) -> list[dict]:
    df, matrix = similar_space()
    row = _row_of(df, movie) if matrix is not None else None
    if row is None:
        return []
    picked = df.iloc[similar_rows(matrix, row, top_n=limit * 2)]
    picked = picked[picked["title"] != df["title"].iloc[row]].drop_duplicates("title")
    return picked.head(limit).to_dict(orient="records")


def _taste(df: pd.DataFrame, matrix) -> tuple[Optional[np.ndarray], set[int]]:
    """Likeness of every row to your likes, and the rows you said weren't for you."""
    prefs = load_user_profile().get("title_preferences") or {}
    if matrix is None or not prefs:
        return None, set()
    by_title = {t: i for i, t in enumerate(df["title"].astype(str).str.strip().str.lower())}
    liked = [by_title[t] for t, w in prefs.items() if t in by_title and float(w) > 0]
    disliked = [by_title[t] for t, w in prefs.items() if t in by_title and float(w) < 0]
    if not liked and not disliked:
        return None, set()
    taste = taste_scores(matrix, liked, disliked)
    # a liked film is perfectly like itself; don't push what you've seen back at you
    taste[liked] = 0.0
    return taste, set(disliked)


def mood_picks(mood: str) -> list[dict]:
    df, matrix = similar_space()
    if df.empty:
        return []
    taste, disliked = _taste(df, matrix)
    # copy: under pandas 3 copy-on-write, to_numpy() hands back a read-only view
    mask = _genre_mask(df, MOOD_TO_GENRES.get(mood, [])).to_numpy(copy=True)
    if disliked:
        mask[list(disliked)] = False
    matches = df[mask]
    if matches.empty:
        return []
    if taste is not None:
        pool = taste[matches.index]
        top = pool.max()
        # raw likeness is small (~0.05-0.2), so measure it against the best candidate
        taste = taste / top if top > 0 else None
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
        taste=None if taste is None else taste[matches.index],
    )
    st.session_state["shown_titles"] = picked["title"].astype(str).tolist()
    records = picked.to_dict(orient="records")
    if taste is not None:
        # order the grid by rating nudged toward your taste (the ranker adds its own boosts)
        for rec, row in zip(records, picked.index):
            rating = pd.to_numeric(rec.get("rating"), errors="coerce")
            rec["score"] = (0.0 if pd.isna(rating) else float(rating)) + TASTE_ORDER * float(taste[row])
    return rerank_results_for_learning(records, mood)


def search_titles(query: str) -> tuple[list[dict], Optional[str]]:
    """Matches for a title, person or genre (typos forgiven), plus a hint when none match."""
    hits, suggestion = search_catalogue(catalogue(), query, limit=GRID_SIZE)
    return hits.to_dict(orient="records"), suggestion


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
def scan_mood(photo: bytes) -> None:
    """Read the mood from a photo taken in the browser; on failure, leave a note and keep the old mood."""
    try:
        details = detect_emotion_from_bytes(photo)
    except ValueError:
        st.session_state["scan_note"] = "That photo couldn’t be read. Take another, or pick a mood."
        return
    except Exception as exc:  # noqa: BLE001 - OpenCV or model failures fall back to the picker
        # log the reason; failing silently once hid a broken face detector
        logger.warning("Mood scan failed: %s", exc)
        st.session_state["scan_note"] = "Mood reading isn’t available right now. Pick a mood instead."
        return
    if not details or not details.get("mood"):
        st.session_state["scan_note"] = "No face found. Face the camera in good light and scan again, or pick a mood."
        return
    st.session_state.update(
        mood=details["mood"],
        # the pills are already drawn this run, so they pick up the mood on the next one
        sync_pick=True,
        mood_source="camera",
        scan_note=None,
        pick_seed=st.session_state["pick_seed"] + 1,
    )


def _on_mood_pick() -> None:
    picked = st.session_state.get("mood_pick")
    st.session_state["mood"] = picked
    st.session_state["mood_source"] = "picked" if picked else None
    st.session_state["scan_note"] = None


# --------------------------------------------------------------------------
# Navigation
# --------------------------------------------------------------------------
def go(view: str, movie: Optional[dict] = None) -> None:
    if view == "movie" and st.session_state.get("view") != "movie":
        # Back returns to the list the film was opened from, even after
        # hopping through "More like this"
        st.session_state["return_view"] = st.session_state.get("view", "home")
    st.session_state["view"] = view
    st.session_state["movie"] = movie


def go_back() -> None:
    back = st.session_state.get("return_view", "home")
    go(back if back in ("home", "genre", "watchlist") else "home")


def go_genre(genre: str) -> None:
    st.session_state["genre"] = genre
    st.session_state["genre_limit"] = GENRE_PAGE_STEP
    go("genre")


def _on_genre_pick() -> None:
    picked = st.session_state.get("genre_pick")
    if picked:  # clicking the selected genre again keeps it
        st.session_state["genre"] = picked
        st.session_state["genre_limit"] = GENRE_PAGE_STEP


def _show_more() -> None:
    st.session_state["genre_limit"] = st.session_state.get("genre_limit", GENRE_PAGE_STEP) + GENRE_PAGE_STEP


def _clear_search() -> None:
    st.session_state["search"] = ""


def _use_suggestion(title: str) -> None:
    st.session_state["search"] = title


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
            with col:
                poster_card(movie, key_prefix, GRID_SIZES_ATTR)


def poster_card(movie: dict, key_prefix: str, sizes_attr: str) -> None:
    title = _text(movie, "title", "Untitled")
    uid = f"{key_prefix}-{zlib.crc32(_movie_key(movie).encode())}"
    with st.container(key=f"card-{uid}"):
        img = poster_img(title, movie.get("poster_path"), CARD_SIZES, sizes_attr,
                         cls="poster", eager=_take_eager())
        st.markdown(f"<div class='poster-frame'>{img}</div>", unsafe_allow_html=True)
        st.button(title, key=f"open-{uid}", on_click=go, args=("movie", movie))
        genre = _text(movie, "genres").split("|")[0]
        rating = _rating_text(movie)
        meta = f"{genre}, {rating}" if genre and rating else genre or rating
        st.markdown(f"<div class='card-meta'>{html.escape(meta)}</div>", unsafe_allow_html=True)


def poster_shelf(heading: str, movies: list[dict], slug: str, genre: Optional[str] = None) -> None:
    """A titled row of posters that scrolls sideways; a genre shelf links to its full page."""
    if not movies:
        return
    with st.container(key=f"shelf-{slug}"):
        with st.container(key=f"shelfhead-{slug}"):
            head, more = st.columns([4, 1], vertical_alignment="center")
            head.markdown(f"<h2 class='shelf-heading'>{html.escape(heading)}</h2>", unsafe_allow_html=True)
            if genre:
                more.button(
                    "See all", key=f"all-{slug}", on_click=go_genre, args=(genre,),
                    help=f"Every {genre.lower()} title in the catalogue",
                )
        cols = st.columns(len(movies))
        for col, movie in zip(cols, movies):
            with col:
                poster_card(movie, f"shelf-{slug}", SHELF_SIZES_ATTR)


def render_browse() -> None:
    poster_shelf("Trending now", trending(), "trending")
    for genre in HOME_SHELVES:
        poster_shelf(genre, genre_titles(genre, limit=SHELF_SIZE), genre.lower().replace("-", ""), genre)


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
                placeholder="Search a title, actor, director or genre",
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
        results, suggestion = search_titles(query)
        st.markdown(f"<h2 class='grid-heading'>Results for “{html.escape(query)}”</h2>", unsafe_allow_html=True)
        st.button("Clear search", key="clear-search", on_click=_clear_search)
        if results:
            poster_grid(results, "search")
        else:
            st.markdown(
                "<p class='empty'>Nothing in the catalogue matches that title, actor, director or genre.</p>",
                unsafe_allow_html=True,
            )
            if suggestion:
                st.button(
                    f"Did you mean {suggestion}?", key="search-suggestion",
                    on_click=_use_suggestion, args=(suggestion,),
                )
        return

    mood = st.session_state.get("mood")
    # the key carries the mood so the page can light itself in the mood's colour
    with st.container(key=f"hero-{mood if mood in MOOD_COPY else 'none'}"):
        if mood in MOOD_COPY:
            line, sub = MOOD_COPY[mood]
            if st.session_state.get("mood_source") != "camera":
                line = f"Feeling {mood}."  # picked by hand, so don't claim we read the face
        else:
            line, sub = "How are you feeling?", "Take a quick photo to read your mood, or pick one yourself."
        st.markdown(f"<h1 class='hero-line'>{line}</h1><p class='hero-sub'>{sub}</p>", unsafe_allow_html=True)

        pills, scan = st.columns([5, 1.2], vertical_alignment="center")
        # Streamlit drops a widget's state while it's off screen (film page, watchlist,
        # search), so re-seed the pills from the mood when the viewer comes back.
        if (st.session_state.pop("sync_pick", False) or "mood_pick" not in st.session_state) and mood in MOOD_COPY:
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
            # The browser grabs one frame with no preview, so this works for every
            # visitor, not just whoever sits at the server's webcam.
            photo, camera_error = mood_scan_button(
                "Scan again" if st.session_state.get("mood_source") == "camera" else "Scan my mood",
                nonce=st.session_state.get("scan_count", 0),
            )
        if photo is not None or camera_error:
            st.session_state["scan_count"] = st.session_state.get("scan_count", 0) + 1
            if camera_error:
                st.session_state["scan_note"] = camera_error
            else:
                with st.spinner("Reading your mood…"):
                    scan_mood(photo)
            st.rerun()  # redraw the hero with the new mood (or the note)
        note = st.session_state.get("scan_note")
        if note:
            st.markdown(f"<p class='hero-source'>{html.escape(note)}</p>", unsafe_allow_html=True)
        elif st.session_state.get("mood_source") == "camera":
            st.markdown(
                "<p class='hero-source'>Read from your photo. Pick another mood if it’s wrong.</p>",
                unsafe_allow_html=True,
            )

    if mood:
        picks = mood_picks(mood)
        if picks:
            article = "an" if mood[:1] in "aeiou" else "a"
            poster_grid(picks, "mood", f"Films for {article} {mood} mood")
        else:
            st.markdown(
                "<p class='empty'>Nothing in the catalogue matches this mood yet. Try another one.</p>",
                unsafe_allow_html=True,
            )
    render_browse()


def render_genre() -> None:
    genre = st.session_state.get("genre")
    if genre not in BROWSE_GENRES:
        genre = HOME_SHELVES[0]
        st.session_state["genre"] = genre
    st.button("Back", key="back-genre", icon=":material/arrow_back:", on_click=go, args=("home",))
    st.markdown(f"<h1 class='grid-heading'>{html.escape(genre)}</h1>", unsafe_allow_html=True)
    # keep the pills on the genre shown, including after a film page and back
    st.session_state["genre_pick"] = genre
    st.pills(
        "Genre",
        options=list(BROWSE_GENRES),
        key="genre_pick",
        on_change=_on_genre_pick,
        label_visibility="collapsed",
    )
    titles = genre_titles(genre, by="rating")
    limit = int(st.session_state.get("genre_limit", GENRE_PAGE_STEP))
    if not titles:
        st.markdown("<p class='empty'>No titles in this genre yet. Pick another one.</p>", unsafe_allow_html=True)
        return
    poster_grid(titles[:limit], "genre", f"{genre} titles, best rated first")
    if len(titles) > limit:
        st.button(f"Show more {genre.lower()}", key="genre-more", on_click=_show_more)


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
    st.button("Back", key="back-movie", icon=":material/arrow_back:", on_click=go_back)
    left, right = st.container(key="film").columns([1, 2.2], gap="large")
    with left:
        img = poster_img(title, movie.get("poster_path"), FILM_SIZES, FILM_SIZES_ATTR, eager=True)
        st.markdown(f"<div class='film-poster'>{img}</div>", unsafe_allow_html=True)
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
        # The chosen verdict shows as pressed and can't stack; the other one switches it.
        verdict = title_verdict(movie)
        with liked:
            if st.button(
                "You liked it" if verdict == "liked" else "I liked it",
                key="liked", icon=":material/thumb_up:", disabled=verdict == "liked",
            ):
                if _feedback(movie, rating=5.0):
                    st.toast("Got it. You’ll see more like this.")
                    st.rerun()
        with disliked:
            if st.button(
                "Not for you" if verdict == "disliked" else "Not for me",
                key="disliked", icon=":material/thumb_down:", disabled=verdict == "disliked",
            ):
                if _feedback(movie, rating=1.0):
                    st.toast("Got it. You’ll see less like this.")
                    st.rerun()

        trailer = trailer_url(title)
        if trailer:
            st.video(trailer)

    poster_shelf("More like this", more_like_this(movie), "similar")


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
_run_daily_scraper()

st.session_state.setdefault("view", "home")
st.session_state.setdefault("pick_seed", int(time.time()))
if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = _load_watchlist()

render_topbar()
view = st.session_state["view"]
if view == "movie" and st.session_state.get("movie"):
    render_movie(st.session_state["movie"])
elif view == "watchlist":
    render_watchlist()
elif view == "genre":
    render_genre()
else:
    render_home()
