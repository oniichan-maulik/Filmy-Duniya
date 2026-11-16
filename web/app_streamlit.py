import streamlit as st
import streamlit.components.v1 as components
import requests
from typing import Optional
from pathlib import Path
import html
import json
import io
from difflib import get_close_matches
import re
import sys
import time
from urllib.parse import quote_plus
import zlib
import numpy as np
import pandas as pd

# Ensure project root is on sys.path so `src` imports work when Streamlit
# runs with a different CWD. This inserts the repo root (one level up from
# this `web/` folder) at the front of sys.path.
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.config import TMDB_API_KEY as DEFAULT_TMDB_KEY
from src.data_loader import load_movies, save_movies
from src.recommender import MOOD_TO_GENRES
from src.scraper import scrape_top_n_movies
from src.user_store import (
    learning_progress,
    learning_summary,
    record_feedback,
    rerank_results_for_learning,
)
from src.emotion_detection import detect_face_mood_backend

# placeholder image used when no poster can be found
PLACEHOLDER_URL = "https://via.placeholder.com/200x300?text=No+Poster"
PUBLIC_DOMAIN_STREAM = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"

# lightweight video source "database" backed by JSON
def _video_store_path() -> Path:
    p = repo_root / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p / "video_sources.json"


def load_video_sources() -> dict[str, str]:
    """Load a title->video_url mapping from data/video_sources.json."""
    try:
        path = _video_store_path()
        if not path.exists():
            return {}
        raw = path.read_text(encoding="utf-8") or "{}"
        data = json.loads(raw)
        if isinstance(data, list):
            out: dict[str, str] = {}
            for item in data:
                if isinstance(item, dict) and item.get("title") and item.get("url"):
                    out[str(item["title"]).lower()] = str(item["url"])
            return out
        if isinstance(data, dict):
            return {str(k).lower(): str(v) for k, v in data.items() if v}
    except Exception:
        pass
    return {}


def save_video_source(title: str, url: str, overwrite: bool = False) -> None:
    """Persist a video URL for a given title."""
    if not title or not url:
        return
    try:
        path = _video_store_path()
        db = load_video_sources()
        key = title.lower()
        if key in db and not overwrite:
            return
        db[key] = url
        path.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _resolve_video_url(title: str, db: dict[str, str]) -> Optional[str]:
    """Resolve a playable URL for a title from the local db (case-insensitive, slug)."""
    if not title:
        return None
    low = title.lower()
    if low in db:
        return db[low]
    slug = _watch_token_from_title(title)
    if slug in db:
        return db[slug]
    # loose match
    for k, v in db.items():
        if low == k or slug == k:
            return v
    return None


def _youtube_embed_url(url: str) -> Optional[str]:
    """Convert a YouTube watch/short link to an embeddable URL."""
    if not url:
        return None
    patterns = [
        r"youtu\.be/([\w-]{6,})",
        r"youtube\.com/watch\?v=([\w-]{6,})",
        r"youtube\.com/embed/([\w-]{6,})",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            vid = m.group(1)
            return f"https://www.youtube.com/embed/{vid}"
    return None


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_trailer_url(title: str, tmdb_api_key: Optional[str]) -> Optional[str]:
    """Try to grab a YouTube trailer URL from TMDB."""
    if not tmdb_api_key or not title:
        return None
    try:
        search = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": tmdb_api_key, "query": title, "page": 1},
            timeout=6,
        )
        search.raise_for_status()
        results = search.json().get("results") or []
        if not results:
            return None
        movie_id = results[0].get("id")
        if not movie_id:
            return None
        vids = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}/videos",
            params={"api_key": tmdb_api_key, "language": "en-US"},
            timeout=6,
        )
        vids.raise_for_status()
        videos = vids.json().get("results") or []
        for v in videos:
            if v.get("site") == "YouTube" and v.get("type") in {"Trailer", "Teaser"} and v.get("key"):
                return f"https://www.youtube.com/watch?v={v['key']}"
    except Exception:
        return None
    return None


def populate_video_db_with_trailers(tmdb_api_key: Optional[str], limit: int = 150) -> int:
    """Fetch YouTube trailers via TMDB and cache them into the local video DB."""
    if not tmdb_api_key:
        return 0
    db = load_video_sources()
    df = load_movies()
    if df.empty or "title" not in df.columns:
        return 0
    added = 0
    for title in df["title"].dropna().astype(str).tolist()[:limit]:
        if title.lower() in db:
            continue
        url = fetch_trailer_url(title, tmdb_api_key)
        if url:
            save_video_source(title, url)
            added += 1
    return added

st.set_page_config(page_title="Filmy Duniya", layout="wide")

# cinematic dark UI styles
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wdth,wght@75..100,400..700&family=DM+Mono:wght@400;500&display=swap');
    /* Look dealt by taste-skill 2.5: palette soot-pale-lime (a dark screening-room ground,
       one lime accent for actions), type Instrument Sans + DM Mono, radius round. */
    /* hide Streamlit's top toolbar (Deploy, menu) and the colored strip; keep sidebar toggle */
    [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
        display: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    :root {
        --bg: #0f0f0f;
        --surface: #1a1a1a;
        --surface-2: #232323;
        --ink: #ffffff;
        --muted: #a3a3a3;
        --accent: #dafa9c;
        --accent-ink: #0f0f0f;
        --line: rgba(255,255,255,0.09);
        --line-strong: rgba(255,255,255,0.22);
        --font-display: "Instrument Sans", system-ui, sans-serif;
        --font-text: "Instrument Sans", system-ui, sans-serif;
        --font-mono: "DM Mono", ui-monospace, Menlo, Consolas, monospace;
        --r-sm: 10px; --r-md: 16px; --r-lg: 24px; --r-pill: 999px;
        --ease: cubic-bezier(.2,.8,.2,1);
        --m-in: 150ms; --m-out: 120ms; --m-press: 90ms;
        --m-image: 600ms; --m-image-out: 400ms;
        /* kept for any older markup that still reads these names */
        --text: var(--ink); --card: var(--surface); --card-2: var(--surface); --stroke: var(--line);
    }
    .stApp {
        background: var(--bg);
        color: var(--ink);
        font-family: var(--font-text);
    }
    .stApp, .stApp p, .stApp label, .stApp input, .stApp textarea, .stApp button,
    .stMarkdown, [data-testid="stSidebar"] {
        font-family: var(--font-text);
    }
    h1, h2, h3, h4,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        font-family: var(--font-display);
        font-weight: 500;
        letter-spacing: -0.028em;
        font-stretch: 96%;
        color: var(--ink);
    }
    .block-container {padding-top: 18px; padding-bottom: 72px;}
    .stMarkdown, .stTextInput label, .stSlider label, .stCheckbox label {color: var(--ink);}
    [data-testid="stCaptionContainer"], .stCaption {color: var(--muted) !important;}
    [data-testid="stSidebar"] {background: #141414; border-right: 1px solid var(--line);}

    /* fields: action border and a soft 3px ring on focus */
    .stTextInput > div > div > input,
    .stTextArea textarea {
        background: var(--surface);
        border: 1px solid var(--line);
        color: var(--ink);
        border-radius: var(--r-pill);
        height: 46px;
        padding-left: 18px;
        transition: border-color 140ms var(--ease), box-shadow 140ms var(--ease);
    }
    .stTextArea textarea {border-radius: var(--r-md); height: auto;}
    .stTextInput > div > div > input::placeholder {color: var(--muted);}
    .stTextInput > div > div:focus-within,
    .stTextInput > div > div > input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(218,250,156,0.18) !important;
        outline: none;
    }

    /* buttons: one pair. secondary = tonal, primary = lime. hover steps 1px up, press sinks */
    .stButton button,
    .stDownloadButton button,
    .stFormSubmitButton button {
        background: var(--surface-2);
        color: var(--ink);
        border: 1px solid var(--line);
        border-radius: var(--r-pill);
        min-height: 44px;
        padding: 10px 18px;
        font-weight: 500;
        letter-spacing: -0.005em;
        box-shadow: none;
        transition: background-color var(--m-out) var(--ease), border-color var(--m-out) var(--ease),
                    transform var(--m-out) var(--ease);
    }
    .stButton button *, .stDownloadButton button *, .stFormSubmitButton button * {color: inherit;}
    .stButton button[kind="primary"],
    .stButton button[data-testid="stBaseButton-primary"],
    .stFormSubmitButton button[kind="primaryFormSubmit"] {
        background: var(--accent);
        color: var(--accent-ink);
        border-color: var(--accent);
    }
    @media (hover: hover) and (pointer: fine) {
        .stButton button:hover,
        .stDownloadButton button:hover,
        .stFormSubmitButton button:hover {
            background: color-mix(in srgb, var(--surface-2) 86%, var(--ink));
            border-color: var(--line-strong);
            color: var(--ink);
            transform: translateY(-1px);
            transition-duration: var(--m-in);
        }
        .stButton button[kind="primary"]:hover,
        .stButton button[data-testid="stBaseButton-primary"]:hover,
        .stFormSubmitButton button[kind="primaryFormSubmit"]:hover {
            background: color-mix(in srgb, var(--accent) 86%, var(--accent-ink));
            border-color: color-mix(in srgb, var(--accent) 86%, var(--accent-ink));
            color: var(--accent-ink);
        }
    }
    .stButton button:active,
    .stDownloadButton button:active,
    .stFormSubmitButton button:active {
        transform: translateY(1px) scale(.985);
        transition-duration: var(--m-press);
    }
    .stButton button:focus-visible,
    .stDownloadButton button:focus-visible,
    .stFormSubmitButton button:focus-visible,
    a:focus-visible {
        outline: 2px solid var(--accent) !important;
        outline-offset: 2px;
        box-shadow: none !important;
    }

    /* st.pills (mood picker): tonal chips; selected = lime */
    [data-testid="stMain"] [data-testid="stButtonGroup"] button {
        background: var(--surface);
        border: 1px solid var(--line);
        color: var(--muted);
        border-radius: var(--r-pill);
        transition: border-color var(--m-out) var(--ease), color var(--m-out) var(--ease),
                    transform var(--m-out) var(--ease);
    }
    [data-testid="stMain"] [data-testid="stButtonGroup"] button * {color: inherit;}
    @media (hover: hover) and (pointer: fine) {
        [data-testid="stMain"] [data-testid="stButtonGroup"] button:hover {
            border-color: var(--line-strong);
            color: var(--ink);
            transform: translateY(-1px);
            transition-duration: var(--m-in);
        }
    }
    [data-testid="stMain"] [data-testid="stButtonGroup"] button:active {
        transform: translateY(1px) scale(.985);
        transition-duration: var(--m-press);
    }
    /* !important: Streamlit colors the selected pill with its own primary color */
    [data-testid="stMain"] [data-testid="stButtonGroup"] button[aria-checked="true"] {
        background: var(--accent) !important;
        border-color: var(--accent) !important;
        color: var(--accent-ink) !important;
        font-weight: 500;
    }
    [data-testid="stMain"] [data-testid="stButtonGroup"] button[aria-checked="true"] * {
        background: transparent !important;
        color: var(--accent-ink) !important;
    }
    [data-testid="stMain"] [data-testid="stElementContainer"]:has([data-testid="stButtonGroup"]) [data-testid="stWidgetLabel"] {
        color: var(--muted);
    }

    [data-testid="stExpander"] details {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: var(--r-md);
    }

    .app-shell {max-width: 1220px; margin: 0 auto;}
    .topbar {
        display:flex; align-items:center; gap:14px; justify-content:space-between;
        background: var(--surface);
        border:1px solid var(--line);
        padding: 14px 18px; border-radius: var(--r-lg); margin-bottom: 20px;
    }
    .brand {
        font-family: var(--font-display);
        font-size: 24px; font-weight: 600; letter-spacing: -0.03em;
        line-height: 44px;
    }
    .brand span {color: var(--accent);}
    .pill-nav {display:flex; flex-wrap:wrap; gap:10px; align-items:center;}
    .pill {
        padding:8px 14px; border-radius: var(--r-pill); border:1px solid var(--line);
        background: var(--surface); color: var(--muted); font-weight:500; font-size:13px;
        transition: border-color var(--m-out) var(--ease), color var(--m-out) var(--ease);
    }
    .pill:hover {border-color: var(--line-strong); color: var(--ink);}
    .pill.active {background: var(--accent); border-color: var(--accent); color: var(--accent-ink);}
    .search-shell {width:100%;}
    .subtext {color: var(--muted); margin: 0 0 12px 2px;}

    /* watch page header */
    .hero {
        border:1px solid var(--line);
        background: var(--surface);
        border-radius: var(--r-lg);
        padding: 22px;
        margin-bottom: 24px;
    }
    .hero-tag {
        display:inline-block; font-family: var(--font-mono); font-size:12px; font-weight:500;
        letter-spacing:0.02em; color: var(--accent-ink); background: var(--accent);
        padding:4px 10px; border-radius: var(--r-pill); margin-bottom:10px;
    }
    .hero-title {
        font-family: var(--font-display); font-size: 40px; font-weight: 500;
        letter-spacing: -0.036em; line-height: 1.05; margin: 14px 0 10px 0;
    }
    .hero-meta {color: var(--muted); font-size: 14px; margin-bottom: 12px;}
    .movie-card {
        border:1px solid var(--line);
        background: var(--surface);
        padding:12px; border-radius: var(--r-md);
    }
    .movie-title {margin: 10px 0 6px 0;}
    .section-title {
        font-family: var(--font-display);
        font-size: 28px; font-weight: 500; letter-spacing: -0.028em; line-height: 1.15;
        margin: 28px 0 14px 0;
    }
    .rating-stars {display:flex; align-items:center; gap:10px; margin: 8px 0 10px 0;}
    .stars {letter-spacing:2px; font-size:15px;}
    .star-on {color: var(--accent);}
    .star-off {color: #3a3a3a;}
    .rating-badge {
        font-family: var(--font-mono); font-size: 13px; font-weight: 500;
        background: var(--surface-2); border:1px solid var(--line);
        padding:5px 10px; border-radius: var(--r-pill);
    }

    /* featured banner: the poster is the picture; a neutral scrim keeps the words readable */
    a:has(> .banner-panel) {display:block; border-radius: var(--r-lg);}
    .banner-panel {
        position: relative;
        aspect-ratio: 16 / 7;
        min-height: 360px;
        border-radius: var(--r-lg);
        border: 1px solid var(--line);
        overflow: hidden;
        margin-bottom: 20px;
        transition: transform 180ms var(--ease), border-color 180ms var(--ease);
    }
    @media (hover: hover) and (pointer: fine) {
        a:hover > .banner-panel {transform: translateY(-2px); border-color: var(--line-strong);}
    }
    .banner-content {
        position: absolute;
        left: 32px;
        bottom: 28px;
        max-width: 58%;
    }
    .banner-kicker {
        display: inline-block;
        font-family: var(--font-mono);
        font-size: 12px;
        font-weight: 500;
        letter-spacing: 0.02em;
        color: var(--accent-ink);
        background: var(--accent);
        padding: 4px 10px;
        border-radius: var(--r-pill);
        margin-bottom: 14px;
    }
    .banner-title {
        font-family: var(--font-display);
        font-size: clamp(34px, 4.6vw, 64px);
        font-weight: 500;
        letter-spacing: -0.036em;
        line-height: 1.0;
        margin: 0 0 12px 0;
        color: var(--ink);
    }
    .banner-sub {
        color: rgba(255,255,255,0.82);
        font-size: 16px;
        line-height: 1.45;
        max-width: 46ch;
    }
    /* opened by st.markdown, so Streamlit closes it at once: it never wraps the checkboxes */
    .genre-strip {display: none;}
    /* multiselect tags (Content type): dark text on the lime fill */
    [data-testid="stMultiSelectTagsContainer"] [data-tag] {background: var(--accent) !important; border-radius: var(--r-pill) !important;}
    [data-testid="stMultiSelectTagsContainer"] [data-tag], [data-testid="stMultiSelectTagsContainer"] [data-tag] * {color: var(--accent-ink) !important;}
    [data-testid="stMultiSelectTagsContainer"] [data-tag] svg, [data-testid="stMultiSelectTagsContainer"] [data-tag] svg * {fill: var(--accent-ink) !important;}
    .strip-title {
        font-family: var(--font-mono);
        font-size: 12px;
        color: var(--muted);
        letter-spacing: 0.02em;
        margin: 0 0 6px 0;
        font-weight: 500;
    }

    /* poster cards: a linked card lifts 2px, the poster zooms 1.03 inside its frame */
    a:has(> .poster-card) {display:block; text-decoration:none; border-radius: var(--r-md);}
    .poster-card {
        position:relative; border-radius: var(--r-md); overflow:hidden;
        border:1px solid var(--line); background: var(--surface);
        width: 100%; margin: 0;
        transition: transform var(--m-out) var(--ease), border-color var(--m-out) var(--ease),
                    box-shadow var(--m-out) var(--ease);
    }
    .poster-frame {position:relative; overflow:hidden; border-radius: var(--r-md) var(--r-md) 0 0;}
    .poster-img {
        display:block; width:100%; object-fit:cover; aspect-ratio: 2 / 3;
        transition: transform var(--m-image-out) var(--ease);
    }
    .poster-overlay {display:none;}
    @media (hover: hover) and (pointer: fine) {
        a:hover > .poster-card {
            transform: translateY(-2px);
            border-color: var(--line-strong);
            box-shadow: 0 12px 24px rgba(0,0,0,0.10);
            transition-duration: 180ms;
        }
        a:hover > .poster-card .poster-img {
            transform: scale(1.03);
            transition-duration: var(--m-image);
        }
        a:hover > .poster-card .poster-title {
            text-decoration: underline; text-underline-offset: 3px; text-decoration-thickness: 1px;
        }
    }
    .poster-link {text-decoration:none; color:inherit; display:block;}
    .poster-meta {padding:10px 12px 12px 12px;}
    .poster-title {
        font-family: var(--font-display);
        font-size:15px; font-weight:500; letter-spacing: -0.01em;
        margin:0 0 4px 0; color: var(--ink);
        overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
    }
    .poster-sub {
        font-family: var(--font-mono);
        color: var(--muted); font-size:12px;
        overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
    }
    .pill-badge {
        padding:4px 10px; border-radius: var(--r-pill); background: var(--surface-2);
        color: var(--muted); font-family: var(--font-mono); font-size:12px; border:1px solid var(--line);
    }
    ::-webkit-scrollbar {height: 8px; width: 8px;}
    ::-webkit-scrollbar-thumb {background: rgba(255,255,255,0.15); border-radius: 999px;}

    @media (max-width: 640px) {
        .banner-panel {aspect-ratio: auto; min-height: 420px;}
        .banner-content {left: 18px; right: 18px; bottom: 20px; max-width: none;}
        .section-title {font-size: 24px;}
    }
    @media (prefers-reduced-motion: reduce) {
        .stApp *, .stApp *::before, .stApp *::after {
            transition-duration: 0ms !important;
            animation-duration: 0ms !important;
        }
        a:hover > .poster-card, a:hover > .banner-panel, a:hover > .poster-card .poster-img,
        .stButton button:hover, .stButton button:active {transform: none !important;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# TMDB key: prefer Streamlit secrets, then env/config, otherwise empty
try:
    secrets_key = str(st.secrets.get("TMDB_API_KEY", "")).strip()
except Exception:
    secrets_key = ""
default_key = secrets_key or DEFAULT_TMDB_KEY or ""
tmdb_input = st.sidebar.text_input("TMDB API key", type="password", value=default_key)
TMDB_API_KEY = (tmdb_input or "").strip()
if TMDB_API_KEY:
    st.sidebar.success("Using TMDB for posters and ratings.")
else:
    st.sidebar.warning("TMDB API key missing; posters/ratings may use placeholders.")

st.sidebar.header("User Sign-In")
st.session_state.setdefault("signed_in", False)
st.session_state.setdefault("user_email", "")
user_email = st.sidebar.text_input("Email", value=st.session_state.get("user_email", ""))
user_pass = st.sidebar.text_input("Password", type="password")
if not st.session_state.get("signed_in"):
    if st.sidebar.button("Sign in"):
        st.session_state["signed_in"] = True
        st.session_state["user_email"] = user_email
        st.sidebar.success(f"Signed in as {user_email or 'guest'}")
else:
    st.sidebar.success(f"Signed in as {st.session_state.get('user_email','guest')}")
    if st.sidebar.button("Sign out"):
        st.session_state["signed_in"] = False
        st.sidebar.info("Signed out")

st.sidebar.header("Recommendation Mode")
mode = "Mood Based"
st.sidebar.caption("Mood-based recommendation is always enabled.")

NUM = st.sidebar.slider("Number of recommendations", min_value=1, max_value=50, value=30)
AUTO_REFRESH = False
POLL_INTERVAL = 10  # refresh interval in seconds
content_types = st.sidebar.multiselect(
    "Content type",
    options=["Movies", "TV Shows"],
    default=["Movies", "TV Shows"],
)

if TMDB_API_KEY:
    with st.sidebar.expander("Trailer prefetch"):
        if st.button("Fetch trailers for all movies", use_container_width=True):
            total = load_movies().shape[0] if not load_movies().empty else 0
            added = populate_video_db_with_trailers(TMDB_API_KEY, limit=max(500, total))
            st.sidebar.success(f"Fetched {added} trailers into video database.")

with st.sidebar.expander("Database manager (edit & save)"):
    try:
        # Edit the full catalog: saving a partial view would overwrite the CSV
        # with only the rows shown.
        df_all = load_movies()
        edited = st.data_editor(df_all, num_rows="dynamic", key="db_editor")
        if st.button("Save edits to data/movies.csv", key="save-db-editor"):
            save_movies(edited)
            st.success(f"Saved {len(edited)} rows to data/movies.csv")
    except Exception as e:
        st.warning(f"DB editor unavailable: {e}")

def _clean_poster_value(value: Optional[str]) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


@st.cache_data(ttl=60 * 60)
def fetch_poster_url(
    title: Optional[str],
    poster_path: Optional[str],
    tmdb_api_key: Optional[str],
    size: str = "w500",
) -> str:
    """Try to fetch a poster URL from TMDB, fall back to placeholder."""

    # normalize title/poster inputs
    title_str = (title or "").strip()
    normalized_poster = _clean_poster_value(poster_path)

    # prefer existing poster reference
    if normalized_poster:
        if normalized_poster.startswith("/"):
            return f"https://image.tmdb.org/t/p/{size}{normalized_poster}"
        return normalized_poster

    if tmdb_api_key and title_str:
        try:
            url = "https://api.themoviedb.org/3/search/movie"
            resp = requests.get(
                url,
                params={"api_key": tmdb_api_key, "query": title_str, "page": 1},
                timeout=5,
            )
            resp.raise_for_status()
            results = resp.json().get("results") or []
            if results:
                tmdb_path = results[0].get("poster_path")
                if tmdb_path:
                    return f"https://image.tmdb.org/t/p/{size}{tmdb_path}"
        except Exception:
            pass

    return PLACEHOLDER_URL


@st.cache_data(ttl=60 * 60)
def fetch_watch_url(
    title: Optional[str],
    tmdb_api_key: Optional[str],
    media_type: Optional[str] = None,
) -> str:
    """Return an internal watch link that opens the in-app player."""
    title_str = (title or "").strip()
    if not title_str:
        return "#"
    token = _watch_token_from_title(title_str)
    return f"?watch_token={token}&watch_title={quote_plus(title_str)}"


@st.cache_data(ttl=10 * 60)
def fetch_banner_image_url(
    title: Optional[str],
    poster_path: Optional[str],
    tmdb_api_key: Optional[str],
    size: str = "w1280",
) -> str:
    """Prefer TMDB backdrop image (landscape) for banner usage."""
    def _fit_for_banner(raw_url: Optional[str]) -> str:
        """Force an input image into banner-friendly dimensions using a free transformer."""
        if not raw_url:
            return PLACEHOLDER_URL
        try:
            return (
                "https://wsrv.nl/?url="
                + quote_plus(raw_url)
                + "&w=1600&h=640&fit=cover&output=webp&q=80"
            )
        except Exception:
            return raw_url

    title_str = (title or "").strip()
    if tmdb_api_key and title_str:
        try:
            url = "https://api.themoviedb.org/3/search/movie"
            resp = requests.get(
                url,
                params={"api_key": tmdb_api_key, "query": title_str, "page": 1},
                timeout=6,
            )
            resp.raise_for_status()
            results = resp.json().get("results") or []
            for r in results:
                backdrop_path = r.get("backdrop_path")
                if backdrop_path:
                    return _fit_for_banner(f"https://image.tmdb.org/t/p/{size}{backdrop_path}")
                # try full image list to find a landscape-friendly backdrop
                movie_id = r.get("id")
                if movie_id:
                    try:
                        imgs = requests.get(
                            f"https://api.themoviedb.org/3/movie/{movie_id}/images",
                            params={"api_key": tmdb_api_key, "include_image_language": "en,null"},
                            timeout=6,
                        )
                        imgs.raise_for_status()
                        backdrops = imgs.json().get("backdrops") or []
                        for b in backdrops:
                            w = b.get("width") or 0
                            h = b.get("height") or 1
                            ar = w / h
                            if w >= 1200 and 1.5 <= ar <= 2.35 and b.get("file_path"):
                                return _fit_for_banner(f"https://image.tmdb.org/t/p/original{b['file_path']}")
                    except Exception:
                        pass
            # if none have backdrops, fall through to fallback
        except Exception:
            pass
    # fallback: fit poster into banner crop if available
    poster_url = fetch_poster_url(title_str, poster_path, tmdb_api_key, size="w780")
    if poster_url and poster_url != PLACEHOLDER_URL:
        return _fit_for_banner(poster_url)
    # last resort: return placeholder so old/non-fitting posters are not reused
    return PLACEHOLDER_URL


def show_movie_card(movie: dict[str, object], tmdb_api_key: Optional[str]):
    """Render a clickable movie poster card with rating."""
    poster_value = movie.get('poster_path')
    title_str = str(movie.get("title", ""))
    media_type = str(movie.get("media_type", "movie")).strip().lower()
    
    poster = fetch_poster_url(
        title_str,
        poster_value if isinstance(poster_value, str) else None,
        tmdb_api_key,
        size="w500",
    )
    if not poster or poster.strip() == PLACEHOLDER_URL:
        poster = PLACEHOLDER_URL
    
    watch_url = fetch_watch_url(title_str, tmdb_api_key, media_type=media_type)
    
    esc = lambda v: html.escape(str(v), quote=True)
    st.markdown(
        f"""
        <a href="{esc(watch_url)}" target="_blank" style="text-decoration:none;">
        <div class='poster-card'>
            <div class='poster-frame'>
                <img src='{esc(poster)}' class='poster-img' alt='{esc(title_str)} poster' loading='lazy'>
                <div class='poster-overlay'></div>
            </div>
            <div class='poster-meta'>
                <div class='poster-title'>{esc(movie.get('title','Untitled'))}</div>
                <div class='poster-sub'>{esc(movie.get('genres','N/A'))}</div>
            </div>
        </div>
        </a>
        """,
        unsafe_allow_html=True,
    )
    
    # Rating comes from TMDB at scrape time; only look it up when it's missing
    rating_val = pd.to_numeric(movie.get("rating"), errors="coerce")
    if pd.isna(rating_val):
        tmdb_info = fetch_tmdb_info(title_str, tmdb_api_key)
        rating_val = pd.to_numeric(tmdb_info.get("rating"), errors="coerce")
    if pd.isna(rating_val):
        rating_val = 0.0
    
    st.markdown(render_star_rating(float(rating_val)), unsafe_allow_html=True)
    st.caption(f"Director: {movie.get('director', '—')}")
    
    # Favorite button
    if st.button("⭐ Add to favorites", key=f"fav-{movie.get('movie_id') or title_str}", use_container_width=True):
        favs = st.session_state.setdefault("favorites", [])
        if not any(m.get('title', '') == title_str for m in favs):
            favs.append(movie)
            st.session_state["favorites"] = favs
            record_feedback(
                mood=st.session_state.get("detected_mood"),
                movie=movie,
                favorite=True,
            )
            st.success("Added to favorites!")

    

@st.cache_data(ttl=10 * 60)
def _get_all_genres() -> list[str]:
    df = _apply_content_filter(load_movies())
    if "genres" not in df.columns or df.empty:
        return []
    genres = set()
    for raw in df["genres"].fillna("").astype(str).tolist():
        if not raw.strip():
            continue
        # support common separators
        for g in raw.replace(",", "|").replace("/", "|").split("|"):
            g = g.strip()
            if g:
                genres.add(g)
    return sorted(genres, key=lambda x: x.lower())

def _resolve_movie_by_title(title: str) -> Optional[dict]:
    """Find a movie dict matching a title (case-insensitive) from the catalog."""
    if not title:
        return None
    try:
        df = load_movies()
        if df.empty or "title" not in df.columns:
            return None
        mask = df["title"].fillna("").str.lower() == title.lower()
        if mask.any():
            return df[mask].iloc[0].to_dict()
        choices = df["title"].dropna().astype(str).tolist()
        match = get_close_matches(title, choices, n=1, cutoff=0.75)
        if match:
            pick = df[df["title"] == match[0]].iloc[0]
            return pick.to_dict()
    except Exception:
        return None
    return None


def _clear_watch_param():
    """Remove watch query param to allow normal navigation."""
    try:
        st.query_params.clear()
    except Exception:
        pass


def _watch_token_from_title(title: str) -> str:
    """Create a stable, URL-safe token from a title."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", title.lower()).strip("-")
    return slug or "untitled"


def _title_from_token(token: str) -> str:
    """Best-effort conversion from a watch token back to a readable title."""
    token = (token or "").strip("-")
    if not token:
        return ""
    words = token.replace("-", " ")
    return " ".join(w.capitalize() for w in words.split())


def _apply_content_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataset by sidebar content type selection when media_type exists."""
    if df.empty:
        return df
    selected = set(content_types or [])
    if selected == {"Movies", "TV Shows"} or not selected:
        return df
    if "media_type" not in df.columns:
        return df
    allowed = set()
    if "Movies" in selected:
        allowed.add("movie")
    if "TV Shows" in selected:
        allowed.add("tv")
    if not allowed:
        return df.iloc[0:0]
    mt = df["media_type"].fillna("").astype(str).str.lower()
    return df[mt.isin(allowed)]


def _pick_varied(df: pd.DataFrame, n: int, seed: int, recent_titles: set[str]) -> pd.DataFrame:
    """Pick n rows at random, favoring higher ratings and avoiding the last batch.

    The same seed gives the same pick, so the list only changes when the caller
    changes the seed (on "Re-check mood"), not on every Streamlit rerun.
    """
    if len(df) <= n:
        return df
    rating = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0).clip(0, 10).to_numpy()
    # each rating point makes a title ~2.7x likelier; tuned so picks average ~8/10
    # while consecutive batches share only ~4 of 30 titles
    weights = np.exp(rating - 10.0)
    shown_last = df["title"].fillna("").astype(str).isin(recent_titles).to_numpy()
    weights = np.where(shown_last, weights * 0.25, weights)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=n, replace=False, p=weights / weights.sum())
    return df.iloc[idx]


def _sort_matches_by_rating(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "rating" in out.columns:
        out.loc[:, "rating"] = pd.to_numeric(out["rating"], errors="coerce")
        out = out.sort_values(by="rating", ascending=False)
    return out


st.markdown("<div class='app-shell'>", unsafe_allow_html=True)
st.session_state.setdefault("page", "home")
topbar_left, topbar_mid, topbar_right = st.columns([1.6, 3.4, 2.4])
with topbar_left:
    st.markdown("<div class='brand'>Filmi<span> Duniya</span></div>", unsafe_allow_html=True)
    if st.session_state.get("signed_in"):
        st.caption(f"Signed in: {st.session_state.get('user_email','guest')}")
    else:
        st.caption("Guest mode • sign in to sync favorites")
with topbar_mid:
    nav_cols = st.columns(3)
    nav_items = [("home", "Home"), ("trending", "Trending"), ("favorites", "Favorites")]
    for (slug, label), col in zip(nav_items, nav_cols):
        with col:
            active = st.session_state.get("page") == slug
            btn = st.button(
                label,
                key=f"nav-{slug}",
                use_container_width=True,
                type="primary" if active else "secondary",
            )
            if btn:
                st.session_state["page"] = slug
                st.session_state.pop("selected_movie", None)
                _clear_watch_param()
with topbar_right:
    st.markdown("<div class='search-shell'>", unsafe_allow_html=True)
    typed_title = st.text_input(
        "Search",
        placeholder="Search",
        label_visibility="collapsed",
        key="search_title",
    )
    st.markdown("</div>", unsafe_allow_html=True)

typed_query = str(typed_title or "").strip()

st.session_state.setdefault("page", "home")
st.session_state.setdefault("user_ratings", {})
selected_genres: list[str] = []

def _build_banner_pool(tmdb_key: Optional[str]) -> list[dict]:
    """Use a stable set of known 16:9 backdrops to avoid missing/blank banners."""
    return [
        {"title": "Inception", "genres": "Action|Sci-Fi", "banner_url": "https://image.tmdb.org/t/p/original/s3TBrRGB1iav7gFOCNx3H31MoES.jpg"},
        {"title": "Interstellar", "genres": "Adventure|Drama", "banner_url": "https://image.tmdb.org/t/p/original/rAiYTfKGqDCRIIqo664sY9XZIvQ.jpg"},
        {"title": "The Dark Knight", "genres": "Action|Crime", "banner_url": "https://image.tmdb.org/t/p/original/hqkIcbrOHL86UncnHIsHVcVmzue.jpg"},
        {"title": "Dune: Part Two", "genres": "Sci-Fi|Adventure", "banner_url": "https://image.tmdb.org/t/p/original/1pdfLvkbY9ohJlCjQH2CZjjYVvJ.jpg"},
        {"title": "Spider-Man: Into the Spider-Verse", "genres": "Animation|Action", "banner_url": "https://image.tmdb.org/t/p/original/iiZZdoQBEYBv6id8su7ImL0oCbD.jpg"},
        {"title": "Mad Max: Fury Road", "genres": "Action|Adventure", "banner_url": "https://image.tmdb.org/t/p/original/1g0dhYtq4irTY1GPXvft6k4YLjm.jpg"},
    ]


banner_movies = _build_banner_pool(TMDB_API_KEY or None)

if "hero_index" not in st.session_state:
    st.session_state["hero_index"] = 0
elif banner_movies:
    st.session_state["hero_index"] = (st.session_state["hero_index"] + 1) % len(banner_movies)

banner_index = st.session_state["hero_index"] % max(1, len(banner_movies))
banner_main = banner_movies[banner_index] if banner_movies else {"title": "Featured Movies", "genres": "Movies|Series"}
banner_token = _watch_token_from_title(str(banner_main.get("title", "")))
st.session_state.setdefault("watch_map", {})[banner_token] = banner_main
banner_poster = (
    banner_main.get("banner_url")
    or fetch_banner_image_url(
        str(banner_main.get("title", "")),
        banner_main.get("poster_path") if isinstance(banner_main.get("poster_path"), str) else None,
        TMDB_API_KEY or None,
        size="original",
    )
)
banner_desc = str(banner_main.get("description", "") or banner_main.get("overview", "")).strip()
if banner_desc:
    banner_desc = banner_desc[:170] + ("..." if len(banner_desc) > 170 else "")
else:
    banner_desc = "Live recommendations based on your mood and learning profile."
banner_watch_url = f"?watch_token={banner_token}&watch_title={quote_plus(str(banner_main.get('title','')))}"
st.markdown(
    f"""
    <a href="{html.escape(banner_watch_url, quote=True)}" target="_blank" style="text-decoration:none;">
    <div class='banner-panel' style="
      background:
        linear-gradient(90deg, rgba(15,15,15,0.88) 0%, rgba(15,15,15,0.55) 45%, rgba(15,15,15,0.05) 100%),
        linear-gradient(0deg, rgba(15,15,15,0.7) 0%, rgba(15,15,15,0) 55%),
        url('{html.escape(str(banner_poster), quote=True)}') center/cover no-repeat;
    ">
      <div class='banner-content'>
        <div class='banner-kicker'>Featured for your mood</div>
        <div class='banner-title'>{html.escape(str(banner_main.get('title','Featured Movies')))}</div>
        <div class='banner-sub'>{html.escape(banner_desc)}</div>
      </div>
    </div>
    </a>
    """,
    unsafe_allow_html=True,
)

all_genres = _get_all_genres()
st.session_state.setdefault("selected_genres_set", set())
selected_genres_set = set(st.session_state.get("selected_genres_set", set()))
if all_genres:
    st.markdown("<div class='genre-strip'>", unsafe_allow_html=True)
    st.markdown("<div class='strip-title'>Genres</div>", unsafe_allow_html=True)
    cols = st.columns(5)
    for idx, g in enumerate(all_genres[:25]):
        with cols[idx % 5]:
            checked = st.checkbox(g, value=g in selected_genres_set, key=f"genre-{g}")
            if checked:
                selected_genres_set.add(g)
            else:
                selected_genres_set.discard(g)
    st.markdown("</div>", unsafe_allow_html=True)
selected_genres = sorted(selected_genres_set)
st.session_state["selected_genres_set"] = selected_genres_set

manual_override = bool(typed_query or selected_genres)

# If a watch param is present, route to watch page (token first, then title)
qp = st.query_params
watch_token = qp.get("watch_token") or None
if watch_token and st.session_state.get("watch_map", {}).get(watch_token):
    st.session_state["selected_movie"] = st.session_state["watch_map"][watch_token]
    st.session_state["page"] = "watch"
    st.query_params.clear()
else:
    target_title = None
    if watch_token:
        target_title = _title_from_token(watch_token)
    if qp.get("watch_title"):
        target_title = qp.get("watch_title")
    if not target_title and qp.get("watch"):
        target_title = qp.get("watch")
    found = _resolve_movie_by_title(target_title) if target_title else None
    if found:
        st.session_state["selected_movie"] = found
        st.session_state["page"] = "watch"
        st.query_params.clear()
    if watch_token or target_title:
        st.query_params.clear()


def render_watch_page(movie: dict[str, object], tmdb_api_key: Optional[str]):
    if st.button("Back to Home"):
        st.session_state["page"] = "home"
        st.session_state.pop("selected_movie", None)
        st.query_params.clear()
        st.rerun()

    tmdb_info = fetch_tmdb_info(str(movie.get("title", "")), tmdb_api_key)
    poster = fetch_poster_url(
        str(movie.get("title", "")),
        movie.get("poster_path") if isinstance(movie.get("poster_path"), str) else None,
        tmdb_api_key,
        size="w1280",
    )
    st.markdown("<div class='hero'>", unsafe_allow_html=True)
    st.image(poster, use_container_width=True)
    st.markdown(
        f"<div class='hero-title'>{html.escape(str(movie.get('title','Untitled')))}</div>",
        unsafe_allow_html=True,
    )
    rating_val = tmdb_info.get("rating") if tmdb_info else movie.get("rating", 0)
    st.markdown(render_star_rating(rating_val), unsafe_allow_html=True)
    desc = tmdb_info.get("overview") if tmdb_info else ""
    if not desc:
        desc = movie.get("description", "") or movie.get("overview", "")
    if desc:
        st.write(desc)
    st.markdown(f"**Genres:** {movie.get('genres', '')}")
    st.markdown(f"**Director:** {movie.get('director', '')}")
    st.markdown("<div class='section-title'>Watch</div>", unsafe_allow_html=True)
    video_db = load_video_sources()
    custom_url = _resolve_video_url(str(movie.get("title", "")), video_db)
    trailer = fetch_trailer_url(str(movie.get("title", "")), tmdb_api_key)
    sources = []
    if custom_url:
        sources.append(("saved", "Saved stream (your URL)", custom_url))
    if trailer:
        sources.append(("trailer", "Trailer (YouTube)", trailer))
    sources.append(("sample", "Sample stream (public domain)", PUBLIC_DOMAIN_STREAM))
    source_labels = {k: label for k, label, _ in sources}
    source_map = {k: url for k, _, url in sources}
    default_idx = 0
    source_choice = st.selectbox(
        "Choose a stream source",
        options=[k for k, _, _ in sources],
        index=default_idx,
        format_func=lambda k: source_labels.get(k, k),
        key=f"source-{movie.get('title','')}",
    )
    play_url = source_map.get(source_choice, PUBLIC_DOMAIN_STREAM)
    yt_embed = _youtube_embed_url(play_url)
    if yt_embed:
        components.html(
            f"""
            <div style='position:relative;padding-top:56.25%;'>
                <iframe src="{yt_embed}" title="Trailer player" frameborder="0"
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                        allowfullscreen
                        style="position:absolute; inset:0; width:100%; height:100%; border-radius:14px;">
                </iframe>
            </div>
            """,
            height=400,
        )
        st.caption("Trailer is embedded from YouTube. If it doesn't load, open directly in a new tab below.")
        st.markdown(f"[Open trailer on YouTube]({play_url})")
    else:
        st.video(play_url)
    st.caption(
        "Click play to start instantly. You can swap the source above or paste your own URL below."
    )

    st.markdown("Add / override stream URL for this title:")
    new_url = st.text_input("Video URL", value=custom_url or "", key=f"vid-url-{movie.get('title','')}")
    if st.button("Save video URL", key=f"save-url-{movie.get('title','')}"):
        if new_url.strip():
            save_video_source(str(movie.get("title", "")), new_url.strip(), overwrite=True)
            st.success("Saved video URL for this movie.")
            st.query_params.clear()
            st.rerun()
    st.markdown("<div class='section-title'>Your Rating</div>", unsafe_allow_html=True)
    current = st.session_state["user_ratings"].get(movie.get("title", ""), 0.0)
    user_rating = st.slider("Rate this movie", 0.0, 5.0, float(current), 0.5, key=f"rating-{movie.get('title','')}")
    col_rate1, col_rate2 = st.columns([1, 1])
    with col_rate1:
        if st.button("👍 Like", key=f"like-{movie.get('title','')}"):
            user_rating = max(user_rating, 4.0)
            st.session_state["user_ratings"][movie.get("title", "")] = user_rating
            record_feedback(
                mood=st.session_state.get("detected_mood"),
                movie=movie,
                rating=float(user_rating),
            )
            st.success(f"Saved as liked ({user_rating:.1f} stars).")
    with col_rate2:
        if st.button("💾 Save Rating", key=f"save-rating-{movie.get('title','')}"):
            st.session_state["user_ratings"][movie.get("title", "")] = user_rating
            record_feedback(
                mood=st.session_state.get("detected_mood"),
                movie=movie,
                rating=float(user_rating),
            )
            st.success(f"Saved your rating: {user_rating:.1f}")


def render_trending_page(tmdb_api_key: Optional[str]):
    st.header("Trending Now")
    items = _get_featured_new_releases(limit=18)
    if not items:
        st.info("Trending data unavailable.")
        return
    per_row = 3
    rows = [items[i:i + per_row] for i in range(0, len(items), per_row)]
    for row in rows:
        cols = st.columns(len(row))
        for c, r in zip(cols, row):
            with c:
                show_movie_card(r, tmdb_api_key)


def render_favorites_page(tmdb_api_key: Optional[str]):
    st.header("Your Favorites")
    favs = st.session_state.get("favorites", [])
    if not favs:
        st.info("No favorites yet. Add some from Home or Trending.")
        return
    per_row = 3
    rows = [favs[i:i + per_row] for i in range(0, len(favs), per_row)]
    for row in rows:
        cols = st.columns(len(row))
        for c, r in zip(cols, row):
            with c:
                show_movie_card(r, tmdb_api_key)
    st.markdown("---")
    if st.button("Export favorites to server", key="export-favs"):
        p = repo_root / "data"
        p.mkdir(exist_ok=True)
        out = p / "favorites.json"
        with out.open("w", encoding="utf8") as f:
            json.dump(favs, f, ensure_ascii=False, indent=2)
        st.success(f"Exported favorites to {out}")
    buf = io.BytesIO()
    buf.write(json.dumps(favs, ensure_ascii=False, indent=2).encode('utf-8'))
    buf.seek(0)
    st.download_button("Download favorites (JSON)", data=buf, file_name="favorites.json", mime="application/json", key="dl-favs")


@st.cache_data(ttl=10 * 60)
def _get_featured_new_release():
    df = _apply_content_filter(load_movies())
    if df.empty:
        return None
    if "popularity" in df.columns:
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
        if df["popularity"].notna().any():
            return df.sort_values(by="popularity", ascending=False).iloc[0].to_dict()
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        if df["rating"].notna().any():
            return df.sort_values(by="rating", ascending=False).iloc[0].to_dict()
    return df.iloc[0].to_dict()


@st.cache_data(ttl=10 * 60)
def _get_featured_new_releases(limit: int = 5) -> list[dict]:
    df = _apply_content_filter(load_movies())
    if df.empty:
        return []
    if "popularity" in df.columns:
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
        df = df.sort_values(by="popularity", ascending=False)
    elif "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df = df.sort_values(by="rating", ascending=False)
    return df.head(limit).to_dict(orient="records")


def _maybe_run_daily_scraper(tmdb_api_key: Optional[str]) -> None:
    """Run scraper once every 24h (movies + TV) to refresh local DB/posters."""
    if not tmdb_api_key:
        return
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    stamp_path = data_dir / ".last_daily_scrape_ts"
    lock_path = data_dir / ".daily_scrape.lock"
    now = time.time()
    interval_seconds = 86400

    # if a recent lock exists, avoid concurrent runs
    if lock_path.exists():
        try:
            if now - lock_path.stat().st_mtime < 2 * 3600:
                return
        except Exception:
            return

    last_ts = 0.0
    if stamp_path.exists():
        try:
            last_ts = float(stamp_path.read_text(encoding="utf-8").strip() or "0")
        except Exception:
            last_ts = 0.0

    if now - last_ts < interval_seconds:
        return

    try:
        lock_path.write_text(str(now), encoding="utf-8")
        scrape_top_n_movies(
            n=400,
            append=True,
            max_per_run=200,
            include_tv=True,
            force=False,
        )
        stamp_path.write_text(str(now), encoding="utf-8")
    except Exception:
        pass
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass


_maybe_run_daily_scraper(TMDB_API_KEY or None)

@st.cache_data(ttl=60 * 60)
def fetch_tmdb_info(title: str, tmdb_api_key: Optional[str]) -> dict:
    if not tmdb_api_key or not title:
        return {}
    try:
        url = "https://api.themoviedb.org/3/search/movie"
        resp = requests.get(
            url,
            params={"api_key": tmdb_api_key, "query": title, "page": 1},
            timeout=5,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if results:
            top = results[0]
            return {
                "rating": top.get("vote_average"),
                "vote_count": top.get("vote_count"),
                "overview": top.get("overview"),
                "poster_path": top.get("poster_path"),
            }
    except Exception:
        pass
    return {}


def render_star_rating(rating_out_of_10: float) -> str:
    try:
        r10 = float(rating_out_of_10)
    except (TypeError, ValueError):
        r10 = 0.0
    r5 = max(0.0, min(5.0, r10 / 2))
    full = int(round(r5))
    stars = ""
    for i in range(5):
        cls = "star-on" if i < full else "star-off"
        stars += f"<span class='{cls}'>★</span>"
    return f"""
    <div class='rating-stars'>
        <div class='rating-badge'>Rating {r5:.1f}</div>
        <div class='stars'>{stars}</div>
    </div>
    """

# ensure watch/detail routing happens after helpers are defined
if st.session_state.get("page") == "watch" and st.session_state.get("selected_movie"):
    render_watch_page(st.session_state["selected_movie"], TMDB_API_KEY or None)
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

if "favorites" not in st.session_state:
    st.session_state["favorites"] = []

if st.session_state.get("page") == "trending":
    render_trending_page(TMDB_API_KEY or None)
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

if st.session_state.get("page") == "favorites":
    render_favorites_page(TMDB_API_KEY or None)
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

st.markdown("<div class='section-title'>Recommendations</div>", unsafe_allow_html=True)

MOOD_EMOJI = {
    "happy": "😄", "sad": "😢", "angry": "😠",
    "anxious": "😰", "excited": "🤩", "calm": "😌",
}


def _on_mood_pick() -> None:
    picked = st.session_state.get("mood_pick")
    if picked:
        # a manual pick wins over the camera until the user asks to scan again
        st.session_state["detected_mood"] = picked
        st.session_state["last_mood_source"] = "picked by you"
        st.session_state["mood_locked"] = True
    else:
        st.session_state["mood_locked"] = False
        st.session_state["last_private_scan_ts"] = 0.0  # scan right away


def _on_rescan() -> None:
    st.session_state["mood_locked"] = False
    st.session_state["mood_pick"] = None


# Mood-based flow using backend-only private sensing
if mode == "Mood Based":
    st.session_state.setdefault("recent_detected_moods", [])
    st.session_state.setdefault("last_private_scan_ts", 0.0)
    st.session_state.setdefault("scan_interval_sec", 6.0)

    now = time.time()
    should_scan = (
        ("detected_mood" not in st.session_state)
        or (
            now - float(st.session_state.get("last_private_scan_ts", 0.0))
            >= float(st.session_state.get("scan_interval_sec", 6.0))
        )
    )

    mood_col, rescan_col = st.columns([4, 1])
    with rescan_col:
        rescan_clicked = st.button(
            "🔄 Re-check mood",
            key="rescan-mood",
            use_container_width=True,
            help="Scan your face again and refresh the suggestions",
            on_click=_on_rescan,
        )

    if manual_override:
        should_scan = False
        mood_col.caption(
            "Mood scanning paused while you search or filter genres. "
            "Clear them to see mood suggestions."
        )
    if st.session_state.get("mood_locked"):
        should_scan = False
    st.session_state.setdefault("mood_pick_seed", int(time.time()))
    if rescan_clicked:
        should_scan = True
        st.session_state["mood_pick_seed"] += 1
    if should_scan:
        st.session_state["last_private_scan_ts"] = now
        try:
            final_details = detect_face_mood_backend()
        except (RuntimeError, OSError) as exc:
            # surface the reason; failing silently here once hid a broken face detector
            mood_col.caption(f"Mood scan unavailable: {exc}")
            final_details = None

        mood = final_details["mood"] if final_details else None
        if mood:
            st.session_state["last_mood_source"] = str(final_details.get("source", "unknown"))
            recent = st.session_state.get("recent_detected_moods", [])
            recent.append(mood)
            st.session_state["recent_detected_moods"] = recent[-6:]
            st.session_state["detected_mood"] = mood
        elif final_details is None and rescan_clicked:
            mood_col.caption("No face found. Face the camera with good lighting and try again.")

    if st.session_state.get("detected_mood"):
        mood_col.caption(
            f"Detected mood: **{str(st.session_state['detected_mood']).upper()}** "
            f"| Source: {st.session_state.get('last_mood_source', 'unknown')}"
        )
    with mood_col:
        st.pills(
            "Not right? Pick your mood",
            options=list(MOOD_TO_GENRES.keys()),
            format_func=lambda m: f"{MOOD_EMOJI.get(m, '')} {m.title()}",
            key="mood_pick",
            on_change=_on_mood_pick,
        )
    if st.session_state.get("mood_locked") and not manual_override:
        mood_col.caption("Camera paused while using your pick. Click 🔄 Re-check mood to scan again.")
    st.markdown("---")

# auto-search on input/genre selection (no button)
detected_mood = st.session_state.get("detected_mood")
title = typed_query

if title:
    base = _apply_content_filter(load_movies())
    exact = base[base["title"].fillna("").str.lower() == title.lower()]
    if exact.empty:
        matches = base[base["title"].fillna("").str.contains(title, case=False, na=False, regex=False)]
    else:
        matches = exact
    if matches.empty:
        results = []
    else:
        matches = _sort_matches_by_rating(matches)
        results = matches.head(NUM).to_dict(orient="records")
        if detected_mood and not manual_override:
            results = rerank_results_for_learning(results, detected_mood)
elif selected_genres:
    df = _apply_content_filter(load_movies())
    mask = df["genres"].fillna("").str.lower().apply(
        lambda x: any(g.lower() in x for g in selected_genres)
    )
    matches = df[mask]
    if matches.empty:
        results = []
    else:
        matches = _sort_matches_by_rating(matches)
        results = matches.head(NUM).to_dict(orient="records")
elif mode == "Mood Based" and detected_mood:
    base = _apply_content_filter(load_movies())
    genres = MOOD_TO_GENRES.get(detected_mood, [])
    if genres:
        mask = base["genres"].fillna("").str.lower().apply(
            lambda x: any(g.lower() in x for g in genres)
        )
        matches = base[mask]
        if matches.empty:
            results = []
        else:
            pick_key = (detected_mood, st.session_state.get("mood_pick_seed", 0))
            if st.session_state.get("mood_pick_key") != pick_key:
                # new batch: remember what the previous one showed so it's avoided
                st.session_state["mood_prev_titles"] = st.session_state.get("mood_shown_titles", [])
                st.session_state["mood_pick_key"] = pick_key
            picked = _pick_varied(
                matches,
                NUM,
                seed=zlib.crc32(f"{pick_key[0]}:{pick_key[1]}".encode()),
                recent_titles=set(st.session_state.get("mood_prev_titles", [])),
            )
            st.session_state["mood_shown_titles"] = picked["title"].astype(str).tolist()
            results = rerank_results_for_learning(picked.to_dict(orient="records"), detected_mood)
    else:
        results = []
else:
    results = []

if title:
    if not results:
        st.info("No local matches for that title. Try another name.")
    else:
        per_row = 3
        rows = [results[i:i+per_row] for i in range(0, len(results), per_row)]
        for row in rows:
            cols = st.columns(len(row))
            for c, r in zip(cols, row):
                with c:
                    show_movie_card(r, TMDB_API_KEY or None)
elif mode == "Mood Based" and detected_mood:
    if not results:
        st.info("No movies found for the detected mood. Try a different title or re-scan.")
    else:
        per_row = 3
        rows = [results[i:i+per_row] for i in range(0, len(results), per_row)]
        for row in rows:
            cols = st.columns(len(row))
            for c, r in zip(cols, row):
                with c:
                    show_movie_card(r, TMDB_API_KEY or None)
elif selected_genres:
    if not results:
        st.info("No movies found for the selected genres. Try different genres.")
    else:
        per_row = 3
        rows = [results[i:i+per_row] for i in range(0, len(results), per_row)]
        for row in rows:
            cols = st.columns(len(row))
            for c, r in zip(cols, row):
                with c:
                    show_movie_card(r, TMDB_API_KEY or None)
else:
    if mode == "Mood Based":
        if not st.session_state.get("private_scan_allowed", False):
            st.warning("Allow to start mood-based recommendations.")  
        else:
            st.warning("Mood scan is running in backend. Recommendations will appear shortly.")
    else:
        st.warning("Type a title or pick genres first.")

if mode == "Mood Based":
    summary = learning_summary()
    progress = learning_progress(target_events=200)
    st.caption(
        f"Learning events: {summary.get('feedback_events', 0)} | Mood history: {summary.get('mood_counts', {})}"
    )
    st.caption(
        f"Model training progress: {progress.get('progress_pct', 0.0):.1f}% ({progress.get('events', 0)}/{progress.get('target_events', 200)}) • Stage: {progress.get('stage', 'cold-start')}"
    )
    st.progress(min(1.0, float(progress.get("progress_pct", 0.0)) / 100.0))

# Manual refresh button
st.sidebar.markdown("---")
st.sidebar.write("No backend API configured — the app reads local CSV and can trigger the embedded scraper.")

# Auto-refresh loop: DISABLED per task requirements

# auto-rotate hero once per refresh (POLL_INTERVAL) to keep movement without extra reloads
if st.session_state.get("page") == "home":
    featured_list = banner_movies or _get_featured_new_releases(limit=8)
    if featured_list:
        hero_index = st.session_state.get("hero_index", 0) % len(featured_list)
        st.session_state["hero_index"] = hero_index

st.markdown("</div>", unsafe_allow_html=True)
