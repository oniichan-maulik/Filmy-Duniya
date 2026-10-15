import streamlit as st
import streamlit.components.v1 as components
import requests
from typing import Optional
from pathlib import Path
import json
import io
from difflib import get_close_matches
import re
import sys
import os
import time
import threading
import queue
from urllib.parse import quote_plus
import pandas as pd

try:
    import sounddevice as _sd  # noqa: F401
    _HAS_SOUNDDEVICE = True
except Exception:
    _HAS_SOUNDDEVICE = False

# Ensure project root is on sys.path so `src` imports work when Streamlit
# runs with a different CWD. This inserts the repo root (one level up from
# this `web/` folder) at the front of sys.path.
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.config import TMDB_API_KEY as DEFAULT_TMDB_KEY
from src.data__loader import load_movies
from src.recomender import MOOD_TO_GENRES
from src.scraper import scrape_top_n_movies
from src.user_store import (
    learning_progress,
    learning_summary,
    record_feedback,
    rerank_results_for_learning,
)
from src.multimodal_mood import (
    detect_multimodal_mood_backend,
)

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

# Configure default emotion model path only if a known local model file exists.
_MODEL_CANDIDATES = [
    repo_root / "models" / "emotion_model.h5",
    repo_root / "src" / "fer2013_mini_XCEPTION.102-0.66.hdf5",
]
for _p in _MODEL_CANDIDATES:
    if _p.exists():
        os.environ.setdefault("EMOTION_MODEL_PATH", str(_p))
        break

st.set_page_config(page_title="Flimi Duniya", layout="wide")

# cinematic dark UI styles
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600&display=swap');
    :root {
        --bg: #05050b;
        --panel: #0b0b15;
        --card: #0f1024;
        --card-2: #0d1324;
        --accent: #9d64ff;
        --accent-2: #6ee3ff;
        --text: #f4f4ff;
        --muted: #b7b9d6;
        --stroke: #1d1d2f;
        --glow: 0 0 28px rgba(157,100,255,0.32);
    }
    .stApp {
        background:
            radial-gradient(1200px 900px at 80% 10%, rgba(157,100,255,0.16), transparent 60%),
            radial-gradient(900px 800px at 6% 16%, rgba(110,227,255,0.12), transparent 55%),
            linear-gradient(180deg, #06060e, var(--bg));
        color: var(--text);
        font-family: "Sora", "Space Grotesk", system-ui, sans-serif;
    }
    h1, h2, h3, h4 {letter-spacing: 0.01em;}
    .block-container {padding-top: 18px; padding-bottom: 72px;}
    .stMarkdown, .stTextInput label, .stSlider label, .stCheckbox label {color: var(--text);}
    .stTextInput > div > div > input {
        background: var(--card);
        border: 1px solid var(--stroke);
        color: var(--text);
        border-radius: 14px;
        height: 46px;
        padding-left: 16px;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
    }
    .stButton > button {
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        color: #06060e;
        border: 0;
        border-radius: 12px;
        padding: 10px 16px;
        font-weight: 700;
        box-shadow: var(--glow);
    }
    .stButton > button:hover {filter: brightness(1.07);}    
    .app-shell {max-width: 1220px; margin: 0 auto;}
    .topbar {
        display:flex; align-items:center; gap:14px; justify-content:space-between;
        background: linear-gradient(90deg, rgba(17,17,30,0.92), rgba(17,18,35,0.82));
        border:1px solid var(--stroke);
        padding: 14px 18px; border-radius: 18px; margin-bottom: 20px;
        backdrop-filter: blur(10px); box-shadow: var(--glow);
    }
    .brand {font-size: 22px; font-weight: 800; letter-spacing: 0.04em;}
    .brand span {color: var(--accent); text-shadow: var(--glow);}
    .pill-nav {display:flex; flex-wrap:wrap; gap:10px; align-items:center;}
    .pill {
        padding:8px 12px; border-radius:12px; border:1px solid var(--stroke);
        background: rgba(255,255,255,0.03); color: var(--muted); font-weight:600; font-size:13px;
        transition: all 0.15s ease;
    }
    .pill:hover {border-color: rgba(157,100,255,0.5); color: var(--text); box-shadow: var(--glow);}
    .pill.active {background: linear-gradient(90deg, rgba(157,100,255,0.18), rgba(110,227,255,0.12)); color: var(--text);}
    .search-shell {width:100%;}
    .subtext {color: var(--muted); margin: 0 0 12px 2px;}
    .hero {
        border:1px solid var(--stroke);
        background: linear-gradient(135deg, rgba(15,15,28,0.92), rgba(12,17,34,0.94));
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 24px 60px rgba(0,0,0,0.4);
        margin-bottom: 24px;
    }
    .hero-tag {
        display:inline-block; font-size:11px; font-weight:700; letter-spacing:0.12em;
        text-transform:uppercase; color:#dfe2ff; background:rgba(157,100,255,0.14);
        border:1px solid rgba(157,100,255,0.4); padding:4px 10px; border-radius:999px; margin-bottom:10px;
    }
    .hero-title {font-size: 34px; margin: 0 0 10px 0;}
    .hero-meta {color: var(--muted); font-size: 13px; margin-bottom: 12px;}
    .movie-card {
        border:1px solid var(--stroke);
        background: linear-gradient(180deg, var(--card), var(--card-2));
        padding:12px; border-radius:16px;
        box-shadow: 0 10px 24px rgba(0,0,0,0.35);
    }
    .movie-title {margin: 10px 0 6px 0;}
    .section-title {font-size: 22px; margin: 10px 0 8px 0; text-transform: uppercase; letter-spacing: 0.06em;}
    .rating-stars {display:flex; align-items:center; gap:8px; margin: 6px 0 8px 0;}
    .stars {letter-spacing:2px; font-size:15px;}
    .star-on {color:#ffb703;}
    .star-off {color:#34344c;}
    .rating-badge {background:#0d0f1e; border:1px solid var(--stroke); padding:6px 10px; border-radius:10px; font-weight:700;}
    .banner-panel {
        position: relative;
        aspect-ratio: 16 / 7;
        min-height: 360px;
        border-radius: 22px;
        border: 1px solid var(--stroke);
        overflow: hidden;
        margin-bottom: 16px;
        box-shadow: 0 24px 56px rgba(0,0,0,0.45);
        filter: brightness(1.05);
    }
    .banner-content {
        position: absolute;
        left: 22px;
        bottom: 20px;
        max-width: 64%;
    }
    .banner-kicker {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: #d9dfff;
        margin-bottom: 6px;
        font-weight: 800;
    }
    .banner-title {
        font-size: 40px;
        line-height: 1.05;
        margin: 0 0 8px 0;
        text-shadow: var(--glow);
    }
    .banner-sub {
        color: #d1d7f5;
        font-size: 13px;
        line-height: 1.4;
    }
    .genre-strip {
        border: 1px solid var(--stroke);
        background: linear-gradient(180deg, rgba(15,15,28,0.92), rgba(15,18,34,0.92));
        border-radius: 12px;
        padding: 8px 10px;
        margin: 10px 0 10px 0;
        max-height: 88px;
        overflow-y: auto;
    }
    .strip-title {
        font-size: 11px;
        color: #d9dfff;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin: 0 0 4px 0;
        font-weight: 800;
    }
    .poster-card {position:relative; border-radius:16px; overflow:hidden; border:1px solid var(--stroke); background: var(--card); box-shadow: var(--glow); max-width: 210px; margin: 0 auto;}
    .poster-frame {position:relative;}
    .poster-img {width:100%; border-radius:14px; object-fit:cover; aspect-ratio: 2 / 3; max-height: 280px;}
    .poster-overlay {position:absolute; inset:0; background: linear-gradient(180deg, transparent 20%, rgba(5,5,11,0.7)); opacity:0; transition:opacity 0.2s ease;}
    .poster-card:hover .poster-overlay {opacity:1;}
    .poster-link {text-decoration:none; color:inherit; display:block;}
    .poster-link:focus {outline:none;}
    .poster-meta {padding:8px 6px 6px 4px;}
    .poster-title {font-size:14px; font-weight:700; margin:4px 0 3px 0; color: var(--text);}   
    .poster-sub {color: var(--muted); font-size:12px; letter-spacing:0.02em;}
    .pill-badge {padding:4px 8px; border-radius:8px; background:rgba(255,255,255,0.04); color:var(--muted); font-size:12px; border:1px solid var(--stroke);}
    ::-webkit-scrollbar {height: 8px; width: 8px;}
    ::-webkit-scrollbar-thumb {background: rgba(255,255,255,0.15); border-radius: 999px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# TMDB key: prefer Streamlit secrets, then env/config, otherwise empty
secrets_key = "785c5f1bd5e3e823f06abdfe6168588e"
try:
    secrets_key = str(st.secrets.get("TMDB_API_KEY", "785c5f1bd5e3e823f06abdfe6168588e")).strip()
except Exception:
    secrets_key = ""
default_key = secrets_key or (
    DEFAULT_TMDB_KEY if DEFAULT_TMDB_KEY and DEFAULT_TMDB_KEY != "785c5f1bd5e3e823f06abdfe6168588e" else ""
)
tmdb_input = st.sidebar.text_input("TMDB API key", type="password", value=default_key)
TMDB_API_KEY = (tmdb_input or "").strip()
if TMDB_API_KEY:
    st.sidebar.success("Using TMDB for posters and ratings.")
else:
    st.sidebar.warning("TMDB API key missing; posters/ratings may use placeholders.")

# Auto-populate trailers into the local video DB (YouTube links via TMDB).
if TMDB_API_KEY and not st.session_state.get("_trailers_prefetched", False):
    try:
        df_tmp = load_movies()
        lim = max(200, len(df_tmp)) if not df_tmp.empty else 200
        populate_video_db_with_trailers(TMDB_API_KEY, limit=lim)
        st.session_state["_trailers_prefetched"] = True
    except Exception:
        st.session_state["_trailers_prefetched"] = True

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

continuous_voice = False
NUM = st.sidebar.slider("Number of recommendations", min_value=1, max_value=50, value=30)
AUTO_REFRESH = False
POLL_INTERVAL = 10  # refresh interval in seconds
content_types = st.sidebar.multiselect(
    "Content type",
    options=["Movies", "TV Shows"],
    default=["Movies", "TV Shows"],
)

st.session_state.setdefault("voice_transcript", "")
st.session_state.setdefault("voice_mood", "calm")

if TMDB_API_KEY:
    with st.sidebar.expander("Trailer prefetch"):
        if st.button("Fetch trailers for all movies", use_container_width=True):
            total = load_movies().shape[0] if not load_movies().empty else 0
            added = populate_video_db_with_trailers(TMDB_API_KEY, limit=max(500, total))
            st.sidebar.success(f"Fetched {added} trailers into video database.")

with st.sidebar.expander("Database manager (edit & save)"):
    try:
        df_preview = load_movies().head(50)
        edited = st.sidebar.data_editor(df_preview, num_rows="dynamic", key="db_editor")
        if st.sidebar.button("Save edits to data/movies.csv", key="save-db-editor"):
            data_path = repo_root / "data" / "movies.csv"
            data_path.parent.mkdir(parents=True, exist_ok=True)
            edited.to_csv(data_path, index=False)
            st.sidebar.success(f"Saved {len(edited)} rows to {data_path}")
    except Exception as e:
        st.sidebar.warning(f"DB editor unavailable: {e}")

def _clean_poster_value(value: Optional[str]) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _friendly_transcript_issue(raw_error: str) -> str:
    text = str(raw_error or "").strip()
    if not text:
        return ""
    low = text.lower()
    if "unknownvalueerror" in low:
        return "Could not understand speech. Please speak a bit slower and closer to the mic."
    if "requesterror" in low or "google:" in low:
        return "Google speech service is unavailable right now."
    if "pocketsphinx not installed" in low:
        return "Offline speech fallback is not installed (PocketSphinx)."
    if "speech_recognition_missing" in low:
        return "Speech recognition package is missing on the server."
    return "Voice transcription is currently unavailable."


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
    
    st.markdown(
        f"""
        <a href="{watch_url}" target="_blank" style="text-decoration:none;">
        <div class='poster-card'>
            <div class='poster-frame'>
                <img src='{poster}' class='poster-img' alt='{title_str} poster' loading='lazy'>
                <div class='poster-overlay'></div>
            </div>
            <div class='poster-meta'>
                <div class='poster-title'>{movie.get('title','Untitled')}</div>
                <div class='poster-sub'>{movie.get('genres','N/A')}</div>
            </div>
        </div>
        </a>
        """,
        unsafe_allow_html=True,
    )
    
    # TMDB rating
    tmdb_info = fetch_tmdb_info(title_str, tmdb_api_key)
    rating_val = tmdb_info.get("rating") if tmdb_info else movie.get("rating", None)
    if rating_val is None or str(rating_val).strip() == "":
        rating_val = 0.0
    
    st.markdown(render_star_rating(float(rating_val)), unsafe_allow_html=True)
    st.caption(f"Director: {movie.get('director', '—')}")
    
    # Favorite button
    if st.button("⭐ Add to favorites", key=f"fav-{title_str[:20]}", use_container_width=True):
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


def _resolve_voice_title(spoken_text: str, df: pd.DataFrame) -> Optional[str]:
    text = str(spoken_text or "").strip()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text or df.empty or "title" not in df.columns:
        return None
    titles = df["title"].fillna("").astype(str).tolist()
    low = text.lower()
    contains = [t for t in titles if low in t.lower() or t.lower() in low]
    if contains:
        return contains[0]
    close = get_close_matches(text, titles, n=1, cutoff=0.55)
    return close[0] if close else None


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
    banner_desc = "Live recommendations based on mood, voice, and your learning profile."
banner_watch_url = f"?watch_token={banner_token}&watch_title={quote_plus(str(banner_main.get('title','')))}"
st.markdown(
    f"""
    <a href="{banner_watch_url}" target="_blank" style="text-decoration:none;">
    <div class='banner-panel' style="
      background:
        linear-gradient(90deg, rgba(8,30,42,0.34), rgba(8,30,42,0.06)),
        linear-gradient(180deg, rgba(150,245,255,0.26), rgba(0,3,8,0.42)),
        url('{banner_poster}') center/cover no-repeat;
    ">
      <div class='banner-content'>
        <div class='banner-kicker'>Cinematic Mood Journey</div>
        <div class='banner-title'>{str(banner_main.get('title','Featured Movies')).upper()}</div>
        <div class='banner-sub'>{banner_desc}</div>
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
watch_token = qp.get("watch_token", [None])[0] if qp.get("watch_token") else None
if watch_token and st.session_state.get("watch_map", {}).get(watch_token):
    st.session_state["selected_movie"] = st.session_state["watch_map"][watch_token]
    st.session_state["page"] = "watch"
    st.query_params.clear()
else:
    target_title = None
    if watch_token:
        target_title = _title_from_token(watch_token)
    if qp.get("watch_title"):
        target_title = qp.get("watch_title", [None])[0]
    if not target_title and qp.get("watch"):
        target_title = qp.get("watch", [None])[0]
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
        f"<div class='hero-title'>{movie.get('title','Untitled')}</div>",
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

    if manual_override:
        should_scan = False
        st.caption("Mood scanning paused while you search or filter genres.")
    if should_scan:
        st.session_state["last_private_scan_ts"] = now
        try:
            final_details = detect_multimodal_mood_backend(
                voice_duration_sec=2.0,
                use_parallel_threads=True,
            )
        except RuntimeError:
            final_details = None

        if final_details:
            spoken_text = str(final_details.get("spoken_text", "")).strip()
            if spoken_text:
                st.session_state["last_voice_transcript"] = spoken_text
                matched = _resolve_voice_title(spoken_text, _apply_content_filter(load_movies()))
                if matched:
                    st.session_state["voice_title_query"] = matched
                else:
                    st.session_state["voice_title_query"] = spoken_text
            transcript_err = str(final_details.get("voice_transcript_error", "")).strip()
            if transcript_err and not spoken_text:
                st.caption(f"Voice transcript issue: {_friendly_transcript_issue(transcript_err)}")

        mood = final_details["mood"] if final_details else None
        if mood:
            st.session_state["last_mood_source"] = str(final_details.get("source", "unknown"))
            top_moods = final_details.get("top_moods", [])
            recent = st.session_state.get("recent_detected_moods", [])
            if len(recent) >= 3 and all(m == mood for m in recent[-3:]) and len(top_moods) > 1:
                first_name, first_score = top_moods[0]
                second_name, second_score = top_moods[1]
                if second_name != first_name and second_score >= (0.75 * first_score):
                    mood = second_name
            recent.append(mood)
            st.session_state["recent_detected_moods"] = recent[-6:]
            st.session_state["detected_mood"] = mood


        st.caption(f"Detected mood: {str(st.session_state.get('detected_mood')).upper()} | Source: {st.session_state.get('last_mood_source', 'unknown')}")
    if not _HAS_SOUNDDEVICE:
        st.caption("Voice backend unavailable in this runtime (install `sounddevice` + PortAudio). Using face-only.")
    st.markdown("---")

# auto-search on input/genre selection (no button)
detected_mood = st.session_state.get("detected_mood")
voice_title = str(st.session_state.get("voice_title_query", "")).strip()
voice_live = st.session_state.get("voice_transcript", "").strip()
title = typed_query or voice_live or voice_title
if voice_live:
    st.caption(f"🔴 Live voice: **{voice_live}**")
if voice_title and not typed_query and not voice_live:
    st.caption(f"Voice movie request: **{voice_title}**")

if title:
    base = _apply_content_filter(load_movies())
    exact = base[base["title"].fillna("").str.lower() == title.lower()]
    if exact.empty:
        matches = base[base["title"].fillna("").str.contains(title, case=False, na=False)]
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
            matches = _sort_matches_by_rating(matches)
            results = matches.head(NUM).to_dict(orient="records")
            results = rerank_results_for_learning(results, detected_mood)
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
