import asyncio
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
import logging
from pydantic import BaseModel, Field

from src.config import (
    REFRESH_INTERVAL_SECONDS,
    TMDB_API_KEY,
    SCRAPE_MOVIE_COUNT,
    SCRAPER_MAX_PER_RUN,
    ENABLE_AUTO_SCRAPER,
)
from src.data_loader import append_movie, load_movies
from src.recommender import build_index, recommend_by_genre, recommend_by_title
from src.scraper import scrape_top_n_movies
from src.user_store import personalization_status, train_personalization_now
from src.emotion_detection import detect_emotion_from_bytes
import base64



class MovieIn(BaseModel):
    title: str = Field(..., min_length=1, description="Movie title")
    genres: Optional[str] = Field(
        default="", description="Pipe or comma separated list of genres"
    )
    director: Optional[str] = Field(default="", description="Director name")
    cast: Optional[str] = Field(default="", description="Pipe separated cast members")
    description: Optional[str] = Field(default="", description="Short summary")
    rating: Optional[float] = Field(default=None, ge=0)
    popularity: Optional[float] = Field(default=None, ge=0)
    poster_path: Optional[str] = Field(default=None, description="Poster URL or TMDB path")


class RecommendResponse(BaseModel):
    results: List[Dict[str, Any]]
    source: str


app = FastAPI(title="Movie Recommender API", version="1.0.0")

logger = logging.getLogger("movie_recommender.api")
logging.basicConfig(level=logging.INFO)

_INDEX_LOCK = threading.Lock()
_INDEX_STATE: Dict[str, Any] = {"index": None, "last_refresh": None}
_TRAIN_STATE: Dict[str, Any] = {"last_events": 0, "last_trained_at": None}


def _scrape_dataset(force: bool = False) -> None:
    if not ENABLE_AUTO_SCRAPER:
        return
    if not TMDB_API_KEY:
        logger.warning("TMDB_API_KEY is not set; skipping automatic scraper run.")
        return
    try:
        logger.info(
            "Running automatic scraper n=%s max_per_run=%s force=%s",
            SCRAPE_MOVIE_COUNT,
            SCRAPER_MAX_PER_RUN,
            force,
        )
        scrape_top_n_movies(
            n=SCRAPE_MOVIE_COUNT,
            append=True,
            max_per_run=SCRAPER_MAX_PER_RUN,
            force=force,
            include_tv=True,
        )
    except Exception as exc:
        logger.warning("Automatic scraper failed: %s", exc)


def _refresh_index() -> None:
    new_index = build_index(method="auto")
    with _INDEX_LOCK:
        _INDEX_STATE["index"] = new_index
        _INDEX_STATE["last_refresh"] = datetime.now(timezone.utc)


def _get_index() -> Dict[str, Any]:
    if _INDEX_STATE["index"] is None:
        _scrape_dataset(force=True)
        _refresh_index()
    return _INDEX_STATE["index"]


async def _run_periodic_refresh() -> None:
    # background coroutine refreshing the in-memory index
    while True:
        await asyncio.sleep(REFRESH_INTERVAL_SECONDS)
        try:
            _scrape_dataset()
            _refresh_index()
        except Exception:
            # log-free fallback; FastAPI will log exception automatically
            pass


async def _run_periodic_training(interval_seconds: int = 300) -> None:
    """Background loop to retrain personalization when new feedback arrives."""
    while True:
        await asyncio.sleep(max(60, int(interval_seconds)))
        try:
            from src.personalization_model import load_feedback_events

            events = load_feedback_events()
            total = len(events)
            last_seen = int(_TRAIN_STATE.get("last_events", 0))
            if total >= 5 and total > last_seen:
                _TRAIN_STATE["last_events"] = total
                _TRAIN_STATE["last_trained_at"] = datetime.now(timezone.utc).isoformat()
                train_personalization_now(min_events=5)
        except Exception:
            # keep running; training is best-effort
            pass


@app.on_event("startup")
async def _startup() -> None:
    _scrape_dataset(force=True)
    _refresh_index()
    if REFRESH_INTERVAL_SECONDS > 0:
        asyncio.create_task(_run_periodic_refresh())
    asyncio.create_task(_run_periodic_training())


@app.get("/health")
async def health() -> Dict[str, Any]:
    df = load_movies()
    last_refresh = _INDEX_STATE.get("last_refresh")
    return {
        "status": "ok",
        "count": int(len(df)),
        "last_refresh": last_refresh.isoformat() if last_refresh else None,
    }


@app.get("/recommend", response_model=RecommendResponse)
async def recommend(title: Optional[str] = None, genre: Optional[str] = None, n: int = 10):
    if not title and not genre:
        raise HTTPException(status_code=400, detail="Provide either title or genre")
    index = _get_index()
    try:
        if title:
            results = recommend_by_title(title, index=index, top_n=n)
            mode = "title"
        else:
            results = recommend_by_genre(genre or "", index=index, top_n=n)
            mode = "genre"
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"results": results, "source": mode}


@app.post("/add_movie")
async def add_movie(movie: MovieIn) -> Dict[str, Any]:
    df = append_movie(movie.dict())
    # refresh index lazily to include the new movie
    _refresh_index()
    added = df[df["title"].str.lower() == movie.title.lower()].tail(1)
    payload = added.to_dict(orient="records")
    return {"status": "ok", "movie": payload[0] if payload else movie.dict()}


@app.post("/refresh")
async def refresh() -> Dict[str, Any]:
    _refresh_index()
    ts = _INDEX_STATE.get("last_refresh")
    return {"status": "ok", "last_refresh": ts.isoformat() if ts else None}


@app.get("/personalization/status")
async def personalization_model_status() -> Dict[str, Any]:
    return personalization_status()


@app.post("/personalization/train")
async def personalization_model_train(min_events: int = 25) -> Dict[str, Any]:
    return train_personalization_now(min_events=min_events)


class AnalyzeFrameRequest(BaseModel):
    image_b64: str = Field(..., description="Base64 encoded image (JPEG/PNG)")


@app.post("/analyze_frame")
async def analyze_frame(req: AnalyzeFrameRequest) -> Dict[str, Any]:
    """Analyze single frame for emotion/mood. Returns emotion details."""
    try:
        # Decode base64 image
        img_data = base64.b64decode(req.image_b64)
        img_array = detect_emotion_from_bytes(img_data)
        if img_array is None:
            return {"error": "No face detected", "mood": None}
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode failed: {str(e)}")
