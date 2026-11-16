import asyncio
import base64
import binascii
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
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

# ~10 MB of base64 is far more than a webcam JPEG needs.
MAX_IMAGE_B64_CHARS = 10_000_000


class MovieIn(BaseModel):
    title: str = Field(..., min_length=1, max_length=300, description="Movie title")
    genres: Optional[str] = Field(
        default="", max_length=500, description="Pipe or comma separated list of genres"
    )
    director: Optional[str] = Field(default="", max_length=300, description="Director name")
    cast: Optional[str] = Field(default="", max_length=2000, description="Pipe separated cast members")
    description: Optional[str] = Field(default="", max_length=5000, description="Short summary")
    rating: Optional[float] = Field(default=None, ge=0, le=10)
    popularity: Optional[float] = Field(default=None, ge=0)
    poster_path: Optional[str] = Field(default=None, max_length=1000, description="Poster URL or TMDB path")


class RecommendResponse(BaseModel):
    results: List[Dict[str, Any]]
    source: str


logger = logging.getLogger("movie_recommender.api")
logging.basicConfig(level=logging.INFO)

_INDEX_LOCK = threading.Lock()
# Held for the whole scrape+build so concurrent requests don't build the index twice.
_BUILD_LOCK = threading.Lock()
_INDEX_STATE: Dict[str, Any] = {"index": None, "last_refresh": None}
_TRAIN_STATE: Dict[str, Any] = {"last_events": 0, "last_trained_at": None}
# Strong references so the background tasks aren't garbage-collected.
_BACKGROUND_TASKS: set = set()


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


def _build_and_store_index() -> None:
    # caller must hold _BUILD_LOCK
    new_index = build_index(method="auto")
    with _INDEX_LOCK:
        _INDEX_STATE["index"] = new_index
        _INDEX_STATE["last_refresh"] = datetime.now(timezone.utc)


def _refresh_index() -> None:
    with _BUILD_LOCK:
        _build_and_store_index()


def _get_index() -> Dict[str, Any]:
    if _INDEX_STATE["index"] is None:
        with _BUILD_LOCK:
            if _INDEX_STATE["index"] is None:
                _scrape_dataset(force=True)
                _build_and_store_index()
    return _INDEX_STATE["index"]


def _scrape_and_refresh() -> None:
    _scrape_dataset()
    _refresh_index()


async def _run_periodic_refresh() -> None:
    # background coroutine refreshing the in-memory index
    while True:
        await asyncio.sleep(REFRESH_INTERVAL_SECONDS)
        try:
            await asyncio.to_thread(_scrape_and_refresh)
        except Exception:
            logger.exception("Periodic index refresh failed")


async def _run_periodic_training(interval_seconds: int = 300) -> None:
    """Background loop to retrain personalization when new feedback arrives."""
    from src.personalization_model import load_feedback_events

    while True:
        await asyncio.sleep(max(60, int(interval_seconds)))
        try:
            total = len(await asyncio.to_thread(load_feedback_events))
            last_seen = int(_TRAIN_STATE.get("last_events", 0))
            if total >= 5 and total > last_seen:
                result = await asyncio.to_thread(train_personalization_now, 5)
                # Only mark events as handled once training actually succeeded,
                # so a failure is retried on the next tick.
                if result.get("trained"):
                    _TRAIN_STATE["last_events"] = total
                    _TRAIN_STATE["last_trained_at"] = datetime.now(timezone.utc).isoformat()
                else:
                    logger.warning("Personalization training did not run: %s", result.get("linear"))
        except Exception:
            logger.exception("Periodic personalization training failed")


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    await asyncio.to_thread(_get_index)
    if REFRESH_INTERVAL_SECONDS > 0:
        _spawn(_run_periodic_refresh())
    _spawn(_run_periodic_training())
    yield
    for task in list(_BACKGROUND_TASKS):
        task.cancel()


app = FastAPI(title="Movie Recommender API", version="1.0.0", lifespan=_lifespan)


# Handlers below are plain `def` on purpose: FastAPI runs them in a threadpool,
# so slow CSV reads, index builds and ONNX inference don't block the event loop.

@app.get("/health")
def health() -> Dict[str, Any]:
    df = load_movies()
    last_refresh = _INDEX_STATE.get("last_refresh")
    return {
        "status": "ok",
        "count": int(len(df)),
        "last_refresh": last_refresh.isoformat() if last_refresh else None,
    }


@app.get("/recommend", response_model=RecommendResponse)
def recommend(
    title: Optional[str] = Query(None, max_length=300),
    genre: Optional[str] = Query(None, max_length=100),
    n: int = Query(10, ge=1, le=50),
):
    title = (title or "").strip()
    genre = (genre or "").strip()
    if not title and not genre:
        raise HTTPException(status_code=400, detail="Provide either title or genre")
    index = _get_index()
    try:
        if title:
            results = recommend_by_title(title, index=index, top_n=n)
            mode = "title"
        else:
            results = recommend_by_genre(genre, index=index, top_n=n)
            mode = "genre"
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"results": results, "source": mode}


@app.post("/add_movie")
def add_movie(movie: MovieIn) -> Dict[str, Any]:
    df = append_movie(movie.model_dump())
    # refresh index to include the new movie
    _refresh_index()
    added = df[df["title"].str.lower() == movie.title.lower()].tail(1)
    if added.empty:
        raise HTTPException(status_code=500, detail="Movie was not saved")
    row = added.astype(object).where(added.notna(), None).to_dict(orient="records")[0]
    return {"status": "ok", "movie": row}


@app.post("/refresh")
def refresh() -> Dict[str, Any]:
    _refresh_index()
    ts = _INDEX_STATE.get("last_refresh")
    return {"status": "ok", "last_refresh": ts.isoformat() if ts else None}


@app.get("/personalization/status")
def personalization_model_status() -> Dict[str, Any]:
    return personalization_status()


@app.post("/personalization/train")
def personalization_model_train(min_events: int = Query(25, ge=5, le=100_000)) -> Dict[str, Any]:
    return train_personalization_now(min_events=min_events)


class AnalyzeFrameRequest(BaseModel):
    image_b64: str = Field(
        ..., max_length=MAX_IMAGE_B64_CHARS, description="Base64 encoded image (JPEG/PNG)"
    )


@app.post("/analyze_frame")
def analyze_frame(req: AnalyzeFrameRequest) -> Dict[str, Any]:
    """Analyze single frame for emotion/mood. Returns emotion details."""
    payload = req.image_b64
    # Accept data URLs ("data:image/jpeg;base64,...") as well as bare base64.
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        img_data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="image_b64 is not valid base64") from exc

    try:
        result = detect_emotion_from_bytes(img_data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Could not decode image") from exc
    except Exception as exc:
        # Missing model files / OpenCV / onnxruntime are server problems, not bad input.
        logger.exception("Emotion detection failed")
        raise HTTPException(status_code=503, detail="Emotion detection unavailable") from exc
    if result is None:
        return {"error": "No face detected", "mood": None}
    return result
