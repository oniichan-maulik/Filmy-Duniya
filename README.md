# Filmy Duniya

A mood-based movie recommender. The Streamlit UI reads your mood from the
webcam (or lets you pick one) and suggests movies and shows from a TMDB-backed
catalogue. Mood picks lean toward films like the ones you liked (and away
from ones you didn't), each film page has a "More like this" shelf, and search
forgives typos and also matches actors, directors and genres. A FastAPI service
exposes the same catalogue and a content-similarity index over HTTP.

## Project layout

```
src/            core library
  api.py                  FastAPI service (recommendations, catalogue, mood from an image)
  recommender.py          TF-IDF / embedding similarity index, title + genre recommendations
  search.py               typo-tolerant catalogue search (rapidfuzz)
  emotion_detection.py    YuNet face detection + HSEmotion mood classifier
  user_store.py           feedback recording and personal re-ranking
  personalization_model.py  trainable ranker over feedback events
  scraper.py              TMDB catalogue scraper (CLI)
  import_feedback_events.py bulk-import interaction CSVs (CLI)
web/app_streamlit.py    Streamlit UI
models/                 pretrained ONNX models (face + emotion)
tests/                  pytest suite (+ opt-in Playwright UI smoke test)
docker/, docker-compose.yml, start_services.sh   ways to run it
```

## Setup

Requires Python 3.10+ (tested on 3.11 and 3.12).

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python3 -m pip install -r requirements.txt
cp .env.example .env               # then add your TMDB_API_KEY
```

Get a TMDB key at <https://www.themoviedb.org/settings/api>. Seed the
catalogue (or copy your own CSV to `data/movies.csv`):

```bash
python3 -m src.scraper --n 300 --append --include-tv
```

## Run

```bash
streamlit run web/app_streamlit.py          # UI on http://localhost:8501
uvicorn src.api:app --port 8000             # API on http://localhost:8000 (optional)
```

On Linux/macOS, `./start_services.sh start|stop|status|restart` runs both in
the background with logs in `logs/`. With Docker:

```bash
docker-compose up -d --build                # backend :8000 (localhost only), UI :8501
docker-compose down
```

### Configuration

Set in `.env` or the environment:

| Variable | Default | Purpose |
|---|---|---|
| `TMDB_API_KEY` | – | TMDB access for scraping, posters and trailers |
| `REFRESH_INTERVAL_SECONDS` | `86400` | API: re-scrape and rebuild the index; `0` disables |
| `ENABLE_AUTO_SCRAPER` | `1` | API: scrape on startup and on refresh |
| `SCRAPE_MOVIE_COUNT` / `SCRAPER_MAX_PER_RUN` | `400` / `100` | API scraper size limits |
| `BLEND_EMBEDDINGS` | `1` | UI: blend sentence embeddings (what the plot means) into TF-IDF for "More like this" and the taste picks; `0` = TF-IDF only. Embeddings are cached in `data/cache/`, so only the first start (~25 s) or new titles run the model |
| `USE_EMBEDDINGS` | `0` | API: `1` = sentence-transformers embeddings instead of TF-IDF |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | embedding model for both of the above |

## API

The API has no authentication; keep it on localhost.

| Method | Path | Description |
|---|---|---|
| GET | `/health` | catalogue size and last index refresh |
| GET | `/recommend?title=…` or `?genre=…&n=10` | similar titles, or top titles in a genre |
| POST | `/add_movie` | append a movie (JSON, `title` required) and rebuild the index |
| POST | `/refresh` | rebuild the index now |
| POST | `/analyze_frame` | `{"image_b64": …}` → detected emotion and mood |
| GET | `/personalization/status` | ranker status |
| POST | `/personalization/train?min_events=25` | train the ranker now |

## Mood detection

"Scan my mood" takes one photo from the visitor's webcam in the browser, with
no preview (`web/mood_scan.py`); the photo is analysed in memory and never saved. Browsers only allow camera
access on `localhost` or over HTTPS, so serve a deployed app over HTTPS.

Two pretrained ONNX models in `models/`, no training needed:

- **Face detection:** [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
  via OpenCV's `FaceDetectorYN`
- **Emotion:** [HSEmotion](https://github.com/HSE-asavchenko/face-emotion-recognition)
  EfficientNet-B2 (7 classes, trained on AffectNet), run with `onnxruntime`

The 7 emotions map to the app's 6 moods (`EMOTION_TO_MOOD` in
`src/emotion_detection.py`). The HSEmotion weights were trained on AffectNet,
which is licensed for non-commercial research use.

## Personalization

Likes, dislikes and watchlist saves are appended to `data/feedback_events.jsonl`.
The API retrains the ranker in the background as events arrive; to train by hand:

```bash
python3 -m src.personalization_model --status
python3 -m src.personalization_model --train --min-events 25
```

The model is written to `data/personalization_model.pkl` and used to re-rank
the UI's mood picks. Existing interaction logs can be imported:

```bash
python3 -m src.import_feedback_events --csv interactions.csv [--overwrite] [--min-abs-signal 0.2]
```

## Evaluating "More like this"

```bash
python3 -m src.evaluate [--k 10] [--worst 10] [--refresh]
```

Scores the similar-titles shelf (`--no-embeddings` scores TF-IDF alone, to
compare) against TMDB's own recommendations (those that
are in the catalogue) and prints hit rate, precision and recall at k, next to
what random picks would score. TMDB leaves a film's own series out of its
recommendations, so the other films of the same TMDB collection also count as
right answers (`--no-series` turns that off). The first run fetches from TMDB
(about a minute) and caches it in `data/cache/`; later runs are offline. Use the
score to compare versions of the recommender, not as an absolute grade.

## Tests

```bash
pytest
```

The Playwright UI smoke test is skipped by default. To run it, install the
browsers (`playwright install`), start the Streamlit UI, then run
`RUN_UI_TESTS=1 pytest`.

## Author

Developed by **Maulik Gupta** ([@oniichan-maulik](https://github.com/oniichan-maulik)).
