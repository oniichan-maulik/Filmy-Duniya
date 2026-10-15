# Running the Movie Recommender Locally

This project has a FastAPI backend and a Streamlit frontend. The instructions below assume you want to run both locally using the project's virtual environment (`.venv`).

Prerequisites
- Python 3.10+ (project was tested with Python 3.12)
- git (optional)

Quick start (from project root):

1) Create and activate a virtualenv (if you don't have `.venv`):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies (and Playwright browsers if you want to run UI tests):

```bash
python3 -m pip install -r requirements.txt
# optional: installs Chromium/WebKit/Firefox used by tests/playwright_ui_test.py
playwright install
```

3) Start the backend (FastAPI / uvicorn):

```bash
# in one terminal (keeps logs in logs/uvicorn.log)
mkdir -p logs
nohup .venv/bin/python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 > logs/uvicorn.log 2>&1 &
```

4) Start the Streamlit frontend:

```bash
# in another terminal; logs go to logs/streamlit.log
nohup .venv/bin/streamlit run web/app_streamlit.py --server.port 8501 --server.headless true > logs/streamlit.log 2>&1 &
```

5) Open the UI in your browser: http://localhost:8501

Troubleshooting
- If the API `/health` endpoint returns 404 or is unreachable, check `logs/uvicorn.log`.
- Ensure `data/movies.csv` exists (used as the catalog). If your CSV has a `movie_id` column, the loader will dedupe on that; otherwise it dedupes on `title`.
- To stop servers, kill their PIDs (see `ps aux | grep uvicorn` or `pgrep -af uvicorn` / `pgrep -af streamlit`).

Auto-refresh and Adding Movies
- The FastAPI backend supports a background periodic refresh of the in-memory index. The refresh interval is controlled by the environment variable `REFRESH_INTERVAL_SECONDS` (default 300 seconds).
- You can trigger an immediate refresh by POSTing to `/refresh` or using the "Refresh server index now" button in the Streamlit sidebar.
- To add a new movie programmatically, POST a JSON object with at least the `title` field to `/add_movie`. The server will append the row to `data/movies.csv` and rebuild the index.

Example add-movie curl:

```bash
curl -X POST -H "Content-Type: application/json" \
	-d '{"title":"My New Movie","genres":"drama","director":"A Director","description":"A short summary."}' \
	http://127.0.0.1:8000/add_movie
```

The Streamlit UI also has an "Add a new movie" form in the sidebar and an "Enable auto-refresh" checkbox which will poll the backend and refresh the UI on the selected interval.

Voice title search
------------------
- Use the **Listen movie name (3s)** button in the Streamlit UI.
- The app records microphone audio from the runtime/server where Streamlit is running.
- Primary transcription uses Google Web Speech via `SpeechRecognition` (internet usually required).
- Offline fallback via Sphinx is attempted automatically if available.

Docker / docker-compose
-----------------------
If you prefer to run both services with Docker, a `docker-compose.yml` is included at the project root.

Quick docker-compose run (from project root):

```bash
# build images and start both services in the foreground
docker-compose up --build

# run in detached mode
docker-compose up -d --build
```

Notes:
- The compose file mounts your project directory into the containers so code and `data/` changes are visible to the running services.
- Ports: backend -> 8000, frontend (streamlit) -> 8501. You can override these in the compose file or with environment variables.
- Provide environment variables (optional): `REFRESH_INTERVAL_SECONDS`, `TMDB_API_KEY`, `API_URL`.

Example to run with a 2-minute refresh interval:

```bash
REFRESH_INTERVAL_SECONDS=120 docker-compose up -d --build
```

To stop and remove containers:

```bash
docker-compose down
```

If you'd like systemd unit files instead (to run the services as system services), tell me and I will add example unit files (`/etc/systemd/system/movie_recommender-backend.service` and `movie_recommender-frontend.service`) and the commands to enable/start them.
I added a systemd unit template for the scraper at `deploy/systemd/movie_recommender-scraper.service` — copy it to `/etc/systemd/system/`, update the `Environment=TMDB_API_KEY=` line, and then enable/start with `systemctl enable --now movie_recommender-scraper.service`.
Two additional example unit files are included for the backend and frontend under `deploy/systemd/`:

- `deploy/systemd/movie_recommender-backend.service`
- `deploy/systemd/movie_recommender-frontend.service`

Copy them to `/etc/systemd/system/`, adjust `User`, `Environment` variables (especially `TMDB_API_KEY` and `API_URL`), then enable/start:

```bash
sudo cp deploy/systemd/movie_recommender-*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now movie_recommender-backend.service movie_recommender-frontend.service movie_recommender-scraper.service
sudo journalctl -u movie_recommender-backend -f
```

If you need me to generate unit files with a specific user/home path or add environment file support (`EnvironmentFile=`), tell me the host layout and I'll produce a ready-to-install set.

Automated test (backend):

```bash
.venv/bin/python - <<'PY'
import requests
print(requests.get('http://127.0.0.1:8000/health').json())
print(requests.get('http://127.0.0.1:8000/recommend', params={'title':'Inception','n':5}).json())
PY
```

Playwright UI smoke test (requires `streamlit run web/app_streamlit.py` to be running and browsers installed via `playwright install`):

```bash
RUN_UI_TESTS=1 pytest
```

Personalization model training:

```bash
# check current status
.venv/bin/python -m src.personalization_model --status

# train from recorded feedback events
.venv/bin/python -m src.personalization_model --train --min-events 25
```

The Streamlit app also exposes a "Train personalization model now" button in the learning section.

Notes
- I added small compatibility wrappers in `src/` so `main.py` and the API import paths match the existing codebase.
- Need pinned dependencies? Run `python3 -m pip freeze > requirements.lock.txt` after creating your environment.
