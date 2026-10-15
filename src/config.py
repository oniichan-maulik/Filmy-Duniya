import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "movies.csv"

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "785c5f1bd5e3e823f06abdfe6168588e")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")  # sentence-transformers
USE_EMBEDDINGS = os.getenv("USE_EMBEDDINGS", "0") == "1"  # toggle to use SBERT
# interval (in seconds) for background data refresh. Set to 0 to disable.
# default: daily refresh
REFRESH_INTERVAL_SECONDS = int(os.getenv("REFRESH_INTERVAL_SECONDS", "86400"))
SCRAPE_MOVIE_COUNT = int(os.getenv("SCRAPE_MOVIE_COUNT", "400"))
SCRAPER_MAX_PER_RUN = int(os.getenv("SCRAPER_MAX_PER_RUN", "100"))
ENABLE_AUTO_SCRAPER = os.getenv("ENABLE_AUTO_SCRAPER", "1") == "1"
