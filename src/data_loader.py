import os
import threading
import time
import uuid
from pathlib import Path

import pandas as pd

from src.config import DATA_PATH

# simple lock to protect concurrent writes to the CSV
_CSV_LOCK = threading.Lock()


def _write_csv_atomic(df: pd.DataFrame, p: Path) -> None:
    """Write via a temp file + rename so readers never see a half-written CSV."""
    tmp = p.with_suffix(p.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    # On Windows the rename fails while another reader has the file open; retry briefly.
    for attempt in range(10):
        try:
            os.replace(tmp, p)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05)


def load_movies(path=DATA_PATH):
    df = pd.read_csv(path)
    # basic dtype normalization
    df['title'] = df['title'].astype(str)
    # some CSVs may not include a `movie_id` column; fall back to title-based dedupe
    if 'movie_id' in df.columns:
        df = df.drop_duplicates(subset=['movie_id'])
    else:
        df = df.drop_duplicates(subset=['title'])
    df = df.reset_index(drop=True)
    return df


def append_movie(movie: dict, path=DATA_PATH):
    """CSV फ़ाइल में एक नई movie (dict) को नई पंक्ति के रूप में जोड़ता है।

    यह फ़ाइल-स्तर का lock लेता है ताकि एक साथ कई API calls से CSV खराब न हो।
    movie dict की कुंजियाँ CSV के कॉलम नामों से मेल खानी चाहिए (कम से कम 'title')।
    जोड़ने के बाद नया DataFrame return करता है।
    """
    p = Path(path)
    # ensure parent dir exists
    p.parent.mkdir(parents=True, exist_ok=True)
    with _CSV_LOCK:
        # load existing (to normalize columns and ordering)
        if p.exists():
            df = pd.read_csv(p)
        else:
            df = pd.DataFrame()

        # create a single-row df for the incoming movie
        row = {k: (v if v is not None else "") for k, v in movie.items()}
        # Manually added movies have no TMDB id; give each a unique one so the
        # movie_id dedupe below doesn't collapse them all into one row.
        if 'movie_id' in df.columns and not str(row.get('movie_id', '') or '').strip():
            row['movie_id'] = f"manual-{uuid.uuid4().hex[:12]}"
        new_df = pd.DataFrame([row])

        # if df empty just write header+row
        if df.empty:
            out = new_df
        else:
            # ensure same columns: union of columns, fill missing with empty
            cols = list(dict.fromkeys(list(df.columns) + list(new_df.columns)))
            out = pd.concat([df.reindex(columns=cols, fill_value=""), new_df.reindex(columns=cols, fill_value="")], ignore_index=True)

        # drop duplicates by movie_id if present otherwise by title
        if 'movie_id' in out.columns:
            out = out.drop_duplicates(subset=['movie_id'])
        else:
            out = out.drop_duplicates(subset=['title'])

        _write_csv_atomic(out, p)
        return out.reset_index(drop=True)


def append_bulk(rows, path=DATA_PATH):
    """Append multiple movie rows at once in a single atomic update.

    This reads existing CSV (if any), concatenates the incoming rows, dedupes
    and writes the resulting CSV while holding the CSV lock to avoid races.
    `rows` is an iterable of dict-like objects.
    Returns the new DataFrame.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _CSV_LOCK:
        if p.exists():
            df = pd.read_csv(p)
        else:
            df = pd.DataFrame()

        new_df = pd.DataFrame(rows)

        if df.empty:
            out = new_df
        else:
            cols = list(dict.fromkeys(list(df.columns) + list(new_df.columns)))
            out = pd.concat([df.reindex(columns=cols, fill_value=""), new_df.reindex(columns=cols, fill_value="")], ignore_index=True)

        if 'movie_id' in out.columns:
            out = out.drop_duplicates(subset=['movie_id'])
        else:
            out = out.drop_duplicates(subset=['title'])

        _write_csv_atomic(out, p)
        return out.reset_index(drop=True)
