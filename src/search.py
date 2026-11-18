"""Forgiving catalogue search: punctuation-blind, typo-tolerant, and not just titles.

A query is scored against each title, its cast and director, and its genres.
Better kinds of match always outrank worse ones (an exact title beats a typo,
a typo in a title beats a cast name), and the rating breaks ties.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

# Score bands, best first. Fuzzy scores (0-100) are squeezed into their band.
EXACT_TITLE = 100.0
TITLE_PREFIX = 96.0
TITLE_CONTAINS = 92.0
TITLE_FUZZY = (70.0, 88.0)
# an exact name beats a title that is only roughly like the query ("nolan" ~ "mobland")
PERSON_CONTAINS = 80.0
PERSON_FUZZY = (55.0, 64.0)
GENRE = 50.0

FUZZY_CUTOFF = 80.0      # below this a fuzzy match is noise
SUGGEST_CUTOFF = 60.0    # looser, only for a "Did you mean" hint
MIN_FUZZY_LEN = 4        # shorter queries match too much by accident
MIN_CONTAINS_LEN = 2

_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_SCI_FI = re.compile(r"^sci\s*fi$")


def normalize(text: object) -> str:
    """Lower case, no accents, punctuation turned into single spaces."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).casefold()
    return _NON_ALNUM.sub(" ", text).strip()


def _compact(text: str) -> str:
    return text.replace(" ", "")


def _band(scores: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    lo, hi = band
    scaled = lo + (scores - FUZZY_CUTOFF) / (100.0 - FUZZY_CUTOFF) * (hi - lo)
    return np.where(scores >= FUZZY_CUTOFF, scaled, 0.0)


def _split(cell: object) -> list[str]:
    return [normalize(p) for p in str(cell).split("|")] if isinstance(cell, str) else []


def _title_scores(query: str, titles: list[str]) -> np.ndarray:
    q, qc = query, _compact(query)
    out = np.zeros(len(titles))
    for i, title in enumerate(titles):
        tc = _compact(title)
        if not tc:
            continue
        if tc == qc:
            out[i] = EXACT_TITLE
        elif tc.startswith(qc):
            out[i] = TITLE_PREFIX
        elif len(qc) >= MIN_CONTAINS_LEN and qc in tc:
            out[i] = TITLE_CONTAINS
    if len(qc) >= MIN_FUZZY_LEN:
        out = np.maximum(out, _band(_title_fuzzy(q, titles), TITLE_FUZZY))
    return out


def _title_fuzzy(query: str, titles: list[str]) -> np.ndarray:
    """0-100 likeness of the query to each title, typos and word order forgiven."""
    whole = process.cdist([query], titles, scorer=fuzz.ratio, dtype=np.float64)[0]
    reordered = process.cdist([query], titles, scorer=fuzz.token_sort_ratio, dtype=np.float64)[0]
    # a misspelt piece of a longer title ("insidius"), but never a short
    # title found inside the query ("X" in "matchbox")
    inside = process.cdist([query], titles, scorer=fuzz.partial_ratio, dtype=np.float64)[0]
    longer = np.array([len(t) >= len(query) for t in titles])
    return np.maximum.reduce([whole, reordered, np.where(longer, inside, 0.0)])


def _person_scores(query: str, people: list[list[str]]) -> np.ndarray:
    qc = _compact(query)
    out = np.zeros(len(people))
    if len(qc) < 3:
        return out
    for i, names in enumerate(people):
        best = 0.0
        for name in names:
            if not name:
                continue
            if qc in _compact(name):
                best = PERSON_CONTAINS
                break
            if len(qc) >= MIN_FUZZY_LEN:
                # whole name ("shahrukh khan") or one part of it ("nolen")
                score = max(fuzz.ratio(query, name), *(fuzz.ratio(query, part) for part in name.split()))
                if score >= FUZZY_CUTOFF:
                    best = max(best, float(_band(np.array([score]), PERSON_FUZZY)[0]))
        out[i] = best
    return out


def _genre_scores(query: str, genres: list[list[str]]) -> np.ndarray:
    qc = _compact(query)
    out = np.zeros(len(genres))
    if len(qc) < 3:
        return out
    for i, names in enumerate(genres):
        # "sci fi" should find Science Fiction, "thriler" should find Thriller
        if any(_compact(g).startswith(qc) or fuzz.ratio(qc, _compact(g)) >= 85 for g in names if g):
            out[i] = GENRE
    return out


def search_catalogue(df: pd.DataFrame, query: str, limit: int = 24) -> tuple[pd.DataFrame, Optional[str]]:
    """Best matches for `query`, and a title to suggest when nothing matched."""
    q = normalize(query)
    if df.empty or not q or "title" not in df.columns:
        return df.iloc[0:0], None

    titles = [normalize(t) for t in df["title"].tolist()]
    score = _title_scores(q, titles)
    people = [
        _split(cast) + _split(director)
        for cast, director in zip(df.get("cast", pd.Series([""] * len(df))), df.get("director", pd.Series([""] * len(df))))
    ]
    score = np.maximum(score, _person_scores(q, people))
    if "genres" in df.columns:
        score = np.maximum(score, _genre_scores(_SCI_FI.sub("science fiction", q), [_split(g) for g in df["genres"]]))

    hits = df.assign(
        _score=score,
        _rating=pd.to_numeric(df.get("rating"), errors="coerce").fillna(0.0),
    )
    hits = hits[hits["_score"] > 0]
    if not hits.empty:
        hits = (
            hits.sort_values(["_score", "_rating"], ascending=False)
            .drop_duplicates("title")
            .head(limit)
            .drop(columns=["_score", "_rating"])
        )
        return hits, None

    likeness = _title_fuzzy(q, titles)
    best = int(np.argmax(likeness))
    suggestion = str(df["title"].iloc[best]) if likeness[best] >= SUGGEST_CUTOFF else None
    return hits.drop(columns=["_score", "_rating"]), suggestion
