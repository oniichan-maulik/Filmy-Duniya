"""Compatibility wrapper: expose expected module name `src.recommender`.

This project contains `src/recomender.py` (typo). Many files import
`src.recommender` (double 'm'); create this thin wrapper to re-export
the expected symbols so the app can run without renaming many files.
"""
from .recomender import (
	build_index,
	genres_for_mood,
	MOOD_TO_GENRES,
	recommend_by_genre,
	recommend_by_mood,
	recommend_by_title,
)
import pandas as pd
from typing import Any, Dict, Optional



def recommend(genre: str, n: int = 5, index: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
	"""Compatibility helper used by `main.py`.

	Now returns recommendations based on a genre string. Calls the underlying
	`recommend_by_genre` and returns a pandas DataFrame so the CLI code in
	`main.py` can iterate with `iterrows()`.
	"""
	results = recommend_by_genre(genre, index=index, top_n=n)
	if not results:
		# return empty DataFrame with expected columns
		return pd.DataFrame(columns=["title", "genres", "director", "score"])
	# ensure a consistent DataFrame (handle list of dicts)
	df = pd.DataFrame(results)
	# normalize column names (lowercase) to match expected display in main.py
	if 'title' not in df.columns and 'Title' in df.columns:
		df = df.rename(columns={'Title': 'title'})
	return df


__all__ = ["build_index", "recommend_by_title", "recommend"]
__all__.append("recommend_by_genre")
__all__.append("recommend_by_mood")
__all__.append("MOOD_TO_GENRES")
__all__.append("genres_for_mood")
