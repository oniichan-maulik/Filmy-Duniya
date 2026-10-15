"""Compatibility wrapper: expose expected module name `src.data_loader`.

The actual file in the repo is `data__loader.py` (double underscore). This
wrapper re-exports `load_movies` from the existing module so imports succeed.
"""
from .data__loader import load_movies

__all__ = ["load_movies"]
