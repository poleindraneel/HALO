"""JSON serialization helpers that handle numpy arrays."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["save_state", "load_state"]


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def save_state(obj: Any, path: str | Path) -> None:
    """Serialize *obj* state to JSON, converting numpy arrays to lists.

    Parameters
    ----------
    obj:
        Any JSON-serialisable object (dicts, lists, numpy arrays).
    path:
        Destination file path; parent directory must exist.
    """
    path = Path(path)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, cls=_NumpyEncoder, indent=2)
    logger.debug("State saved to %s", path)


def load_state(path: str | Path) -> Any:
    """Deserialise state from JSON produced by :func:`save_state`.

    Parameters
    ----------
    path:
        Path to the JSON file.

    Returns
    -------
    Any
        The deserialised Python object.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    logger.debug("State loaded from %s", path)
    return data
