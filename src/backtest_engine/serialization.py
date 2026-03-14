from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np


def to_json_safe(value: Any) -> Any:
    """
    Normalizes Python, numpy, and datetime values into strict JSON-safe types.

    Methodology:
    Metrics and manifests must remain parseable under strict RFC-style JSON
    parsers. Finite numeric values stay numeric, while non-finite numbers are
    converted to null. This keeps the artifact contract stable across single and
    portfolio exporters.

    Args:
        value: Arbitrary scalar or nested structure to normalize.

    Returns:
        A JSON-serializable representation using only built-in safe types.
    """
    if isinstance(value, dict):
        return {str(key): to_json_safe(val) for key, val in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(item) for item in value]

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, bool) or value is None:
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    return str(value)


def dumps_json(data: Any, *, indent: int = 2) -> str:
    """Serializes data with strict JSON compliance and shared normalization."""
    return json.dumps(to_json_safe(data), indent=indent, allow_nan=False)
