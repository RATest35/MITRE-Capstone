"""
utils.py
--------
Shared utility helpers used across the pipeline.

Keeping these here avoids circular imports and gives every module a single
place to import generic helpers from.
"""

import logging

logger = logging.getLogger(__name__)


def safe_float(value, default: float = 0.0) -> float:
    """
    Safely convert *value* to a float.

    Unlike the original ``to_float`` which returned ``None`` on failure
    (because ``print()`` returns ``None``), this function:
      - returns *default* on failure so callers always get a numeric value,
      - logs a warning so failures are visible without crashing the pipeline.

    Parameters
    ----------
    value:
        The value to convert.
    default:
        The fallback value when conversion fails (default ``0.0``).

    Returns
    -------
    float
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("Could not convert %r to float; using default %s", value, default)
        return default


def configure_logging(level: int = logging.INFO) -> None:
    """
    Set up a basic logging configuration for the pipeline.

    Call once from ``main.py`` before anything else runs.
    """
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )
