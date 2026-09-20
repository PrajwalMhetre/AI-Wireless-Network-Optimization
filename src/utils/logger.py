"""Logging helpers."""

from __future__ import annotations

import logging


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Create a configured logger with a consistent project format."""

    logger = logging.getLogger(name)
    logger.setLevel(level.upper())

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)

    return logger

