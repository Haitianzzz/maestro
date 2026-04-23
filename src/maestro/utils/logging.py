"""Structured logging setup for Maestro.

All modules should obtain a logger via ``get_logger(__name__)`` and emit events
as structured key/value pairs (``logger.info("event_name", key=value, ...)``).

The log level is controlled by the ``MAESTRO_LOG_LEVEL`` environment variable
(default ``INFO``). When stdout is a TTY we render human-friendly colored output;
otherwise we render JSON lines suitable for ingestion by log aggregators and the
benchmark harness.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog
from structlog.types import Processor

_CONFIGURED = False


def _resolve_level() -> int:
    raw = os.getenv("MAESTRO_LOG_LEVEL", "INFO").upper()
    return getattr(logging, raw, logging.INFO)


def configure_logging(*, force: bool = False) -> None:
    """Configure structlog + stdlib logging. Idempotent unless ``force=True``."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    level = _resolve_level()

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if sys.stdout.isatty():
        renderer: Processor = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        level=level,
        stream=sys.stdout,
        force=True,
    )

    _CONFIGURED = True


def get_logger(name: str | None = None, **initial_values: Any) -> structlog.stdlib.BoundLogger:
    """Return a structured logger bound with initial context values."""
    if not _CONFIGURED:
        configure_logging()
    logger = structlog.get_logger(name)
    if initial_values:
        logger = logger.bind(**initial_values)
    return logger  # type: ignore[no-any-return]
