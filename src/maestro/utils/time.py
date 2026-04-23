"""Time utilities shared across Maestro modules."""

from __future__ import annotations

from datetime import UTC, datetime


def utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware ``datetime``."""
    return datetime.now(UTC)


__all__ = ["utcnow"]
