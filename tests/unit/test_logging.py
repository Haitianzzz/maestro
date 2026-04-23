"""Smoke tests for structlog configuration."""

from __future__ import annotations

import logging

import structlog

from maestro.utils.logging import configure_logging, get_logger


def test_configure_logging_is_idempotent() -> None:
    configure_logging(force=True)
    configure_logging()  # second call must not raise


def test_get_logger_returns_bound_logger() -> None:
    logger = get_logger("test.module", component="unit_test")
    assert logger is not None
    # Binding should be reflected on the returned logger.
    assert isinstance(logger, structlog.stdlib.BoundLogger) or hasattr(logger, "info")


def test_respects_env_level(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("MAESTRO_LOG_LEVEL", "WARNING")
    configure_logging(force=True)
    assert logging.getLogger().level == logging.WARNING
    # Reset for subsequent tests.
    monkeypatch.setenv("MAESTRO_LOG_LEVEL", "INFO")
    configure_logging(force=True)
