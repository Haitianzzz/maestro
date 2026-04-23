"""Exceptions raised by the Maestro LLM client (spec 02 §6)."""

from __future__ import annotations


class LLMCallError(Exception):
    """Base class for all LLM call failures."""


class LLMRetryExhaustedError(LLMCallError):
    """Raised when all HTTP-level retries are exhausted without success."""

    def __init__(self, message: str, last_error: BaseException | None = None) -> None:
        super().__init__(message)
        self.last_error = last_error


class LLMOutputParseError(LLMCallError):
    """Raised when the LLM output cannot be parsed as the target schema."""

    def __init__(self, message: str, raw_output: str = "") -> None:
        super().__init__(message)
        self.raw_output = raw_output


class LLMConfigError(LLMCallError):
    """Raised for configuration errors (missing role, bad YAML, etc.)."""


__all__ = [
    "LLMCallError",
    "LLMConfigError",
    "LLMOutputParseError",
    "LLMRetryExhaustedError",
]
