"""Unified LLM client for Maestro (spec 02 §3).

All LLM calls from any Maestro module MUST go through this client. It provides:

* Uniform token / cost / latency accounting (``CostReport``).
* Structured-output enforcement via Pydantic model classes.
* HTTP-level retry with exponential backoff for 5xx / 429 / timeout.
* A one-shot parse retry when the LLM's output fails schema validation.
* A global ``asyncio.Semaphore`` to keep concurrent in-flight calls within
  the provider's rate-limit budget.
* A ``dry_run`` mode that returns dummy objects matching the schema — used
  by the benchmark harness to exercise the pipeline without API spend.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Iterable
from datetime import datetime
from typing import Any, TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from maestro.utils.logging import get_logger
from maestro.utils.time import utcnow

from .config import ClientConfig, ModelConfig, Role
from .errors import LLMCallError, LLMOutputParseError, LLMRetryExhaustedError

T = TypeVar("T", bound=BaseModel)

_logger = get_logger("maestro.llm")


# ---------------------------------------------------------------------------
# Call metadata + cost report models
# ---------------------------------------------------------------------------


class LLMCallMetadata(BaseModel):
    """Metadata recorded for each LLM call (spec 02 §3).

    ``http_retry_count`` is the number of HTTP-level retries this single call
    performed internally. This is distinct from ``SubAgentResult.retry_count``
    which counts subtask-level retries after verification failure (see M7).
    """

    model_config = ConfigDict(frozen=True)

    model_name: str
    role: Role
    tokens_input: int = Field(ge=0)
    tokens_output: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    cost: float = Field(ge=0.0)
    currency: str
    called_at: datetime
    success: bool
    http_retry_count: int = Field(ge=0)


class ToolCallRequest(BaseModel):
    """A single tool invocation requested by the model.

    Arguments are the parsed JSON object already decoded from the raw
    ``arguments`` string returned by the provider. If the provider emits an
    unparseable argument blob we still surface the raw text via
    ``arguments_raw`` so the caller can decide how to handle it.
    """

    model_config = ConfigDict(frozen=True)

    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    arguments_raw: str = ""


class ToolCallResponse(BaseModel):
    """Combined text + tool-call output from :meth:`LLMClient.call_with_tools`.

    Exactly one of ``text`` (free-form reply, no tool call) or
    ``tool_calls`` (at least one tool invocation) will typically be
    populated, mirroring OpenAI-style completions. Both may be non-empty
    when the model narrates before dispatching tools.
    """

    model_config = ConfigDict(frozen=True)

    text: str = ""
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    finish_reason: str | None = None


class ModelCostSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str  # role or model_name depending on aggregation axis
    model_name: str
    role: Role
    call_count: int = Field(ge=0)
    tokens_input: int = Field(ge=0)
    tokens_output: int = Field(ge=0)
    cost: float = Field(ge=0.0)
    avg_latency_ms: float = Field(ge=0.0)


class CostReport(BaseModel):
    """Aggregate cost report across all LLM calls in a session."""

    model_config = ConfigDict(frozen=True)

    currency: str
    total_calls: int = Field(ge=0)
    total_tokens_input: int = Field(ge=0)
    total_tokens_output: int = Field(ge=0)
    total_cost: float = Field(ge=0.0)
    per_model: list[ModelCostSummary]
    per_role: list[ModelCostSummary]

    def to_markdown(self) -> str:
        lines = [
            f"# LLM Cost Report ({self.currency})",
            "",
            f"- Total calls: **{self.total_calls}**",
            f"- Total tokens in/out: **{self.total_tokens_input} / {self.total_tokens_output}**",
            f"- Total cost: **{self.total_cost:.4f} {self.currency}**",
            "",
            "## Per role",
            "",
            "| Role | Model | Calls | Tokens in | Tokens out | Cost | Avg latency (ms) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for r in self.per_role:
            lines.append(
                f"| {r.role} | {r.model_name} | {r.call_count} | "
                f"{r.tokens_input} | {r.tokens_output} | {r.cost:.4f} | {r.avg_latency_ms:.1f} |"
            )
        lines += [
            "",
            "## Per model",
            "",
            "| Model | Calls | Tokens in | Tokens out | Cost | Avg latency (ms) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for m in self.per_model:
            lines.append(
                f"| {m.model_name} | {m.call_count} | {m.tokens_input} | "
                f"{m.tokens_output} | {m.cost:.4f} | {m.avg_latency_ms:.1f} |"
            )
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_RETRIABLE_STATUS = {429, 500, 502, 503, 504}


def _is_retriable(exc: BaseException) -> bool:
    """Return True iff an HTTP-level error should be retried with backoff."""
    if isinstance(exc, APITimeoutError | APIConnectionError | RateLimitError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in _RETRIABLE_STATUS
    return False


def _compute_cost(model_cfg: ModelConfig, tokens_input: int, tokens_output: int) -> float:
    return (
        tokens_input * model_cfg.price_input_per_mtok / 1_000_000
        + tokens_output * model_cfg.price_output_per_mtok / 1_000_000
    )


def _dummy_instance(schema: type[T]) -> T:
    """Construct a best-effort dummy instance of a Pydantic model.

    Used only by ``dry_run``. We fill string fields with ``"dummy"``, numeric
    fields with ``0``, and use schema defaults wherever available.
    """
    fields: dict[str, Any] = {}
    for name, info in schema.model_fields.items():
        if not info.is_required():
            continue
        annotation = info.annotation
        fields[name] = _dummy_value(annotation)
    return schema(**fields)


def _dummy_value(annotation: Any) -> Any:
    if annotation is str:
        return "dummy"
    if annotation in (int, float):
        return 0
    if annotation is bool:
        return False
    # list[...] / dict[...] / optional
    origin = getattr(annotation, "__origin__", None)
    if origin is list:
        return []
    if origin is dict:
        return {}
    # Fall back to zero-like; Pydantic will raise if the caller passes an
    # invalid shape, which is acceptable in dry-run.
    return None


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------


class LLMClient:
    """Unified async LLM client for Maestro.

    Wrap all OpenAI-compatible calls. Business modules MUST use this class;
    they must never import ``openai`` directly.
    """

    def __init__(
        self,
        config: ClientConfig,
        *,
        dry_run: bool = False,
        now: Any = utcnow,
    ) -> None:
        self._config = config
        self._client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)
        self._semaphore = asyncio.Semaphore(config.global_semaphore_limit)
        self._call_log: list[LLMCallMetadata] = []
        self._log_lock = asyncio.Lock()
        self._dry_run = dry_run
        self._now = now

    @property
    def config(self) -> ClientConfig:
        """Public read-only view of the client config.

        Sub-agent and other modules read ``client.config.models[role].name`` to
        record which model produced a given patch; exposing this as a property
        makes that intent explicit.
        """
        return self._config

    @property
    def in_flight_limit(self) -> int:
        return self._config.global_semaphore_limit

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def call_structured(
        self,
        *,
        role: Role,
        messages: list[dict[str, Any]],
        output_schema: type[T],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> tuple[T, LLMCallMetadata]:
        """Call the LLM and parse its output into ``output_schema``."""
        model_cfg = self._config.get_model(role)

        if self._dry_run:
            return self._dry_run_structured(role, model_cfg, output_schema)

        async with self._semaphore:
            return await self._do_call_structured(
                role=role,
                model_cfg=model_cfg,
                messages=list(messages),
                output_schema=output_schema,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    async def call_text(
        self,
        *,
        role: Role,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> tuple[str, LLMCallMetadata]:
        """Call the LLM and return plain text. Prefer ``call_structured``."""
        model_cfg = self._config.get_model(role)

        if self._dry_run:
            return self._dry_run_text(role, model_cfg)

        async with self._semaphore:
            return await self._do_call_text(
                role=role,
                model_cfg=model_cfg,
                messages=list(messages),
                temperature=temperature,
                max_tokens=max_tokens,
            )

    async def call_with_tools(
        self,
        *,
        role: Role,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> tuple[ToolCallResponse, LLMCallMetadata]:
        """Call the LLM with OpenAI-style function-calling tools enabled.

        The sub-agent's Explore phase (spec 05 §4.1) uses this to dispatch
        ``read_file`` tool calls. The response carries any free-form text
        AND the list of tool invocations the model wants to make; the
        caller is responsible for executing them and feeding results back
        via follow-up messages.
        """
        model_cfg = self._config.get_model(role)

        if self._dry_run:
            return self._dry_run_tool_call(role, model_cfg)

        async with self._semaphore:
            return await self._do_call_with_tools(
                role=role,
                model_cfg=model_cfg,
                messages=list(messages),
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    async def get_cost_report(self) -> CostReport:
        """Return a snapshot cost report aggregated across all recorded calls."""
        async with self._log_lock:
            log = list(self._call_log)
        return _build_cost_report(log, currency=self._config.currency)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _do_call_structured(
        self,
        *,
        role: Role,
        model_cfg: ModelConfig,
        messages: list[dict[str, Any]],
        output_schema: type[T],
        temperature: float,
        max_tokens: int | None,
    ) -> tuple[T, LLMCallMetadata]:
        schema = output_schema.model_json_schema()
        response_format: dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": output_schema.__name__,
                "schema": schema,
                "strict": True,
            },
        }

        raw_text = ""
        parse_attempts = 0
        last_parse_error: Exception | None = None

        while parse_attempts < 2:  # one parse retry on validation failure
            raw_text, _tool_calls, _finish, metadata = await self._send(
                role=role,
                model_cfg=model_cfg,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
            try:
                parsed = output_schema.model_validate_json(raw_text)
            except ValidationError as exc:
                last_parse_error = exc
                parse_attempts += 1
                _logger.warning(
                    "llm_parse_retry",
                    role=role,
                    model=model_cfg.name,
                    attempt=parse_attempts,
                    error=str(exc),
                )
                messages.append({"role": "assistant", "content": raw_text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous output failed to parse against the required schema. "
                            "Please regenerate the response strictly matching the JSON schema."
                        ),
                    }
                )
                continue
            except json.JSONDecodeError as exc:
                last_parse_error = exc
                parse_attempts += 1
                messages.append({"role": "assistant", "content": raw_text})
                messages.append(
                    {
                        "role": "user",
                        "content": "Your previous output was not valid JSON. Respond with JSON only.",
                    }
                )
                continue

            await self._record(metadata)
            return parsed, metadata

        raise LLMOutputParseError(
            f"Failed to parse LLM output for role={role} after 2 attempts: {last_parse_error}",
            raw_output=raw_text,
        )

    async def _do_call_text(
        self,
        *,
        role: Role,
        model_cfg: ModelConfig,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int | None,
    ) -> tuple[str, LLMCallMetadata]:
        raw_text, _tool_calls, _finish, metadata = await self._send(
            role=role,
            model_cfg=model_cfg,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=None,
        )
        await self._record(metadata)
        return raw_text, metadata

    async def _do_call_with_tools(
        self,
        *,
        role: Role,
        model_cfg: ModelConfig,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str,
        temperature: float,
        max_tokens: int | None,
    ) -> tuple[ToolCallResponse, LLMCallMetadata]:
        text, tool_calls, finish_reason, metadata = await self._send(
            role=role,
            model_cfg=model_cfg,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=None,
            tools=tools,
            tool_choice=tool_choice,
        )
        await self._record(metadata)
        return (
            ToolCallResponse(text=text, tool_calls=tool_calls, finish_reason=finish_reason),
            metadata,
        )

    async def _send(
        self,
        *,
        role: Role,
        model_cfg: ModelConfig,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int | None,
        response_format: dict[str, Any] | None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> tuple[str, list[ToolCallRequest], str | None, LLMCallMetadata]:
        """Low-level send with exponential backoff on retriable errors."""
        max_retries = self._config.max_retries
        attempt = 0
        last_exc: BaseException | None = None

        while attempt <= max_retries:
            start = time.perf_counter()
            try:
                kwargs: dict[str, Any] = {
                    "model": model_cfg.name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens or model_cfg.max_tokens_output,
                }
                if response_format is not None:
                    kwargs["response_format"] = response_format
                if tools is not None:
                    kwargs["tools"] = tools
                    if tool_choice is not None:
                        kwargs["tool_choice"] = tool_choice

                response = await self._client.chat.completions.create(**kwargs)
            except Exception as exc:
                last_exc = exc
                latency_ms = int((time.perf_counter() - start) * 1000)
                if _is_retriable(exc) and attempt < max_retries:
                    backoff = _backoff_delay(attempt)
                    _logger.warning(
                        "llm_retry",
                        role=role,
                        model=model_cfg.name,
                        attempt=attempt + 1,
                        backoff_s=round(backoff, 3),
                        error=exc.__class__.__name__,
                        latency_ms=latency_ms,
                    )
                    await asyncio.sleep(backoff)
                    attempt += 1
                    continue
                _logger.error(
                    "llm_call_failed",
                    role=role,
                    model=model_cfg.name,
                    attempt=attempt + 1,
                    error=exc.__class__.__name__,
                    error_message=str(exc),
                    latency_ms=latency_ms,
                )
                raise LLMCallError(
                    f"LLM call failed for role={role} model={model_cfg.name}: {exc}"
                ) from exc

            latency_ms = int((time.perf_counter() - start) * 1000)
            text, tool_calls, finish_reason, tokens_in, tokens_out = _extract_response(response)
            cost = _compute_cost(model_cfg, tokens_in, tokens_out)
            metadata = LLMCallMetadata(
                model_name=model_cfg.name,
                role=role,
                tokens_input=tokens_in,
                tokens_output=tokens_out,
                latency_ms=latency_ms,
                cost=cost,
                currency=self._config.currency,
                called_at=self._now(),
                success=True,
                http_retry_count=attempt,
            )
            _logger.info(
                "llm_call",
                role=role,
                model=model_cfg.name,
                tokens_input=tokens_in,
                tokens_output=tokens_out,
                latency_ms=latency_ms,
                cost=cost,
                currency=self._config.currency,
                http_retry_count=attempt,
                tool_calls=len(tool_calls),
            )
            return text, tool_calls, finish_reason, metadata

        raise LLMRetryExhaustedError(
            f"Exhausted {max_retries} retries for role={role} model={model_cfg.name}",
            last_error=last_exc,
        )

    async def _record(self, metadata: LLMCallMetadata) -> None:
        async with self._log_lock:
            self._call_log.append(metadata)

    def _dry_run_structured(
        self, role: Role, model_cfg: ModelConfig, schema: type[T]
    ) -> tuple[T, LLMCallMetadata]:
        parsed = _dummy_instance(schema)
        metadata = self._dry_run_metadata(role, model_cfg)
        return parsed, metadata

    def _dry_run_text(self, role: Role, model_cfg: ModelConfig) -> tuple[str, LLMCallMetadata]:
        metadata = self._dry_run_metadata(role, model_cfg)
        return "dummy text", metadata

    def _dry_run_tool_call(
        self, role: Role, model_cfg: ModelConfig
    ) -> tuple[ToolCallResponse, LLMCallMetadata]:
        """Return a dummy tool-call response (no tool calls, empty text)."""
        metadata = self._dry_run_metadata(role, model_cfg)
        return ToolCallResponse(text="", tool_calls=[], finish_reason="stop"), metadata

    def _dry_run_metadata(self, role: Role, model_cfg: ModelConfig) -> LLMCallMetadata:
        return LLMCallMetadata(
            model_name=model_cfg.name,
            role=role,
            tokens_input=0,
            tokens_output=0,
            latency_ms=0,
            cost=0.0,
            currency=self._config.currency,
            called_at=self._now(),
            success=True,
            http_retry_count=0,
        )


# ---------------------------------------------------------------------------
# Aggregation helpers (pure functions, exposed for testing)
# ---------------------------------------------------------------------------


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter: 0.5 * 2^attempt seconds ± 20%."""
    base = 0.5 * (2**attempt)
    jitter = random.uniform(0.8, 1.2)
    return float(base * jitter)


def _extract_response(
    response: Any,
) -> tuple[str, list[ToolCallRequest], str | None, int, int]:
    """Extract text, tool calls, finish_reason, and token counts.

    Robust to tool-free responses (``message.tool_calls`` absent or None) and
    to providers that return ``arguments`` as an already-decoded object
    instead of a JSON string.
    """
    choice = response.choices[0]
    message = choice.message
    text = message.content or ""
    finish_reason = getattr(choice, "finish_reason", None)

    tool_calls: list[ToolCallRequest] = []
    raw_tool_calls = getattr(message, "tool_calls", None) or []
    for tc in raw_tool_calls:
        call_id = getattr(tc, "id", None) or ""
        function = getattr(tc, "function", None)
        name = getattr(function, "name", None) or ""
        raw_args = getattr(function, "arguments", "") or ""
        parsed_args: dict[str, Any]
        if isinstance(raw_args, dict):
            parsed_args = raw_args
            raw_args_str = json.dumps(raw_args)
        else:
            raw_args_str = str(raw_args)
            try:
                decoded = json.loads(raw_args_str) if raw_args_str else {}
                parsed_args = decoded if isinstance(decoded, dict) else {}
            except json.JSONDecodeError:
                parsed_args = {}
        tool_calls.append(
            ToolCallRequest(
                call_id=call_id,
                name=name,
                arguments=parsed_args,
                arguments_raw=raw_args_str,
            )
        )

    usage = getattr(response, "usage", None)
    tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    tokens_out = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    return text, tool_calls, finish_reason, tokens_in, tokens_out


def _build_cost_report(records: Iterable[LLMCallMetadata], *, currency: str) -> CostReport:
    by_model: dict[str, dict[str, Any]] = {}
    by_role: dict[str, dict[str, Any]] = {}
    total_calls = 0
    total_in = 0
    total_out = 0
    total_cost = 0.0

    for rec in records:
        if rec.currency != currency:
            # Sessions should be single-currency; if a mismatch sneaks in,
            # we still tally tokens but flag cost at zero to avoid mixing.
            continue
        total_calls += 1
        total_in += rec.tokens_input
        total_out += rec.tokens_output
        total_cost += rec.cost

        m_bucket = by_model.setdefault(
            rec.model_name,
            {
                "role": rec.role,
                "call_count": 0,
                "tokens_input": 0,
                "tokens_output": 0,
                "cost": 0.0,
                "latency_sum": 0,
            },
        )
        m_bucket["call_count"] += 1
        m_bucket["tokens_input"] += rec.tokens_input
        m_bucket["tokens_output"] += rec.tokens_output
        m_bucket["cost"] += rec.cost
        m_bucket["latency_sum"] += rec.latency_ms

        r_bucket = by_role.setdefault(
            rec.role,
            {
                "model_name": rec.model_name,
                "call_count": 0,
                "tokens_input": 0,
                "tokens_output": 0,
                "cost": 0.0,
                "latency_sum": 0,
            },
        )
        r_bucket["call_count"] += 1
        r_bucket["tokens_input"] += rec.tokens_input
        r_bucket["tokens_output"] += rec.tokens_output
        r_bucket["cost"] += rec.cost
        r_bucket["latency_sum"] += rec.latency_ms

    per_model = [
        ModelCostSummary(
            key=model_name,
            model_name=model_name,
            role=bucket["role"],
            call_count=bucket["call_count"],
            tokens_input=bucket["tokens_input"],
            tokens_output=bucket["tokens_output"],
            cost=round(bucket["cost"], 6),
            avg_latency_ms=(bucket["latency_sum"] / bucket["call_count"])
            if bucket["call_count"]
            else 0.0,
        )
        for model_name, bucket in sorted(by_model.items())
    ]
    per_role = [
        ModelCostSummary(
            key=role,
            model_name=bucket["model_name"],
            role=role,
            call_count=bucket["call_count"],
            tokens_input=bucket["tokens_input"],
            tokens_output=bucket["tokens_output"],
            cost=round(bucket["cost"], 6),
            avg_latency_ms=(bucket["latency_sum"] / bucket["call_count"])
            if bucket["call_count"]
            else 0.0,
        )
        for role, bucket in sorted(by_role.items())
    ]

    return CostReport(
        currency=currency,
        total_calls=total_calls,
        total_tokens_input=total_in,
        total_tokens_output=total_out,
        total_cost=round(total_cost, 6),
        per_model=per_model,
        per_role=per_role,
    )


__all__ = [
    "CostReport",
    "LLMCallMetadata",
    "LLMClient",
    "ModelCostSummary",
]
