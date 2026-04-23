"""Tests for ``maestro.llm.client`` (spec 02 §7).

These tests mock the ``AsyncOpenAI`` object so no real API calls are made.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import APIStatusError, APITimeoutError
from pydantic import BaseModel

from maestro.llm.client import (
    CostReport,
    LLMCallMetadata,
    LLMClient,
    _build_cost_report,
)
from maestro.llm.config import ClientConfig, ModelConfig
from maestro.llm.errors import LLMCallError, LLMOutputParseError

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class Echo(BaseModel):
    message: str
    score: float


def _make_config(currency: str = "RMB") -> ClientConfig:
    return ClientConfig(
        base_url="https://example.com/v1",
        api_key="fake",
        currency=currency,  # type: ignore[arg-type]
        max_retries=2,
        global_semaphore_limit=3,
        models={
            "planner": ModelConfig(
                name="qwen3-max",
                display_name="Qwen3-Max",
                price_input_per_mtok=2.8,
                price_output_per_mtok=8.4,
            ),
            "subagent": ModelConfig(
                name="qwen3-coder-plus",
                display_name="Qwen3-Coder-Plus",
                price_input_per_mtok=0.84,
                price_output_per_mtok=3.36,
            ),
            "judge": ModelConfig(
                name="deepseek-v3",
                display_name="DeepSeek-V3",
                price_input_per_mtok=0.28,
                price_output_per_mtok=1.12,
            ),
        },
    )


def _mock_chat_response(content: str, prompt_tokens: int = 100, completion_tokens: int = 50) -> Any:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message = MagicMock()
    response.choices[0].message.content = content
    response.usage = MagicMock()
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response


def _install_client_mock(client: LLMClient, handler: Callable[..., Awaitable[Any]]) -> AsyncMock:
    """Replace the inner AsyncOpenAI so each ``create`` call routes to ``handler``."""
    mock_create = AsyncMock(side_effect=handler)
    client._client.chat.completions.create = mock_create  # type: ignore[method-assign]
    return mock_create


def _fixed_now() -> datetime:
    return datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# call_structured — happy path
# ---------------------------------------------------------------------------


async def test_call_structured_success_records_metadata() -> None:
    cfg = _make_config()
    client = LLMClient(cfg, now=_fixed_now)

    async def handler(**_: Any) -> Any:
        return _mock_chat_response('{"message": "hi", "score": 0.9}')

    _install_client_mock(client, handler)

    parsed, meta = await client.call_structured(
        role="planner",
        messages=[{"role": "user", "content": "hi"}],
        output_schema=Echo,
    )
    assert parsed.message == "hi"
    assert parsed.score == pytest.approx(0.9)
    assert meta.model_name == "qwen3-max"
    assert meta.role == "planner"
    assert meta.tokens_input == 100
    assert meta.tokens_output == 50
    assert meta.http_retry_count == 0
    assert meta.currency == "RMB"
    # cost = 100 * 2.8/1e6 + 50 * 8.4/1e6 = 0.00028 + 0.00042 = 0.0007
    assert meta.cost == pytest.approx(0.0007)

    report = await client.get_cost_report()
    assert report.total_calls == 1
    assert report.total_cost == pytest.approx(0.0007)


# ---------------------------------------------------------------------------
# call_structured — parse-retry path
# ---------------------------------------------------------------------------


async def test_call_structured_retries_on_parse_failure() -> None:
    cfg = _make_config()
    client = LLMClient(cfg, now=_fixed_now)

    call_count = {"n": 0}

    async def handler(**_: Any) -> Any:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _mock_chat_response("not-json")
        return _mock_chat_response('{"message": "ok", "score": 1.0}')

    _install_client_mock(client, handler)

    parsed, _ = await client.call_structured(
        role="planner",
        messages=[{"role": "user", "content": "hi"}],
        output_schema=Echo,
    )
    assert parsed.message == "ok"
    assert call_count["n"] == 2


async def test_call_structured_raises_after_two_parse_failures() -> None:
    cfg = _make_config()
    client = LLMClient(cfg, now=_fixed_now)

    async def handler(**_: Any) -> Any:
        return _mock_chat_response("still-not-json")

    _install_client_mock(client, handler)

    with pytest.raises(LLMOutputParseError) as excinfo:
        await client.call_structured(
            role="planner",
            messages=[{"role": "user", "content": "hi"}],
            output_schema=Echo,
        )
    assert "still-not-json" in excinfo.value.raw_output


# ---------------------------------------------------------------------------
# Retry / backoff
# ---------------------------------------------------------------------------


async def test_call_retries_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_config()
    client = LLMClient(cfg, now=_fixed_now)

    # Avoid actually sleeping in the test.
    monkeypatch.setattr(
        "maestro.llm.client.asyncio.sleep",
        AsyncMock(return_value=None),
    )

    attempts = {"n": 0}

    def _make_429() -> APIStatusError:
        return APIStatusError(
            "rate limited",
            response=MagicMock(status_code=429),
            body=None,
        )

    async def handler(**_: Any) -> Any:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _make_429()
        return _mock_chat_response('{"message": "ok", "score": 0.5}')

    _install_client_mock(client, handler)

    parsed, meta = await client.call_structured(
        role="planner",
        messages=[{"role": "user", "content": "hi"}],
        output_schema=Echo,
    )
    assert parsed.message == "ok"
    assert attempts["n"] == 3
    # http_retry_count reflects the number of *retries* performed (attempts - 1)
    assert meta.http_retry_count == 2


async def test_call_non_retriable_error_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_config()
    client = LLMClient(cfg, now=_fixed_now)

    monkeypatch.setattr("maestro.llm.client.asyncio.sleep", AsyncMock(return_value=None))

    def _make_400() -> APIStatusError:
        return APIStatusError(
            "bad request",
            response=MagicMock(status_code=400),
            body=None,
        )

    async def handler(**_: Any) -> Any:
        raise _make_400()

    _install_client_mock(client, handler)

    with pytest.raises(LLMCallError):
        await client.call_structured(
            role="planner",
            messages=[{"role": "user", "content": "hi"}],
            output_schema=Echo,
        )


async def test_call_timeout_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_config()
    client = LLMClient(cfg, now=_fixed_now)

    monkeypatch.setattr("maestro.llm.client.asyncio.sleep", AsyncMock(return_value=None))

    attempts = {"n": 0}

    async def handler(**_: Any) -> Any:
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise APITimeoutError(request=MagicMock())
        return _mock_chat_response('{"message": "ok", "score": 0.0}')

    _install_client_mock(client, handler)
    await client.call_structured(
        role="planner",
        messages=[{"role": "user", "content": "hi"}],
        output_schema=Echo,
    )
    assert attempts["n"] == 2


# ---------------------------------------------------------------------------
# Semaphore concurrency
# ---------------------------------------------------------------------------


async def test_semaphore_limits_concurrent_inflight(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_config()
    assert cfg.global_semaphore_limit == 3
    client = LLMClient(cfg, now=_fixed_now)

    in_flight = 0
    peak = 0
    gate = asyncio.Event()

    async def handler(**_: Any) -> Any:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await gate.wait()
        in_flight -= 1
        return _mock_chat_response('{"message": "ok", "score": 0.0}')

    _install_client_mock(client, handler)

    async def one_call() -> None:
        await client.call_structured(
            role="planner",
            messages=[{"role": "user", "content": "hi"}],
            output_schema=Echo,
        )

    tasks = [asyncio.create_task(one_call()) for _ in range(10)]
    # Give the event loop a few ticks to fan out and block.
    for _ in range(20):
        await asyncio.sleep(0)
    gate.set()
    await asyncio.gather(*tasks)

    assert peak <= cfg.global_semaphore_limit
    assert peak == cfg.global_semaphore_limit  # saturated


# ---------------------------------------------------------------------------
# call_text
# ---------------------------------------------------------------------------


async def test_call_text_returns_plain_string() -> None:
    cfg = _make_config()
    client = LLMClient(cfg, now=_fixed_now)

    async def handler(**_: Any) -> Any:
        return _mock_chat_response("free-form response")

    _install_client_mock(client, handler)
    text, meta = await client.call_text(
        role="subagent",
        messages=[{"role": "user", "content": "plan"}],
    )
    assert text == "free-form response"
    assert meta.role == "subagent"


# ---------------------------------------------------------------------------
# call_with_tools (M1)
# ---------------------------------------------------------------------------


def _mock_tool_response(
    *,
    content: str | None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "tool_calls",
    prompt_tokens: int = 120,
    completion_tokens: int = 40,
) -> Any:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message = MagicMock()
    response.choices[0].message.content = content

    if tool_calls is None:
        response.choices[0].message.tool_calls = None
    else:
        calls: list[Any] = []
        for tc in tool_calls:
            m = MagicMock()
            m.id = tc["id"]
            m.function = MagicMock()
            m.function.name = tc["name"]
            m.function.arguments = tc["arguments"]
            calls.append(m)
        response.choices[0].message.tool_calls = calls

    response.choices[0].finish_reason = finish_reason
    response.usage = MagicMock()
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response


async def test_call_with_tools_parses_tool_calls() -> None:
    cfg = _make_config()
    client = LLMClient(cfg, now=_fixed_now)

    async def handler(**_: Any) -> Any:
        return _mock_tool_response(
            content=None,
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "read_file",
                    "arguments": '{"path": "src/app.py"}',
                }
            ],
            finish_reason="tool_calls",
        )

    mock = _install_client_mock(client, handler)

    resp, meta = await client.call_with_tools(
        role="subagent",
        messages=[{"role": "user", "content": "read the file"}],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
    )
    assert resp.text == ""
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].call_id == "call_1"
    assert resp.tool_calls[0].name == "read_file"
    assert resp.tool_calls[0].arguments == {"path": "src/app.py"}
    assert resp.finish_reason == "tool_calls"
    assert meta.role == "subagent"
    assert mock.call_count == 1
    # Assert tools + tool_choice were forwarded.
    forwarded = mock.call_args.kwargs
    assert forwarded["tools"][0]["function"]["name"] == "read_file"
    assert forwarded["tool_choice"] == "auto"


async def test_call_with_tools_returns_text_when_no_tool_calls() -> None:
    cfg = _make_config()
    client = LLMClient(cfg, now=_fixed_now)

    async def handler(**_: Any) -> Any:
        return _mock_tool_response(
            content="I have enough context, stopping exploration.",
            tool_calls=None,
            finish_reason="stop",
        )

    _install_client_mock(client, handler)
    resp, _ = await client.call_with_tools(
        role="subagent",
        messages=[{"role": "user", "content": "explore"}],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
    )
    assert resp.tool_calls == []
    assert "stopping exploration" in resp.text
    assert resp.finish_reason == "stop"


async def test_call_with_tools_tolerates_unparseable_arguments() -> None:
    cfg = _make_config()
    client = LLMClient(cfg, now=_fixed_now)

    async def handler(**_: Any) -> Any:
        return _mock_tool_response(
            content=None,
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "read_file",
                    "arguments": "this is not json",
                }
            ],
        )

    _install_client_mock(client, handler)
    resp, _ = await client.call_with_tools(
        role="subagent",
        messages=[],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
    )
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].arguments == {}
    assert resp.tool_calls[0].arguments_raw == "this is not json"


async def test_call_with_tools_records_metadata() -> None:
    cfg = _make_config()
    client = LLMClient(cfg, now=_fixed_now)

    async def handler(**_: Any) -> Any:
        return _mock_tool_response(
            content=None,
            tool_calls=[{"id": "c", "name": "read_file", "arguments": "{}"}],
        )

    _install_client_mock(client, handler)
    _, meta = await client.call_with_tools(
        role="subagent",
        messages=[],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
    )
    report = await client.get_cost_report()
    assert report.total_calls == 1
    assert meta.tokens_input == 120
    assert meta.tokens_output == 40


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------


async def test_dry_run_does_not_call_api() -> None:
    cfg = _make_config()
    client = LLMClient(cfg, dry_run=True, now=_fixed_now)

    async def handler(**_: Any) -> Any:
        raise AssertionError("dry-run must not call the API")

    _install_client_mock(client, handler)

    parsed, meta = await client.call_structured(
        role="planner",
        messages=[{"role": "user", "content": "x"}],
        output_schema=Echo,
    )
    assert isinstance(parsed, Echo)
    assert meta.cost == 0.0
    assert meta.tokens_input == 0

    text, _ = await client.call_text(
        role="subagent",
        messages=[{"role": "user", "content": "x"}],
    )
    assert text == "dummy text"


# ---------------------------------------------------------------------------
# CostReport aggregation
# ---------------------------------------------------------------------------


def test_cost_report_aggregates_per_role_and_model() -> None:
    records = [
        LLMCallMetadata(
            model_name="qwen3-max",
            role="planner",
            tokens_input=100,
            tokens_output=50,
            latency_ms=200,
            cost=0.5,
            currency="RMB",
            called_at=_fixed_now(),
            success=True,
            http_retry_count=0,
        ),
        LLMCallMetadata(
            model_name="qwen3-max",
            role="planner",
            tokens_input=200,
            tokens_output=80,
            latency_ms=300,
            cost=0.9,
            currency="RMB",
            called_at=_fixed_now(),
            success=True,
            http_retry_count=1,
        ),
        LLMCallMetadata(
            model_name="deepseek-v3",
            role="judge",
            tokens_input=1000,
            tokens_output=200,
            latency_ms=150,
            cost=0.2,
            currency="RMB",
            called_at=_fixed_now(),
            success=True,
            http_retry_count=0,
        ),
    ]
    report = _build_cost_report(records, currency="RMB")
    assert report.total_calls == 3
    assert report.total_tokens_input == 1300
    assert report.total_tokens_output == 330
    assert report.total_cost == pytest.approx(1.6)

    planner = next(r for r in report.per_role if r.role == "planner")
    assert planner.call_count == 2
    assert planner.cost == pytest.approx(1.4)
    assert planner.avg_latency_ms == pytest.approx(250.0)


def test_cost_report_markdown_mentions_currency() -> None:
    report = CostReport(
        currency="RMB",
        total_calls=0,
        total_tokens_input=0,
        total_tokens_output=0,
        total_cost=0.0,
        per_model=[],
        per_role=[],
    )
    md = report.to_markdown()
    assert "(RMB)" in md
    assert "Total calls" in md
