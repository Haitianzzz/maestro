"""Unit tests for ``maestro.subagent.subagent`` (spec 05 §8)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from maestro.llm.client import (
    LLMCallMetadata,
    LLMClient,
    ToolCallRequest,
    ToolCallResponse,
)
from maestro.llm.config import ClientConfig, ModelConfig
from maestro.models import (
    LLMJudgeDetail,
    SubAgentOutput,
    SubAgentResult,
    SubTask,
    TierResult,
    VerificationResult,
)
from maestro.sandbox.workspace import WorkspaceManager
from maestro.subagent.subagent import SubAgent, SubAgentFactory

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")
    (repo / "src" / "utils.py").write_text("VERSION = '0.1'\n", encoding="utf-8")
    return repo


def _make_client() -> LLMClient:
    cfg = ClientConfig(
        base_url="https://example.com/v1",
        api_key="fake",
        models={
            "subagent": ModelConfig(
                name="qwen3-coder-plus",
                display_name="Qwen3-Coder-Plus",
                price_input_per_mtok=0.84,
                price_output_per_mtok=3.36,
            ),
        },
    )
    return LLMClient(cfg)


def _fake_meta(tokens_in: int = 10, tokens_out: int = 5) -> LLMCallMetadata:
    return LLMCallMetadata(
        model_name="qwen3-coder-plus",
        role="subagent",
        tokens_input=tokens_in,
        tokens_output=tokens_out,
        latency_ms=10,
        cost=0.0,
        currency="RMB",
        called_at=datetime(2026, 4, 23, tzinfo=UTC),
        success=True,
        http_retry_count=0,
    )


def _stub_tools(
    responses: list[ToolCallResponse],
) -> Callable[..., Awaitable[tuple[ToolCallResponse, LLMCallMetadata]]]:
    iterator = iter(responses)

    async def fake(**_: Any) -> tuple[ToolCallResponse, LLMCallMetadata]:
        try:
            return next(iterator), _fake_meta()
        except StopIteration:
            # Once exhausted, return "stop" so the loop can't spin forever.
            return ToolCallResponse(text="done", finish_reason="stop"), _fake_meta()

    return fake


def _install(
    client: LLMClient,
    *,
    tools: Callable[..., Awaitable[tuple[ToolCallResponse, LLMCallMetadata]]] | None = None,
    text: Callable[..., Awaitable[tuple[str, LLMCallMetadata]]] | None = None,
    structured: Callable[..., Awaitable[tuple[Any, LLMCallMetadata]]] | None = None,
) -> None:
    if tools is not None:
        client.call_with_tools = AsyncMock(side_effect=tools)  # type: ignore[method-assign]
    if text is not None:
        client.call_text = AsyncMock(side_effect=text)  # type: ignore[method-assign]
    if structured is not None:
        client.call_structured = AsyncMock(side_effect=structured)  # type: ignore[method-assign]


def _subtask(
    *,
    reads: list[str] | None = None,
    writes: list[str] | None = None,
) -> SubTask:
    return SubTask(
        subtask_id="t-001",
        description="Rewrite the greeting",
        reads=reads or ["src/app.py"],
        writes=writes or ["src/app.py"],
    )


async def _make_iso(sample_repo: Path, subtask: SubTask) -> tuple[WorkspaceManager, Any]:
    mgr = WorkspaceManager(sample_repo, task_id="t")
    iso = await mgr.create_isolated(subtask)
    return mgr, iso


_GOOD_DIFF = """\
--- a/src/app.py
+++ b/src/app.py
@@ -1,2 +1,2 @@
 def hello():
-    return 'world'
+    return 'maestro'
"""


_OUT_OF_WRITES_DIFF = """\
--- a/src/utils.py
+++ b/src/utils.py
@@ -1 +1 @@
-VERSION = '0.1'
+VERSION = '0.2'
"""


_BAD_HUNK_DIFF = """\
--- a/src/app.py
+++ b/src/app.py
@@ -1,2 +1,2 @@
 def hello():
-    return 'WRONG CONTEXT'
+    return 'x'
"""


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_subagent_success_path(sample_repo: Path) -> None:
    subtask = _subtask()
    mgr, iso = await _make_iso(sample_repo, subtask)
    try:
        client = _make_client()

        # Explore: one read_file round, then stop.
        tool_responses = [
            ToolCallResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="call_1",
                        name="read_file",
                        arguments={"path": "src/app.py"},
                        arguments_raw='{"path": "src/app.py"}',
                    ),
                ],
                finish_reason="tool_calls",
            ),
        ]

        async def text_responder(**_: Any) -> tuple[str, LLMCallMetadata]:
            return "I will change the return value.", _fake_meta()

        async def structured_responder(**_: Any) -> tuple[SubAgentOutput, LLMCallMetadata]:
            out = SubAgentOutput(
                status="success",
                diff=_GOOD_DIFF,
                modified_files=["src/app.py"],
                rationale="changed greeting",
                confidence=0.9,
            )
            return out, _fake_meta()

        _install(
            client,
            tools=_stub_tools(tool_responses),
            text=text_responder,
            structured=structured_responder,
        )

        agent = SubAgent(subtask, iso, client)
        result = await agent.run(global_context="add greeting")

        assert isinstance(result, SubAgentResult)
        assert result.status == "success"
        assert result.modified_files == ["src/app.py"]
        assert result.retry_count == 0
        assert result.model_used == "qwen3-coder-plus"
        # Diff was applied to iso.
        assert "maestro" in (iso.path / "src" / "app.py").read_text(encoding="utf-8")
    finally:
        mgr.cleanup()


async def test_subagent_skips_explore_when_no_tool_calls(sample_repo: Path) -> None:
    """Model may decide to skip Explore entirely — single-round then stop."""
    subtask = _subtask()
    mgr, iso = await _make_iso(sample_repo, subtask)
    try:
        client = _make_client()
        tool_call_count = 0

        async def tools_responder(**_: Any) -> tuple[ToolCallResponse, LLMCallMetadata]:
            nonlocal tool_call_count
            tool_call_count += 1
            return ToolCallResponse(
                text="no need to read anything", finish_reason="stop"
            ), _fake_meta()

        async def text_responder(**_: Any) -> tuple[str, LLMCallMetadata]:
            return "plan", _fake_meta()

        async def structured_responder(**_: Any) -> tuple[SubAgentOutput, LLMCallMetadata]:
            out = SubAgentOutput(
                status="success",
                diff=_GOOD_DIFF,
                modified_files=["src/app.py"],
                rationale="r",
                confidence=0.7,
            )
            return out, _fake_meta()

        _install(
            client,
            tools=tools_responder,
            text=text_responder,
            structured=structured_responder,
        )

        agent = SubAgent(subtask, iso, client)
        result = await agent.run(global_context="do thing")
        assert result.status == "success"
        assert tool_call_count == 1  # stopped after round 1
    finally:
        mgr.cleanup()


async def test_subagent_caps_explore_at_max_rounds(sample_repo: Path) -> None:
    """Even if the model keeps issuing tool calls, we exit after max_rounds."""
    subtask = _subtask()
    mgr, iso = await _make_iso(sample_repo, subtask)
    try:
        client = _make_client()

        tool_call_count = 0

        async def tools_responder(**_: Any) -> tuple[ToolCallResponse, LLMCallMetadata]:
            nonlocal tool_call_count
            tool_call_count += 1
            return ToolCallResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id=f"call_{tool_call_count}",
                        name="read_file",
                        arguments={"path": "src/app.py"},
                        arguments_raw='{"path": "src/app.py"}',
                    )
                ],
                finish_reason="tool_calls",
            ), _fake_meta()

        async def text_responder(**_: Any) -> tuple[str, LLMCallMetadata]:
            return "plan", _fake_meta()

        async def structured_responder(**_: Any) -> tuple[SubAgentOutput, LLMCallMetadata]:
            out = SubAgentOutput(
                status="success",
                diff=_GOOD_DIFF,
                modified_files=["src/app.py"],
                rationale="r",
                confidence=0.7,
            )
            return out, _fake_meta()

        _install(
            client,
            tools=tools_responder,
            text=text_responder,
            structured=structured_responder,
        )

        agent = SubAgent(subtask, iso, client, max_explore_rounds=3)
        await agent.run(global_context="x")
        assert tool_call_count == 3
    finally:
        mgr.cleanup()


# ---------------------------------------------------------------------------
# Permission + diff validation
# ---------------------------------------------------------------------------


async def test_read_file_permission_denied_surfaces_to_llm(
    sample_repo: Path,
) -> None:
    """The sub-agent catches PermissionError and feeds 'permission denied' back."""
    subtask = _subtask(reads=["src/app.py"], writes=["src/app.py"])
    mgr, iso = await _make_iso(sample_repo, subtask)
    try:
        client = _make_client()

        captured_tool_messages: list[str] = []

        round_counter = {"n": 0}

        async def tools_responder(**kwargs: Any) -> tuple[ToolCallResponse, LLMCallMetadata]:
            # On the second call, the previous tool-result must be in messages.
            for msg in kwargs["messages"]:
                if msg.get("role") == "tool":
                    captured_tool_messages.append(str(msg.get("content", "")))
            round_counter["n"] += 1
            if round_counter["n"] == 1:
                # First call: ask to read an undeclared file.
                return ToolCallResponse(
                    tool_calls=[
                        ToolCallRequest(
                            call_id="call_1",
                            name="read_file",
                            arguments={"path": "src/utils.py"},  # not in reads
                            arguments_raw='{"path": "src/utils.py"}',
                        )
                    ],
                    finish_reason="tool_calls",
                ), _fake_meta()
            return ToolCallResponse(text="done", finish_reason="stop"), _fake_meta()

        async def text_responder(**_: Any) -> tuple[str, LLMCallMetadata]:
            return "plan", _fake_meta()

        async def structured_responder(**_: Any) -> tuple[SubAgentOutput, LLMCallMetadata]:
            out = SubAgentOutput(
                status="success",
                diff=_GOOD_DIFF,
                modified_files=["src/app.py"],
                rationale="r",
                confidence=0.7,
            )
            return out, _fake_meta()

        _install(
            client,
            tools=tools_responder,
            text=text_responder,
            structured=structured_responder,
        )

        agent = SubAgent(subtask, iso, client)
        await agent.run(global_context="x")
        joined = "\n".join(captured_tool_messages)
        assert "permission denied" in joined
    finally:
        mgr.cleanup()


async def test_subagent_rejects_diff_touching_file_outside_writes(
    sample_repo: Path,
) -> None:
    subtask = _subtask(writes=["src/app.py"])  # utils.py NOT in writes
    mgr, iso = await _make_iso(sample_repo, subtask)
    try:
        client = _make_client()

        async def tools_responder(**_: Any) -> tuple[ToolCallResponse, LLMCallMetadata]:
            return ToolCallResponse(text="", finish_reason="stop"), _fake_meta()

        async def text_responder(**_: Any) -> tuple[str, LLMCallMetadata]:
            return "plan", _fake_meta()

        async def structured_responder(**_: Any) -> tuple[SubAgentOutput, LLMCallMetadata]:
            out = SubAgentOutput(
                status="success",
                diff=_OUT_OF_WRITES_DIFF,
                modified_files=["src/utils.py"],
                rationale="",
                confidence=0.5,
            )
            return out, _fake_meta()

        _install(
            client,
            tools=tools_responder,
            text=text_responder,
            structured=structured_responder,
        )

        agent = SubAgent(subtask, iso, client)
        result = await agent.run(global_context="x")
        assert result.status == "rejected"
        assert "outside writes" in result.rationale
        # Iso workspace must be unchanged.
        assert (iso.path / "src" / "utils.py").read_text(encoding="utf-8") == "VERSION = '0.1'\n"
    finally:
        mgr.cleanup()


async def test_subagent_catches_diff_forgery_via_modified_files(
    sample_repo: Path,
) -> None:
    """A lying ``modified_files`` list can't bypass the diff-header check.

    This verifies the dual-source detection: modified_files lies that
    src/app.py was touched, but the diff header --- a/src/utils.py /
    +++ b/src/utils.py exposes the real target, which is outside writes.
    _find_illegal_writes must parse both sources to catch this.
    """
    subtask = _subtask(writes=["src/app.py"])
    mgr, iso = await _make_iso(sample_repo, subtask)
    try:
        client = _make_client()

        async def tools_responder(**_: Any) -> tuple[ToolCallResponse, LLMCallMetadata]:
            return ToolCallResponse(text="", finish_reason="stop"), _fake_meta()

        async def text_responder(**_: Any) -> tuple[str, LLMCallMetadata]:
            return "plan", _fake_meta()

        async def structured_responder(**_: Any) -> tuple[SubAgentOutput, LLMCallMetadata]:
            out = SubAgentOutput(
                status="success",
                diff=_OUT_OF_WRITES_DIFF,
                modified_files=["src/app.py"],  # claims app.py, diff touches utils.py
                rationale="",
                confidence=0.5,
            )
            return out, _fake_meta()

        _install(
            client,
            tools=tools_responder,
            text=text_responder,
            structured=structured_responder,
        )

        agent = SubAgent(subtask, iso, client)
        result = await agent.run(global_context="x")
        assert result.status == "rejected"
        assert "outside writes" in result.rationale
    finally:
        mgr.cleanup()


async def test_subagent_rejects_unapplyable_diff(sample_repo: Path) -> None:
    subtask = _subtask(writes=["src/app.py"])
    mgr, iso = await _make_iso(sample_repo, subtask)
    try:
        client = _make_client()

        async def tools_responder(**_: Any) -> tuple[ToolCallResponse, LLMCallMetadata]:
            return ToolCallResponse(text="", finish_reason="stop"), _fake_meta()

        async def text_responder(**_: Any) -> tuple[str, LLMCallMetadata]:
            return "plan", _fake_meta()

        async def structured_responder(**_: Any) -> tuple[SubAgentOutput, LLMCallMetadata]:
            out = SubAgentOutput(
                status="success",
                diff=_BAD_HUNK_DIFF,
                modified_files=["src/app.py"],
                rationale="",
                confidence=0.5,
            )
            return out, _fake_meta()

        _install(
            client,
            tools=tools_responder,
            text=text_responder,
            structured=structured_responder,
        )

        agent = SubAgent(subtask, iso, client)
        result = await agent.run(global_context="x")
        assert result.status == "rejected"
        assert "apply failed" in result.rationale
    finally:
        mgr.cleanup()


# ---------------------------------------------------------------------------
# Prior-failure feedback
# ---------------------------------------------------------------------------


async def test_prior_failure_injected_into_write_prompt(
    sample_repo: Path,
) -> None:
    subtask = _subtask(writes=["src/app.py"])
    mgr, iso = await _make_iso(sample_repo, subtask)
    try:
        client = _make_client()

        prior_result = SubAgentResult(
            subtask_id=subtask.subtask_id,
            status="success",
            diff="old diff content",
            modified_files=["src/app.py"],
            rationale="",
            confidence=0.5,
            retry_count=0,
            tokens_input=10,
            tokens_output=10,
            latency_ms=100,
            model_used="qwen3-coder-plus",
            created_at=datetime(2026, 4, 23, tzinfo=UTC),
        )
        prior_verify = VerificationResult(
            subtask_id=subtask.subtask_id,
            overall_passed=False,
            tiers=[
                TierResult(
                    tier="deterministic",
                    passed=False,
                    details="ruff failed: E501 line too long",
                    latency_ms=5,
                )
            ],
            judge_detail=LLMJudgeDetail(
                samples=[0.2, 0.3, 0.4],
                mean_score=0.3,
                disagreement=0.1,
                is_uncertain=False,
                judge_model="deepseek-v3",
            ),
            total_latency_ms=5,
        )

        async def tools_responder(**_: Any) -> tuple[ToolCallResponse, LLMCallMetadata]:
            return ToolCallResponse(text="", finish_reason="stop"), _fake_meta()

        async def text_responder(**_: Any) -> tuple[str, LLMCallMetadata]:
            return "plan", _fake_meta()

        captured: dict[str, Any] = {}

        async def structured_responder(
            **kwargs: Any,
        ) -> tuple[SubAgentOutput, LLMCallMetadata]:
            captured["messages"] = kwargs["messages"]
            out = SubAgentOutput(
                status="success",
                diff=_GOOD_DIFF,
                modified_files=["src/app.py"],
                rationale="fixed it",
                confidence=0.9,
            )
            return out, _fake_meta()

        _install(
            client,
            tools=tools_responder,
            text=text_responder,
            structured=structured_responder,
        )

        agent = SubAgent(
            subtask,
            iso,
            client,
            prior_attempt=prior_result,
            prior_failure=prior_verify,
        )
        result = await agent.run(global_context="x")
        assert result.retry_count == 1

        write_prompt = captured["messages"][-1]["content"]
        assert "Prior attempt failed" in write_prompt
        assert "ruff failed" in write_prompt
        assert "old diff content" in write_prompt
    finally:
        mgr.cleanup()


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------


async def test_subagent_aggregates_tokens_across_all_phases(sample_repo: Path) -> None:
    """Tokens from Explore, Plan, and Write must all land in SubAgentResult.

    This is the foundation of the benchmark's cost column — if we silently
    only record the last phase's usage, the Pareto frontier numbers go
    invisible-wrong. Guard it here.
    """
    subtask = _subtask(writes=["src/app.py"])
    mgr, iso = await _make_iso(sample_repo, subtask)
    try:
        client = _make_client()

        explore_round = {"n": 0}

        async def tools_responder(**_: Any) -> tuple[ToolCallResponse, LLMCallMetadata]:
            explore_round["n"] += 1
            if explore_round["n"] == 1:
                return ToolCallResponse(
                    tool_calls=[
                        ToolCallRequest(
                            call_id="call_1",
                            name="read_file",
                            arguments={"path": "src/app.py"},
                            arguments_raw='{"path": "src/app.py"}',
                        )
                    ],
                    finish_reason="tool_calls",
                ), _fake_meta(tokens_in=100, tokens_out=20)
            # Second round: model stops exploring (no tool calls). To keep
            # total = 100 + 50 + 80 = 230, this terminal round must report
            # zero tokens.
            return ToolCallResponse(text="done", finish_reason="stop"), _fake_meta(
                tokens_in=0, tokens_out=0
            )

        async def text_responder(**_: Any) -> tuple[str, LLMCallMetadata]:
            return "plan", _fake_meta(tokens_in=50, tokens_out=30)

        async def structured_responder(**_: Any) -> tuple[SubAgentOutput, LLMCallMetadata]:
            out = SubAgentOutput(
                status="success",
                diff=_GOOD_DIFF,
                modified_files=["src/app.py"],
                rationale="r",
                confidence=0.9,
            )
            return out, _fake_meta(tokens_in=80, tokens_out=150)

        _install(
            client,
            tools=tools_responder,
            text=text_responder,
            structured=structured_responder,
        )

        agent = SubAgent(subtask, iso, client)
        result = await agent.run(global_context="x")
        assert result.status == "success"
        assert result.tokens_input == 100 + 50 + 80 == 230
        assert result.tokens_output == 20 + 30 + 150 == 200
    finally:
        mgr.cleanup()


async def test_subagent_aggregates_tokens_across_multiple_explore_rounds(
    sample_repo: Path,
) -> None:
    """Explore's per-round tokens must accumulate, not be overwritten."""
    subtask = _subtask(writes=["src/app.py"])
    mgr, iso = await _make_iso(sample_repo, subtask)
    try:
        client = _make_client()

        round_n = {"n": 0}

        async def tools_responder(**_: Any) -> tuple[ToolCallResponse, LLMCallMetadata]:
            round_n["n"] += 1
            if round_n["n"] <= 3:
                return ToolCallResponse(
                    tool_calls=[
                        ToolCallRequest(
                            call_id=f"call_{round_n['n']}",
                            name="read_file",
                            arguments={"path": "src/app.py"},
                            arguments_raw='{"path": "src/app.py"}',
                        )
                    ],
                    finish_reason="tool_calls",
                ), _fake_meta(tokens_in=10, tokens_out=5)
            return ToolCallResponse(text="enough", finish_reason="stop"), _fake_meta(
                tokens_in=10, tokens_out=5
            )

        async def text_responder(**_: Any) -> tuple[str, LLMCallMetadata]:
            return "plan", _fake_meta(tokens_in=40, tokens_out=25)

        async def structured_responder(**_: Any) -> tuple[SubAgentOutput, LLMCallMetadata]:
            out = SubAgentOutput(
                status="success",
                diff=_GOOD_DIFF,
                modified_files=["src/app.py"],
                rationale="r",
                confidence=0.9,
            )
            return out, _fake_meta(tokens_in=70, tokens_out=110)

        _install(
            client,
            tools=tools_responder,
            text=text_responder,
            structured=structured_responder,
        )

        agent = SubAgent(subtask, iso, client)
        result = await agent.run(global_context="x")
        assert result.status == "success"
        # 4 explore rounds (3 with tool calls + 1 terminal) * (10 in, 5 out)
        # + plan (40, 25) + write (70, 110)
        expected_in = 4 * 10 + 40 + 70
        expected_out = 4 * 5 + 25 + 110
        assert result.tokens_input == expected_in
        assert result.tokens_output == expected_out
    finally:
        mgr.cleanup()


# ---------------------------------------------------------------------------
# LLM failure → graceful failed result
# ---------------------------------------------------------------------------


async def test_llm_exception_produces_failed_result(sample_repo: Path) -> None:
    subtask = _subtask()
    mgr, iso = await _make_iso(sample_repo, subtask)
    try:
        client = _make_client()

        async def tools_responder(**_: Any) -> tuple[ToolCallResponse, LLMCallMetadata]:
            raise RuntimeError("LLM exploded")

        _install(client, tools=tools_responder)
        agent = SubAgent(subtask, iso, client)
        result = await agent.run(global_context="x")
        assert result.status == "failed"
        assert "LLM exploded" in result.rationale
    finally:
        mgr.cleanup()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


async def test_factory_creates_sub_agent_with_prior_failure(sample_repo: Path) -> None:
    subtask = _subtask()
    mgr, iso = await _make_iso(sample_repo, subtask)
    try:
        factory = SubAgentFactory(max_explore_rounds=2)
        agent = factory.create(subtask, iso, _make_client())
        assert isinstance(agent, SubAgent)
        assert agent._max_explore_rounds == 2
    finally:
        mgr.cleanup()
