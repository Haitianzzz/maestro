"""Unit tests for ``maestro.verifier.llm_judge`` (spec 06 §5)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from maestro.llm.client import LLMCallMetadata, LLMClient
from maestro.llm.config import ClientConfig, ModelConfig
from maestro.models import (
    JudgeOutput,
    SubAgentResult,
    SubTask,
    TaskSpec,
    generate_task_id,
)
from maestro.sandbox.workspace import WorkspaceManager
from maestro.verifier.llm_judge import (
    LLMJudgeVerifier,
    compute_disagreement,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    # The marker appears ONLY in the pre-patch file and is NOT referenced by
    # ``_sub_result().diff`` below. That way
    # test_judge_prompt_includes_pre_patch_content can assert the marker is
    # present in the prompt and we know it came from the pre-patch read,
    # not from the diff section of the prompt.
    (repo / "src" / "app.py").write_text(
        "PRE_PATCH_MARKER_XYZ = True\n\ndef double(x):\n    return x * 2\n",
        encoding="utf-8",
    )
    return repo


def _make_client() -> LLMClient:
    cfg = ClientConfig(
        base_url="https://example.com/v1",
        api_key="fake",
        models={
            "judge": ModelConfig(
                name="deepseek-v3",
                display_name="DeepSeek-V3",
                price_input_per_mtok=0.28,
                price_output_per_mtok=1.12,
            ),
        },
    )
    return LLMClient(cfg)


def _spec(*, k: int = 3, threshold: float = 0.3) -> TaskSpec:
    return TaskSpec(
        task_id=generate_task_id(),
        description="test",
        repo_path=Path("/tmp"),
        judge_samples=k,
        judge_disagreement_threshold=threshold,
    )


def _sub_result() -> SubAgentResult:
    return SubAgentResult(
        subtask_id="t-001",
        status="success",
        diff="--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-def double(x): return x*2\n+def double(x): return x * 2\n",
        modified_files=["src/app.py"],
        rationale="rename internal variable",
        confidence=0.9,
        retry_count=0,
        tokens_input=10,
        tokens_output=10,
        latency_ms=1,
        model_used="qwen3-coder-plus",
        created_at=datetime(2026, 4, 23, tzinfo=UTC),
    )


def _subtask() -> SubTask:
    return SubTask(
        subtask_id="t-001",
        description="double x",
        reads=["src/app.py"],
        writes=["src/app.py"],
    )


def _meta(cost: float = 0.001) -> LLMCallMetadata:
    return LLMCallMetadata(
        model_name="deepseek-v3",
        role="judge",
        tokens_input=100,
        tokens_output=40,
        latency_ms=10,
        cost=cost,
        currency="RMB",
        called_at=datetime(2026, 4, 23, tzinfo=UTC),
        success=True,
        http_retry_count=0,
    )


def _install_judge(
    client: LLMClient,
    responder: Callable[..., Awaitable[tuple[JudgeOutput, LLMCallMetadata]]],
) -> AsyncMock:
    mock = AsyncMock(side_effect=responder)
    client.call_structured = mock  # type: ignore[method-assign]
    return mock


def _constant_scores(
    scores: list[float], passes: list[bool]
) -> Callable[..., Awaitable[tuple[JudgeOutput, LLMCallMetadata]]]:
    """Responder that returns one pre-baked JudgeOutput per call."""
    calls = {"n": 0}

    async def fake(**_: Any) -> tuple[JudgeOutput, LLMCallMetadata]:
        i = calls["n"]
        calls["n"] += 1
        out = JudgeOutput(
            score=scores[i],
            passes_requirements=passes[i],
            reasoning="rationale",
            detected_issues=[],
        )
        return out, _meta()

    return fake


# ---------------------------------------------------------------------------
# compute_disagreement — pure helper
# ---------------------------------------------------------------------------


def test_disagreement_all_agree_is_near_zero() -> None:
    d = compute_disagreement([0.9, 0.9, 0.9], [True, True, True])
    assert d == pytest.approx(0.0)


def test_disagreement_binary_split_raises_value() -> None:
    d = compute_disagreement([0.7, 0.3, 0.7, 0.3], [True, False, True, False])
    # Strong binary disagreement; stdev also non-trivial.
    assert d > 0.3


def test_disagreement_score_spread_without_vote_split() -> None:
    # Everyone votes pass, but scores spread widely. stdev dominates.
    d = compute_disagreement([0.2, 0.5, 0.9], [True, True, True])
    assert d > 0.0


def test_disagreement_single_sample_is_zero() -> None:
    d = compute_disagreement([0.8], [True])
    assert d == pytest.approx(0.0)


def test_disagreement_empty_is_zero() -> None:
    assert compute_disagreement([], []) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# LLMJudgeVerifier — pass / fail / uncertain paths
# ---------------------------------------------------------------------------


async def test_judge_passes_when_all_agree_high(sample_repo: Path) -> None:
    client = _make_client()
    _install_judge(
        client,
        _constant_scores([0.9, 0.85, 0.9], [True, True, True]),
    )
    with WorkspaceManager(sample_repo, task_id="t-ok") as mgr:
        iso = await mgr.create_isolated(_subtask())
        verifier = LLMJudgeVerifier(client, _spec(), workspace_manager=mgr)
        tier, detail = await verifier.run(_subtask(), iso, _sub_result())

    assert tier.passed is True
    assert tier.tier == "llm_judge"
    assert detail.is_uncertain is False
    assert detail.mean_score == pytest.approx((0.9 + 0.85 + 0.9) / 3)
    assert tier.cost_usd > 0


async def test_judge_fails_on_low_mean(sample_repo: Path) -> None:
    client = _make_client()
    _install_judge(
        client,
        _constant_scores([0.3, 0.4, 0.2], [False, False, False]),
    )
    with WorkspaceManager(sample_repo, task_id="t-low") as mgr:
        iso = await mgr.create_isolated(_subtask())
        verifier = LLMJudgeVerifier(client, _spec(), workspace_manager=mgr)
        tier, detail = await verifier.run(_subtask(), iso, _sub_result())

    assert tier.passed is False
    assert detail.mean_score < 0.6


async def test_judge_flags_uncertainty_on_split_votes(sample_repo: Path) -> None:
    """K=3 with 2 pass + 1 fail at wildly different scores → uncertain."""
    client = _make_client()
    _install_judge(
        client,
        _constant_scores([0.9, 0.2, 0.8], [True, False, True]),
    )
    with WorkspaceManager(sample_repo, task_id="t-split") as mgr:
        iso = await mgr.create_isolated(_subtask())
        verifier = LLMJudgeVerifier(client, _spec(k=3, threshold=0.3), workspace_manager=mgr)
        tier, detail = await verifier.run(_subtask(), iso, _sub_result())

    assert detail.is_uncertain is True
    assert tier.passed is False  # uncertain overrides mean/majority


async def test_judge_minority_pass_fails_even_without_uncertainty(
    sample_repo: Path,
) -> None:
    """Only 1/3 vote pass → majority check fails even at mean=0.7."""
    client = _make_client()
    _install_judge(
        client,
        _constant_scores([0.8, 0.7, 0.55], [True, False, False]),
    )
    with WorkspaceManager(sample_repo, task_id="t-minority") as mgr:
        iso = await mgr.create_isolated(_subtask())
        verifier = LLMJudgeVerifier(client, _spec(), workspace_manager=mgr)
        tier, _ = await verifier.run(_subtask(), iso, _sub_result())

    assert tier.passed is False


async def test_judge_cost_accumulates_across_samples(sample_repo: Path) -> None:
    client = _make_client()

    async def fake(**_: Any) -> tuple[JudgeOutput, LLMCallMetadata]:
        out = JudgeOutput(score=0.8, passes_requirements=True, reasoning="", detected_issues=[])
        return out, _meta(cost=0.01)

    _install_judge(client, fake)
    with WorkspaceManager(sample_repo, task_id="t-cost") as mgr:
        iso = await mgr.create_isolated(_subtask())
        verifier = LLMJudgeVerifier(client, _spec(k=3), workspace_manager=mgr)
        tier, _ = await verifier.run(_subtask(), iso, _sub_result())

    # 3 samples * 0.01 each
    assert tier.cost_usd == pytest.approx(0.03)


async def test_judge_fans_out_parallel(sample_repo: Path) -> None:
    """TaskGroup should dispatch K parallel LLM calls."""
    client = _make_client()
    call_count = {"n": 0}

    async def fake(**_: Any) -> tuple[JudgeOutput, LLMCallMetadata]:
        call_count["n"] += 1
        return (
            JudgeOutput(score=0.9, passes_requirements=True, reasoning="", detected_issues=[]),
            _meta(),
        )

    _install_judge(client, fake)
    with WorkspaceManager(sample_repo, task_id="t-fan") as mgr:
        iso = await mgr.create_isolated(_subtask())
        verifier = LLMJudgeVerifier(client, _spec(k=3), workspace_manager=mgr)
        await verifier.run(_subtask(), iso, _sub_result())

    assert call_count["n"] == 3


async def test_judge_prompt_includes_pre_patch_content(sample_repo: Path) -> None:
    """Verify the judge sees the *pre-diff* file content (M3)."""
    client = _make_client()
    captured: dict[str, Any] = {}

    async def fake(**kwargs: Any) -> tuple[JudgeOutput, LLMCallMetadata]:
        captured.setdefault("messages", kwargs["messages"])
        return (
            JudgeOutput(score=0.9, passes_requirements=True, reasoning="", detected_issues=[]),
            _meta(),
        )

    _install_judge(client, fake)
    with WorkspaceManager(sample_repo, task_id="t-prepatch") as mgr:
        iso = await mgr.create_isolated(_subtask())
        verifier = LLMJudgeVerifier(client, _spec(k=1), workspace_manager=mgr)
        await verifier.run(_subtask(), iso, _sub_result())

    user_msg = captured["messages"][-1]["content"]
    assert "src/app.py" in user_msg
    # The marker only appears in the pre-patch file, never in ``_sub_result``'s
    # diff section, so this assertion genuinely verifies that pre-patch
    # content was injected (M3). Asserting "return x * 2" would be unreliable
    # because that string also appears in the diff's ``+`` line.
    assert "PRE_PATCH_MARKER_XYZ" in user_msg
    assert "PRE_PATCH_MARKER_XYZ" not in _sub_result().diff  # sanity


async def test_judge_handles_missing_workspace_manager(sample_repo: Path) -> None:
    """When no workspace manager is provided, the judge still runs."""
    client = _make_client()
    _install_judge(
        client,
        _constant_scores([0.85, 0.85, 0.85], [True, True, True]),
    )
    with WorkspaceManager(sample_repo, task_id="t-noworkspace") as mgr:
        iso = await mgr.create_isolated(_subtask())
        verifier = LLMJudgeVerifier(client, _spec(), workspace_manager=None)
        tier, _ = await verifier.run(_subtask(), iso, _sub_result())

    assert tier.passed is True


# ---------------------------------------------------------------------------
# Temperatures spread
# ---------------------------------------------------------------------------


async def test_judge_uses_varied_temperatures(sample_repo: Path) -> None:
    client = _make_client()
    seen_temps: list[float] = []

    async def fake(**kwargs: Any) -> tuple[JudgeOutput, LLMCallMetadata]:
        seen_temps.append(kwargs["temperature"])
        return (
            JudgeOutput(score=0.8, passes_requirements=True, reasoning="", detected_issues=[]),
            _meta(),
        )

    _install_judge(client, fake)
    with WorkspaceManager(sample_repo, task_id="t-temps") as mgr:
        iso = await mgr.create_isolated(_subtask())
        verifier = LLMJudgeVerifier(client, _spec(k=3), workspace_manager=mgr)
        await verifier.run(_subtask(), iso, _sub_result())

    # Three distinct temperatures emitted.
    assert len(set(seen_temps)) == 3
