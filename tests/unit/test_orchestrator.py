"""Unit tests for ``maestro.orchestrator`` (LangGraph state machine)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from maestro.llm.client import LLMCallMetadata, LLMClient
from maestro.llm.config import ClientConfig, ModelConfig
from maestro.models import (
    PlannerLLMOutput,
    PlannerLLMSubTask,
    SubAgentResult,
    SubTask,
    TaskResult,
    TaskSpec,
    TierResult,
    VerificationResult,
    generate_task_id,
)
from maestro.orchestrator import OrchestratorDeps, OrchestratorState, build_graph
from maestro.planner.planner import Planner
from maestro.sandbox.workspace import IsolatedWorkspace, WorkspaceManager
from maestro.subagent.subagent import SubAgent, SubAgentFactory

# ---------------------------------------------------------------------------
# Fake sub-agent + verifier (reused shape from module 9 tests)
# ---------------------------------------------------------------------------


class _FakeAgent:
    def __init__(
        self,
        subtask: SubTask,
        workspace: IsolatedWorkspace,
        behaviour: Callable[[SubTask], SubAgentResult],
    ) -> None:
        self._subtask = subtask
        self._workspace = workspace
        self._behaviour = behaviour

    async def run(self, global_context: str) -> SubAgentResult:
        """Apply the diff the real sub-agent would have applied.

        The real :class:`SubAgent` writes its diff into the iso workspace before
        returning; otherwise ``merge_patches`` sees an unchanged copy of main
        and produces an empty final diff. Keep that invariant here so the
        orchestrator test genuinely exercises the merge path.
        """
        del global_context
        result = self._behaviour(self._subtask)
        if result.status == "success" and result.diff:
            ok, err = self._workspace.apply_diff(result.diff)
            assert ok, f"fake-agent diff apply failed: {err}"
        return result


class _FakeFactory(SubAgentFactory):
    def __init__(self, behaviour: Callable[[SubTask], SubAgentResult]) -> None:
        super().__init__()
        self._behaviour = behaviour

    def create(  # type: ignore[override]
        self,
        subtask: SubTask,
        workspace: IsolatedWorkspace,
        llm_client: LLMClient,
        *,
        prior_attempt: SubAgentResult | None = None,
        prior_failure: VerificationResult | None = None,
    ) -> SubAgent:
        del llm_client, prior_attempt, prior_failure
        return _FakeAgent(subtask, workspace, self._behaviour)  # type: ignore[return-value]


class _FakeVerifier:
    def __init__(self, verdict: Callable[[SubTask], bool] | None = None) -> None:
        self._verdict = verdict or (lambda _s: True)
        self.calls: list[str] = []

    async def verify(
        self,
        subtask: SubTask,
        workspace: IsolatedWorkspace,
        sub_result: SubAgentResult,
    ) -> VerificationResult:
        del workspace, sub_result
        passed = self._verdict(subtask)
        self.calls.append(subtask.subtask_id)
        return VerificationResult(
            subtask_id=subtask.subtask_id,
            overall_passed=passed,
            tiers=[
                TierResult(
                    tier="deterministic",
                    passed=passed,
                    details="ok" if passed else "fail",
                    latency_ms=1,
                    cost_usd=0.05 if passed else 0.0,
                )
            ],
            total_latency_ms=1,
            total_cost_usd=0.05 if passed else 0.0,
        )


def _success_subagent(diff: str) -> Callable[[SubTask], SubAgentResult]:
    def _b(subtask: SubTask) -> SubAgentResult:
        return SubAgentResult(
            subtask_id=subtask.subtask_id,
            status="success",
            diff=diff,
            modified_files=list(subtask.writes),
            rationale="ok",
            confidence=0.9,
            retry_count=0,
            tokens_input=12,
            tokens_output=8,
            latency_ms=1,
            model_used="qwen3-coder-plus",
            created_at=datetime(2026, 4, 23, tzinfo=UTC),
        )

    return _b


def _failing_subagent() -> Callable[[SubTask], SubAgentResult]:
    def _b(subtask: SubTask) -> SubAgentResult:
        return SubAgentResult(
            subtask_id=subtask.subtask_id,
            status="failed",
            diff="",
            modified_files=[],
            rationale="broken",
            confidence=0.0,
            retry_count=0,
            tokens_input=0,
            tokens_output=0,
            latency_ms=1,
            model_used="qwen3-coder-plus",
            created_at=datetime(2026, 4, 23, tzinfo=UTC),
        )

    return _b


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("X = 1\n", encoding="utf-8")
    (repo / "src" / "other.py").write_text("Y = 2\n", encoding="utf-8")
    return repo


def _make_llm_client() -> LLMClient:
    cfg = ClientConfig(
        base_url="https://example.com/v1",
        api_key="fake",
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
        },
    )
    return LLMClient(cfg)


def _planner_with_output(
    llm: LLMClient, output: PlannerLLMOutput, *, max_retries: int = 2
) -> Planner:
    async def responder(**_: Any) -> tuple[PlannerLLMOutput, LLMCallMetadata]:
        return output, LLMCallMetadata(
            model_name="qwen3-max",
            role="planner",
            tokens_input=30,
            tokens_output=20,
            latency_ms=5,
            cost=0.01,
            currency="RMB",
            called_at=datetime(2026, 4, 23, tzinfo=UTC),
            success=True,
            http_retry_count=0,
        )

    llm.call_structured = AsyncMock(side_effect=responder)  # type: ignore[method-assign]
    return Planner(llm, max_retries=max_retries)


def _simple_plan_output() -> PlannerLLMOutput:
    return PlannerLLMOutput(
        subtasks=[
            PlannerLLMSubTask(
                index=0,
                description="Touch app.py",
                reads=["src/app.py"],
                writes=["src/app.py"],
            ),
            PlannerLLMSubTask(
                index=1,
                description="Touch other.py",
                reads=["src/other.py"],
                writes=["src/other.py"],
            ),
        ],
        global_context="test task",
        planning_rationale="two parallel edits",
    )


_DIFF_APP = """\
--- a/src/app.py
+++ b/src/app.py
@@ -1 +1 @@
-X = 1
+X = 2
"""

_DIFF_OTHER = """\
--- a/src/other.py
+++ b/src/other.py
@@ -1 +1 @@
-Y = 2
+Y = 3
"""


def _spec(repo: Path) -> TaskSpec:
    return TaskSpec(
        task_id=generate_task_id(),
        description="two edits",
        repo_path=repo,
        max_parallel=4,
        max_retries_per_subtask=0,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_graph_full_happy_path(sample_repo: Path) -> None:
    llm = _make_llm_client()
    planner = _planner_with_output(llm, _simple_plan_output())

    def behaviour(subtask: SubTask) -> SubAgentResult:
        diff = _DIFF_APP if "app.py" in subtask.writes[0] else _DIFF_OTHER
        return _success_subagent(diff)(subtask)

    factory = _FakeFactory(behaviour)
    verifier = _FakeVerifier()

    with WorkspaceManager(sample_repo, task_id="t-orch-happy") as ws:
        deps = OrchestratorDeps(
            planner=planner,
            verifier=verifier,
            workspace=ws,
            llm_client=llm,
            subagent_factory=factory,
        )
        graph = build_graph(deps)
        spec = _spec(sample_repo)
        final_state: OrchestratorState = await graph.ainvoke({"task_spec": spec})

    assert final_state.get("error") is None
    result = final_state["final_result"]
    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.task_id == spec.task_id
    assert len(result.batches) == 1
    assert sorted(result.batches[0].merged_patches) == sorted(
        [f"{spec.task_id}-000", f"{spec.task_id}-001"]
    )
    # Token + cost aggregation:
    assert result.total_tokens_input == 24  # 2 subagents * 12
    assert result.total_tokens_output == 16  # 2 subagents * 8
    assert result.total_cost_usd == pytest.approx(0.10)  # 2 * 0.05 verify cost
    assert "X = 2" in result.final_diff
    assert "Y = 3" in result.final_diff


# ---------------------------------------------------------------------------
# Planner failure → error node
# ---------------------------------------------------------------------------


async def test_graph_planner_failure_routes_to_error(sample_repo: Path) -> None:
    llm = _make_llm_client()
    bad_plan = PlannerLLMOutput(
        subtasks=[
            PlannerLLMSubTask(
                index=0,
                description="nope",
                reads=["src/does_not_exist.py"],
                writes=["src/ghost.py"],
            ),
        ],
        global_context="bad",
        planning_rationale="bad",
    )
    # Disable retries so the planner gives up quickly.
    planner = _planner_with_output(llm, bad_plan, max_retries=0)

    factory = _FakeFactory(_success_subagent(_DIFF_APP))
    verifier = _FakeVerifier()

    with WorkspaceManager(sample_repo, task_id="t-orch-plan-fail") as ws:
        deps = OrchestratorDeps(
            planner=planner,
            verifier=verifier,
            workspace=ws,
            llm_client=llm,
            subagent_factory=factory,
        )
        graph = build_graph(deps)
        final_state = await graph.ainvoke({"task_spec": _spec(sample_repo)})

    assert final_state.get("final_result") is None
    assert "Planner failed" in final_state["error"]
    # Scheduler/verifier must not have been touched.
    assert verifier.calls == []


# ---------------------------------------------------------------------------
# Partial success status
# ---------------------------------------------------------------------------


async def test_graph_partial_status_when_one_subtask_fails(
    sample_repo: Path,
) -> None:
    llm = _make_llm_client()
    planner = _planner_with_output(llm, _simple_plan_output())

    def behaviour(subtask: SubTask) -> SubAgentResult:
        # First subtask fails at sub-agent level; second succeeds.
        if subtask.subtask_id.endswith("-000"):
            return _failing_subagent()(subtask)
        return _success_subagent(_DIFF_OTHER)(subtask)

    factory = _FakeFactory(behaviour)
    verifier = _FakeVerifier()

    with WorkspaceManager(sample_repo, task_id="t-orch-partial") as ws:
        deps = OrchestratorDeps(
            planner=planner,
            verifier=verifier,
            workspace=ws,
            llm_client=llm,
            subagent_factory=factory,
        )
        graph = build_graph(deps)
        final_state = await graph.ainvoke({"task_spec": _spec(sample_repo)})

    result = final_state["final_result"]
    assert result.status == "partial"
    merged = result.batches[0].merged_patches
    failed = result.batches[0].failed_patches
    assert len(merged) == 1
    assert len(failed) == 1


# ---------------------------------------------------------------------------
# Failed status when nothing merges
# ---------------------------------------------------------------------------


async def test_graph_failed_status_when_all_subagents_fail(
    sample_repo: Path,
) -> None:
    llm = _make_llm_client()
    planner = _planner_with_output(llm, _simple_plan_output())
    factory = _FakeFactory(_failing_subagent())
    verifier = _FakeVerifier()

    with WorkspaceManager(sample_repo, task_id="t-orch-allfail") as ws:
        deps = OrchestratorDeps(
            planner=planner,
            verifier=verifier,
            workspace=ws,
            llm_client=llm,
            subagent_factory=factory,
        )
        graph = build_graph(deps)
        final_state = await graph.ainvoke({"task_spec": _spec(sample_repo)})

    result = final_state["final_result"]
    assert result.status == "failed"
    # Verifier never ran (sub-agent failed short-circuited it).
    assert verifier.calls == []


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


async def test_graph_aggregates_tokens_and_cost_across_batches(
    sample_repo: Path,
) -> None:
    """Verify TaskResult sums tokens from every SubAgentResult and verifier."""
    llm = _make_llm_client()
    plan = PlannerLLMOutput(
        subtasks=[
            PlannerLLMSubTask(
                index=0, description="a", writes=["src/app.py"], reads=["src/app.py"]
            ),
            PlannerLLMSubTask(
                index=1,
                description="b",
                writes=["src/other.py"],
                depends_on_indices=[0],
            ),
        ],
        global_context="sequential",
        planning_rationale="",
    )
    planner = _planner_with_output(llm, plan)

    def behaviour(subtask: SubTask) -> SubAgentResult:
        diff = _DIFF_APP if subtask.writes[0] == "src/app.py" else _DIFF_OTHER
        return _success_subagent(diff)(subtask)

    factory = _FakeFactory(behaviour)
    verifier = _FakeVerifier()

    with WorkspaceManager(sample_repo, task_id="t-orch-agg") as ws:
        deps = OrchestratorDeps(
            planner=planner,
            verifier=verifier,
            workspace=ws,
            llm_client=llm,
            subagent_factory=factory,
        )
        graph = build_graph(deps)
        final_state = await graph.ainvoke({"task_spec": _spec(sample_repo)})

    result = final_state["final_result"]
    # 2 batches, each with 1 subagent (12 in, 8 out) and 1 verify (0.05 cost).
    assert result.total_tokens_input == 24
    assert result.total_tokens_output == 16
    assert result.total_cost_usd == pytest.approx(0.10)
    assert len(result.batches) == 2
