"""Unit tests for ``maestro.scheduler.scheduler`` (spec 04 §7)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from maestro.llm.client import LLMClient
from maestro.llm.config import ClientConfig, ModelConfig
from maestro.models import (
    SubAgentResult,
    SubTask,
    TaskDAG,
    TaskSpec,
    TierResult,
    VerificationResult,
)
from maestro.sandbox.workspace import IsolatedWorkspace, WorkspaceManager
from maestro.scheduler.scheduler import Scheduler
from maestro.subagent.subagent import SubAgent, SubAgentFactory

# ---------------------------------------------------------------------------
# Fake helpers
# ---------------------------------------------------------------------------


class _FakeSubAgent:
    """Stand-in sub-agent whose behaviour is driven by a registry."""

    def __init__(
        self,
        subtask: SubTask,
        workspace: IsolatedWorkspace,
        *,
        behaviour: Callable[[SubTask, int], SubAgentResult],
        call_log: list[tuple[str, int]],
        attempts: dict[str, int],
        prior_attempt: SubAgentResult | None,
    ) -> None:
        self._subtask = subtask
        self._workspace = workspace
        self._behaviour = behaviour
        self._call_log = call_log
        self._attempts = attempts
        self._attempt_index = attempts.get(subtask.subtask_id, 0)
        self._prior = prior_attempt

    async def run(self, global_context: str) -> SubAgentResult:
        del global_context
        idx = self._attempt_index
        self._attempts[self._subtask.subtask_id] = idx + 1
        self._call_log.append((self._subtask.subtask_id, idx))
        result = self._behaviour(self._subtask, idx)
        return result.model_copy(update={"retry_count": idx})


class _FakeFactory(SubAgentFactory):
    def __init__(
        self,
        *,
        behaviour: Callable[[SubTask, int], SubAgentResult],
        call_log: list[tuple[str, int]],
    ) -> None:
        super().__init__()
        self._behaviour = behaviour
        self._call_log = call_log
        self._attempts: dict[str, int] = {}

    def create(  # type: ignore[override]
        self,
        subtask: SubTask,
        workspace: IsolatedWorkspace,
        llm_client: LLMClient,
        *,
        prior_attempt: SubAgentResult | None = None,
        prior_failure: VerificationResult | None = None,
    ) -> SubAgent:
        del llm_client, prior_failure
        return _FakeSubAgent(  # type: ignore[return-value]
            subtask,
            workspace,
            behaviour=self._behaviour,
            call_log=self._call_log,
            attempts=self._attempts,
            prior_attempt=prior_attempt,
        )


class _FakeVerifier:
    """Per-subtask verifier driven by a registry of verdicts."""

    def __init__(
        self,
        *,
        verdicts: Callable[[SubTask, SubAgentResult], bool] | None = None,
    ) -> None:
        self._verdicts = verdicts or (lambda _s, _r: True)
        self.calls: list[tuple[str, int]] = []  # (subtask_id, retry_count)

    async def verify(
        self,
        subtask: SubTask,
        workspace: IsolatedWorkspace,
        sub_result: SubAgentResult,
    ) -> VerificationResult:
        del workspace
        passed = self._verdicts(subtask, sub_result)
        self.calls.append((subtask.subtask_id, sub_result.retry_count))
        return VerificationResult(
            subtask_id=subtask.subtask_id,
            overall_passed=passed,
            tiers=[
                TierResult(
                    tier="deterministic",
                    passed=passed,
                    details="ok" if passed else "fail",
                    latency_ms=1,
                )
            ],
            total_latency_ms=1,
        )


def _success_behaviour(
    diff: str = "--- a/x\n+++ b/x\n", modified: list[str] | None = None
) -> Callable[[SubTask, int], SubAgentResult]:
    def _b(subtask: SubTask, attempt: int) -> SubAgentResult:
        del attempt
        return SubAgentResult(
            subtask_id=subtask.subtask_id,
            status="success",
            diff=diff,
            modified_files=list(modified or subtask.writes),
            rationale="ok",
            confidence=0.9,
            retry_count=0,
            tokens_input=10,
            tokens_output=10,
            latency_ms=1,
            model_used="qwen3-coder-plus",
            created_at=datetime(2026, 4, 23, tzinfo=UTC),
        )

    return _b


def _sub_failure(subtask: SubTask, _attempt: int) -> SubAgentResult:
    return SubAgentResult(
        subtask_id=subtask.subtask_id,
        status="failed",
        diff="",
        modified_files=[],
        rationale="LLM broke",
        confidence=0.0,
        retry_count=0,
        tokens_input=0,
        tokens_output=0,
        latency_ms=0,
        model_used="qwen3-coder-plus",
        created_at=datetime(2026, 4, 23, tzinfo=UTC),
    )


def _spec(repo: Path, *, max_parallel: int = 4, max_retries: int = 2) -> TaskSpec:
    return TaskSpec(
        task_id="task-sched",
        description="test",
        repo_path=repo,
        max_parallel=max_parallel,
        max_retries_per_subtask=max_retries,
    )


def _dag(subtasks: list[SubTask]) -> TaskDAG:
    return TaskDAG(task_id="task-sched", subtasks=subtasks, global_context="")


def _st(
    sid: str,
    *,
    writes: list[str] | None = None,
    deps: list[str] | None = None,
    priority: int = 0,
) -> SubTask:
    return SubTask(
        subtask_id=sid,
        description=sid,
        reads=[],
        writes=writes or [f"src/{sid}.py"],
        depends_on=deps or [],
        priority=priority,
    )


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    return repo


def _make_llm_client() -> LLMClient:
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


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


async def test_scheduler_runs_all_subtasks_in_single_batch(sample_repo: Path) -> None:
    dag = _dag([_st("a"), _st("b"), _st("c")])
    call_log: list[tuple[str, int]] = []
    factory = _FakeFactory(behaviour=_success_behaviour(), call_log=call_log)
    verifier = _FakeVerifier()

    with WorkspaceManager(sample_repo, task_id="t") as ws:
        sch = Scheduler(
            dag=dag,
            task_spec=_spec(sample_repo),
            subagent_factory=factory,
            verifier=verifier,
            workspace_manager=ws,
            llm_client=_make_llm_client(),
        )
        results = await sch.execute()

    assert len(results) == 1
    batch = results[0]
    assert sorted(batch.merged_patches) == ["a", "b", "c"]
    assert batch.failed_patches == []
    assert len(batch.subtask_results) == 3
    assert {c[0] for c in call_log} == {"a", "b", "c"}


async def test_scheduler_walks_diamond_batches(sample_repo: Path) -> None:
    dag = _dag(
        [
            _st("a"),
            _st("b", deps=["a"]),
            _st("c", deps=["a"]),
            _st("d", deps=["b", "c"]),
        ]
    )
    factory = _FakeFactory(behaviour=_success_behaviour(), call_log=[])
    verifier = _FakeVerifier()

    with WorkspaceManager(sample_repo, task_id="t-diamond") as ws:
        sch = Scheduler(
            dag=dag,
            task_spec=_spec(sample_repo),
            subagent_factory=factory,
            verifier=verifier,
            workspace_manager=ws,
            llm_client=_make_llm_client(),
        )
        results = await sch.execute()

    assert [br.batch_index for br in results] == [0, 1, 2]
    assert [sorted(br.merged_patches) for br in results] == [["a"], ["b", "c"], ["d"]]


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


async def test_scheduler_retries_on_verify_failure(sample_repo: Path) -> None:
    def verdict(_s: SubTask, sub_result: SubAgentResult) -> bool:
        del _s
        # sub_result.retry_count counts attempts (0, 1, 2 ...).
        return sub_result.retry_count >= 2

    dag = _dag([_st("x")])
    factory = _FakeFactory(behaviour=_success_behaviour(), call_log=[])
    verifier = _FakeVerifier(verdicts=verdict)

    with WorkspaceManager(sample_repo, task_id="t-retry") as ws:
        sch = Scheduler(
            dag=dag,
            task_spec=_spec(sample_repo, max_retries=2),
            subagent_factory=factory,
            verifier=verifier,
            workspace_manager=ws,
            llm_client=_make_llm_client(),
        )
        results = await sch.execute()

    batch = results[0]
    assert batch.merged_patches == ["x"]
    assert batch.retried_patches == ["x"]
    assert len(verifier.calls) == 3  # attempt 0 (fail), attempt 1 (fail), attempt 2 (pass)


async def test_scheduler_respects_max_retries(sample_repo: Path) -> None:
    def verdict(_s: SubTask, _r: SubAgentResult) -> bool:
        return False  # never pass

    dag = _dag([_st("x")])
    factory = _FakeFactory(behaviour=_success_behaviour(), call_log=[])
    verifier = _FakeVerifier(verdicts=verdict)

    with WorkspaceManager(sample_repo, task_id="t-max") as ws:
        sch = Scheduler(
            dag=dag,
            task_spec=_spec(sample_repo, max_retries=2),
            subagent_factory=factory,
            verifier=verifier,
            workspace_manager=ws,
            llm_client=_make_llm_client(),
        )
        results = await sch.execute()

    batch = results[0]
    assert batch.failed_patches == ["x"]
    assert batch.merged_patches == []
    # Initial attempt + 2 retries = 3 verify calls.
    assert len(verifier.calls) == 3


async def test_subagent_failure_short_circuits_verify(sample_repo: Path) -> None:
    dag = _dag([_st("x")])
    factory = _FakeFactory(behaviour=_sub_failure, call_log=[])
    verifier = _FakeVerifier()

    with WorkspaceManager(sample_repo, task_id="t-agentfail") as ws:
        sch = Scheduler(
            dag=dag,
            task_spec=_spec(sample_repo),
            subagent_factory=factory,
            verifier=verifier,
            workspace_manager=ws,
            llm_client=_make_llm_client(),
        )
        results = await sch.execute()

    batch = results[0]
    assert batch.failed_patches == ["x"]
    assert verifier.calls == []  # never invoked


# ---------------------------------------------------------------------------
# Write-conflict defer
# ---------------------------------------------------------------------------


async def test_scheduler_defers_lower_priority_write_conflict(
    sample_repo: Path,
) -> None:
    # Two subtasks in the same batch write the same file; lower priority
    # should be deferred to batch 1.
    dag = _dag(
        [
            _st("a", writes=["src/shared.py"], priority=10),
            _st("b", writes=["src/shared.py"], priority=1),
        ]
    )
    factory = _FakeFactory(behaviour=_success_behaviour(), call_log=[])
    verifier = _FakeVerifier()

    with WorkspaceManager(sample_repo, task_id="t-defer") as ws:
        sch = Scheduler(
            dag=dag,
            task_spec=_spec(sample_repo),
            subagent_factory=factory,
            verifier=verifier,
            workspace_manager=ws,
            llm_client=_make_llm_client(),
        )
        results = await sch.execute()

    assert [br.batch_index for br in results] == [0, 1]
    assert results[0].merged_patches == ["a"]
    assert results[1].merged_patches == ["b"]
    # Conflict recorded on the batch that detected it.
    assert ("a", "b") in results[0].conflicts_detected


# ---------------------------------------------------------------------------
# Abort-on-full-batch-failure
# ---------------------------------------------------------------------------


async def test_scheduler_aborts_when_entire_batch_fails(sample_repo: Path) -> None:
    dag = _dag(
        [
            _st("a"),
            _st("downstream", deps=["a"]),
        ]
    )

    def verdict(subtask: SubTask, _r: SubAgentResult) -> bool:
        return subtask.subtask_id != "a"  # a fails always; downstream unreachable

    factory = _FakeFactory(behaviour=_success_behaviour(), call_log=[])
    verifier = _FakeVerifier(verdicts=verdict)

    with WorkspaceManager(sample_repo, task_id="t-abort") as ws:
        sch = Scheduler(
            dag=dag,
            task_spec=_spec(sample_repo, max_retries=0),
            subagent_factory=factory,
            verifier=verifier,
            workspace_manager=ws,
            llm_client=_make_llm_client(),
        )
        results = await sch.execute()

    assert len(results) == 1  # downstream batch never ran
    assert results[0].failed_patches == ["a"]


# ---------------------------------------------------------------------------
# Parallelism respects max_parallel
# ---------------------------------------------------------------------------


async def test_scheduler_respects_max_parallel(sample_repo: Path) -> None:
    gate = asyncio.Event()
    in_flight = 0
    peak = 0

    def behaviour_factory() -> Callable[[SubTask, int], SubAgentResult]:
        base = _success_behaviour()

        def _b(subtask: SubTask, attempt: int) -> SubAgentResult:
            return base(subtask, attempt)

        return _b

    class CountingFactory(SubAgentFactory):
        def create(  # type: ignore[override]
            self,
            subtask: SubTask,
            workspace: IsolatedWorkspace,
            llm_client: LLMClient,
            *,
            prior_attempt: SubAgentResult | None = None,
            prior_failure: VerificationResult | None = None,
        ) -> SubAgent:
            del llm_client, prior_failure, prior_attempt

            class Agent:
                async def run(self, global_context: str) -> SubAgentResult:
                    nonlocal in_flight, peak
                    del self, global_context
                    in_flight += 1
                    peak = max(peak, in_flight)
                    await gate.wait()
                    in_flight -= 1
                    return behaviour_factory()(subtask, 0)

            return Agent()  # type: ignore[return-value]

    dag = _dag([_st(f"s{i}") for i in range(6)])
    factory = CountingFactory()
    verifier = _FakeVerifier()

    with WorkspaceManager(sample_repo, task_id="t-para") as ws:
        sch = Scheduler(
            dag=dag,
            task_spec=_spec(sample_repo, max_parallel=2),
            subagent_factory=factory,
            verifier=verifier,
            workspace_manager=ws,
            llm_client=_make_llm_client(),
        )
        exec_task = asyncio.create_task(sch.execute())
        # Let the scheduler fan out as much as it can; 50 ticks is ample to
        # drain every reachable coroutine to its next await point. We do NOT
        # early-break on peak == 2 because that would mask a regression
        # where the semaphore gets bypassed and peak reaches 3+.
        for _ in range(50):
            await asyncio.sleep(0)
        assert peak == 2  # capped by max_parallel=2
        gate.set()
        results = await exec_task
        assert sorted(results[0].merged_patches) == sorted([f"s{i}" for i in range(6)])
