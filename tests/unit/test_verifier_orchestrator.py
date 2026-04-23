"""Unit tests for ``maestro.verifier.orchestrator`` (spec 06 §2)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from maestro.llm.client import LLMClient
from maestro.llm.config import ClientConfig, ModelConfig
from maestro.models import (
    LLMJudgeDetail,
    SubAgentResult,
    SubTask,
    TaskSpec,
    TierResult,
    generate_task_id,
)
from maestro.sandbox.workspace import IsolatedWorkspace, WorkspaceManager
from maestro.verifier import Verifier

# ---------------------------------------------------------------------------
# Fake tier shims (avoid real ruff / pytest / LLM)
# ---------------------------------------------------------------------------


class _FakeDet:
    def __init__(self, *, passed: bool, details: str = "stubbed ruff") -> None:
        self._passed = passed
        self._details = details
        self.calls = 0

    async def run(self, workspace: IsolatedWorkspace, sub_result: SubAgentResult) -> TierResult:
        del workspace, sub_result
        self.calls += 1
        return TierResult(
            tier="deterministic",
            passed=self._passed,
            details=self._details,
            latency_ms=1,
            cost_usd=0.0,
        )


class _FakeTest:
    def __init__(self, *, passed: bool) -> None:
        self._passed = passed
        self.calls = 0

    async def run(self, workspace: IsolatedWorkspace, sub_result: SubAgentResult) -> TierResult:
        del workspace, sub_result
        self.calls += 1
        return TierResult(
            tier="test_based",
            passed=self._passed,
            details="stubbed pytest",
            latency_ms=1,
            cost_usd=0.0,
        )


class _FakeJudge:
    def __init__(
        self,
        *,
        passed: bool,
        mean: float = 0.85,
        disagreement: float = 0.1,
        is_uncertain: bool = False,
        cost: float = 0.05,
    ) -> None:
        self._passed = passed
        self._mean = mean
        self._disagreement = disagreement
        self._uncertain = is_uncertain
        self._cost = cost
        self.calls = 0

    async def run(
        self,
        subtask: SubTask,
        workspace: IsolatedWorkspace,
        sub_result: SubAgentResult,
    ) -> tuple[TierResult, LLMJudgeDetail]:
        del workspace, sub_result
        self.calls += 1
        tier = TierResult(
            tier="llm_judge",
            passed=self._passed,
            details="stubbed judge",
            latency_ms=2,
            cost_usd=self._cost,
        )
        detail = LLMJudgeDetail(
            samples=[self._mean] * 3,
            mean_score=self._mean,
            disagreement=self._disagreement,
            is_uncertain=self._uncertain,
            judge_model="fake-judge",
        )
        return tier, detail


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("X = 1\n", encoding="utf-8")
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
            "judge": ModelConfig(
                name="deepseek-v3",
                display_name="DeepSeek-V3",
                price_input_per_mtok=0.28,
                price_output_per_mtok=1.12,
            ),
        },
    )
    return LLMClient(cfg)


def _spec() -> TaskSpec:
    return TaskSpec(
        task_id=generate_task_id(),
        description="test",
        repo_path=Path("/tmp"),
    )


def _subtask() -> SubTask:
    return SubTask(subtask_id="t-001", description="x", writes=["src/app.py"])


def _sub_result() -> SubAgentResult:
    return SubAgentResult(
        subtask_id="t-001",
        status="success",
        diff="",
        modified_files=["src/app.py"],
        rationale="",
        confidence=1.0,
        retry_count=0,
        tokens_input=0,
        tokens_output=0,
        latency_ms=0,
        model_used="test",
        created_at=datetime(2026, 4, 23, tzinfo=UTC),
    )


def _build(
    sample_repo: Path,
    *,
    det: _FakeDet,
    tst: _FakeTest,
    jdg: _FakeJudge,
    enabled: set[str] | None = None,
) -> Callable[[], Verifier]:
    """Return a factory that builds a Verifier bound to a fresh workspace."""

    def _factory() -> Verifier:
        client = _make_client()
        return Verifier(
            client,
            _spec(),
            workspace_manager=None,
            enabled_tiers=enabled,  # type: ignore[arg-type]
            tier1=det,  # type: ignore[arg-type]
            tier2=tst,  # type: ignore[arg-type]
            tier3=jdg,  # type: ignore[arg-type]
        )

    return _factory


# ---------------------------------------------------------------------------
# Short-circuit + happy path
# ---------------------------------------------------------------------------


async def test_full_pass_runs_all_three_tiers(sample_repo: Path) -> None:
    det = _FakeDet(passed=True)
    tst = _FakeTest(passed=True)
    jdg = _FakeJudge(passed=True, cost=0.03)
    make = _build(sample_repo, det=det, tst=tst, jdg=jdg)
    with WorkspaceManager(sample_repo, task_id="t-ok") as mgr:
        iso = await mgr.create_isolated(_subtask())
        vr = await make().verify(_subtask(), iso, _sub_result())

    assert vr.overall_passed is True
    assert [t.tier for t in vr.tiers] == ["deterministic", "test_based", "llm_judge"]
    assert vr.judge_detail is not None
    assert vr.total_cost_usd == pytest.approx(0.03)
    assert det.calls == tst.calls == jdg.calls == 1


async def test_tier1_failure_short_circuits(sample_repo: Path) -> None:
    det = _FakeDet(passed=False, details="ruff error")
    tst = _FakeTest(passed=True)
    jdg = _FakeJudge(passed=True)
    make = _build(sample_repo, det=det, tst=tst, jdg=jdg)
    with WorkspaceManager(sample_repo, task_id="t-t1-fail") as mgr:
        iso = await mgr.create_isolated(_subtask())
        vr = await make().verify(_subtask(), iso, _sub_result())

    assert vr.overall_passed is False
    assert [t.tier for t in vr.tiers] == ["deterministic"]
    assert tst.calls == 0
    assert jdg.calls == 0


async def test_tier2_failure_short_circuits(sample_repo: Path) -> None:
    det = _FakeDet(passed=True)
    tst = _FakeTest(passed=False)
    jdg = _FakeJudge(passed=True)
    make = _build(sample_repo, det=det, tst=tst, jdg=jdg)
    with WorkspaceManager(sample_repo, task_id="t-t2-fail") as mgr:
        iso = await mgr.create_isolated(_subtask())
        vr = await make().verify(_subtask(), iso, _sub_result())

    assert vr.overall_passed is False
    assert [t.tier for t in vr.tiers] == ["deterministic", "test_based"]
    assert jdg.calls == 0


async def test_tier3_failure_is_reported_with_detail(sample_repo: Path) -> None:
    det = _FakeDet(passed=True)
    tst = _FakeTest(passed=True)
    jdg = _FakeJudge(passed=False, is_uncertain=True, disagreement=0.5)
    make = _build(sample_repo, det=det, tst=tst, jdg=jdg)
    with WorkspaceManager(sample_repo, task_id="t-t3-fail") as mgr:
        iso = await mgr.create_isolated(_subtask())
        vr = await make().verify(_subtask(), iso, _sub_result())

    assert vr.overall_passed is False
    assert vr.judge_detail is not None
    assert vr.judge_detail.is_uncertain is True


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------


async def test_ablation_tier1_only(sample_repo: Path) -> None:
    det = _FakeDet(passed=True)
    tst = _FakeTest(passed=True)
    jdg = _FakeJudge(passed=True)
    make = _build(sample_repo, det=det, tst=tst, jdg=jdg, enabled={"deterministic"})
    with WorkspaceManager(sample_repo, task_id="t-abl-t1") as mgr:
        iso = await mgr.create_isolated(_subtask())
        vr = await make().verify(_subtask(), iso, _sub_result())

    assert vr.overall_passed is True
    assert [t.tier for t in vr.tiers] == ["deterministic"]
    assert tst.calls == 0
    assert jdg.calls == 0
    assert vr.judge_detail is None


async def test_ablation_t1_t2(sample_repo: Path) -> None:
    det = _FakeDet(passed=True)
    tst = _FakeTest(passed=True)
    jdg = _FakeJudge(passed=True)
    make = _build(
        sample_repo,
        det=det,
        tst=tst,
        jdg=jdg,
        enabled={"deterministic", "test_based"},
    )
    with WorkspaceManager(sample_repo, task_id="t-abl-t12") as mgr:
        iso = await mgr.create_isolated(_subtask())
        vr = await make().verify(_subtask(), iso, _sub_result())

    assert vr.overall_passed is True
    assert [t.tier for t in vr.tiers] == ["deterministic", "test_based"]
    assert jdg.calls == 0


async def test_ablation_judge_only(sample_repo: Path) -> None:
    det = _FakeDet(passed=False)  # would fail if run, but it's disabled
    tst = _FakeTest(passed=False)
    jdg = _FakeJudge(passed=True)
    make = _build(sample_repo, det=det, tst=tst, jdg=jdg, enabled={"llm_judge"})
    with WorkspaceManager(sample_repo, task_id="t-abl-j") as mgr:
        iso = await mgr.create_isolated(_subtask())
        vr = await make().verify(_subtask(), iso, _sub_result())

    assert vr.overall_passed is True
    assert [t.tier for t in vr.tiers] == ["llm_judge"]
    assert det.calls == 0
    assert tst.calls == 0


async def test_ablation_empty_tiers_always_passes(sample_repo: Path) -> None:
    det = _FakeDet(passed=False)
    tst = _FakeTest(passed=False)
    jdg = _FakeJudge(passed=False)
    make = _build(sample_repo, det=det, tst=tst, jdg=jdg, enabled=set())
    with WorkspaceManager(sample_repo, task_id="t-abl-none") as mgr:
        iso = await mgr.create_isolated(_subtask())
        vr = await make().verify(_subtask(), iso, _sub_result())

    assert vr.overall_passed is True
    assert vr.tiers == []
    assert det.calls == tst.calls == jdg.calls == 0


# ---------------------------------------------------------------------------
# Cost accumulation across tiers
# ---------------------------------------------------------------------------


async def test_total_cost_sums_across_tiers(sample_repo: Path) -> None:
    det = _FakeDet(passed=True)
    tst = _FakeTest(passed=True)
    jdg = _FakeJudge(passed=True, cost=0.04)
    make = _build(sample_repo, det=det, tst=tst, jdg=jdg)
    with WorkspaceManager(sample_repo, task_id="t-cost") as mgr:
        iso = await mgr.create_isolated(_subtask())
        vr = await make().verify(_subtask(), iso, _sub_result())

    # T1 and T2 are zero-cost; only T3 contributes.
    assert vr.total_cost_usd == pytest.approx(0.04)
