"""Unit tests for ``maestro.models`` (spec 01 §4.3)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from maestro.models import (
    BatchResult,
    JudgeOutput,
    LLMJudgeDetail,
    PlannerLLMOutput,
    PlannerLLMSubTask,
    PlannerOutput,
    SubAgentOutput,
    SubAgentResult,
    SubTask,
    TaskDAG,
    TaskResult,
    TaskSpec,
    TierResult,
    VerificationResult,
    generate_subtask_id,
    generate_task_id,
)

# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------


def test_generate_task_id_is_unique_and_prefixed() -> None:
    a = generate_task_id()
    b = generate_task_id()
    assert a != b
    assert a.startswith("task-")


def test_generate_subtask_id_pads_to_three_digits() -> None:
    assert generate_subtask_id("task-abc", 0) == "task-abc-000"
    assert generate_subtask_id("task-abc", 7) == "task-abc-007"
    assert generate_subtask_id("task-abc", 123) == "task-abc-123"


# ---------------------------------------------------------------------------
# TaskSpec
# ---------------------------------------------------------------------------


def test_taskspec_defaults_and_creation(tmp_path: Path) -> None:
    spec = TaskSpec(
        task_id=generate_task_id(),
        description="Add login",
        repo_path=tmp_path,
    )
    assert spec.max_parallel == 4
    assert spec.judge_samples == 3
    assert spec.judge_disagreement_threshold == 0.3
    assert spec.auto_gen_tests is False  # M4: default off on benchmark
    assert spec.created_at.tzinfo is not None


def test_taskspec_is_frozen(tmp_path: Path) -> None:
    spec = TaskSpec(task_id="t1", description="x", repo_path=tmp_path)
    with pytest.raises(ValidationError):
        spec.description = "y"  # type: ignore[misc]


def test_taskspec_rejects_invalid_threshold(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        TaskSpec(
            task_id="t1",
            description="x",
            repo_path=tmp_path,
            judge_disagreement_threshold=1.5,
        )


def test_taskspec_rejects_zero_parallel(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        TaskSpec(task_id="t1", description="x", repo_path=tmp_path, max_parallel=0)


# ---------------------------------------------------------------------------
# SubTask
# ---------------------------------------------------------------------------


def test_subtask_minimal_construction() -> None:
    st = SubTask(subtask_id="t-001", description="do thing")
    assert st.reads == []
    assert st.writes == []
    assert st.depends_on == []
    assert st.priority == 0


def test_subtask_missing_required_fields() -> None:
    with pytest.raises(ValidationError):
        SubTask(description="no id")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TaskDAG.validate_dag
# ---------------------------------------------------------------------------


def _st(sid: str, deps: list[str] | None = None) -> SubTask:
    return SubTask(subtask_id=sid, description=sid, depends_on=deps or [])


def test_validate_empty_dag() -> None:
    TaskDAG(task_id="t", subtasks=[], global_context="").validate_dag()


def test_validate_single_node_dag() -> None:
    TaskDAG(task_id="t", subtasks=[_st("a")], global_context="").validate_dag()


def test_validate_linear_chain_dag() -> None:
    dag = TaskDAG(
        task_id="t",
        subtasks=[_st("a"), _st("b", ["a"]), _st("c", ["b"])],
        global_context="",
    )
    dag.validate_dag()


def test_validate_diamond_dag() -> None:
    dag = TaskDAG(
        task_id="t",
        subtasks=[
            _st("a"),
            _st("b", ["a"]),
            _st("c", ["a"]),
            _st("d", ["b", "c"]),
        ],
        global_context="",
    )
    dag.validate_dag()


def test_validate_fully_parallel_dag() -> None:
    dag = TaskDAG(
        task_id="t",
        subtasks=[_st("a"), _st("b"), _st("c")],
        global_context="",
    )
    dag.validate_dag()


def test_validate_detects_cycle() -> None:
    dag = TaskDAG(
        task_id="t",
        subtasks=[_st("a", ["b"]), _st("b", ["a"])],
        global_context="",
    )
    with pytest.raises(ValueError, match="cycle"):
        dag.validate_dag()


def test_validate_detects_missing_dep() -> None:
    dag = TaskDAG(
        task_id="t",
        subtasks=[_st("a", ["ghost"])],
        global_context="",
    )
    with pytest.raises(ValueError, match="missing id"):
        dag.validate_dag()


def test_validate_detects_self_dependency() -> None:
    dag = TaskDAG(task_id="t", subtasks=[_st("a", ["a"])], global_context="")
    with pytest.raises(ValueError, match="cannot depend on itself"):
        dag.validate_dag()


def test_validate_detects_duplicate_ids() -> None:
    dag = TaskDAG(
        task_id="t",
        subtasks=[_st("a"), _st("a")],
        global_context="",
    )
    with pytest.raises(ValueError, match="unique"):
        dag.validate_dag()


# ---------------------------------------------------------------------------
# SubAgentResult + SubAgentOutput
# ---------------------------------------------------------------------------


def _sub_result_kwargs() -> dict[str, object]:
    return dict(
        subtask_id="t-001",
        status="success",
        diff="--- a/x\n+++ b/x\n",
        modified_files=["x"],
        rationale="because",
        confidence=0.8,
        retry_count=0,
        tokens_input=100,
        tokens_output=50,
        latency_ms=1234,
        model_used="qwen3-coder-plus",
    )


def test_subagent_result_valid() -> None:
    r = SubAgentResult(**_sub_result_kwargs())  # type: ignore[arg-type]
    assert r.status == "success"
    assert r.model_used == "qwen3-coder-plus"


def test_subagent_result_rejects_out_of_range_confidence() -> None:
    kwargs = _sub_result_kwargs()
    kwargs["confidence"] = 1.5
    with pytest.raises(ValidationError):
        SubAgentResult(**kwargs)  # type: ignore[arg-type]


def test_subagent_result_rejects_negative_tokens() -> None:
    kwargs = _sub_result_kwargs()
    kwargs["tokens_input"] = -1
    with pytest.raises(ValidationError):
        SubAgentResult(**kwargs)  # type: ignore[arg-type]


def test_subagent_output_rejects_invalid_status() -> None:
    with pytest.raises(ValidationError):
        SubAgentOutput(
            status="maybe",  # type: ignore[arg-type]
            diff="",
            modified_files=[],
            rationale="",
            confidence=0.5,
        )


# ---------------------------------------------------------------------------
# Verifier models
# ---------------------------------------------------------------------------


def test_tier_result_requires_valid_tier() -> None:
    with pytest.raises(ValidationError):
        TierResult(
            tier="semantic",  # type: ignore[arg-type]
            passed=True,
            details="",
            latency_ms=0,
        )


def test_verification_result_optional_judge_detail() -> None:
    vr = VerificationResult(
        subtask_id="t-001",
        overall_passed=True,
        tiers=[TierResult(tier="deterministic", passed=True, details="ok", latency_ms=10)],
        total_latency_ms=10,
    )
    assert vr.judge_detail is None
    assert vr.total_cost_usd == 0.0


def test_llm_judge_detail_valid() -> None:
    d = LLMJudgeDetail(
        samples=[0.8, 0.7, 0.9],
        mean_score=0.8,
        disagreement=0.08,
        is_uncertain=False,
        judge_model="deepseek-v3",
    )
    assert d.mean_score == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# BatchResult / TaskResult
# ---------------------------------------------------------------------------


def test_batch_result_defaults_empty_lists() -> None:
    br = BatchResult(batch_index=0, subtask_results=[], verification_results=[])
    assert br.merged_patches == []
    assert br.conflicts_detected == []


def test_task_result_rejects_reversed_timestamps(tmp_path: Path) -> None:
    start = datetime.now(UTC)
    end = start - timedelta(seconds=5)
    with pytest.raises(ValidationError):
        TaskResult(
            task_id="t",
            status="success",
            batches=[],
            final_diff="",
            final_workspace=tmp_path,
            total_wall_clock_ms=0,
            total_tokens_input=0,
            total_tokens_output=0,
            total_cost_usd=0.0,
            started_at=start,
            finished_at=end,
        )


# ---------------------------------------------------------------------------
# Planner LLM / post-processed output (M2 decision)
# ---------------------------------------------------------------------------


def test_planner_llm_output_uses_indices_not_ids() -> None:
    raw = PlannerLLMOutput(
        subtasks=[
            PlannerLLMSubTask(index=0, description="signup", writes=["auth/signup.py"]),
            PlannerLLMSubTask(
                index=1,
                description="register",
                writes=["app.py"],
                depends_on_indices=[0],
            ),
        ],
        global_context="ctx",
        planning_rationale="rationale",
    )
    assert raw.subtasks[1].depends_on_indices == [0]
    # M2: LLM-facing schema must not reference subtask_id strings.
    assert not hasattr(raw.subtasks[0], "subtask_id")


def test_planner_output_carries_canonical_subtask_ids() -> None:
    task_id = "task-abc123"
    post = PlannerOutput(
        subtasks=[
            SubTask(
                subtask_id=generate_subtask_id(task_id, 0),
                description="signup",
                writes=["auth/signup.py"],
            ),
            SubTask(
                subtask_id=generate_subtask_id(task_id, 1),
                description="register",
                writes=["app.py"],
                depends_on=[generate_subtask_id(task_id, 0)],
            ),
        ],
        global_context="ctx",
        planning_rationale="rationale",
    )
    assert post.subtasks[0].subtask_id == "task-abc123-000"
    assert post.subtasks[1].depends_on == ["task-abc123-000"]


# ---------------------------------------------------------------------------
# JSON round-trip + LLM schema surfaces
# ---------------------------------------------------------------------------


def test_task_result_json_roundtrip(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    tr = TaskResult(
        task_id="t",
        status="success",
        batches=[],
        final_diff="",
        final_workspace=tmp_path,
        total_wall_clock_ms=100,
        total_tokens_input=10,
        total_tokens_output=20,
        total_cost_usd=0.01,
        started_at=now,
        finished_at=now,
    )
    data = tr.model_dump_json()
    restored = TaskResult.model_validate_json(data)
    assert restored.task_id == tr.task_id
    assert restored.total_cost_usd == pytest.approx(0.01)


def test_llm_schema_exports_are_valid_json_schema() -> None:
    # Smoke test: each Pydantic class used for structured output must produce
    # a valid JSON schema (used by LLMClient.call_structured).
    for cls in (PlannerLLMOutput, SubAgentOutput, JudgeOutput):
        schema = cls.model_json_schema()
        assert schema["type"] == "object"
        assert "properties" in schema
