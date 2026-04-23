"""Pydantic data models for Maestro.

This module is the single source of truth for data that crosses module
boundaries. All models are frozen (immutable) to avoid accidental mutation
in concurrent sub-agent execution.

Spec reference: ``specs/01-data-models.md``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

JudgeTier = Literal["deterministic", "test_based", "llm_judge"]
SubAgentStatus = Literal["success", "failed", "rejected"]
TaskStatus = Literal["success", "partial", "failed"]
Difficulty = Literal["easy", "medium", "hard"]


def generate_task_id() -> str:
    """Generate a UUID4 task id, prefixed for readability in logs."""
    return f"task-{uuid.uuid4().hex[:12]}"


def generate_subtask_id(task_id: str, index: int) -> str:
    """Generate a subtask id in the canonical ``{task_id}-{index:03d}`` format."""
    return f"{task_id}-{index:03d}"


def _utcnow() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Task-level input
# ---------------------------------------------------------------------------


class TaskSpec(BaseModel):
    """Top-level task input from user.

    ``auto_gen_tests`` is off by default because on the benchmark auto-generated
    tests risk confirming the sub-agent's own bias (see spec 06 §4.4). The CLI
    can flip it on for production use.
    """

    model_config = ConfigDict(frozen=True)

    task_id: str
    description: str
    repo_path: Path
    target_files_hint: list[str] | None = None
    max_parallel: int = Field(default=4, ge=1)
    max_retries_per_subtask: int = Field(default=2, ge=0)
    judge_samples: int = Field(default=3, ge=1)
    judge_disagreement_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    auto_gen_tests: bool = False
    created_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# DAG + subtasks
# ---------------------------------------------------------------------------


class SubTask(BaseModel):
    """One unit of work produced by the Planner (post-processed form).

    ``writes`` declares ground truth — the sub-agent is only allowed to modify
    these files. ``reads`` is advisory (used to assemble context for the
    sub-agent) and is not enforced at write-time.
    """

    model_config = ConfigDict(frozen=True)

    subtask_id: str
    description: str
    reads: list[str] = Field(default_factory=list)
    writes: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    priority: int = 0
    estimated_difficulty: Difficulty = "medium"


class TaskDAG(BaseModel):
    """Full DAG of subtasks with a shared global context."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    subtasks: list[SubTask]
    global_context: str

    def validate_dag(self) -> None:
        """Validate structural integrity: unique ids, valid deps, no cycle.

        Raises ``DAGError`` (defined in ``scheduler.dag``) when called from the
        scheduler; here we raise ``ValueError`` so ``models.py`` stays free of
        cross-module imports.
        """
        ids = [s.subtask_id for s in self.subtasks]
        unique_ids = set(ids)
        if len(ids) != len(unique_ids):
            raise ValueError("SubTask ids must be unique within a TaskDAG")

        for subtask in self.subtasks:
            for dep in subtask.depends_on:
                if dep not in unique_ids:
                    raise ValueError(f"SubTask {subtask.subtask_id} depends on missing id {dep!r}")
                if dep == subtask.subtask_id:
                    raise ValueError(f"SubTask {subtask.subtask_id} cannot depend on itself")

        # Kahn's algorithm for cycle detection.
        in_degree = {s.subtask_id: 0 for s in self.subtasks}
        adj: dict[str, list[str]] = {s.subtask_id: [] for s in self.subtasks}
        for subtask in self.subtasks:
            for dep in subtask.depends_on:
                adj[dep].append(subtask.subtask_id)
                in_degree[subtask.subtask_id] += 1

        ready = [sid for sid, d in in_degree.items() if d == 0]
        visited = 0
        while ready:
            node = ready.pop()
            visited += 1
            for nxt in adj[node]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    ready.append(nxt)
        if visited != len(self.subtasks):
            raise ValueError("TaskDAG contains a cycle")


# ---------------------------------------------------------------------------
# Sub-agent results
# ---------------------------------------------------------------------------


class SubAgentResult(BaseModel):
    """Framework-wrapped result of one sub-agent execution.

    ``retry_count`` here is the *subtask-level* retry count (how many times
    this subtask was re-attempted after verification failure). This is distinct
    from ``LLMCallMetadata.http_retry_count`` which counts in-client HTTP-level
    retries for a single LLM call.
    """

    model_config = ConfigDict(frozen=True)

    subtask_id: str
    status: SubAgentStatus
    diff: str
    modified_files: list[str]
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    retry_count: int = Field(ge=0)

    tokens_input: int = Field(ge=0)
    tokens_output: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    model_used: str

    created_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


class TierResult(BaseModel):
    """Result of a single verifier tier."""

    model_config = ConfigDict(frozen=True)

    tier: JudgeTier
    passed: bool
    details: str
    latency_ms: int = Field(ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)


class LLMJudgeDetail(BaseModel):
    """Per-sample detail emitted by the multi-sample LLM judge (spec 06 §5)."""

    model_config = ConfigDict(frozen=True)

    samples: list[float]
    mean_score: float = Field(ge=0.0, le=1.0)
    disagreement: float = Field(ge=0.0)
    is_uncertain: bool
    judge_model: str


class VerificationResult(BaseModel):
    """Aggregate verification result for one patch.

    TODO(haitian): when module 11-13 (verifier) is built, WorkspaceManager must
    expose ``get_pre_patch_content(subtask, path)`` so the LLM judge prompt can
    render the pre-diff file content alongside the diff (spec 06 §5.6).
    """

    model_config = ConfigDict(frozen=True)

    subtask_id: str
    overall_passed: bool
    tiers: list[TierResult]
    judge_detail: LLMJudgeDetail | None = None
    total_latency_ms: int = Field(ge=0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)


# ---------------------------------------------------------------------------
# Batch + final result
# ---------------------------------------------------------------------------


class BatchResult(BaseModel):
    """Aggregate result of executing one parallel batch."""

    model_config = ConfigDict(frozen=True)

    batch_index: int = Field(ge=0)
    subtask_results: list[SubAgentResult]
    verification_results: list[VerificationResult]
    merged_patches: list[str] = Field(default_factory=list)
    retried_patches: list[str] = Field(default_factory=list)
    failed_patches: list[str] = Field(default_factory=list)
    conflicts_detected: list[tuple[str, str]] = Field(default_factory=list)


class TaskResult(BaseModel):
    """Final result of a complete task execution."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    status: TaskStatus
    batches: list[BatchResult]
    final_diff: str
    final_workspace: Path

    total_wall_clock_ms: int = Field(ge=0)
    total_tokens_input: int = Field(ge=0)
    total_tokens_output: int = Field(ge=0)
    total_cost_usd: float = Field(ge=0.0)

    started_at: datetime
    finished_at: datetime

    @model_validator(mode="after")
    def _validate_timestamps(self) -> TaskResult:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must be >= started_at")
        return self


# ---------------------------------------------------------------------------
# LLM I/O schemas (structured output contracts)
# ---------------------------------------------------------------------------


class PlannerLLMSubTask(BaseModel):
    """Sub-task as emitted directly by the Planner LLM.

    The LLM does not know the system-generated ``task_id`` (a UUID) and therefore
    cannot assemble the canonical ``subtask_id``. It only produces an ``index``
    that the Planner code maps to ``subtask_id = {task_id}-{index:03d}`` during
    post-processing. See ``PlannerOutput`` for the post-processed form.
    """

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    description: str
    reads: list[str] = Field(default_factory=list)
    writes: list[str] = Field(default_factory=list)
    depends_on_indices: list[int] = Field(default_factory=list)
    priority: int = 0
    estimated_difficulty: Difficulty = "medium"


class PlannerLLMOutput(BaseModel):
    """Raw structured output produced by the Planner LLM.

    This is what the LLM returns (before code post-processing). It uses
    integer ``index`` identifiers. The Planner module converts it to
    ``PlannerOutput`` with canonical ``subtask_id`` strings.
    """

    model_config = ConfigDict(frozen=True)

    subtasks: list[PlannerLLMSubTask]
    global_context: str
    planning_rationale: str


class PlannerOutput(BaseModel):
    """Post-processed planner output with canonical subtask ids.

    ``subtasks`` here contain ``SubTask`` instances whose ``subtask_id`` is the
    canonical ``{task_id}-{idx:03d}`` form, and whose ``depends_on`` references
    those ids (translated from the LLM's integer indices).
    """

    model_config = ConfigDict(frozen=True)

    subtasks: list[SubTask]
    global_context: str
    planning_rationale: str


class SubAgentOutput(BaseModel):
    """Structured output from a Sub-agent's Write phase (spec 05 §3.1).

    Unlike ``SubAgentResult``, this is the raw LLM output without framework
    metadata (no token counts, no retry_count, no latency). The Sub-agent code
    wraps this into ``SubAgentResult``.
    """

    model_config = ConfigDict(frozen=True)

    status: Literal["success", "failed"]
    diff: str
    modified_files: list[str]
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class JudgeOutput(BaseModel):
    """Per-sample structured output from the LLM Judge (spec 06 §5)."""

    model_config = ConfigDict(frozen=True)

    score: float = Field(ge=0.0, le=1.0)
    passes_requirements: bool
    reasoning: str
    detected_issues: list[str] = Field(default_factory=list)


__all__ = [
    "BatchResult",
    "Difficulty",
    "JudgeOutput",
    "JudgeTier",
    "LLMJudgeDetail",
    "PlannerLLMOutput",
    "PlannerLLMSubTask",
    "PlannerOutput",
    "SubAgentOutput",
    "SubAgentResult",
    "SubAgentStatus",
    "SubTask",
    "TaskDAG",
    "TaskResult",
    "TaskSpec",
    "TaskStatus",
    "TierResult",
    "VerificationResult",
    "generate_subtask_id",
    "generate_task_id",
]
