"""Pydantic models for the benchmark harness (spec 09)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

BenchStatus = Literal["resolved", "unresolved", "apply_failed", "error"]


class BenchmarkTask(BaseModel):
    """One benchmark task — the on-disk ``task.json`` format (spec 09 §2.3).

    Matches the real-PR layout described in the spec. Fields are kept
    permissive (``difficulty`` is a free string, ``files_hint`` is optional)
    because the spec expects some fields to be populated by hand during
    the task curation pass in Week 5 Day 1.
    """

    model_config = ConfigDict(frozen=True)

    task_id: str
    repo: str
    pr_url: str = ""
    issue_url: str = ""
    description: str
    natural_language_prompt: str
    failing_tests: list[str] = Field(default_factory=list)
    expected_modified_files: list[str] = Field(default_factory=list)
    files_hint: list[str] = Field(default_factory=list)
    difficulty: str = "medium"
    source_commit: str = ""


class EvalResult(BaseModel):
    """Outcome of applying Maestro's diff to ``before/`` and running tests."""

    model_config = ConfigDict(frozen=True)

    resolved: bool
    patch_similarity: float = Field(ge=0.0, le=1.0)
    files_modified_match: bool = False
    extra_files_modified: int = Field(default=0, ge=0)
    error: str | None = None


class TaskBenchmarkResult(BaseModel):
    """Per-task benchmark outcome (spec 09 §3.2)."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    config_name: str
    status: BenchStatus
    resolved: bool
    wall_clock_ms: int = Field(ge=0)
    total_cost: float = Field(ge=0.0)
    total_tokens_input: int = Field(ge=0)
    total_tokens_output: int = Field(ge=0)
    patch_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    files_modified_match: bool = False
    extra_files_modified: int = Field(default=0, ge=0)
    error: str | None = None


class BenchmarkReport(BaseModel):
    """Aggregate report for one full benchmark run (spec 09 §5.1)."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    config_name: str
    task_count: int = Field(ge=0)
    resolve_rate: float = Field(ge=0.0, le=1.0)
    avg_wall_clock_ms: float = Field(ge=0.0)
    avg_cost: float = Field(ge=0.0)
    avg_tokens_input: float = Field(ge=0.0)
    avg_tokens_output: float = Field(ge=0.0)
    per_task_results: list[TaskBenchmarkResult]
    started_at: datetime
    finished_at: datetime


__all__ = [
    "BenchStatus",
    "BenchmarkReport",
    "BenchmarkTask",
    "EvalResult",
    "TaskBenchmarkResult",
]
