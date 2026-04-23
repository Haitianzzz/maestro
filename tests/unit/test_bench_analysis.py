"""Unit tests for ``benchmark.analysis.analyze``."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from maestro.benchmark.analysis import (
    ExperimentAnalyzer,
    compute_pareto_frontier,
    compute_speedups,
    render_comparison_markdown,
)
from maestro.benchmark.models import BenchmarkReport, TaskBenchmarkResult


def _result(
    *,
    task_id: str,
    config: str,
    resolved: bool,
    cost: float = 0.1,
    wall_ms: int = 1000,
) -> TaskBenchmarkResult:
    return TaskBenchmarkResult(
        task_id=task_id,
        config_name=config,
        status="resolved" if resolved else "unresolved",
        resolved=resolved,
        wall_clock_ms=wall_ms,
        total_cost=cost,
        total_tokens_input=100,
        total_tokens_output=50,
        patch_similarity=0.8 if resolved else 0.2,
    )


def _report(
    *,
    name: str,
    resolved: int,
    total: int,
    avg_cost: float,
    avg_wall_ms: float,
    per_task: list[TaskBenchmarkResult],
) -> BenchmarkReport:
    return BenchmarkReport(
        run_id=f"run-{name}",
        config_name=name,
        task_count=total,
        resolve_rate=resolved / total if total else 0.0,
        avg_wall_clock_ms=avg_wall_ms,
        avg_cost=avg_cost,
        avg_tokens_input=100.0,
        avg_tokens_output=50.0,
        per_task_results=per_task,
        started_at=datetime(2026, 4, 23, tzinfo=UTC),
        finished_at=datetime(2026, 4, 23, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# compute_speedups
# ---------------------------------------------------------------------------


def test_speedups_relative_to_baseline() -> None:
    reports = {
        "baseline": _report(
            name="baseline",
            resolved=1,
            total=2,
            avg_cost=0.0,
            avg_wall_ms=2000.0,
            per_task=[
                _result(task_id="t-1", config="baseline", resolved=True),
                _result(task_id="t-2", config="baseline", resolved=False),
            ],
        ),
        "full": _report(
            name="full",
            resolved=2,
            total=2,
            avg_cost=0.5,
            avg_wall_ms=1000.0,
            per_task=[
                _result(task_id="t-1", config="full", resolved=True),
                _result(task_id="t-2", config="full", resolved=True),
            ],
        ),
    }
    speedups = compute_speedups(reports)
    assert speedups["baseline"] == pytest.approx(1.0)
    assert speedups["full"] == pytest.approx(2.0)


def test_speedups_missing_baseline_returns_empty() -> None:
    reports = {
        "full": _report(
            name="full",
            resolved=1,
            total=1,
            avg_cost=0.1,
            avg_wall_ms=1000.0,
            per_task=[_result(task_id="t-1", config="full", resolved=True)],
        )
    }
    assert compute_speedups(reports) == {}


# ---------------------------------------------------------------------------
# compute_pareto_frontier
# ---------------------------------------------------------------------------


def test_pareto_flags_dominated_points() -> None:
    # ``weak`` is strictly worse than ``full`` on both axes → dominated.
    reports = {
        "weak": _report(
            name="weak",
            resolved=0,
            total=1,
            avg_cost=1.0,
            avg_wall_ms=1000.0,
            per_task=[_result(task_id="t-1", config="weak", resolved=False, cost=1.0)],
        ),
        "full": _report(
            name="full",
            resolved=1,
            total=1,
            avg_cost=0.5,
            avg_wall_ms=800.0,
            per_task=[_result(task_id="t-1", config="full", resolved=True, cost=0.5)],
        ),
        "cheap_unresolved": _report(
            name="cheap_unresolved",
            resolved=0,
            total=1,
            avg_cost=0.0,
            avg_wall_ms=2000.0,
            per_task=[_result(task_id="t-1", config="cheap_unresolved", resolved=False, cost=0.0)],
        ),
    }
    pareto = {p.config_name: p for p in compute_pareto_frontier(reports)}
    assert pareto["weak"].dominated is True
    # Full dominates weak; cheap_unresolved dominates nothing on cost axis
    # alone because its resolve is worse. Both live on the frontier.
    assert pareto["full"].dominated is False
    assert pareto["cheap_unresolved"].dominated is False


# ---------------------------------------------------------------------------
# ExperimentAnalyzer + rendering
# ---------------------------------------------------------------------------


def test_task_win_matrix_spans_all_tasks() -> None:
    reports = {
        "baseline": _report(
            name="baseline",
            resolved=1,
            total=2,
            avg_cost=0.0,
            avg_wall_ms=1000.0,
            per_task=[
                _result(task_id="t-1", config="baseline", resolved=True),
                _result(task_id="t-2", config="baseline", resolved=False),
            ],
        ),
        "full": _report(
            name="full",
            resolved=2,
            total=2,
            avg_cost=0.5,
            avg_wall_ms=500.0,
            per_task=[
                _result(task_id="t-1", config="full", resolved=True),
                _result(task_id="t-2", config="full", resolved=True),
            ],
        ),
    }
    analyzer = ExperimentAnalyzer(reports)
    matrix = analyzer.task_win_matrix()
    assert matrix["t-1"] == {"baseline": True, "full": True}
    assert matrix["t-2"] == {"baseline": False, "full": True}


def test_render_comparison_markdown_contains_expected_sections() -> None:
    reports = {
        "baseline": _report(
            name="baseline",
            resolved=1,
            total=2,
            avg_cost=0.0,
            avg_wall_ms=2000.0,
            per_task=[
                _result(task_id="t-1", config="baseline", resolved=True),
                _result(task_id="t-2", config="baseline", resolved=False),
            ],
        ),
        "full": _report(
            name="full",
            resolved=2,
            total=2,
            avg_cost=0.5,
            avg_wall_ms=1000.0,
            per_task=[
                _result(task_id="t-1", config="full", resolved=True),
                _result(task_id="t-2", config="full", resolved=True),
            ],
        ),
    }
    md = render_comparison_markdown(reports)
    # Header
    assert "Maestro benchmark" in md
    # Speedup column rendered
    assert "2.00x" in md
    # Both configs appear as rows
    assert "| baseline |" in md
    assert "| full |" in md
    # Task win matrix block
    assert "Task x config win matrix" in md or "win matrix" in md.lower()
