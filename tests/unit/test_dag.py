"""Unit tests for ``maestro.scheduler.dag`` (spec 04 §7)."""

from __future__ import annotations

import pytest

from maestro.models import SubTask, TaskDAG
from maestro.scheduler.dag import (
    DAGError,
    defer_lower_priority_on_conflicts,
    detect_write_conflicts,
    topological_batches,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _st(
    sid: str,
    *,
    deps: list[str] | None = None,
    writes: list[str] | None = None,
    priority: int = 0,
) -> SubTask:
    return SubTask(
        subtask_id=sid,
        description=sid,
        depends_on=deps or [],
        writes=writes or [],
        priority=priority,
    )


def _dag(subtasks: list[SubTask]) -> TaskDAG:
    return TaskDAG(task_id="t", subtasks=subtasks, global_context="")


# ---------------------------------------------------------------------------
# topological_batches
# ---------------------------------------------------------------------------


def test_empty_dag_returns_no_batches() -> None:
    assert topological_batches(_dag([])) == []


def test_single_node_dag() -> None:
    batches = topological_batches(_dag([_st("a")]))
    assert len(batches) == 1
    assert [s.subtask_id for s in batches[0]] == ["a"]


def test_linear_chain() -> None:
    batches = topological_batches(_dag([_st("a"), _st("b", deps=["a"]), _st("c", deps=["b"])]))
    assert [[s.subtask_id for s in b] for b in batches] == [["a"], ["b"], ["c"]]


def test_fully_parallel() -> None:
    batches = topological_batches(_dag([_st("a"), _st("b"), _st("c")]))
    assert len(batches) == 1
    assert sorted(s.subtask_id for s in batches[0]) == ["a", "b", "c"]


def test_diamond_dag() -> None:
    dag = _dag(
        [
            _st("a"),
            _st("b", deps=["a"]),
            _st("c", deps=["a"]),
            _st("d", deps=["b", "c"]),
        ]
    )
    batches = [[s.subtask_id for s in b] for b in topological_batches(dag)]
    assert batches == [["a"], ["b", "c"], ["d"]]


def test_batch_order_respects_priority() -> None:
    dag = _dag(
        [
            _st("low", priority=0),
            _st("high", priority=10),
            _st("mid", priority=5),
        ]
    )
    batches = topological_batches(dag)
    assert [s.subtask_id for s in batches[0]] == ["high", "mid", "low"]


def test_batch_order_ties_by_subtask_id() -> None:
    dag = _dag(
        [
            _st("zeta", priority=1),
            _st("alpha", priority=1),
            _st("middle", priority=1),
        ]
    )
    batches = topological_batches(dag)
    assert [s.subtask_id for s in batches[0]] == ["alpha", "middle", "zeta"]


def test_cycle_raises_dag_error() -> None:
    dag = _dag([_st("a", deps=["b"]), _st("b", deps=["a"])])
    with pytest.raises(DAGError, match="cycle"):
        topological_batches(dag)


def test_missing_dependency_raises_dag_error() -> None:
    dag = _dag([_st("a", deps=["ghost"])])
    with pytest.raises(DAGError, match="missing id"):
        topological_batches(dag)


def test_duplicate_id_raises_dag_error() -> None:
    dag = _dag([_st("a"), _st("a")])
    with pytest.raises(DAGError, match="unique"):
        topological_batches(dag)


# ---------------------------------------------------------------------------
# detect_write_conflicts
# ---------------------------------------------------------------------------


def test_detect_no_conflicts_with_disjoint_writes() -> None:
    batch = [
        _st("a", writes=["src/a.py"]),
        _st("b", writes=["src/b.py"]),
    ]
    assert detect_write_conflicts(batch) == []


def test_detect_conflict_with_shared_write() -> None:
    batch = [
        _st("a", writes=["src/app.py"]),
        _st("b", writes=["src/app.py"]),
    ]
    assert detect_write_conflicts(batch) == [("a", "b")]


def test_detect_conflict_three_way_collision() -> None:
    """Three subtasks writing the same file produce all 3 pairs."""
    batch = [
        _st("a", writes=["src/app.py"]),
        _st("b", writes=["src/app.py"]),
        _st("c", writes=["src/app.py"]),
    ]
    assert detect_write_conflicts(batch) == [("a", "b"), ("a", "c"), ("b", "c")]


def test_detect_conflict_ordering_is_deterministic() -> None:
    """Input order of ``batch`` must not change the output."""
    forward = detect_write_conflicts([_st("x", writes=["f.py"]), _st("y", writes=["f.py"])])
    reverse = detect_write_conflicts([_st("y", writes=["f.py"]), _st("x", writes=["f.py"])])
    assert forward == reverse == [("x", "y")]


def test_detect_empty_writes_never_conflicts() -> None:
    batch = [_st("a", writes=[]), _st("b", writes=[])]
    assert detect_write_conflicts(batch) == []


# ---------------------------------------------------------------------------
# defer_lower_priority_on_conflicts
# ---------------------------------------------------------------------------


def test_defer_returns_input_when_no_conflicts() -> None:
    batch = [_st("a", writes=["x.py"]), _st("b", writes=["y.py"])]
    keep, deferred = defer_lower_priority_on_conflicts(batch, [])
    assert [s.subtask_id for s in keep] == ["a", "b"]
    assert deferred == []


def test_defer_drops_lower_priority_on_conflict() -> None:
    batch = [
        _st("a", writes=["app.py"], priority=10),
        _st("b", writes=["app.py"], priority=1),
    ]
    keep, deferred = defer_lower_priority_on_conflicts(batch, detect_write_conflicts(batch))
    assert [s.subtask_id for s in keep] == ["a"]
    assert [s.subtask_id for s in deferred] == ["b"]


def test_defer_breaks_priority_ties_by_subtask_id() -> None:
    batch = [
        _st("alpha", writes=["app.py"], priority=5),
        _st("zeta", writes=["app.py"], priority=5),
    ]
    keep, deferred = defer_lower_priority_on_conflicts(batch, detect_write_conflicts(batch))
    assert [s.subtask_id for s in keep] == ["alpha"]
    assert [s.subtask_id for s in deferred] == ["zeta"]


def test_defer_three_way_collision_keeps_highest_priority() -> None:
    batch = [
        _st("a", writes=["app.py"], priority=1),
        _st("b", writes=["app.py"], priority=10),
        _st("c", writes=["app.py"], priority=5),
    ]
    keep, deferred = defer_lower_priority_on_conflicts(batch, detect_write_conflicts(batch))
    assert [s.subtask_id for s in keep] == ["b"]
    assert sorted(s.subtask_id for s in deferred) == ["a", "c"]


def test_defer_preserves_non_conflicting_subtasks() -> None:
    batch = [
        _st("a", writes=["app.py"], priority=10),
        _st("b", writes=["app.py"], priority=1),
        _st("c", writes=["db.py"], priority=0),
    ]
    keep, deferred = defer_lower_priority_on_conflicts(batch, detect_write_conflicts(batch))
    assert [s.subtask_id for s in keep] == ["a", "c"]
    assert [s.subtask_id for s in deferred] == ["b"]


def test_defer_deferred_order_is_priority_then_id() -> None:
    """``deferred`` is sorted the same way a batch would be: high-priority first."""
    batch = [
        _st("keeper", writes=["app.py"], priority=100),
        _st("loser_low", writes=["app.py"], priority=0),
        _st("loser_mid", writes=["app.py"], priority=5),
    ]
    _keep, deferred = defer_lower_priority_on_conflicts(batch, detect_write_conflicts(batch))
    assert [s.subtask_id for s in deferred] == ["loser_mid", "loser_low"]
