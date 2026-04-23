"""DAG primitives for Maestro's scheduler (spec 04 §2).

This module is pure: given a :class:`~maestro.models.TaskDAG` it produces
batches of :class:`~maestro.models.SubTask` instances that can be executed
in parallel, plus helpers to detect and resolve runtime write-conflicts.

Algorithms
----------

* :func:`topological_batches` — layered topological sort. Each batch contains
  every subtask whose dependencies are fully resolved by the preceding
  batches. Within a batch, subtasks are sorted by ``priority`` descending,
  breaking ties by ``subtask_id``.
* :func:`detect_write_conflicts` — O(n²) pair enumeration (n ≤ ~8 per spec
  03, so this is fine) returning every pair of subtasks in the same batch
  that declare a common write target.
* :func:`defer_lower_priority_on_conflicts` — given a conflict list, splits
  the batch into ``(keep, deferred)`` where the lower-priority subtask of
  each conflicting pair moves to ``deferred``. The scheduler then appends
  ``deferred`` to the next batch, effectively inserting an implicit
  dependency. Priority ties fall back to ``subtask_id`` for determinism.

``DAGError`` wraps the validation errors that :meth:`TaskDAG.validate_dag`
raises; the scheduler only ever catches this type so unrelated exceptions
stay visible.
"""

from __future__ import annotations

from collections import defaultdict

from maestro.models import SubTask, TaskDAG


class DAGError(Exception):
    """Raised for any structural DAG error."""


# ---------------------------------------------------------------------------
# Topological batching
# ---------------------------------------------------------------------------


def topological_batches(dag: TaskDAG) -> list[list[SubTask]]:
    """Group ``dag`` subtasks into layered parallel batches.

    The algorithm is Kahn-ish: we compute indegree per node, emit every
    indegree-zero node as one batch, decrement indegrees of their
    successors, and repeat until the graph is empty. Within a batch, order
    is ``priority`` descending then ``subtask_id`` ascending for stable
    output.

    Raises :class:`DAGError` on duplicate ids, missing references, self-loops
    or cycles (detected via :meth:`TaskDAG.validate_dag`).
    """
    try:
        dag.validate_dag()
    except ValueError as exc:
        raise DAGError(str(exc)) from exc

    if not dag.subtasks:
        return []

    by_id: dict[str, SubTask] = {s.subtask_id: s for s in dag.subtasks}
    remaining_deps: dict[str, set[str]] = {s.subtask_id: set(s.depends_on) for s in dag.subtasks}
    successors: dict[str, list[str]] = defaultdict(list)
    for s in dag.subtasks:
        for dep in s.depends_on:
            successors[dep].append(s.subtask_id)

    batches: list[list[SubTask]] = []
    remaining_ids: set[str] = set(by_id)

    while remaining_ids:
        ready_ids = [sid for sid in remaining_ids if not remaining_deps[sid]]
        if not ready_ids:
            # validate_dag should have caught cycles, but guard defensively.
            raise DAGError(
                f"Scheduler found subtasks with unresolvable dependencies: {sorted(remaining_ids)}"
            )
        ready = [by_id[sid] for sid in ready_ids]
        ready.sort(key=_batch_sort_key)
        batches.append(ready)
        for sid in ready_ids:
            remaining_ids.discard(sid)
            for succ in successors.get(sid, []):
                remaining_deps[succ].discard(sid)

    return batches


def _batch_sort_key(subtask: SubTask) -> tuple[int, str]:
    """Higher priority first; ties broken by subtask_id ascending."""
    return (-subtask.priority, subtask.subtask_id)


# ---------------------------------------------------------------------------
# Write-conflict detection
# ---------------------------------------------------------------------------


def detect_write_conflicts(batch: list[SubTask]) -> list[tuple[str, str]]:
    """Return every pair of subtasks in ``batch`` that share a write target.

    The returned tuples are ordered ``(lhs, rhs)`` such that ``lhs.subtask_id
    < rhs.subtask_id`` so the output is deterministic regardless of the input
    order of ``batch``.
    """
    # Map each written file to the list of subtask ids that declared it.
    writers: dict[str, list[str]] = defaultdict(list)
    for subtask in batch:
        for path in subtask.writes:
            writers[path].append(subtask.subtask_id)

    conflicts: set[tuple[str, str]] = set()
    for ids in writers.values():
        if len(ids) < 2:
            continue
        sorted_ids = sorted(ids)
        for i, a in enumerate(sorted_ids):
            for b in sorted_ids[i + 1 :]:
                conflicts.add((a, b))
    return sorted(conflicts)


# ---------------------------------------------------------------------------
# Conflict resolution (defer lower-priority to next batch)
# ---------------------------------------------------------------------------


def defer_lower_priority_on_conflicts(
    batch: list[SubTask], conflicts: list[tuple[str, str]]
) -> tuple[list[SubTask], list[SubTask]]:
    """Split ``batch`` into ``(keep, deferred)`` based on ``conflicts``.

    For each conflicting pair we defer the lower-priority subtask. If both
    have the same priority, the one with the larger ``subtask_id`` is
    deferred (this matches the tiebreak used when assembling the batch, so
    the chosen survivor is the one that would have been listed first).

    The resulting ``keep`` list retains its original input order minus the
    deferred ids; ``deferred`` contains the dropped subtasks in the same
    relative order, sorted by priority descending then id for determinism.
    """
    if not conflicts:
        return list(batch), []

    by_id = {s.subtask_id: s for s in batch}
    deferred_ids: set[str] = set()

    for lhs_id, rhs_id in conflicts:
        # If either side is already deferred, propagate the decision — this
        # handles three-way (or more) write collisions on the same file:
        # any surviving subtask still conflicts with any already-deferred
        # one only through that file, and deferring both would be wrong.
        if lhs_id in deferred_ids or rhs_id in deferred_ids:
            continue
        lhs = by_id[lhs_id]
        rhs = by_id[rhs_id]
        loser = _pick_loser(lhs, rhs)
        deferred_ids.add(loser.subtask_id)

    keep = [s for s in batch if s.subtask_id not in deferred_ids]
    deferred = [s for s in batch if s.subtask_id in deferred_ids]
    deferred.sort(key=_batch_sort_key)
    return keep, deferred


def _pick_loser(lhs: SubTask, rhs: SubTask) -> SubTask:
    """Return whichever of ``lhs``/``rhs`` should be deferred."""
    if lhs.priority != rhs.priority:
        return lhs if lhs.priority < rhs.priority else rhs
    # Same priority: defer the one with the larger subtask_id so the
    # "first" one (by canonical id order) survives.
    return lhs if lhs.subtask_id > rhs.subtask_id else rhs


__all__ = [
    "DAGError",
    "defer_lower_priority_on_conflicts",
    "detect_write_conflicts",
    "topological_batches",
]
