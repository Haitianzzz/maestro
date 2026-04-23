"""DAG scheduler main loop (spec 04 §3).

The :class:`Scheduler` coordinates the live execution of a ``TaskDAG``:

1. Break the DAG into topological batches via :func:`topological_batches`.
2. For each batch, detect runtime write-conflicts and defer lower-priority
   subtasks into the next batch (spec 04 §2.2).
3. Run the surviving subtasks in parallel under a :class:`PrioritySemaphore`
   capped at ``TaskSpec.max_parallel``. Each subtask gets a fresh isolated
   workspace, a sub-agent instance, and its output is pushed through the
   verifier.
4. On verification failure, the scheduler retries the subtask — feeding the
   prior attempt and failure back into the sub-agent so it can correct
   course. ``TaskSpec.max_retries_per_subtask`` bounds retry depth.
5. Successful patches are merged into the main workspace before the next
   batch begins (see :class:`~maestro.sandbox.workspace.WorkspaceManager`'s
   concurrency contract).

Design notes
------------

* This module is deliberately thin on business logic. All conflict
  semantics live in :mod:`maestro.scheduler.dag`; all diff / permission
  logic lives in the workspace and sub-agent modules. The scheduler only
  wires these together.
* ``asyncio.TaskGroup`` is used for the per-batch parallel fan-out
  (Python 3.11+; DESIGN §4). If one task raises, sibling tasks are
  cancelled and the cancellation is visible as a clean ``failed`` state
  on their ``SubAgentResult``.
* The verifier parameter is typed as a ``Protocol`` so module 9 can ship
  before modules 11-13. The final :class:`~maestro.verifier.Verifier`
  satisfies the protocol structurally.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

from maestro.llm.client import LLMClient
from maestro.models import (
    BatchResult,
    SubAgentResult,
    SubTask,
    TaskDAG,
    TaskSpec,
    VerificationResult,
)
from maestro.sandbox.workspace import IsolatedWorkspace, WorkspaceManager
from maestro.subagent.subagent import SubAgentFactory
from maestro.utils.logging import get_logger
from maestro.utils.priority_queue import PrioritySemaphore

from .dag import (
    defer_lower_priority_on_conflicts,
    detect_write_conflicts,
    topological_batches,
)

_logger = get_logger("maestro.scheduler")


class VerifierProtocol(Protocol):
    """Structural interface the scheduler expects from a verifier.

    The concrete verifier (modules 11-13) implements this; we declare it
    here so the scheduler can land before its verifier dependency.
    """

    async def verify(
        self,
        subtask: SubTask,
        workspace: IsolatedWorkspace,
        sub_result: SubAgentResult,
    ) -> VerificationResult: ...


class Scheduler:
    """Parallel DAG scheduler for sub-agent execution."""

    def __init__(
        self,
        dag: TaskDAG,
        task_spec: TaskSpec,
        subagent_factory: SubAgentFactory,
        verifier: VerifierProtocol,
        workspace_manager: WorkspaceManager,
        llm_client: LLMClient,
    ) -> None:
        self._dag = dag
        self._spec = task_spec
        self._factory = subagent_factory
        self._verifier = verifier
        self._workspace = workspace_manager
        self._llm = llm_client

        self._prio_semaphore = PrioritySemaphore(task_spec.max_parallel)
        # Retry count is keyed by subtask_id and is strictly monotonic across
        # the entire task lifetime; a subtask that exhausts its budget in one
        # invocation is never re-run by the scheduler itself.
        self._retry_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def execute(self) -> list[BatchResult]:
        """Execute every batch in topological order, returning per-batch results."""
        batches = topological_batches(self._dag)
        results: list[BatchResult] = []

        batch_index = 0
        pending_batches: list[list[SubTask]] = [list(b) for b in batches]

        while batch_index < len(pending_batches):
            batch = pending_batches[batch_index]

            conflicts = detect_write_conflicts(batch)
            if conflicts:
                keep, deferred = defer_lower_priority_on_conflicts(batch, conflicts)
                _logger.warning(
                    "write_conflicts_detected",
                    task_id=self._spec.task_id,
                    batch_index=batch_index,
                    pairs=conflicts,
                    deferred=[s.subtask_id for s in deferred],
                )
                batch = keep
                if deferred:
                    if batch_index + 1 < len(pending_batches):
                        pending_batches[batch_index + 1].extend(deferred)
                    else:
                        pending_batches.append(deferred)

            _logger.info(
                "batch_start",
                task_id=self._spec.task_id,
                batch_index=batch_index,
                subtasks=[s.subtask_id for s in batch],
                conflicts=len(conflicts),
            )

            batch_result = await self._execute_batch(
                batch_index=batch_index,
                subtasks=batch,
                conflicts=conflicts,
            )
            results.append(batch_result)

            # Merge successful patches before the next batch opens (this is
            # the serialisation point documented in WorkspaceManager's
            # "Concurrency contract").
            if batch_result.merged_patches:
                merged_results = [
                    r
                    for r in batch_result.subtask_results
                    if r.subtask_id in batch_result.merged_patches
                ]
                await self._workspace.merge_patches(merged_results)

            _logger.info(
                "batch_done",
                task_id=self._spec.task_id,
                batch_index=batch_index,
                merged=len(batch_result.merged_patches),
                retried=len(batch_result.retried_patches),
                failed=len(batch_result.failed_patches),
            )

            # If every subtask in a batch failed (and there were subtasks
            # at all), bail out — subsequent batches would build on a
            # broken workspace.
            if batch and not batch_result.merged_patches and batch_result.failed_patches:
                _logger.warning(
                    "batch_all_failed_aborting",
                    task_id=self._spec.task_id,
                    batch_index=batch_index,
                )
                break

            batch_index += 1

        return results

    # ------------------------------------------------------------------
    # Batch execution
    # ------------------------------------------------------------------

    async def _execute_batch(
        self,
        *,
        batch_index: int,
        subtasks: list[SubTask],
        conflicts: list[tuple[str, str]],
    ) -> BatchResult:
        if not subtasks:
            return BatchResult(
                batch_index=batch_index,
                subtask_results=[],
                verification_results=[],
                merged_patches=[],
                retried_patches=[],
                failed_patches=[],
                conflicts_detected=conflicts,
            )

        # Fan out in parallel; cancel siblings on any unexpected exception.
        tasks: dict[str, asyncio.Task[tuple[SubAgentResult, VerificationResult]]] = {}
        async with asyncio.TaskGroup() as tg:
            for subtask in subtasks:
                tasks[subtask.subtask_id] = tg.create_task(self._run_subtask_with_verify(subtask))

        subtask_results: list[SubAgentResult] = []
        verification_results: list[VerificationResult] = []
        merged: list[str] = []
        retried: list[str] = []
        failed: list[str] = []

        for subtask in subtasks:
            sub_res, verify_res = tasks[subtask.subtask_id].result()
            subtask_results.append(sub_res)
            verification_results.append(verify_res)

            if self._retry_counts.get(subtask.subtask_id, 0) > 0:
                retried.append(subtask.subtask_id)

            if sub_res.status == "success" and verify_res.overall_passed:
                merged.append(subtask.subtask_id)
            else:
                failed.append(subtask.subtask_id)

        return BatchResult(
            batch_index=batch_index,
            subtask_results=subtask_results,
            verification_results=verification_results,
            merged_patches=merged,
            retried_patches=retried,
            failed_patches=failed,
            conflicts_detected=conflicts,
        )

    # ------------------------------------------------------------------
    # Per-subtask
    # ------------------------------------------------------------------

    async def _run_subtask_with_verify(
        self,
        subtask: SubTask,
    ) -> tuple[SubAgentResult, VerificationResult]:
        """Run one subtask through the sub-agent + verifier pipeline.

        Held under the priority semaphore so high-priority subtasks get
        slots first when the batch is over-subscribed.
        """
        async with self._prio_semaphore.acquire(priority=subtask.priority):
            return await self._attempt(
                subtask=subtask,
                prior_attempt=None,
                prior_failure=None,
            )

    async def _attempt(
        self,
        *,
        subtask: SubTask,
        prior_attempt: SubAgentResult | None,
        prior_failure: VerificationResult | None,
    ) -> tuple[SubAgentResult, VerificationResult]:
        workspace = await self._workspace.create_isolated(subtask)
        agent = self._factory.create(
            subtask,
            workspace,
            self._llm,
            prior_attempt=prior_attempt,
            prior_failure=prior_failure,
        )
        sub_result = await agent.run(self._dag.global_context)

        if sub_result.status != "success":
            return sub_result, _empty_verification(subtask.subtask_id)

        verify_result = await self._verifier.verify(subtask, workspace, sub_result)

        if verify_result.overall_passed:
            return sub_result, verify_result

        retry = await self._handle_verify_failure(
            subtask=subtask, sub_result=sub_result, verify_result=verify_result
        )
        if retry is not None:
            return retry
        return sub_result, verify_result

    async def _handle_verify_failure(
        self,
        *,
        subtask: SubTask,
        sub_result: SubAgentResult,
        verify_result: VerificationResult,
    ) -> tuple[SubAgentResult, VerificationResult] | None:
        current = self._retry_counts.get(subtask.subtask_id, 0)
        if current >= self._spec.max_retries_per_subtask:
            _logger.warning(
                "retry_exhausted",
                task_id=self._spec.task_id,
                subtask_id=subtask.subtask_id,
                retries=current,
            )
            return None

        self._retry_counts[subtask.subtask_id] = current + 1
        _logger.info(
            "subtask_retry",
            task_id=self._spec.task_id,
            subtask_id=subtask.subtask_id,
            retry=current + 1,
            max_retries=self._spec.max_retries_per_subtask,
        )
        return await self._attempt(
            subtask=subtask,
            prior_attempt=sub_result,
            prior_failure=verify_result,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_verification(subtask_id: str) -> VerificationResult:
    """Placeholder verification result when sub-agent failed before verify ran."""
    return VerificationResult(
        subtask_id=subtask_id,
        overall_passed=False,
        tiers=[],
        judge_detail=None,
        total_latency_ms=0,
        total_cost_usd=0.0,
    )


__all__ = ["Scheduler", "VerifierProtocol"]
