"""Orchestrator: LangGraph state machine that wires the pipeline (spec 04 §5).

Graph shape::

    START → plan_node → schedule_node → merge_node → report_node → END
                   ↘──────────── error_node ────────────↗

Node responsibilities
---------------------

* ``plan_node`` calls :class:`~maestro.planner.Planner.plan` and stores the
  ``TaskDAG`` into the state. On ``PlanningError`` it forwards to
  ``error_node``.
* ``schedule_node`` runs :class:`~maestro.scheduler.Scheduler.execute`, which
  drives the sub-agent + verifier loop and returns a list of
  ``BatchResult``.
* ``merge_node`` pulls the final diff from the workspace and assembles a
  :class:`~maestro.models.TaskResult` with aggregate token / cost / latency
  metrics. It decides ``status`` from the batch-level outcome:

    - all subtasks across every batch merged cleanly → ``success``
    - some merged, some failed → ``partial``
    - nothing merged → ``failed``

* ``report_node`` emits a structlog summary with the three numbers a
  benchmark report eventually wants (resolve count, tokens, cost). CLI
  rendering happens elsewhere (module 14).
* ``error_node`` records the error message and short-circuits to END with
  ``final_result=None``. Any node may route here by returning a state
  update that sets ``error``.

Why LangGraph for the outer loop?
---------------------------------

The outer pipeline is a textbook branchy state machine (plan → either
execute or error → report); LangGraph expresses that cleanly and
provides tracing that is painful to replicate by hand. The *inner*
parallel scheduling stays in plain asyncio because that's what asyncio
is good at — LangGraph would only add ceremony there (DESIGN §3.6 /
spec 04 §5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from maestro.llm.client import LLMClient
from maestro.models import (
    BatchResult,
    TaskDAG,
    TaskResult,
    TaskSpec,
    TaskStatus,
)
from maestro.planner.planner import Planner, PlanningError
from maestro.sandbox.workspace import WorkspaceManager
from maestro.scheduler.scheduler import Scheduler, VerifierProtocol
from maestro.subagent.subagent import SubAgentFactory
from maestro.utils.logging import get_logger
from maestro.utils.time import utcnow

_logger = get_logger("maestro.orchestrator")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class OrchestratorState(TypedDict, total=False):
    """Mutable state passed between LangGraph nodes.

    ``total=False`` lets each node return only the keys it wants to update;
    LangGraph merges these partial updates into the prior state.
    """

    task_spec: TaskSpec
    dag: TaskDAG | None
    batch_results: list[BatchResult]
    final_result: TaskResult | None
    error: str | None


# ---------------------------------------------------------------------------
# Dependencies bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrchestratorDeps:
    """Concrete dependencies the graph nodes need at runtime.

    Passed into :func:`build_graph` so tests can inject fakes without
    monkey-patching module globals.
    """

    planner: Planner
    verifier: VerifierProtocol
    workspace: WorkspaceManager
    llm_client: LLMClient
    subagent_factory: SubAgentFactory


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(deps: OrchestratorDeps) -> Any:
    """Return a compiled LangGraph app for the orchestrator.

    The graph's input is an :class:`OrchestratorState` containing at least
    ``task_spec``; the output is the state with ``final_result`` populated
    (or ``error`` if the pipeline failed).
    """
    graph: StateGraph[OrchestratorState, Any, OrchestratorState, OrchestratorState] = StateGraph(
        OrchestratorState
    )

    graph.add_node("plan", _make_plan_node(deps))
    graph.add_node("schedule", _make_schedule_node(deps))
    graph.add_node("merge", _make_merge_node(deps))
    graph.add_node("report", _report_node)
    graph.add_node("error", _error_node)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", _branch_on_error, {"ok": "schedule", "error": "error"})
    graph.add_conditional_edges("schedule", _branch_on_error, {"ok": "merge", "error": "error"})
    graph.add_conditional_edges("merge", _branch_on_error, {"ok": "report", "error": "error"})
    graph.add_edge("report", END)
    graph.add_edge("error", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------


def _make_plan_node(deps: OrchestratorDeps):  # type: ignore[no-untyped-def]
    async def plan_node(state: OrchestratorState) -> OrchestratorState:
        spec = _require_task_spec(state)
        try:
            dag = await deps.planner.plan(spec)
        except PlanningError as exc:
            _logger.error("plan_failed", task_id=spec.task_id, error=str(exc))
            return {"error": f"Planner failed: {exc}"}
        _logger.info(
            "plan_ok",
            task_id=spec.task_id,
            subtasks=len(dag.subtasks),
        )
        return {"dag": dag}

    return plan_node


def _make_schedule_node(deps: OrchestratorDeps):  # type: ignore[no-untyped-def]
    async def schedule_node(state: OrchestratorState) -> OrchestratorState:
        spec = _require_task_spec(state)
        dag = state.get("dag")
        if dag is None:
            return {"error": "Scheduler reached without a DAG"}
        try:
            scheduler = Scheduler(
                dag=dag,
                task_spec=spec,
                subagent_factory=deps.subagent_factory,
                verifier=deps.verifier,
                workspace_manager=deps.workspace,
                llm_client=deps.llm_client,
            )
            batch_results = await scheduler.execute()
        except Exception as exc:
            _logger.error(
                "schedule_failed",
                task_id=spec.task_id,
                error=exc.__class__.__name__,
                message=str(exc),
            )
            return {"error": f"Scheduler failed: {exc}"}
        return {"batch_results": batch_results}

    return schedule_node


def _make_merge_node(deps: OrchestratorDeps):  # type: ignore[no-untyped-def]
    async def merge_node(state: OrchestratorState) -> OrchestratorState:
        spec = _require_task_spec(state)
        batches = state.get("batch_results") or []
        started = spec.created_at
        finished = utcnow()
        try:
            final_diff = deps.workspace.get_final_diff()
        except Exception as exc:
            _logger.error("merge_diff_failed", task_id=spec.task_id, error=str(exc))
            return {"error": f"Final diff generation failed: {exc}"}

        metrics = _aggregate_metrics(batches)
        status = _overall_status(batches)
        wall_clock_ms = int((finished - started).total_seconds() * 1000)
        result = TaskResult(
            task_id=spec.task_id,
            status=status,
            batches=batches,
            final_diff=final_diff,
            final_workspace=deps.workspace.main_path,
            total_wall_clock_ms=max(0, wall_clock_ms),
            total_tokens_input=metrics.tokens_in,
            total_tokens_output=metrics.tokens_out,
            total_cost_usd=metrics.cost,
            started_at=started,
            finished_at=finished,
        )
        return {"final_result": result}

    return merge_node


async def _report_node(state: OrchestratorState) -> OrchestratorState:
    result = state.get("final_result")
    if result is None:
        return {}
    _logger.info(
        "task_report",
        task_id=result.task_id,
        status=result.status,
        batches=len(result.batches),
        merged=sum(len(b.merged_patches) for b in result.batches),
        failed=sum(len(b.failed_patches) for b in result.batches),
        tokens_input=result.total_tokens_input,
        tokens_output=result.total_tokens_output,
        cost=result.total_cost_usd,
        wall_clock_ms=result.total_wall_clock_ms,
    )
    return {}


async def _error_node(state: OrchestratorState) -> OrchestratorState:
    err = state.get("error") or "unknown orchestrator error"
    task_id = state.get("task_spec")
    _logger.error(
        "task_aborted",
        task_id=task_id.task_id if isinstance(task_id, TaskSpec) else None,
        error=err,
    )
    # Leave ``final_result`` unset so callers can distinguish failure from
    # success by ``final_result is None``.
    return {"final_result": None}


# ---------------------------------------------------------------------------
# Routing + helpers
# ---------------------------------------------------------------------------


def _branch_on_error(state: OrchestratorState) -> str:
    return "error" if state.get("error") else "ok"


def _require_task_spec(state: OrchestratorState) -> TaskSpec:
    spec = state.get("task_spec")
    if spec is None:
        raise RuntimeError("Orchestrator state missing task_spec")
    return spec


@dataclass(frozen=True)
class _AggregateMetrics:
    tokens_in: int
    tokens_out: int
    cost: float


def _aggregate_metrics(batches: list[BatchResult]) -> _AggregateMetrics:
    """Aggregate per-subtask metrics from batch results.

    LIMITATION: This only sums SubAgentResult tokens and VerificationResult
    cost — it does NOT include:

    - Planner's LLM tokens (one call per task, recorded on LLMClient but
      not on any batch)
    - LLM-Judge's tokens (judge cost_usd is in VerificationResult, but
      the token counts currently are not)

    TODO(haitian, module 11-13): when verifier lands, extend TaskResult
    schema with per-role breakdown (planner / subagent / judge) pulled
    from LLMClient.get_cost_report(). This keeps total-task cost honest
    and enables per-role ablation in benchmark analysis.
    """
    tokens_in = 0
    tokens_out = 0
    cost = 0.0
    for batch in batches:
        for sub in batch.subtask_results:
            tokens_in += sub.tokens_input
            tokens_out += sub.tokens_output
        for v in batch.verification_results:
            cost += v.total_cost_usd
    return _AggregateMetrics(tokens_in=tokens_in, tokens_out=tokens_out, cost=cost)


def _overall_status(batches: list[BatchResult]) -> TaskStatus:
    total_subtasks = 0
    merged = 0
    failed = 0
    for batch in batches:
        total_subtasks += len(batch.subtask_results)
        merged += len(batch.merged_patches)
        failed += len(batch.failed_patches)
    if merged == 0 and failed == 0 and total_subtasks == 0:
        return "failed"  # nothing ran
    if failed == 0 and merged == total_subtasks:
        return "success"
    if merged == 0:
        return "failed"
    return "partial"


__all__ = [
    "OrchestratorDeps",
    "OrchestratorState",
    "build_graph",
]
