"""Three-tier verifier orchestrator (spec 06 §2).

Runs enabled tiers in order ``deterministic → test_based → llm_judge`` and
short-circuits on the first failure (spec 06 §1). Uncertainty from the
judge is treated as a failure — the mean score may look fine, but the
samples disagree, so we decline to pass.

``Verifier`` satisfies the :class:`~maestro.scheduler.VerifierProtocol`
structurally, so the scheduler accepts it without any additional glue.

Ablation support
----------------

``enabled_tiers`` lets the benchmark harness run tier-level ablations
without spinning up a different class. Spec 09 §4 / spec 10 §2 rely on
this to enumerate E1..E6:

- ``{"deterministic"}`` — Tier 1 only
- ``{"deterministic", "test_based"}`` — T1 + T2
- ``{"deterministic", "test_based", "llm_judge"}`` — full (default)
- ``{"llm_judge"}`` — judge-only, useful for the judge-parameter
  sensitivity experiments in spec 10 §2.2
- ``set()`` or ``None`` with ``enabled_tiers=set()`` — no tiers; the
  verifier will always pass. Used by the ``parallel_only`` config
  in spec 10 §2.1.
"""

from __future__ import annotations

import time

from maestro.llm.client import LLMClient
from maestro.models import (
    JudgeTier,
    LLMJudgeDetail,
    SubAgentResult,
    SubTask,
    TaskSpec,
    TierResult,
    VerificationResult,
)
from maestro.sandbox.workspace import IsolatedWorkspace, WorkspaceManager
from maestro.utils.logging import get_logger

from .deterministic import DeterministicVerifier
from .llm_judge import LLMJudgeVerifier
from .test_based import TestBasedVerifier

_logger = get_logger("maestro.verifier")

_DEFAULT_TIERS: frozenset[JudgeTier] = frozenset({"deterministic", "test_based", "llm_judge"})


class Verifier:
    """Orchestrates the three verification tiers with short-circuit semantics."""

    def __init__(
        self,
        llm_client: LLMClient,
        task_spec: TaskSpec,
        *,
        workspace_manager: WorkspaceManager | None = None,
        enabled_tiers: set[JudgeTier] | None = None,
        tier1: DeterministicVerifier | None = None,
        tier2: TestBasedVerifier | None = None,
        tier3: LLMJudgeVerifier | None = None,
    ) -> None:
        self._llm = llm_client
        self._spec = task_spec
        self._enabled: frozenset[JudgeTier] = (
            frozenset(enabled_tiers) if enabled_tiers is not None else _DEFAULT_TIERS
        )
        self._tier1 = tier1 or DeterministicVerifier()
        self._tier2 = tier2 or TestBasedVerifier(
            llm_client=llm_client, auto_gen_tests=task_spec.auto_gen_tests
        )
        self._tier3 = tier3 or LLMJudgeVerifier(
            llm_client, task_spec, workspace_manager=workspace_manager
        )

    @property
    def enabled_tiers(self) -> frozenset[JudgeTier]:
        return self._enabled

    async def verify(
        self,
        subtask: SubTask,
        workspace: IsolatedWorkspace,
        sub_result: SubAgentResult,
    ) -> VerificationResult:
        """Run enabled tiers in order, short-circuit on first failure."""
        start = time.perf_counter()
        tiers: list[TierResult] = []
        judge_detail: LLMJudgeDetail | None = None
        total_cost = 0.0

        if "deterministic" in self._enabled:
            t1 = await self._tier1.run(workspace, sub_result)
            tiers.append(t1)
            total_cost += t1.cost_usd
            if not t1.passed:
                return _assemble(subtask, tiers, judge_detail, start, total_cost, passed=False)

        if "test_based" in self._enabled:
            t2 = await self._tier2.run(workspace, sub_result)
            tiers.append(t2)
            total_cost += t2.cost_usd
            if not t2.passed:
                return _assemble(subtask, tiers, judge_detail, start, total_cost, passed=False)

        if "llm_judge" in self._enabled:
            t3, judge_detail = await self._tier3.run(subtask, workspace, sub_result)
            tiers.append(t3)
            total_cost += t3.cost_usd
            if not t3.passed:
                # Judge's ``passed`` already folds in disagreement/uncertainty.
                return _assemble(subtask, tiers, judge_detail, start, total_cost, passed=False)

        _logger.info(
            "verifier_passed",
            subtask_id=subtask.subtask_id,
            tiers=[t.tier for t in tiers],
            cost=total_cost,
        )
        return _assemble(subtask, tiers, judge_detail, start, total_cost, passed=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assemble(
    subtask: SubTask,
    tiers: list[TierResult],
    judge_detail: LLMJudgeDetail | None,
    start: float,
    total_cost: float,
    *,
    passed: bool,
) -> VerificationResult:
    return VerificationResult(
        subtask_id=subtask.subtask_id,
        overall_passed=passed,
        tiers=tiers,
        judge_detail=judge_detail,
        total_latency_ms=max(0, int((time.perf_counter() - start) * 1000)),
        total_cost_usd=total_cost,
    )


__all__ = ["Verifier"]
