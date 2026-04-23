"""Tier 3 verifier: multi-sample LLM-as-Judge with disagreement detection.

Spec reference: ``specs/06-verifier.md`` §5.

Design motivation
-----------------

LLM-as-Judge on verifiable tasks exhibits *silent correctness leakage*:
on any given verdict the judge may be confidently wrong, and a single
sample provides no signal for that confidence. We answer with K parallel
samples at varied temperatures, compute a disagreement metric, and flag
"uncertain" when the samples fail to agree even if the mean score looks
good. Pass criteria require mean ≥ 0.6 AND majority pass AND *not*
uncertain.

This directly materialises the engineering claim of the author's EMNLP
2026 paper: single-sample judges leak, aggregation + disagreement is the
practical mitigation.

Budget knobs (spec 06 §2.1):

* ``TaskSpec.judge_samples`` — K, default 3 (spec 06 §5.3 — 3x cost is
  controlled; 5x is diminishing returns).
* ``TaskSpec.judge_disagreement_threshold`` — the weighted (0.5·stdev +
  0.5·binary-disagreement) cutoff, default 0.3.

The judge uses :meth:`LLMClient.call_structured` under the ``judge`` role
so cost statistics land in the right per-role bucket
(``CostReport.per_role``).
"""

from __future__ import annotations

import asyncio
import statistics
import time

from maestro.llm.client import LLMCallMetadata, LLMClient
from maestro.models import (
    JudgeOutput,
    LLMJudgeDetail,
    SubAgentResult,
    SubTask,
    TaskSpec,
    TierResult,
)
from maestro.sandbox.workspace import IsolatedWorkspace, WorkspaceManager
from maestro.utils.logging import get_logger

_logger = get_logger("maestro.verifier.llm_judge")

_MEAN_SCORE_THRESHOLD = 0.6
_MAX_ORIGINAL_FILES_BYTES = 4_000  # per-file cap in judge prompt
_MAX_ORIGINAL_FILES_TOTAL = 20_000  # total budget for "original files" section


JUDGE_SYSTEM_PROMPT = """\
You are Maestro Judge, an independent code reviewer.

You are given:
1. A subtask description (what was supposed to be done)
2. A diff produced by a coding agent
3. The original files before the diff was applied

Your job: evaluate whether the diff correctly implements the subtask.

Score 0.0-1.0:
- 1.0: Perfectly implements the subtask, no bugs, good style
- 0.8: Correct implementation with minor issues (style, naming)
- 0.6: Mostly correct but has a notable issue (missing edge case)
- 0.4: Partially correct, significant issue
- 0.2: Largely incorrect
- 0.0: Completely wrong or does nothing

passes_requirements = true iff score >= 0.6 AND you are confident the
diff does not introduce bugs.

Output JSON matching the JudgeOutput schema. Be concise in reasoning
(under 200 words).
"""


JUDGE_USER_PROMPT_TEMPLATE = """\
# Subtask
ID: {subtask_id}
Description: {description}
Files to modify: {writes}

# Agent's rationale
{rationale}

# Agent's diff
```diff
{diff}
```

# Original files (before diff)
{formatted_original_files}

# Instructions
Evaluate correctness. Return JSON matching the JudgeOutput schema.
"""


class LLMJudgeVerifier:
    """Multi-sample LLM judge with disagreement detection."""

    def __init__(
        self,
        llm_client: LLMClient,
        task_spec: TaskSpec,
        *,
        workspace_manager: WorkspaceManager | None = None,
        temperatures: list[float] | None = None,
    ) -> None:
        self._llm = llm_client
        self._spec = task_spec
        # Optional: needed to surface pre-diff file content to the judge
        # prompt (M3). When absent we just omit the "Original files"
        # section and fall back to ``(unavailable)``.
        self._workspace_manager = workspace_manager
        self._k = task_spec.judge_samples
        self._threshold = task_spec.judge_disagreement_threshold
        self._temperatures = temperatures or _default_temperatures(self._k)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        subtask: SubTask,
        workspace: IsolatedWorkspace,
        sub_result: SubAgentResult,
    ) -> tuple[TierResult, LLMJudgeDetail]:
        """Run K judge calls, aggregate, and emit (tier, detail) pair.

        Passing requires:
        - ``mean_score >= _MEAN_SCORE_THRESHOLD`` (0.6)
        - at least ``ceil(K/2) + 1 when even, (K+1)//2 when odd`` samples
          voting ``passes_requirements=True`` (strict majority)
        - ``not is_uncertain`` (disagreement <= threshold)
        """
        start = time.perf_counter()

        prompt = self._build_user_prompt(subtask, sub_result)

        # Fan out K calls in parallel. Use TaskGroup so an exception in
        # any sample cancels the siblings cleanly. ``except*`` can't hold
        # a ``return``, so we catch the ExceptionGroup directly and
        # unwrap nested ExceptionGroups to surface a meaningful leaf error
        # (TaskGroup may wrap children inside sub-groups, in which case
        # ``excgroup.exceptions[0]`` is itself a group, not a useful
        # error to render in logs / tier details).
        sample_results: list[JudgeOutput] = []
        total_cost = 0.0
        sampling_error: BaseException | None = None
        try:
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(self._single_judge_call(prompt, temp))
                    for temp in self._temperatures[: self._k]
                ]
            for t in tasks:
                output, meta = t.result()
                sample_results.append(output)
                total_cost += meta.cost
        except BaseExceptionGroup as excgroup:
            sampling_error = _first_leaf_exception(excgroup)

        if sampling_error is not None:
            _logger.warning(
                "judge_sample_failed",
                subtask_id=subtask.subtask_id,
                error=sampling_error.__class__.__name__,
                message=str(sampling_error),
            )
            return _fail_tier(
                f"judge sampling failed: {sampling_error}", start, total_cost
            ), LLMJudgeDetail(
                samples=[s.score for s in sample_results],
                mean_score=0.0,
                disagreement=0.0,
                is_uncertain=True,
                judge_model=self._judge_model_name(),
            )

        scores = [s.score for s in sample_results]
        passes = [s.passes_requirements for s in sample_results]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        disagreement = compute_disagreement(scores, passes)
        is_uncertain = disagreement > self._threshold

        majority_threshold = self._k // 2 + 1
        majority_pass = sum(passes) >= majority_threshold

        passed = mean_score >= _MEAN_SCORE_THRESHOLD and majority_pass and not is_uncertain

        detail = LLMJudgeDetail(
            samples=scores,
            mean_score=mean_score,
            disagreement=disagreement,
            is_uncertain=is_uncertain,
            judge_model=self._judge_model_name(),
        )
        tier = TierResult(
            tier="llm_judge",
            passed=passed,
            details=_format_details(sample_results, mean_score, disagreement, is_uncertain),
            latency_ms=_elapsed_ms(start),
            cost_usd=total_cost,
        )
        _logger.info(
            "judge_verdict",
            subtask_id=subtask.subtask_id,
            passed=passed,
            mean_score=round(mean_score, 3),
            disagreement=round(disagreement, 3),
            is_uncertain=is_uncertain,
            samples=len(scores),
            cost=total_cost,
        )
        return tier, detail

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _single_judge_call(
        self, prompt_context: str, temperature: float
    ) -> tuple[JudgeOutput, LLMCallMetadata]:
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_context},
        ]
        return await self._llm.call_structured(
            role="judge",
            messages=messages,
            output_schema=JudgeOutput,
            temperature=temperature,
        )

    def _build_user_prompt(self, subtask: SubTask, sub_result: SubAgentResult) -> str:
        originals = self._collect_original_files(subtask)
        return JUDGE_USER_PROMPT_TEMPLATE.format(
            subtask_id=subtask.subtask_id,
            description=subtask.description,
            writes=", ".join(subtask.writes) or "(none)",
            rationale=sub_result.rationale or "(none)",
            diff=sub_result.diff,
            formatted_original_files=originals,
        )

    def _collect_original_files(self, subtask: SubTask) -> str:
        """Render the pre-diff content of each ``writes`` file.

        Uses :meth:`WorkspaceManager.get_pre_patch_content` (M3). Files
        that don't exist in ``main/`` yet (sub-agent is creating them)
        render as ``(new file)``. If no workspace manager was provided
        we emit ``(unavailable)`` so the judge still has some context.
        """
        if self._workspace_manager is None:
            return "(unavailable: no workspace manager wired to judge)"
        if not subtask.writes:
            return "(no files declared in writes)"

        blocks: list[str] = []
        total_bytes = 0
        for rel in subtask.writes:
            try:
                content = self._workspace_manager.get_pre_patch_content(rel)
            except Exception as exc:
                blocks.append(f"## {rel}\n(error: {exc})")
                continue
            if content is None:
                blocks.append(f"## {rel}\n(new file — does not exist before diff)")
                continue
            if len(content) > _MAX_ORIGINAL_FILES_BYTES:
                content = content[:_MAX_ORIGINAL_FILES_BYTES] + "\n... (truncated)"
            body = f"## {rel}\n```\n{content}\n```"
            if total_bytes + len(body) > _MAX_ORIGINAL_FILES_TOTAL:
                blocks.append("(remaining files omitted to stay within prompt budget)")
                break
            blocks.append(body)
            total_bytes += len(body)
        return "\n\n".join(blocks)

    def _judge_model_name(self) -> str:
        try:
            return self._llm.config.models["judge"].name
        except KeyError:
            return "unknown"


# ---------------------------------------------------------------------------
# Pure helpers (exposed for direct testing)
# ---------------------------------------------------------------------------


def compute_disagreement(scores: list[float], passes: list[bool]) -> float:
    """Combined disagreement metric: 0.5·stdev + 0.5·binary-disagreement.

    * Continuous signal: standard deviation of the sample scores. If we have
      fewer than two samples the stdev term is zero.
    * Binary signal: how far the pass/fail vote is from unanimous. Zero when
      everyone agrees, one when it is a dead 50/50 split.

    The 0.5 / 0.5 weighting is a reasonable default — either axis alone would
    miss a class of failures (unanimous pass with wide scores, or matching
    scores with split votes).
    """
    if not scores:
        return 0.0
    std = statistics.stdev(scores) if len(scores) >= 2 else 0.0
    pass_rate = sum(passes) / len(passes) if passes else 0.0
    binary_disagreement = 1.0 - abs(pass_rate - 0.5) * 2
    return 0.5 * std + 0.5 * binary_disagreement


def _default_temperatures(k: int) -> list[float]:
    """Pick a reasonable spread of temperatures for K samples."""
    base = [0.1, 0.5, 0.9, 0.3, 0.7]
    if k <= len(base):
        return base[:k]
    # Extend by cycling if someone asks for more than we hard-coded.
    return [base[i % len(base)] for i in range(k)]


def _format_details(
    samples: list[JudgeOutput],
    mean_score: float,
    disagreement: float,
    is_uncertain: bool,
) -> str:
    lines = [
        f"mean_score={mean_score:.2f} disagreement={disagreement:.2f} is_uncertain={is_uncertain}",
    ]
    for i, s in enumerate(samples):
        issues = f" issues={s.detected_issues}" if s.detected_issues else ""
        lines.append(f"  sample[{i}] score={s.score:.2f} pass={s.passes_requirements}{issues}")
    return "\n".join(lines)


def _fail_tier(details: str, start: float, cost: float) -> TierResult:
    return TierResult(
        tier="llm_judge",
        passed=False,
        details=details,
        latency_ms=_elapsed_ms(start),
        cost_usd=cost,
    )


def _elapsed_ms(start: float) -> int:
    return max(0, int((time.perf_counter() - start) * 1000))


def _first_leaf_exception(eg: BaseExceptionGroup[BaseException]) -> BaseException:
    """Recursively unwrap to find the first leaf (non-group) exception."""
    for exc in eg.exceptions:
        if isinstance(exc, BaseExceptionGroup):
            return _first_leaf_exception(exc)
        return exc
    return eg  # empty group, degenerate case


__all__ = [
    "JUDGE_SYSTEM_PROMPT",
    "JUDGE_USER_PROMPT_TEMPLATE",
    "LLMJudgeVerifier",
    "compute_disagreement",
]
