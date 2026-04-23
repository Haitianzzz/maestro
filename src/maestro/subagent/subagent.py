"""Sub-agent: executes one subtask in an isolated workspace (spec 05).

Architecture: a **controlled 3-phase loop** — Explore → Plan → Write.

* **Explore** (phase 1): the LLM may call ``read_file`` at most
  :attr:`SubAgent.max_explore_rounds` times. Each tool call goes through
  :class:`~maestro.sandbox.workspace.IsolatedWorkspace.read_file` which
  enforces the ``reads + writes`` permission and safe-file rules.
* **Plan** (phase 2): one free-text LLM call producing a short internal
  plan. This is pure CoT — we do not feed it downstream, we keep it in
  the message history so the Write phase can reference it.
* **Write** (phase 3): one structured LLM call producing
  :class:`~maestro.models.SubAgentOutput`. The diff is validated against
  ``writes`` before being applied; permission violations reject the whole
  patch without modifying the isolated workspace.

Failure recovery: if a prior attempt failed verification, the scheduler
passes the ``prior_attempt`` and ``prior_failure`` into the factory. Their
contents are inserted into the Write prompt as a "prior attempt failed"
section (spec 05 §5.4) so the LLM sees why it must correct course.

All output is wrapped in :class:`~maestro.models.SubAgentResult` carrying
framework metadata (tokens, latency, retry_count, model_used) that the
scheduler needs for reporting and the benchmark harness needs for cost
accounting.
"""

from __future__ import annotations

import time
from typing import Any

from maestro.llm.client import LLMClient
from maestro.models import (
    SubAgentOutput,
    SubAgentResult,
    SubTask,
    VerificationResult,
)
from maestro.sandbox.workspace import IsolatedWorkspace
from maestro.utils.logging import get_logger
from maestro.utils.time import utcnow

from .prompts import (
    EXPLORE_SYSTEM_PROMPT,
    EXPLORE_USER_PROMPT_TEMPLATE,
    PLAN_PROMPT_TEMPLATE,
    PRIOR_FAILURE_SECTION_TEMPLATE,
    READ_FILE_TOOL_SCHEMA,
    WRITE_PROMPT_TEMPLATE,
)

_logger = get_logger("maestro.subagent")

_DEFAULT_MAX_EXPLORE_ROUNDS = 5


class SubAgent:
    """Execute one :class:`~maestro.models.SubTask` in an isolated workspace."""

    def __init__(
        self,
        subtask: SubTask,
        workspace: IsolatedWorkspace,
        llm_client: LLMClient,
        *,
        prior_attempt: SubAgentResult | None = None,
        prior_failure: VerificationResult | None = None,
        max_explore_rounds: int = _DEFAULT_MAX_EXPLORE_ROUNDS,
    ) -> None:
        self._subtask = subtask
        self._workspace = workspace
        self._llm = llm_client
        self._prior = prior_attempt
        self._prior_failure = prior_failure
        self._max_explore_rounds = max_explore_rounds

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, global_context: str) -> SubAgentResult:
        """Run the 3-phase loop and return the wrapped result."""
        start = time.perf_counter()
        tokens_in = 0
        tokens_out = 0

        try:
            explore_messages, round_tokens = await self._explore(global_context)
            tokens_in += round_tokens[0]
            tokens_out += round_tokens[1]

            plan_text, plan_in, plan_out = await self._plan(explore_messages)
            tokens_in += plan_in
            tokens_out += plan_out

            output, write_in, write_out = await self._write(explore_messages, plan_text)
            tokens_in += write_in
            tokens_out += write_out
        except Exception as exc:
            _logger.warning(
                "subagent_llm_failed",
                subtask_id=self._subtask.subtask_id,
                error=exc.__class__.__name__,
                message=str(exc),
            )
            return self._build_failed_result(
                status="failed",
                rationale=f"LLM pipeline failed: {exc}",
                tokens_input=tokens_in,
                tokens_output=tokens_out,
                start=start,
            )

        # Validate: does the diff touch any file outside writes?
        illegal = _find_illegal_writes(output, self._subtask.writes)
        if illegal:
            return self._build_failed_result(
                status="rejected",
                rationale=(
                    f"Sub-agent attempted to modify files outside writes: {sorted(illegal)}"
                ),
                tokens_input=tokens_in,
                tokens_output=tokens_out,
                start=start,
                diff=output.diff,
                modified_files=output.modified_files,
                confidence=output.confidence,
            )

        # Apply the diff to the isolated workspace.
        apply_ok, apply_err = self._workspace.apply_diff(output.diff)
        if not apply_ok:
            return self._build_failed_result(
                status="rejected",
                rationale=f"Diff apply failed: {apply_err}",
                tokens_input=tokens_in,
                tokens_output=tokens_out,
                start=start,
                diff=output.diff,
                modified_files=output.modified_files,
                confidence=output.confidence,
            )

        latency_ms = _elapsed_ms(start)
        retry_count = _next_retry_count(self._prior)
        return SubAgentResult(
            subtask_id=self._subtask.subtask_id,
            status="success" if output.status == "success" else "failed",
            diff=output.diff,
            modified_files=list(output.modified_files),
            rationale=output.rationale,
            confidence=output.confidence,
            retry_count=retry_count,
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            latency_ms=latency_ms,
            model_used=self._resolve_model_name(),
            created_at=utcnow(),
        )

    # ------------------------------------------------------------------
    # Phase 1: Explore
    # ------------------------------------------------------------------

    async def _explore(self, global_context: str) -> tuple[list[dict[str, Any]], tuple[int, int]]:
        system = EXPLORE_SYSTEM_PROMPT.format(max_rounds=self._max_explore_rounds)
        user = EXPLORE_USER_PROMPT_TEMPLATE.format(
            global_context=global_context,
            description=self._subtask.description,
            reads=", ".join(self._subtask.reads) or "(none)",
            writes=", ".join(self._subtask.writes) or "(none)",
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        tokens_in = 0
        tokens_out = 0

        for round_num in range(self._max_explore_rounds):
            response, meta = await self._llm.call_with_tools(
                role="subagent",
                messages=messages,
                tools=[READ_FILE_TOOL_SCHEMA],
                tool_choice="auto",
                temperature=0.2,
            )
            tokens_in += meta.tokens_input
            tokens_out += meta.tokens_output

            if not response.tool_calls:
                # Model chose to stop exploring.
                if response.text:
                    messages.append({"role": "assistant", "content": response.text})
                break

            # Record the assistant message that issued the tool calls.
            messages.append(
                {
                    "role": "assistant",
                    "content": response.text or None,
                    "tool_calls": [
                        {
                            "id": tc.call_id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments_raw or "{}",
                            },
                        }
                        for tc in response.tool_calls
                    ],
                }
            )

            for tc in response.tool_calls:
                result_content = self._dispatch_tool_call(tc.name, tc.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.call_id,
                        "content": result_content,
                    }
                )

            _logger.debug(
                "subagent_explore_round",
                subtask_id=self._subtask.subtask_id,
                round=round_num,
                tool_calls=len(response.tool_calls),
            )

        return messages, (tokens_in, tokens_out)

    def _dispatch_tool_call(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool call and return a string for the tool-result message."""
        if name != "read_file":
            return f"error: unknown tool {name!r}"
        path = arguments.get("path")
        if not isinstance(path, str) or not path:
            return "error: read_file requires a 'path' string argument"
        try:
            content = self._workspace.read_file(path)
        except PermissionError as exc:
            return f"permission denied: {exc}"
        except FileNotFoundError:
            return f"not found: {path}"
        except Exception as exc:
            return f"error: {exc}"
        # Cap very large reads — spec 05 §6 read_file should surface content.
        if len(content) > 50_000:
            content = content[:50_000] + "\n... (truncated)"
        return content

    # ------------------------------------------------------------------
    # Phase 2: Plan
    # ------------------------------------------------------------------

    async def _plan(self, messages: list[dict[str, Any]]) -> tuple[str, int, int]:
        plan_prompt = PLAN_PROMPT_TEMPLATE.format(
            description=self._subtask.description,
            writes=", ".join(self._subtask.writes) or "(none)",
            reads=", ".join(self._subtask.reads) or "(none)",
        )
        plan_messages = [*messages, {"role": "user", "content": plan_prompt}]
        plan_text, meta = await self._llm.call_text(
            role="subagent",
            messages=plan_messages,
            temperature=0.3,
        )
        # Mutate the caller's message history so Phase 3 sees the plan.
        messages.append({"role": "user", "content": plan_prompt})
        messages.append({"role": "assistant", "content": plan_text})
        return plan_text, meta.tokens_input, meta.tokens_output

    # ------------------------------------------------------------------
    # Phase 3: Write
    # ------------------------------------------------------------------

    async def _write(
        self, messages: list[dict[str, Any]], plan_text: str
    ) -> tuple[SubAgentOutput, int, int]:
        prompt = WRITE_PROMPT_TEMPLATE.format(
            writes=", ".join(self._subtask.writes) or "(none)",
        )
        if self._prior_failure is not None and self._prior is not None:
            prompt = (
                PRIOR_FAILURE_SECTION_TEMPLATE.format(
                    failure_details=_format_failure(self._prior_failure),
                    prior_diff=self._prior.diff,
                )
                + "\n"
                + prompt
            )

        write_messages = [*messages, {"role": "user", "content": prompt}]
        output, meta = await self._llm.call_structured(
            role="subagent",
            messages=write_messages,
            output_schema=SubAgentOutput,
            temperature=0.2,
        )
        return output, meta.tokens_input, meta.tokens_output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_model_name(self) -> str:
        try:
            return self._llm.config.models["subagent"].name
        except KeyError:
            return "unknown"

    def _build_failed_result(
        self,
        *,
        status: str,
        rationale: str,
        tokens_input: int,
        tokens_output: int,
        start: float,
        diff: str = "",
        modified_files: list[str] | None = None,
        confidence: float = 0.0,
    ) -> SubAgentResult:
        return SubAgentResult(
            subtask_id=self._subtask.subtask_id,
            status=status,  # type: ignore[arg-type]
            diff=diff,
            modified_files=list(modified_files or []),
            rationale=rationale,
            confidence=confidence,
            retry_count=_next_retry_count(self._prior),
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            latency_ms=_elapsed_ms(start),
            model_used=self._resolve_model_name(),
            created_at=utcnow(),
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class SubAgentFactory:
    """Thin factory wrapping :class:`SubAgent` construction (spec 05 §3)."""

    def __init__(self, *, max_explore_rounds: int = _DEFAULT_MAX_EXPLORE_ROUNDS) -> None:
        self._max_explore_rounds = max_explore_rounds

    def create(
        self,
        subtask: SubTask,
        workspace: IsolatedWorkspace,
        llm_client: LLMClient,
        *,
        prior_attempt: SubAgentResult | None = None,
        prior_failure: VerificationResult | None = None,
    ) -> SubAgent:
        return SubAgent(
            subtask,
            workspace,
            llm_client,
            prior_attempt=prior_attempt,
            prior_failure=prior_failure,
            max_explore_rounds=self._max_explore_rounds,
        )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _elapsed_ms(start: float) -> int:
    return max(0, int((time.perf_counter() - start) * 1000))


def _next_retry_count(prior: SubAgentResult | None) -> int:
    if prior is None:
        return 0
    return prior.retry_count + 1


def _find_illegal_writes(output: SubAgentOutput, writes: list[str]) -> set[str]:
    """Return the set of modified files that are not in ``writes``."""
    writes_set = set(writes)
    touched = set(output.modified_files)
    # Pull paths straight out of the diff headers too, so a lying
    # modified_files list can't bypass the check.
    for line in output.diff.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            path = line.split(maxsplit=1)[1] if len(line.split()) > 1 else ""
            path = _strip_diff_prefix(path)
            if path and path != "/dev/null":
                touched.add(path)
    return {p for p in touched if p and p not in writes_set}


def _strip_diff_prefix(path: str) -> str:
    """Strip ``a/``/``b/`` markers that unified diffs emit."""
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def _format_failure(result: VerificationResult) -> str:
    lines = [f"- overall_passed: {result.overall_passed}"]
    for tier in result.tiers:
        lines.append(f"- tier={tier.tier} passed={tier.passed} details={tier.details[:300]!r}")
    if result.judge_detail is not None:
        lines.append(
            f"- judge mean_score={result.judge_detail.mean_score:.2f} "
            f"disagreement={result.judge_detail.disagreement:.2f} "
            f"uncertain={result.judge_detail.is_uncertain}"
        )
    return "\n".join(lines)


__all__ = ["SubAgent", "SubAgentFactory"]
