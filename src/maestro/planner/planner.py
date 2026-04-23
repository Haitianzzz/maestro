"""Planner — decomposes a user task into a validated :class:`TaskDAG` (spec 03).

The planner runs the following pipeline:

1. Scan the repo into a :class:`RepoContext` (file tree, key files, AST
   signatures) budgeted by ``max_context_tokens``.
2. Build the planner prompt (system + user message) with the task
   description plus repo context.
3. Call the LLM for a structured :class:`PlannerLLMOutput`.
4. Post-process: translate integer ``index`` identifiers to canonical
   ``{task_id}-{idx:03d}`` subtask ids (M2), emit :class:`SubTask` instances.
5. Validate: DAG acyclicity, id uniqueness, path safety, file existence.
6. If validation fails, feed the error list back to the LLM and retry up to
   ``max_retries`` times. Exhaustion raises :class:`PlanningError`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from maestro.llm.client import LLMClient
from maestro.models import (
    PlannerLLMOutput,
    PlannerLLMSubTask,
    PlannerOutput,
    SubTask,
    TaskDAG,
    TaskSpec,
    generate_subtask_id,
)
from maestro.utils.logging import get_logger

from .prompts import (
    PLANNER_RETRY_PROMPT_TEMPLATE,
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_PROMPT_TEMPLATE,
    format_hint_section,
    format_key_files,
    format_signatures,
)
from .repo_scanner import RepoContext, RepoScanner

_logger = get_logger("maestro.planner")

# Paths we never allow the planner to write to, regardless of repo layout.
_FORBIDDEN_PREFIXES = (".git/", ".venv/", "venv/", "node_modules/", ".maestro/")

# Max number of retries after the initial LLM call.
_DEFAULT_MAX_RETRIES = 2


class PlanningError(Exception):
    """Raised when the planner cannot produce a valid plan after retries."""


class Planner:
    """Decompose a high-level task into a DAG of subtasks."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        max_context_tokens: int = 60_000,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> None:
        self._llm = llm_client
        self._max_context_tokens = max_context_tokens
        self._max_retries = max_retries

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def plan(self, task_spec: TaskSpec) -> TaskDAG:
        """Main entry. Raises :class:`PlanningError` on unrecoverable failure."""
        scanner = RepoScanner(task_spec.repo_path, self._max_context_tokens)
        repo_ctx = scanner.scan(task_spec.target_files_hint)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(task_spec, repo_ctx)},
        ]

        attempt = 0
        last_issues: list[str] = []
        while attempt <= self._max_retries:
            llm_output, _meta = await self._llm.call_structured(
                role="planner",
                messages=messages,
                output_schema=PlannerLLMOutput,
                temperature=0.2,
            )

            post: PlannerOutput | None
            try:
                post = _post_process(task_spec.task_id, llm_output)
                issues = _validate_plan(post, task_spec.repo_path)
            except PlanningError as exc:
                # Structural failure that cannot be expressed as issues
                # (e.g. the LLM returned 0 subtasks). No point retrying
                # beyond one feedback round.
                issues = [str(exc)]
                post = None

            if not issues and post is not None:
                _logger.info(
                    "plan_accepted",
                    task_id=task_spec.task_id,
                    subtasks=len(post.subtasks),
                    attempt=attempt,
                )
                return TaskDAG(
                    task_id=task_spec.task_id,
                    subtasks=post.subtasks,
                    global_context=post.global_context,
                )

            _logger.warning(
                "plan_rejected",
                task_id=task_spec.task_id,
                attempt=attempt,
                issue_count=len(issues),
            )
            last_issues = issues
            if attempt >= self._max_retries:
                break

            # Feed the error list back and ask the LLM to fix it.
            messages.append({"role": "assistant", "content": llm_output.model_dump_json()})
            messages.append(
                {
                    "role": "user",
                    "content": PLANNER_RETRY_PROMPT_TEMPLATE.format(
                        issues="\n".join(f"- {msg}" for msg in issues),
                    ),
                }
            )
            attempt += 1

        raise PlanningError(
            "Planner failed after "
            f"{self._max_retries + 1} attempts. Last issues: " + "; ".join(last_issues)
        )


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def _build_user_prompt(task_spec: TaskSpec, ctx: RepoContext) -> str:
    return PLANNER_USER_PROMPT_TEMPLATE.format(
        task_description=task_spec.description,
        file_tree=ctx.file_tree,
        key_files_formatted=format_key_files(ctx.key_files),
        signatures_formatted=format_signatures(ctx.file_signatures),
        hint_section=format_hint_section(task_spec.target_files_hint),
    )


# ---------------------------------------------------------------------------
# Post-processing & validation
# ---------------------------------------------------------------------------


def _post_process(task_id: str, llm_output: PlannerLLMOutput) -> PlannerOutput:
    """Translate LLM integer indices into canonical SubTask instances (M2)."""
    if not llm_output.subtasks:
        raise PlanningError("LLM produced an empty subtask list")

    # LLM emits integer ``index``; de-duplicate and map to canonical ids.
    indices_seen: set[int] = set()
    index_to_id: dict[int, str] = {}
    canonical: list[PlannerLLMSubTask] = []
    for sub in llm_output.subtasks:
        if sub.index in indices_seen:
            raise PlanningError(f"Planner output contains duplicate index {sub.index}")
        indices_seen.add(sub.index)
        canonical.append(sub)

    # Sort by index so ids are stable.
    canonical.sort(key=lambda s: s.index)
    for ordinal, sub in enumerate(canonical):
        index_to_id[sub.index] = generate_subtask_id(task_id, ordinal)

    subtasks: list[SubTask] = []
    for ordinal, sub in enumerate(canonical):
        deps: list[str] = []
        for dep_idx in sub.depends_on_indices:
            if dep_idx not in index_to_id:
                raise PlanningError(f"Subtask index={sub.index} depends on unknown index={dep_idx}")
            deps.append(index_to_id[dep_idx])
        subtasks.append(
            SubTask(
                subtask_id=generate_subtask_id(task_id, ordinal),
                description=sub.description,
                reads=list(sub.reads),
                writes=list(sub.writes),
                depends_on=deps,
                priority=sub.priority,
                estimated_difficulty=sub.estimated_difficulty,
            )
        )
    return PlannerOutput(
        subtasks=subtasks,
        global_context=llm_output.global_context,
        planning_rationale=llm_output.planning_rationale,
    )


def _validate_plan(plan: PlannerOutput, repo_root: Path) -> list[str]:
    """Return a list of human-readable issues. Empty list = plan is valid."""
    issues: list[str] = []

    # Id uniqueness + DAG shape are enforced by TaskDAG.validate_dag, but we
    # build a lightweight dag first so we can surface *all* issues instead of
    # bailing at the first one.
    ids = {s.subtask_id for s in plan.subtasks}
    for subtask in plan.subtasks:
        for dep in subtask.depends_on:
            if dep not in ids:
                issues.append(f"Subtask {subtask.subtask_id} depends on missing id {dep!r}")
            if dep == subtask.subtask_id:
                issues.append(f"Subtask {subtask.subtask_id} cannot depend on itself")

    # Structural cycle check via validate_dag on a trial TaskDAG.
    try:
        TaskDAG(
            task_id="_tmp",
            subtasks=plan.subtasks,
            global_context=plan.global_context,
        ).validate_dag()
    except ValueError as exc:
        issues.append(str(exc))

    # Path checks for every subtask.
    for subtask in plan.subtasks:
        for p in subtask.writes:
            issues.extend(_check_write_path(subtask.subtask_id, p))
        writes_set = set(subtask.writes)
        for p in subtask.reads:
            issues.extend(_check_read_path(subtask.subtask_id, p, repo_root, writes_set))

    return issues


def _check_write_path(subtask_id: str, path: str) -> list[str]:
    issues: list[str] = []
    if not path or path.startswith("/"):
        issues.append(f"{subtask_id} writes absolute/empty path {path!r}")
    if ".." in Path(path).parts:
        issues.append(f"{subtask_id} writes path with parent traversal: {path!r}")
    if any(path.startswith(prefix) for prefix in _FORBIDDEN_PREFIXES):
        issues.append(f"{subtask_id} writes a forbidden path: {path!r}")
    return issues


def _check_read_path(
    subtask_id: str, path: str, repo_root: Path, writes_set: set[str]
) -> list[str]:
    issues: list[str] = []
    if not path or path.startswith("/"):
        issues.append(f"{subtask_id} reads absolute/empty path {path!r}")
        return issues
    if ".." in Path(path).parts:
        issues.append(f"{subtask_id} reads path with parent traversal: {path!r}")
        return issues
    if path in writes_set:
        # Sub-agent may read the writes file's pre-modification content; don't
        # require existence here.
        return issues
    if not (repo_root / path).exists():
        issues.append(f"{subtask_id} reads nonexistent file: {path!r}")
    return issues


__all__ = ["Planner", "PlanningError"]
