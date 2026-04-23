"""Prompt templates for the Maestro planner (spec 03 §4)."""

from __future__ import annotations

PLANNER_SYSTEM_PROMPT = """\
You are Maestro Planner, a coding task decomposition agent.

Your job is to decompose a user's high-level programming task into a DAG of
subtasks that can be executed by independent sub-agents.

HARD RULES:
1. Each subtask MUST declare `reads` and `writes` as file paths relative to \
the repo root.
2. `writes` MUST be precise — sub-agents are only allowed to modify files in \
`writes`. If you leave a file out, the sub-agent cannot modify it.
3. Use `depends_on_indices` ONLY for true dependencies (e.g., file A is \
imported by file B; A must be written first). Refer to earlier subtasks by \
their integer `index`.
4. Subtasks with no file-write conflict AND no dependency SHOULD be \
parallelizable.
5. Keep subtask granularity at "one logical feature unit" level — not \
per-function, not per-file unless a file is a whole feature.
6. Output MUST conform to the provided JSON schema strictly.

Produce a clear `planning_rationale` describing your decomposition strategy.
"""


PLANNER_USER_PROMPT_TEMPLATE = """\
# Task
{task_description}

# Repo structure
{file_tree}

# Key files
{key_files_formatted}

# File signatures (for context)
{signatures_formatted}

# Instructions
Decompose this task into 2-8 subtasks. Prioritize parallelism when possible.
Return JSON matching the PlannerLLMOutput schema.
{hint_section}
"""


PLANNER_RETRY_PROMPT_TEMPLATE = """\
Your previous plan had the following issues:
{issues}

Please fix every issue and re-output JSON matching the PlannerLLMOutput schema.
"""


def format_key_files(key_files: dict[str, str]) -> str:
    if not key_files:
        return "(none)"
    blocks: list[str] = []
    for path, content in sorted(key_files.items()):
        blocks.append(f"## {path}\n```\n{content}\n```")
    return "\n\n".join(blocks)


def format_signatures(signatures: dict[str, str]) -> str:
    if not signatures:
        return "(none)"
    blocks: list[str] = []
    for path, sigs in sorted(signatures.items()):
        blocks.append(f"### {path}\n{sigs}")
    return "\n\n".join(blocks)


def format_hint_section(hint: list[str] | None) -> str:
    if not hint:
        return ""
    joined = "\n".join(f"- {p}" for p in hint)
    return (
        "\n# Files the user suggests you focus on\n"
        "These are suggestions — you may still touch other files if needed:\n"
        f"{joined}\n"
    )


__all__ = [
    "PLANNER_RETRY_PROMPT_TEMPLATE",
    "PLANNER_SYSTEM_PROMPT",
    "PLANNER_USER_PROMPT_TEMPLATE",
    "format_hint_section",
    "format_key_files",
    "format_signatures",
]
