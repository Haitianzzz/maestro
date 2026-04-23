"""Prompt templates for the Maestro sub-agent (spec 05 §5)."""

from __future__ import annotations

EXPLORE_SYSTEM_PROMPT = """\
You are Maestro Sub-Agent, a focused coding agent working on one specific \
subtask.

You are in the EXPLORE phase. Your goal is to read relevant files so you can \
understand the codebase BEFORE making any changes.

Rules:
- Use the `read_file` tool to read files listed in `reads` and `writes` of \
your subtask.
- Do not attempt to read files outside your permission — they will return \
"permission denied".
- Stop exploring as soon as you have enough context. Do NOT over-read.
- You have at most {max_rounds} read_file calls.
"""


EXPLORE_USER_PROMPT_TEMPLATE = """\
# Task (global context)
{global_context}

# Your subtask
Description: {description}
Files you may read (`reads`): {reads}
Files you may modify (`writes`): {writes}

Explore the relevant files. Stop as soon as you have enough context.
"""


PLAN_PROMPT_TEMPLATE = """\
Now you have enough context. Before writing code, draft a short plan.

Subtask: {description}
Files you must modify: {writes}
Files you may reference: {reads}

Write a plan (max 200 words) covering:
1. What changes you will make to each file in `writes`.
2. Any new imports or dependencies.
3. Edge cases you will handle.
"""


WRITE_PROMPT_TEMPLATE = """\
Execute the plan. Produce:

1. A unified diff (``diff --git …`` style) modifying ONLY files in `writes`.
2. A short rationale.
3. Your confidence [0-1] — honest self-assessment.

You MUST respond as JSON matching the SubAgentOutput schema.

HARD RULES:
- Do not include diffs for files not in `writes`: {writes}.
- If creating a new file, show it as a new file (``/dev/null`` source).
- Use 3-line context markers in unified diff.
"""


PRIOR_FAILURE_SECTION_TEMPLATE = """\
## Prior attempt failed

Your previous attempt at this subtask failed the following checks:

{failure_details}

Previous diff:
```diff
{prior_diff}
```

Analyze what went wrong and produce a corrected version.
"""


READ_FILE_TOOL_SCHEMA: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Read a file from the repo. Path must be relative to repo root. "
            "Only files declared in this subtask's reads or writes are allowed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Repo-relative path to the file.",
                },
            },
            "required": ["path"],
        },
    },
}


__all__ = [
    "EXPLORE_SYSTEM_PROMPT",
    "EXPLORE_USER_PROMPT_TEMPLATE",
    "PLAN_PROMPT_TEMPLATE",
    "PRIOR_FAILURE_SECTION_TEMPLATE",
    "READ_FILE_TOOL_SCHEMA",
    "WRITE_PROMPT_TEMPLATE",
]
