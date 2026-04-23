"""Evaluate Maestro's diff against a benchmark task (spec 09 §3.3).

Flow:

1. Copy the task's ``before/`` directory to a scratch path.
2. Apply Maestro's ``final_diff`` using the same ``IsolatedWorkspace``
   machinery the sub-agent uses (spec 07 §5). We construct a synthetic
   ``SubTask`` whose ``writes`` white-list is the diff's own file set,
   so the workspace accepts it.
3. Run the task's ``failing_tests`` via pytest. If they pass, the task
   is ``resolved``.
4. Compute a cheap ``patch_similarity`` against the ground-truth ``after/``
   directory (line-level Jaccard on ``git diff --no-index`` lines per
   spec 09 §3.3).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from unidiff import PatchSet

from maestro.sandbox.workspace import IsolatedWorkspace
from maestro.utils.logging import get_logger
from maestro.verifier.deterministic import run_subprocess

from .models import BenchmarkTask, EvalResult

_logger = get_logger("benchmark.evaluator")

_PYTEST_TIMEOUT_S = 180.0


async def evaluate(
    task: BenchmarkTask,
    before_dir: Path,
    after_dir: Path,
    final_diff: str,
) -> EvalResult:
    """Run the full evaluation. ``before_dir``/``after_dir`` are not mutated.

    A fresh copy of ``before_dir`` is made in a scratch tempdir and all
    mutations (``apply_diff``, pytest output, etc.) land there.
    """
    if not final_diff.strip():
        return EvalResult(
            resolved=False,
            patch_similarity=0.0,
            files_modified_match=False,
            extra_files_modified=0,
            error="empty diff from Maestro",
        )

    with tempfile.TemporaryDirectory(prefix="bench-eval-") as tmpdir:
        eval_dir = Path(tmpdir) / "repo"
        shutil.copytree(before_dir, eval_dir)

        apply_ok, apply_err = _apply_diff_via_workspace(eval_dir, final_diff)
        if not apply_ok:
            return EvalResult(
                resolved=False,
                patch_similarity=await _patch_similarity(before_dir, after_dir, final_diff),
                files_modified_match=False,
                extra_files_modified=0,
                error=f"diff apply failed: {apply_err}",
            )

        pytest_result = await run_subprocess(
            ["pytest", "-x", "--tb=short", *task.failing_tests],
            cwd=eval_dir,
            timeout=_PYTEST_TIMEOUT_S,
        )
        resolved = pytest_result.ok
        err: str | None = None
        if pytest_result.timed_out:
            err = "pytest timed out"
        elif not resolved:
            err = pytest_result.combined[-500:]

        touched = _diff_targets(final_diff)
        expected = set(task.expected_modified_files)
        files_match = bool(expected) and touched == expected
        extras = max(0, len(touched - expected))

    similarity = await _patch_similarity(before_dir, after_dir, final_diff)
    return EvalResult(
        resolved=resolved,
        patch_similarity=similarity,
        files_modified_match=files_match,
        extra_files_modified=extras,
        error=None if resolved else err,
    )


# ---------------------------------------------------------------------------
# Diff application
# ---------------------------------------------------------------------------


def _apply_diff_via_workspace(eval_dir: Path, diff: str) -> tuple[bool, str | None]:
    """Reuse :class:`IsolatedWorkspace.apply_diff` with a synthetic SubTask.

    The synthetic subtask's ``writes`` list is the diff's own targets, so
    the workspace's writes-whitelist check always passes for legitimately
    parsed diffs. We still get the unified-diff parser's hunk-mismatch
    detection and atomic failure semantics.
    """
    from maestro.models import SubTask  # local import to dodge package cycles

    targets = _diff_targets(diff)
    subtask = SubTask(
        subtask_id="eval-synthetic",
        description="benchmark eval",
        writes=list(targets),
    )
    iso = IsolatedWorkspace(eval_dir, subtask)
    return iso.apply_diff(diff)


def _diff_targets(diff: str) -> set[str]:
    try:
        patch_set = PatchSet.from_string(diff)
    except Exception:
        return set()
    out: set[str] = set()
    for patched in patch_set:
        target = patched.target_file
        if patched.is_removed_file:
            target = patched.source_file
        if target.startswith(("a/", "b/")):
            target = target[2:]
        if target and target != "/dev/null":
            out.add(target)
    return out


# ---------------------------------------------------------------------------
# Patch similarity (line-level Jaccard)
# ---------------------------------------------------------------------------


async def _patch_similarity(before_dir: Path, after_dir: Path, final_diff: str) -> float:
    """Jaccard similarity between the agent's and ground-truth diff lines.

    Both diffs are normalised to a set of ``(+line)`` / ``(-line)`` tokens
    so whitespace-only differences and header metadata don't dominate.
    """
    gt_diff = await _git_diff(before_dir, after_dir)
    agent_lines = _extract_changed_lines(final_diff)
    gt_lines = _extract_changed_lines(gt_diff)
    if not agent_lines and not gt_lines:
        return 1.0
    if not agent_lines or not gt_lines:
        return 0.0
    intersection = len(agent_lines & gt_lines)
    union = len(agent_lines | gt_lines)
    return intersection / union if union else 0.0


async def _git_diff(before: Path, after: Path) -> str:
    """Run ``git diff --no-index`` asynchronously via :func:`run_subprocess`.

    ``rc == 1`` is normal — it just means the diff is non-empty (same
    convention as :meth:`maestro.sandbox.workspace.WorkspaceManager.get_final_diff`).
    Only ``rc >= 2`` is a real failure.
    """
    result = await run_subprocess(
        ["git", "diff", "--no-index", "--", str(before), str(after)],
        cwd=before.parent if before.parent.exists() else before,
        timeout=30.0,
    )
    if result.returncode >= 2:
        _logger.warning("eval_git_diff_failed", stderr=result.stderr[-200:])
        return ""
    return result.stdout


def _extract_changed_lines(diff_text: str) -> set[str]:
    lines: set[str] = set()
    for raw in diff_text.splitlines():
        if raw.startswith("+++ ") or raw.startswith("--- "):
            continue
        if raw.startswith(("+", "-")):
            stripped = raw[1:].strip()
            if stripped:
                lines.add(f"{raw[0]}{stripped}")
    return lines


__all__ = ["evaluate"]
