"""File-system-level sandbox / workspace manager (spec 07).

The :class:`WorkspaceManager` owns a single ``tempfile`` root directory and
two kinds of sub-directories inside it:

* ``main/`` — the cumulative workspace. Initialised as a copy of the user's
  source repo (minus vendored / build / VCS dirs) and progressively updated
  after each batch merges its successful patches.
* ``iso/<subtask_id>/`` — per-subtask isolated workspaces. Each isolated
  workspace is copied from the *current* ``main/`` (not the source repo) so
  that batch ``N`` sees the patches merged by batches ``0..N-1``.

Design decisions
----------------

* **No Docker.** DESIGN §3.7 rejects container isolation — container spin-up
  would dwarf the wall-clock parallelism wins. We rely on the fact that
  Maestro never runs untrusted code here.
* **Read permission is enforced at runtime**, not merely via prompt. A sub-agent
  that tries to read a file outside its ``reads + writes`` declaration gets a
  :class:`PermissionError`, which the sub-agent loop converts into a
  ``permission denied`` tool response (spec 05 §4.1).
* **Diff application validates ``writes`` up-front**: if the proposed diff
  touches a file not in the subtask's ``writes``, the workspace rejects the
  whole patch and returns ``(False, reason)`` — we never partially apply.
* **``create_isolated`` is async but the underlying ``shutil.copytree`` is
  synchronous** (M6). We wrap it in :func:`asyncio.to_thread` so the event
  loop isn't blocked for large repos.

``get_pre_patch_content`` (M3)
------------------------------
The LLM judge (spec 06 §5.6) needs the file content *before* the sub-agent's
diff was applied. At the moment the verifier runs, ``main/`` still holds the
pre-patch content for files within the batch (the batch's patches have not
been merged yet); the isolated workspace holds the post-patch content. We
therefore provide :meth:`WorkspaceManager.get_pre_patch_content` that reads
from ``main/``.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from unidiff import PatchSet

from maestro.models import SubAgentResult, SubTask
from maestro.utils.logging import get_logger

if TYPE_CHECKING:
    from types import TracebackType

_logger = get_logger("maestro.sandbox")

# Files we never copy into the main workspace; also reject reads/writes here.
_IGNORE_DIRS = (
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "*.pyc",
    "node_modules",
    ".tox",
    "dist",
    "build",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".maestro",
)

_SAFE_EXTENSIONS = frozenset(
    {".py", ".md", ".txt", ".cfg", ".toml", ".ini", ".yml", ".yaml", ".json", ".rst"}
)
_MAX_FILE_BYTES = 500_000

_FORBIDDEN_PREFIXES = (".git/", ".venv/", "venv/", "node_modules/", ".maestro/")


class WorkspaceError(Exception):
    """Base class for workspace-level failures."""


class MergeConflictError(WorkspaceError):
    """Raised when merging a sub-agent patch back into ``main/`` conflicts."""

    def __init__(self, *, file: str, reason: str) -> None:
        super().__init__(f"Merge conflict at {file!r}: {reason}")
        self.file = file
        self.reason = reason


@dataclass(frozen=True)
class MergeConflict:
    """One unresolved merge conflict (data class, no behaviour)."""

    subtask_id: str
    file: str
    reason: str


@dataclass(frozen=True)
class MergeReport:
    """Aggregate result of merging a batch's patches into ``main/``."""

    merged: list[str]
    conflicts: list[MergeConflict]


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------


def _strip_git_prefix(path: str) -> str:
    """Strip the ``a/``/``b/`` prefix that unified diffs use."""
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def _is_path_safe(relative: str) -> bool:
    """Reject absolute paths, parent-traversal, and blessed-bad dirs."""
    if not relative or relative.startswith("/"):
        return False
    if ".." in Path(relative).parts:
        return False
    return not any(relative.startswith(prefix) for prefix in _FORBIDDEN_PREFIXES)


def _is_safe_file(abs_path: Path) -> bool:
    if abs_path.suffix.lower() not in _SAFE_EXTENSIONS:
        return False
    try:
        size = abs_path.stat().st_size
    except OSError:
        return False
    return size <= _MAX_FILE_BYTES


def _ignore_patterns() -> Callable[[str, list[str]], Iterable[str]]:
    return shutil.ignore_patterns(*_IGNORE_DIRS)


# ---------------------------------------------------------------------------
# IsolatedWorkspace
# ---------------------------------------------------------------------------


class IsolatedWorkspace:
    """One sub-agent's isolated view of the repo."""

    def __init__(self, path: Path, subtask: SubTask) -> None:
        self._path = path
        self._subtask = subtask

    @property
    def path(self) -> Path:
        return self._path

    @property
    def subtask(self) -> SubTask:
        return self._subtask

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def read_file(self, relative_path: str) -> str:
        """Read a file from this workspace.

        Permission rule: ``relative_path`` must be in ``reads + writes``.
        Files outside that set — even if present on disk — are rejected.
        """
        if not _is_path_safe(relative_path):
            raise PermissionError(f"Unsafe path: {relative_path!r}")

        allowed = set(self._subtask.reads) | set(self._subtask.writes)
        if relative_path not in allowed:
            raise PermissionError(
                f"Sub-agent for {self._subtask.subtask_id} attempted to read "
                f"{relative_path!r} which is not in reads or writes."
            )

        abs_path = self._path / relative_path
        if not abs_path.exists():
            raise FileNotFoundError(relative_path)
        if not _is_safe_file(abs_path):
            raise PermissionError(f"{relative_path} is not a safe file (type or size)")

        return abs_path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Diff application
    # ------------------------------------------------------------------

    def apply_diff(self, diff_text: str) -> tuple[bool, str | None]:
        """Apply a unified diff to this workspace.

        Returns ``(True, None)`` on success, ``(False, reason)`` on failure.
        Failure modes: parse error, file-outside-writes, missing target,
        hunk mismatch. On failure the workspace is left unchanged.
        """
        if not diff_text.strip():
            return False, "Empty diff"

        try:
            patch_set = PatchSet.from_string(diff_text)
        except Exception as exc:
            return False, f"Failed to parse diff: {exc}"

        if not patch_set:
            return False, "Diff contained no hunks"

        writes_set = set(self._subtask.writes)

        # Validate all files up-front so we never partially apply.
        targets: list[tuple[Any, str]] = []
        for patched_file in patch_set:
            target = _strip_git_prefix(patched_file.target_file)
            if patched_file.is_removed_file:
                target = _strip_git_prefix(patched_file.source_file)
            if not _is_path_safe(target):
                return False, f"Diff targets unsafe path {target!r}"
            if target not in writes_set:
                return False, (
                    f"Diff targets {target!r} which is not in writes {sorted(writes_set)}"
                )
            targets.append((patched_file, target))

        # Pre-compute new content for existing-file modifications so hunk
        # mismatches are caught before we touch the filesystem.
        pending: list[tuple[Any, str, str | None]] = []  # (patched_file, target, new_text | None)
        for patched_file, target in targets:
            abs_target = self._path / target
            if patched_file.is_added_file:
                new_text = _reconstruct_added_file(patched_file)
                pending.append((patched_file, target, new_text))
            elif patched_file.is_removed_file:
                if not abs_target.exists():
                    return False, f"Cannot remove missing file {target!r}"
                pending.append((patched_file, target, None))
            else:
                if not abs_target.exists():
                    return False, f"Target file {target!r} does not exist"
                original = abs_target.read_text(encoding="utf-8")
                try:
                    new_text = _apply_hunks_to_text(original, patched_file)
                except HunkMismatchError as exc:
                    return False, f"Hunk mismatch at {target!r}: {exc}"
                pending.append((patched_file, target, new_text))

        # All validated — now commit writes.
        for _patched_file, commit_target, commit_text in pending:
            abs_target = self._path / commit_target
            if commit_text is None:
                abs_target.unlink(missing_ok=True)
            else:
                abs_target.parent.mkdir(parents=True, exist_ok=True)
                abs_target.write_text(commit_text, encoding="utf-8")

        return True, None


# ---------------------------------------------------------------------------
# Hunk application
# ---------------------------------------------------------------------------


class HunkMismatchError(Exception):
    """Raised when a hunk's context does not match the original file."""


def _reconstruct_added_file(patched_file: Any) -> str:
    """Reconstruct a brand-new file from a single-hunk diff."""
    lines: list[str] = []
    for hunk in patched_file:
        for line in hunk:
            if line.is_added or line.is_context:
                lines.append(line.value.rstrip("\n"))
    return "\n".join(lines) + ("\n" if lines else "")


def _apply_hunks_to_text(original: str, patched_file: Any) -> str:
    """Apply each hunk of ``patched_file`` to ``original`` (strict matching)."""
    # We keep a newline-preserving view so the reconstructed file keeps its
    # trailing-newline convention.
    original_lines = original.splitlines(keepends=True)
    result = list(original_lines)
    # Track the cumulative drift between the original line numbers (which the
    # diff references) and the current index in ``result``.
    drift = 0

    for hunk in patched_file:
        # unidiff's source_start is 1-based.
        start = hunk.source_start - 1 + drift
        end = start + hunk.source_length

        expected = [line.value for line in hunk if line.is_context or line.is_removed]
        actual = [_strip_newline_marker(line) for line in result[start:end]]
        if [_normalise(line) for line in expected] != [_normalise(line) for line in actual]:
            raise HunkMismatchError(
                f"hunk at line {hunk.source_start} does not match original content"
            )

        new_block: list[str] = []
        for line in hunk:
            if line.is_context or line.is_added:
                new_block.append(line.value if line.value.endswith("\n") else line.value + "\n")

        # Preserve whether the last line had a trailing newline.
        if end >= len(result) and not original.endswith("\n") and new_block:
            new_block[-1] = new_block[-1].rstrip("\n")

        result[start:end] = new_block
        drift += len(new_block) - (end - start)

    return "".join(result)


def _normalise(value: str) -> str:
    return value.rstrip("\r\n")


def _strip_newline_marker(line: str) -> str:
    return line


# ---------------------------------------------------------------------------
# WorkspaceManager
# ---------------------------------------------------------------------------


class WorkspaceManager:
    """Owns the temp root, the main workspace, and per-subtask iso workspaces.

    Concurrency contract
    --------------------
    ``WorkspaceManager`` does not use locks to guard ``main/``. Thread-safety
    relies on the scheduler's calling pattern:

    * All ``create_isolated`` calls for a given batch complete before any
      sub-agent in that batch starts modifying its iso workspace.
    * ``merge_patches`` is only invoked after every subtask in the batch has
      finished (successfully or not). No ``create_isolated`` for the next
      batch begins until the merge is complete.

    In other words, reads from ``main/`` (via ``create_isolated`` or
    ``get_pre_patch_content``) and writes to ``main/`` (via ``merge_patches``)
    are serialised by the scheduler, not by this class. If you call the API
    concurrently outside that pattern you will see torn reads.
    """

    def __init__(
        self,
        repo_path: Path,
        task_id: str,
        *,
        keep_on_exit: bool = False,
    ) -> None:
        if not repo_path.exists() or not repo_path.is_dir():
            raise WorkspaceError(f"Source repo {repo_path} is not a directory")

        self._source_repo = repo_path
        self._task_id = task_id
        self._keep_on_exit = keep_on_exit
        self._root = Path(tempfile.mkdtemp(prefix=f"maestro-{task_id}-"))
        self._main_workspace = self._root / "main"
        # Baseline is an unchanging snapshot taken right after init with the
        # same ignore rules as ``main/``. ``get_final_diff`` diffs baseline
        # against main, so ignored directories that exist in the source repo
        # but not in main (e.g. ``.git/``) never show up as spurious deletes.
        self._baseline = self._root / "baseline"
        self._isolated_workspaces: dict[str, IsolatedWorkspace] = {}

        self._initialise_main()
        _logger.info(
            "workspace_initialised",
            task_id=task_id,
            root=str(self._root),
            main=str(self._main_workspace),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def root(self) -> Path:
        return self._root

    @property
    def main_path(self) -> Path:
        return self._main_workspace

    @property
    def source_repo(self) -> Path:
        return self._source_repo

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> WorkspaceManager:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if not self._keep_on_exit:
            self.cleanup()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialise_main(self) -> None:
        shutil.copytree(
            self._source_repo,
            self._main_workspace,
            ignore=_ignore_patterns(),
        )
        shutil.copytree(
            self._source_repo,
            self._baseline,
            ignore=_ignore_patterns(),
        )

    # ------------------------------------------------------------------
    # Isolated workspace creation
    # ------------------------------------------------------------------

    async def create_isolated(self, subtask: SubTask) -> IsolatedWorkspace:
        """Create a fresh isolated workspace for ``subtask``.

        The copy is taken from the *current* ``main/`` so batch N sees the
        merged patches of batches 0..N-1. The synchronous ``shutil.copytree``
        is dispatched to a thread to avoid blocking the event loop (M6).

        See class docstring — 'Concurrency contract' — for thread-safety
        assumptions.
        """
        iso_path = self._root / "iso" / subtask.subtask_id
        iso_path.parent.mkdir(parents=True, exist_ok=True)
        if iso_path.exists():
            # Re-creation (e.g. for a retry attempt): purge the stale copy.
            await asyncio.to_thread(shutil.rmtree, iso_path, ignore_errors=True)
        await asyncio.to_thread(
            shutil.copytree,
            self._main_workspace,
            iso_path,
            ignore=_ignore_patterns(),
        )
        ws = IsolatedWorkspace(iso_path, subtask)
        self._isolated_workspaces[subtask.subtask_id] = ws
        return ws

    def get_isolated(self, subtask_id: str) -> IsolatedWorkspace:
        """Return the previously-created isolated workspace for a subtask."""
        try:
            return self._isolated_workspaces[subtask_id]
        except KeyError as exc:
            raise WorkspaceError(f"No isolated workspace for subtask {subtask_id!r}") from exc

    # ------------------------------------------------------------------
    # Judge pre-patch lookup (M3)
    # ------------------------------------------------------------------

    def get_pre_patch_content(self, relative_path: str) -> str | None:
        """Return the file content in ``main/`` (i.e. before the batch merged).

        Used by the LLM judge to render ``# Original files (before diff)`` in
        its prompt (spec 06 §5.6). Returns ``None`` if the file doesn't exist
        in ``main/`` yet (e.g. the subtask is creating a brand-new file).
        """
        if not _is_path_safe(relative_path):
            raise WorkspaceError(f"Unsafe path: {relative_path!r}")
        abs_path = self._main_workspace / relative_path
        if not abs_path.exists():
            return None
        return abs_path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Patch merging
    # ------------------------------------------------------------------

    async def merge_patches(self, results: list[SubAgentResult]) -> MergeReport:
        """Merge successful sub-agent patches into ``main/``.

        Within a batch we already rely on the scheduler's write-conflict check
        to prevent two patches touching the same file. If a cross-batch
        conflict somehow slips through (should not happen by construction), we
        collect it rather than raise so the scheduler can classify.

        See class docstring — 'Concurrency contract' — for thread-safety
        assumptions.
        """
        merged: list[str] = []
        conflicts: list[MergeConflict] = []

        for result in results:
            if result.status != "success":
                continue
            try:
                iso_ws = self.get_isolated(result.subtask_id)
            except WorkspaceError:
                conflicts.append(
                    MergeConflict(
                        subtask_id=result.subtask_id,
                        file="<unknown>",
                        reason="No isolated workspace found",
                    )
                )
                continue

            try:
                await asyncio.to_thread(self._apply_workspace_delta, iso_ws, result.modified_files)
                merged.append(result.subtask_id)
            except MergeConflictError as exc:
                conflicts.append(
                    MergeConflict(
                        subtask_id=result.subtask_id,
                        file=exc.file,
                        reason=exc.reason,
                    )
                )

        _logger.info(
            "patches_merged",
            task_id=self._task_id,
            merged=len(merged),
            conflicts=len(conflicts),
        )
        return MergeReport(merged=merged, conflicts=conflicts)

    def _apply_workspace_delta(self, iso_ws: IsolatedWorkspace, modified_files: list[str]) -> None:
        for rel_path in modified_files:
            if not _is_path_safe(rel_path):
                raise MergeConflictError(file=rel_path, reason="unsafe path")
            iso_file = iso_ws.path / rel_path
            main_file = self._main_workspace / rel_path

            if not iso_file.exists():
                main_file.unlink(missing_ok=True)
                continue

            main_file.parent.mkdir(parents=True, exist_ok=True)
            main_file.write_bytes(iso_file.read_bytes())

    # ------------------------------------------------------------------
    # Final diff
    # ------------------------------------------------------------------

    def get_final_diff(self) -> str:
        """Produce the unified diff between the initial repo state and ``main/``.

        We diff the internal ``baseline/`` copy (taken at init with the same
        ignore patterns as ``main/``) against the current ``main/``. This
        avoids spurious deletions for files that the copy skipped (``.git/``,
        ``__pycache__/``, ...).
        """
        proc = subprocess.run(
            [
                "git",
                "diff",
                "--no-index",
                "--",
                str(self._baseline),
                str(self._main_workspace),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        # ``git diff`` exits with code 1 when the diff is non-empty, which is
        # *not* an error for our purposes. Only treat ≥2 as failure.
        if proc.returncode >= 2:
            _logger.error(
                "git_diff_failed",
                returncode=proc.returncode,
                stderr=proc.stderr[-200:],
            )
            raise WorkspaceError(
                f"git diff --no-index failed (rc={proc.returncode}): {proc.stderr}"
            )
        return proc.stdout

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        if self._root.exists():
            shutil.rmtree(self._root, ignore_errors=True)
            _logger.info("workspace_cleaned", task_id=self._task_id)


__all__ = [
    "IsolatedWorkspace",
    "MergeConflict",
    "MergeConflictError",
    "MergeReport",
    "WorkspaceError",
    "WorkspaceManager",
]
