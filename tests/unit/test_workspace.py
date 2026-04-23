"""Unit tests for ``maestro.sandbox.workspace`` (spec 07 §9)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from maestro.models import SubAgentResult, SubTask
from maestro.sandbox.workspace import (
    IsolatedWorkspace,
    MergeConflict,
    WorkspaceError,
    WorkspaceManager,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """A tiny repo with a couple of files and directories we must skip."""
    repo = tmp_path / "sample_repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")
    (repo / "src" / "utils.py").write_text("VERSION = '0.1'\n", encoding="utf-8")
    (repo / "README.md").write_text("# Sample\n", encoding="utf-8")
    # Directories we must ignore:
    (repo / ".git").mkdir()
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (repo / "__pycache__").mkdir()
    (repo / "__pycache__" / "cache.pyc").write_text("binary", encoding="utf-8")
    return repo


def _subtask(
    subtask_id: str = "t-001",
    reads: list[str] | None = None,
    writes: list[str] | None = None,
) -> SubTask:
    return SubTask(
        subtask_id=subtask_id,
        description="demo",
        reads=reads or [],
        writes=writes or [],
    )


def _subagent_result(subtask_id: str, modified_files: list[str]) -> SubAgentResult:
    return SubAgentResult(
        subtask_id=subtask_id,
        status="success",
        diff="irrelevant-for-merge",
        modified_files=modified_files,
        rationale="",
        confidence=1.0,
        retry_count=0,
        tokens_input=0,
        tokens_output=0,
        latency_ms=0,
        model_used="test",
        created_at=datetime(2026, 4, 23, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# WorkspaceManager init
# ---------------------------------------------------------------------------


def test_init_rejects_missing_repo(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceError):
        WorkspaceManager(tmp_path / "does-not-exist", task_id="t")


def test_init_copies_repo_and_respects_ignore(sample_repo: Path) -> None:
    with WorkspaceManager(sample_repo, task_id="t-abc") as mgr:
        main = mgr.main_path
        assert (main / "src" / "app.py").exists()
        assert (main / "README.md").exists()
        assert not (main / ".git").exists()
        assert not (main / "__pycache__").exists()


def test_cleanup_removes_root(sample_repo: Path) -> None:
    mgr = WorkspaceManager(sample_repo, task_id="t-abc")
    root = mgr.root
    assert root.exists()
    mgr.cleanup()
    assert not root.exists()


def test_context_manager_keeps_on_exit_when_requested(sample_repo: Path) -> None:
    with WorkspaceManager(sample_repo, task_id="t-keep", keep_on_exit=True) as mgr:
        root = mgr.root
    assert root.exists()
    # Clean up manually so the test doesn't leak.
    import shutil

    shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Isolated workspaces
# ---------------------------------------------------------------------------


async def test_create_isolated_independent_of_main(sample_repo: Path) -> None:
    with WorkspaceManager(sample_repo, task_id="t-iso") as mgr:
        st = _subtask(writes=["src/app.py"], reads=["src/utils.py"])
        iso = await mgr.create_isolated(st)
        assert iso.path != mgr.main_path
        assert (iso.path / "src" / "app.py").read_text(encoding="utf-8") == (
            "def hello():\n    return 'world'\n"
        )
        # Mutate the iso copy; main should remain unchanged.
        (iso.path / "src" / "app.py").write_text("# changed\n", encoding="utf-8")
        assert (mgr.main_path / "src" / "app.py").read_text(
            encoding="utf-8"
        ) == "def hello():\n    return 'world'\n"


async def test_get_isolated_returns_same_instance(sample_repo: Path) -> None:
    with WorkspaceManager(sample_repo, task_id="t-iso2") as mgr:
        st = _subtask()
        iso = await mgr.create_isolated(st)
        assert mgr.get_isolated(st.subtask_id) is iso


async def test_get_isolated_unknown_subtask_raises(sample_repo: Path) -> None:
    with WorkspaceManager(sample_repo, task_id="t-iso3") as mgr, pytest.raises(WorkspaceError):
        mgr.get_isolated("ghost")


# ---------------------------------------------------------------------------
# Read permission
# ---------------------------------------------------------------------------


async def test_read_file_allowed_via_reads(sample_repo: Path) -> None:
    with WorkspaceManager(sample_repo, task_id="t-read") as mgr:
        st = _subtask(reads=["src/app.py"])
        iso = await mgr.create_isolated(st)
        assert "hello" in iso.read_file("src/app.py")


async def test_read_file_allowed_via_writes(sample_repo: Path) -> None:
    with WorkspaceManager(sample_repo, task_id="t-read2") as mgr:
        st = _subtask(writes=["src/app.py"])
        iso = await mgr.create_isolated(st)
        assert "hello" in iso.read_file("src/app.py")


async def test_read_file_blocks_undeclared_file(sample_repo: Path) -> None:
    with WorkspaceManager(sample_repo, task_id="t-read3") as mgr:
        st = _subtask(reads=["src/app.py"])
        iso = await mgr.create_isolated(st)
        with pytest.raises(PermissionError, match="not in reads or writes"):
            iso.read_file("src/utils.py")


async def test_read_file_rejects_path_traversal(sample_repo: Path) -> None:
    with WorkspaceManager(sample_repo, task_id="t-read4") as mgr:
        st = _subtask(reads=["../etc/passwd"])
        iso = await mgr.create_isolated(st)
        with pytest.raises(PermissionError, match="Unsafe path"):
            iso.read_file("../etc/passwd")


async def test_read_file_rejects_unsafe_extension(sample_repo: Path, tmp_path: Path) -> None:
    # Plant an unsafe file in the repo so we can target it.
    (sample_repo / "binary.exe").write_bytes(b"\x00\x01")
    with WorkspaceManager(sample_repo, task_id="t-read5") as mgr:
        st = _subtask(reads=["binary.exe"])
        iso = await mgr.create_isolated(st)
        with pytest.raises(PermissionError, match="not a safe file"):
            iso.read_file("binary.exe")


# ---------------------------------------------------------------------------
# Diff application
# ---------------------------------------------------------------------------


_DIFF_MODIFY_APP = """\
--- a/src/app.py
+++ b/src/app.py
@@ -1,2 +1,2 @@
 def hello():
-    return 'world'
+    return 'maestro'
"""


_DIFF_ADD_NEW = """\
--- /dev/null
+++ b/src/new_file.py
@@ -0,0 +1,2 @@
+def added():
+    return 42
"""


_DIFF_REMOVE = """\
--- a/src/utils.py
+++ /dev/null
@@ -1 +0,0 @@
-VERSION = '0.1'
"""


_DIFF_OUT_OF_WRITES = """\
--- a/src/utils.py
+++ b/src/utils.py
@@ -1 +1 @@
-VERSION = '0.1'
+VERSION = '0.2'
"""


_DIFF_HUNK_MISMATCH = """\
--- a/src/app.py
+++ b/src/app.py
@@ -1,2 +1,2 @@
 def hello():
-    return 'COMPLETELY WRONG CONTEXT'
+    return 'x'
"""


async def test_apply_diff_modifies_existing_file(sample_repo: Path) -> None:
    with WorkspaceManager(sample_repo, task_id="t-apply1") as mgr:
        st = _subtask(writes=["src/app.py"])
        iso = await mgr.create_isolated(st)
        ok, err = iso.apply_diff(_DIFF_MODIFY_APP)
        assert ok, err
        assert (iso.path / "src" / "app.py").read_text(encoding="utf-8") == (
            "def hello():\n    return 'maestro'\n"
        )


async def test_apply_diff_creates_new_file(sample_repo: Path) -> None:
    with WorkspaceManager(sample_repo, task_id="t-apply2") as mgr:
        st = _subtask(writes=["src/new_file.py"])
        iso = await mgr.create_isolated(st)
        ok, err = iso.apply_diff(_DIFF_ADD_NEW)
        assert ok, err
        content = (iso.path / "src" / "new_file.py").read_text(encoding="utf-8")
        assert "def added" in content
        assert "return 42" in content


async def test_apply_diff_removes_file(sample_repo: Path) -> None:
    with WorkspaceManager(sample_repo, task_id="t-apply3") as mgr:
        st = _subtask(writes=["src/utils.py"])
        iso = await mgr.create_isolated(st)
        ok, err = iso.apply_diff(_DIFF_REMOVE)
        assert ok, err
        assert not (iso.path / "src" / "utils.py").exists()


async def test_apply_diff_rejects_out_of_writes(sample_repo: Path) -> None:
    with WorkspaceManager(sample_repo, task_id="t-apply4") as mgr:
        st = _subtask(writes=["src/app.py"])  # utils.py NOT in writes
        iso = await mgr.create_isolated(st)
        ok, err = iso.apply_diff(_DIFF_OUT_OF_WRITES)
        assert not ok
        assert err is not None
        assert "not in writes" in err
        # utils.py must be untouched.
        assert (iso.path / "src" / "utils.py").read_text(encoding="utf-8") == "VERSION = '0.1'\n"


async def test_apply_diff_rejects_hunk_mismatch(sample_repo: Path) -> None:
    with WorkspaceManager(sample_repo, task_id="t-apply5") as mgr:
        st = _subtask(writes=["src/app.py"])
        iso = await mgr.create_isolated(st)
        ok, err = iso.apply_diff(_DIFF_HUNK_MISMATCH)
        assert not ok
        assert err is not None
        assert "Hunk mismatch" in err


async def test_apply_diff_rejects_unparseable(sample_repo: Path) -> None:
    with WorkspaceManager(sample_repo, task_id="t-apply6") as mgr:
        st = _subtask(writes=["src/app.py"])
        iso = await mgr.create_isolated(st)
        ok, err = iso.apply_diff("not a diff")
        assert not ok
        assert err is not None


async def test_apply_diff_rejects_empty(sample_repo: Path) -> None:
    with WorkspaceManager(sample_repo, task_id="t-apply7") as mgr:
        st = _subtask(writes=["src/app.py"])
        iso = await mgr.create_isolated(st)
        ok, err = iso.apply_diff("   \n")
        assert not ok
        assert err == "Empty diff"


# ---------------------------------------------------------------------------
# merge_patches + get_pre_patch_content
# ---------------------------------------------------------------------------


async def test_merge_patches_applies_successful_iso_edits_to_main(sample_repo: Path) -> None:
    with WorkspaceManager(sample_repo, task_id="t-merge1") as mgr:
        st = _subtask(writes=["src/app.py"])
        iso = await mgr.create_isolated(st)
        assert iso.apply_diff(_DIFF_MODIFY_APP)[0]

        report = await mgr.merge_patches([_subagent_result(st.subtask_id, ["src/app.py"])])
        assert report.merged == [st.subtask_id]
        assert report.conflicts == []
        # main should now reflect the iso change.
        assert "maestro" in (mgr.main_path / "src" / "app.py").read_text(encoding="utf-8")


async def test_merge_patches_skips_failed_subagents(sample_repo: Path) -> None:
    with WorkspaceManager(sample_repo, task_id="t-merge2") as mgr:
        st = _subtask(writes=["src/app.py"])
        await mgr.create_isolated(st)
        failed = SubAgentResult(
            subtask_id=st.subtask_id,
            status="failed",
            diff="",
            modified_files=[],
            rationale="",
            confidence=0.0,
            retry_count=0,
            tokens_input=0,
            tokens_output=0,
            latency_ms=0,
            model_used="test",
            created_at=datetime(2026, 4, 23, tzinfo=UTC),
        )
        report = await mgr.merge_patches([failed])
        assert report.merged == []
        assert report.conflicts == []


async def test_merge_patches_records_missing_isolated_as_conflict(sample_repo: Path) -> None:
    with WorkspaceManager(sample_repo, task_id="t-merge3") as mgr:
        # Never created the isolated workspace for this subtask.
        report = await mgr.merge_patches([_subagent_result("ghost-001", ["src/app.py"])])
        assert report.merged == []
        assert len(report.conflicts) == 1
        assert isinstance(report.conflicts[0], MergeConflict)


async def test_merge_patches_rejects_unsafe_paths(sample_repo: Path) -> None:
    with WorkspaceManager(sample_repo, task_id="t-merge4") as mgr:
        st = _subtask(writes=["src/app.py"])
        await mgr.create_isolated(st)
        report = await mgr.merge_patches([_subagent_result(st.subtask_id, ["../evil.py"])])
        assert report.merged == []
        assert len(report.conflicts) == 1
        assert "unsafe" in report.conflicts[0].reason


def test_get_pre_patch_content_reads_main(sample_repo: Path) -> None:
    with WorkspaceManager(sample_repo, task_id="t-pre1") as mgr:
        content = mgr.get_pre_patch_content("src/app.py")
        assert content is not None
        assert "hello" in content


def test_get_pre_patch_content_returns_none_for_missing(sample_repo: Path) -> None:
    with WorkspaceManager(sample_repo, task_id="t-pre2") as mgr:
        assert mgr.get_pre_patch_content("src/does_not_exist.py") is None


def test_get_pre_patch_content_rejects_unsafe(sample_repo: Path) -> None:
    with WorkspaceManager(sample_repo, task_id="t-pre3") as mgr, pytest.raises(WorkspaceError):
        mgr.get_pre_patch_content("../etc/passwd")


# ---------------------------------------------------------------------------
# Final diff
# ---------------------------------------------------------------------------


async def test_get_final_diff_produces_patch_after_merge(sample_repo: Path) -> None:
    with WorkspaceManager(sample_repo, task_id="t-finaldiff") as mgr:
        st = _subtask(writes=["src/app.py"])
        iso = await mgr.create_isolated(st)
        assert iso.apply_diff(_DIFF_MODIFY_APP)[0]
        await mgr.merge_patches([_subagent_result(st.subtask_id, ["src/app.py"])])

        diff = mgr.get_final_diff()
        assert diff  # non-empty
        assert "maestro" in diff
        assert "world" in diff


def test_get_final_diff_returns_empty_when_unchanged(sample_repo: Path) -> None:
    with WorkspaceManager(sample_repo, task_id="t-finaldiff2") as mgr:
        assert mgr.get_final_diff() == ""


# ---------------------------------------------------------------------------
# IsolatedWorkspace surface
# ---------------------------------------------------------------------------


async def test_isolated_workspace_exposes_path_and_subtask(sample_repo: Path) -> None:
    with WorkspaceManager(sample_repo, task_id="t-surface") as mgr:
        st = _subtask(writes=["src/app.py"])
        iso = await mgr.create_isolated(st)
        assert isinstance(iso, IsolatedWorkspace)
        assert iso.path.is_dir()
        assert iso.subtask.subtask_id == st.subtask_id
