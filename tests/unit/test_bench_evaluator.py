"""Unit tests for ``benchmark.evaluator``."""

from __future__ import annotations

from pathlib import Path

from maestro.benchmark.evaluator import evaluate
from maestro.benchmark.models import BenchmarkTask

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "bench_tiny" / "task-001"


def _load_task() -> BenchmarkTask:
    return BenchmarkTask.model_validate_json((_FIXTURE / "task.json").read_text(encoding="utf-8"))


_FIX_DIFF = """\
--- a/src/app.py
+++ b/src/app.py
@@ -1,2 +1,2 @@
 def double(x):
-    return x + 2
+    return x * 2
"""


async def test_evaluate_resolved_when_diff_fixes_failing_tests() -> None:
    task = _load_task()
    result = await evaluate(
        task,
        before_dir=_FIXTURE / "before",
        after_dir=_FIXTURE / "after",
        final_diff=_FIX_DIFF,
    )
    assert result.resolved is True
    assert result.error is None
    assert result.files_modified_match is True
    # The fix is identical to ground-truth so similarity should be high.
    assert result.patch_similarity >= 0.5


async def test_evaluate_unresolved_when_diff_is_wrong() -> None:
    task = _load_task()
    wrong_diff = """\
--- a/src/app.py
+++ b/src/app.py
@@ -1,2 +1,2 @@
 def double(x):
-    return x + 2
+    return x + 3
"""
    result = await evaluate(
        task,
        before_dir=_FIXTURE / "before",
        after_dir=_FIXTURE / "after",
        final_diff=wrong_diff,
    )
    assert result.resolved is False
    assert result.error is not None


async def test_evaluate_rejects_empty_diff() -> None:
    task = _load_task()
    result = await evaluate(
        task,
        before_dir=_FIXTURE / "before",
        after_dir=_FIXTURE / "after",
        final_diff="",
    )
    assert result.resolved is False
    assert result.error == "empty diff from Maestro"


async def test_evaluate_reports_apply_failure() -> None:
    task = _load_task()
    malformed = """\
--- a/src/app.py
+++ b/src/app.py
@@ -1,2 +1,2 @@
 def double(x):
-    return NOT_THE_REAL_CONTENT
+    return x * 2
"""
    result = await evaluate(
        task,
        before_dir=_FIXTURE / "before",
        after_dir=_FIXTURE / "after",
        final_diff=malformed,
    )
    assert result.resolved is False
    assert result.error is not None
    assert "apply failed" in result.error


async def test_evaluate_tracks_extra_modified_files(tmp_path: Path) -> None:
    """Diff touching files outside expected_modified_files inflates extras."""
    # Build a before dir with two source files.
    before = tmp_path / "before"
    (before / "src").mkdir(parents=True)
    (before / "src" / "app.py").write_text("X = 1\n", encoding="utf-8")
    (before / "src" / "other.py").write_text("Y = 2\n", encoding="utf-8")
    (before / "tests").mkdir()
    (before / "tests" / "test_app.py").write_text(
        "def test_placeholder():\n    assert True\n", encoding="utf-8"
    )
    after = tmp_path / "after"
    (after / "src").mkdir(parents=True)
    (after / "src" / "app.py").write_text("X = 2\n", encoding="utf-8")
    (after / "src" / "other.py").write_text("Y = 2\n", encoding="utf-8")
    (after / "tests").mkdir()
    (after / "tests" / "test_app.py").write_text(
        "def test_placeholder():\n    assert True\n", encoding="utf-8"
    )

    task = BenchmarkTask(
        task_id="extra",
        repo="fake/repo",
        description="",
        natural_language_prompt="",
        failing_tests=["tests/test_app.py"],
        expected_modified_files=["src/app.py"],
    )
    diff = """\
--- a/src/app.py
+++ b/src/app.py
@@ -1 +1 @@
-X = 1
+X = 2
--- a/src/other.py
+++ b/src/other.py
@@ -1 +1 @@
-Y = 2
+Y = 3
"""
    result = await evaluate(
        task,
        before_dir=before,
        after_dir=after,
        final_diff=diff,
    )
    assert result.extra_files_modified == 1
    assert result.files_modified_match is False
