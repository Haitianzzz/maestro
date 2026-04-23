"""Benchmark task builder (spec 09 §2.5).

Scaffolds the on-disk ``<task_set>/<task_id>/`` layout from a JSON
descriptor so Week 5 Day 1 can curate the 30-task corpus by hand. The
actual GitHub-API pipeline described in spec 09 §2.5 (list merged PRs,
clone, extract failing tests, filter by line count) is *not* implemented
here — spec 09 §2.5 flags it as "requires GitHub credentials and a
human-review pass". What we ship is the deterministic tail: given an
already-curated descriptor + ``before/`` + ``after/`` directories, emit
a validated task directory.

The intended workflow::

    # 1. Prepare descriptor.json + before/ + after/ locally
    # 2. Invoke build_task_dir(descriptor, before_dir, after_dir, out_dir)
    # 3. Run `pytest` in ``before/`` to confirm the failing test is red
    # 4. Run `pytest` in ``after/`` to confirm it's green
    # 5. Commit to benchmark/tasks/

Step 3 + 4 are provided by :func:`validate_task_dir` — it's fast enough
that the curator can loop locally.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from maestro.utils.logging import get_logger
from maestro.verifier.deterministic import run_subprocess

from .models import BenchmarkTask

_logger = get_logger("benchmark.build")

_PYTEST_TIMEOUT_S = 120.0


def build_task_dir(
    task: BenchmarkTask,
    *,
    before_source: Path,
    after_source: Path,
    out_dir: Path,
    overwrite: bool = False,
) -> Path:
    """Write the canonical task layout for ``task`` into ``out_dir``.

    Produces ``out_dir/<task.task_id>/{task.json, before/, after/}``.

    Raises :class:`FileExistsError` if the target exists and
    ``overwrite=False``; that's deliberate — we don't want a rebuild to
    silently shadow a hand-tuned ``before/`` tree.
    """
    target = out_dir / task.task_id
    if target.exists():
        if not overwrite:
            raise FileExistsError(
                f"Task dir {target} already exists; pass overwrite=True to replace"
            )
        shutil.rmtree(target)

    target.mkdir(parents=True)
    (target / "task.json").write_text(task.model_dump_json(indent=2), encoding="utf-8")
    shutil.copytree(before_source, target / "before")
    shutil.copytree(after_source, target / "after")
    _logger.info(
        "bench_task_built",
        task_id=task.task_id,
        path=str(target),
    )
    return target


async def validate_task_dir(task_dir: Path) -> tuple[bool, str]:
    """Confirm that ``before/`` fails the declared tests and ``after/`` passes.

    Returns ``(ok, message)``. ``ok=True`` only when both pytest runs
    behave as the spec requires (before red, after green).
    """
    task_json = task_dir / "task.json"
    if not task_json.exists():
        return False, f"missing task.json at {task_json}"
    task = BenchmarkTask.model_validate_json(task_json.read_text(encoding="utf-8"))

    if not task.failing_tests:
        return False, "task.json declares no failing_tests"

    before = task_dir / "before"
    after = task_dir / "after"
    if not before.exists() or not after.exists():
        return False, "missing before/ or after/ directory"

    # Expect RED before the fix.
    before_pytest = await run_subprocess(
        ["pytest", "--tb=no", "-q", *task.failing_tests],
        cwd=before,
        timeout=_PYTEST_TIMEOUT_S,
    )
    if before_pytest.ok or before_pytest.timed_out:
        return False, (
            "before/ pytest did not fail as expected "
            f"(rc={before_pytest.returncode}, timeout={before_pytest.timed_out})"
        )

    # Expect GREEN after the fix.
    after_pytest = await run_subprocess(
        ["pytest", "--tb=no", "-q", *task.failing_tests],
        cwd=after,
        timeout=_PYTEST_TIMEOUT_S,
    )
    if not after_pytest.ok:
        return False, (
            "after/ pytest did not pass "
            f"(rc={after_pytest.returncode}):\n{after_pytest.combined[-500:]}"
        )

    return True, "before red, after green"


__all__ = ["build_task_dir", "validate_task_dir"]
