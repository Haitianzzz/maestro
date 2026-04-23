"""Unit tests for ``benchmark.harness`` — the dry-run smoke path.

Real LLM wiring requires API credentials and is out of scope for unit
tests (spec 09 §3.1 runs are live in Week 5-6). This module exercises
the harness structure via ``dry_run=True`` so the LLMClient returns
dummy responses and we validate the assembly + persistence logic
without the network.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from maestro.benchmark.harness import BenchmarkHarness
from maestro.benchmark.models import BenchmarkReport, BenchmarkTask, TaskBenchmarkResult

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "bench_tiny"


def _copy_fixture_to(tmp_path: Path) -> Path:
    target = tmp_path / "tasks"
    shutil.copytree(_FIXTURE, target)
    return target


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_harness_rejects_missing_task_set(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        BenchmarkHarness(
            task_set_dir=tmp_path / "ghost",
            output_dir=tmp_path / "out",
            config_name="baseline",
        )


def test_harness_rejects_unknown_config_name(tmp_path: Path) -> None:
    task_set = _copy_fixture_to(tmp_path)
    with pytest.raises(KeyError):
        BenchmarkHarness(
            task_set_dir=task_set,
            output_dir=tmp_path / "out",
            config_name="no_such_config",
        )


# ---------------------------------------------------------------------------
# End-to-end dry-run
# ---------------------------------------------------------------------------


async def test_harness_dry_run_produces_report(tmp_path: Path) -> None:
    task_set = _copy_fixture_to(tmp_path)
    out = tmp_path / "results"

    harness = BenchmarkHarness(
        task_set_dir=task_set,
        output_dir=out,
        config_name="baseline",
        dry_run=True,
    )
    report = await harness.run_all()
    assert isinstance(report, BenchmarkReport)
    assert report.config_name == "baseline"
    assert report.task_count == 1

    # The single task must have produced a TaskBenchmarkResult even if the
    # dry-run orchestrator errored out — dry_run returns dummy content that
    # doesn't actually fix `double()`, so the resolved flag can go either
    # way. The important invariant is that ``run_all`` completes cleanly.
    assert len(report.per_task_results) == 1
    assert report.per_task_results[0].task_id == "task-001"

    # Persistence: both JSON + MD artifacts written.
    assert (out / "baseline.json").exists()
    assert (out / "baseline.md").exists()
    body = (out / "baseline.md").read_text(encoding="utf-8")
    assert "baseline" in body
    assert "task-001" in body


async def test_harness_error_status_when_before_dir_missing(tmp_path: Path) -> None:
    task_set = _copy_fixture_to(tmp_path)
    # Remove the task's before/ so the harness must emit an error record.
    shutil.rmtree(task_set / "task-001" / "before")
    harness = BenchmarkHarness(
        task_set_dir=task_set,
        output_dir=tmp_path / "out",
        config_name="baseline",
        dry_run=True,
    )
    report = await harness.run_all()
    assert report.task_count == 1
    assert report.per_task_results[0].status == "error"
    assert "missing" in (report.per_task_results[0].error or "")


async def test_harness_respects_limit(tmp_path: Path) -> None:
    task_set = _copy_fixture_to(tmp_path)
    # Clone the single fixture task under a second id so we have 2.
    shutil.copytree(task_set / "task-001", task_set / "task-002")
    (task_set / "task-002" / "task.json").write_text(
        '{"task_id":"task-002","repo":"fake/tiny","description":"dup",'
        '"natural_language_prompt":"dup","failing_tests":["tests/test_app.py"],'
        '"expected_modified_files":["src/app.py"]}',
        encoding="utf-8",
    )

    harness = BenchmarkHarness(
        task_set_dir=task_set,
        output_dir=tmp_path / "out",
        config_name="baseline",
        dry_run=True,
    )
    report = await harness.run_all(limit=1)
    assert report.task_count == 1


# ---------------------------------------------------------------------------
# Regression guard — baseline / parallel_only MUST NOT silently enable tiers
# ---------------------------------------------------------------------------


async def _run_and_capture_tiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_name: str
) -> object:
    """Run one dry-run harness task and capture what Verifier was constructed with."""
    import maestro.benchmark.harness as harness_mod
    from maestro.verifier import Verifier as RealVerifier

    captured: dict[str, object] = {}

    class _SpyVerifier(RealVerifier):  # type: ignore[misc,valid-type]
        def __init__(self, *args: object, **kwargs: object) -> None:
            # Record the tiers argument *exactly* as passed so we can tell
            # ``None`` (= default, all three tiers) from ``set()`` (= all
            # disabled). This distinction is the whole point of the test.
            captured.setdefault("enabled_tiers", kwargs.get("enabled_tiers"))
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(harness_mod, "Verifier", _SpyVerifier)

    task_set = _copy_fixture_to(tmp_path)
    harness = BenchmarkHarness(
        task_set_dir=task_set,
        output_dir=tmp_path / "out",
        config_name=config_name,
        dry_run=True,
    )
    await harness.run_all()
    return captured.get("enabled_tiers", "never-called")


async def test_baseline_config_really_disables_all_tiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: ``baseline`` used to collapse set() to None, re-enabling
    every tier. Guard the fix here — the benchmark's "no-verify baseline"
    numbers are meaningless if this silently flips back on."""
    observed = await _run_and_capture_tiers(tmp_path, monkeypatch, "baseline")
    assert observed == set(), (
        f"baseline must pass an empty set, got {observed!r}. If this becomes "
        "None, Verifier falls back to ALL tiers and the baseline contrast is fake."
    )


async def test_parallel_only_config_really_disables_all_tiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same regression guard for ``parallel_only``."""
    observed = await _run_and_capture_tiers(tmp_path, monkeypatch, "parallel_only")
    assert observed == set(), (
        f"parallel_only must pass an empty set, got {observed!r}. If this "
        "becomes None, Verifier falls back to ALL tiers and the parallel-vs-"
        "verify isolation experiment is contaminated."
    )


# ---------------------------------------------------------------------------
# Model round-trip (sanity that the persistence schema is stable)
# ---------------------------------------------------------------------------


def test_benchmark_report_roundtrip() -> None:
    now = datetime(2026, 4, 23, tzinfo=UTC)
    report = BenchmarkReport(
        run_id="run-abc",
        config_name="full",
        task_count=1,
        resolve_rate=1.0,
        avg_wall_clock_ms=500.0,
        avg_cost=0.04,
        avg_tokens_input=100.0,
        avg_tokens_output=50.0,
        per_task_results=[
            TaskBenchmarkResult(
                task_id="task-001",
                config_name="full",
                status="resolved",
                resolved=True,
                wall_clock_ms=500,
                total_cost=0.04,
                total_tokens_input=100,
                total_tokens_output=50,
                patch_similarity=0.9,
                files_modified_match=True,
                extra_files_modified=0,
            )
        ],
        started_at=now,
        finished_at=now,
    )
    blob = report.model_dump_json()
    restored = BenchmarkReport.model_validate_json(blob)
    assert restored.run_id == report.run_id
    assert restored.per_task_results[0].resolved is True


# ---------------------------------------------------------------------------
# BenchmarkTask round-trip (wants the fixture's task.json to be valid)
# ---------------------------------------------------------------------------


def test_fixture_task_json_parses_into_model() -> None:
    task = BenchmarkTask.model_validate_json(
        (_FIXTURE / "task-001" / "task.json").read_text(encoding="utf-8")
    )
    assert task.task_id == "task-001"
    assert task.failing_tests == ["tests/test_app.py"]
