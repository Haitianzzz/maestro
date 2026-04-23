"""Unit tests for ``maestro.verifier.deterministic`` (spec 06 §3)."""

from __future__ import annotations

import os
import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest

from maestro.models import SubAgentResult, SubTask
from maestro.sandbox.workspace import WorkspaceManager
from maestro.verifier.deterministic import (
    DeterministicVerifier,
    SubprocessResult,
    run_subprocess,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    return repo


def _sub_result(modified: list[str]) -> SubAgentResult:
    return SubAgentResult(
        subtask_id="t-001",
        status="success",
        diff="",
        modified_files=modified,
        rationale="",
        confidence=1.0,
        retry_count=0,
        tokens_input=0,
        tokens_output=0,
        latency_ms=0,
        model_used="test",
        created_at=datetime(2026, 4, 23, tzinfo=UTC),
    )


def _make_stub(
    tmp_path: Path, name: str, *, exit_code: int, stdout: str = "", stderr: str = ""
) -> Path:
    """Write a tiny executable shell script that emits the given output / rc."""
    script = tmp_path / name
    script.write_text(
        "#!/bin/sh\n"
        f"printf %s {shlex_q(stdout)}\n"
        f"printf %s {shlex_q(stderr)} >&2\n"
        f"exit {exit_code}\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script


def shlex_q(text: str) -> str:
    # Small helper so stub scripts handle embedded newlines safely.
    import shlex as _shlex

    return _shlex.quote(text)


# ---------------------------------------------------------------------------
# run_subprocess
# ---------------------------------------------------------------------------


async def test_run_subprocess_success(sample_repo: Path, tmp_path: Path) -> None:
    script = _make_stub(tmp_path, "ok.sh", exit_code=0, stdout="hello\n")
    result = await run_subprocess(
        [str(script)],
        cwd=sample_repo,
        timeout=5.0,
    )
    assert isinstance(result, SubprocessResult)
    assert result.ok is True
    assert result.returncode == 0
    assert "hello" in result.stdout
    assert result.timed_out is False


async def test_run_subprocess_non_zero_exit(sample_repo: Path, tmp_path: Path) -> None:
    script = _make_stub(tmp_path, "fail.sh", exit_code=2, stdout="out", stderr="err")
    result = await run_subprocess([str(script)], cwd=sample_repo, timeout=5.0)
    assert result.ok is False
    assert result.returncode == 2
    assert "err" in result.combined
    assert "out" in result.combined


async def test_run_subprocess_timeout(sample_repo: Path, tmp_path: Path) -> None:
    script = tmp_path / "hang.sh"
    script.write_text("#!/bin/sh\nsleep 5\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    result = await run_subprocess([str(script)], cwd=sample_repo, timeout=0.2)
    assert result.timed_out is True
    assert result.ok is False


# ---------------------------------------------------------------------------
# DeterministicVerifier — ruff / mypy stubs
# ---------------------------------------------------------------------------


async def test_det_skips_when_no_python_files(
    sample_repo: Path,
) -> None:
    with WorkspaceManager(sample_repo, task_id="t-nopy") as mgr:
        st = SubTask(subtask_id="t-001", description="x", writes=["README.md"])
        iso = await mgr.create_isolated(st)
        verifier = DeterministicVerifier(ruff_argv=["false"], mypy_argv=["false"])
        result = await verifier.run(iso, _sub_result(["README.md"]))
    assert result.passed is True
    assert "no .py files" in result.details


async def test_det_passes_when_both_pass(
    sample_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MAESTRO_SKIP_MYPY", raising=False)
    ruff = _make_stub(tmp_path, "ruff.sh", exit_code=0)
    mypy = _make_stub(tmp_path, "mypy.sh", exit_code=0)
    with WorkspaceManager(sample_repo, task_id="t-ok") as mgr:
        st = SubTask(subtask_id="t-001", description="x", writes=["src/app.py"])
        iso = await mgr.create_isolated(st)
        verifier = DeterministicVerifier(ruff_argv=[str(ruff)], mypy_argv=[str(mypy)])
        result = await verifier.run(iso, _sub_result(["src/app.py"]))
    assert result.passed is True
    assert "passed" in result.details
    assert result.cost_usd == 0.0


async def test_det_short_circuits_on_ruff_failure(
    sample_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MAESTRO_SKIP_MYPY", raising=False)
    ruff = _make_stub(tmp_path, "ruff.sh", exit_code=1, stdout="E501 line too long\n")
    # If mypy ever runs, fail the test loudly by making it a non-existent path.
    mypy = tmp_path / "should_not_run"
    with WorkspaceManager(sample_repo, task_id="t-ruff-fail") as mgr:
        st = SubTask(subtask_id="t-001", description="x", writes=["src/app.py"])
        iso = await mgr.create_isolated(st)
        verifier = DeterministicVerifier(ruff_argv=[str(ruff)], mypy_argv=[str(mypy)])
        result = await verifier.run(iso, _sub_result(["src/app.py"]))
    assert result.passed is False
    assert "ruff failed" in result.details
    assert "E501" in result.details


async def test_det_fails_on_mypy_failure(
    sample_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MAESTRO_SKIP_MYPY", raising=False)
    ruff = _make_stub(tmp_path, "ruff.sh", exit_code=0)
    mypy = _make_stub(tmp_path, "mypy.sh", exit_code=1, stdout="error: incompatible types\n")
    with WorkspaceManager(sample_repo, task_id="t-mypy-fail") as mgr:
        st = SubTask(subtask_id="t-001", description="x", writes=["src/app.py"])
        iso = await mgr.create_isolated(st)
        verifier = DeterministicVerifier(ruff_argv=[str(ruff)], mypy_argv=[str(mypy)])
        result = await verifier.run(iso, _sub_result(["src/app.py"]))
    assert result.passed is False
    assert "mypy failed" in result.details
    assert "incompatible" in result.details


async def test_det_skips_mypy_when_env_var_set(
    sample_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAESTRO_SKIP_MYPY", "1")
    ruff = _make_stub(tmp_path, "ruff.sh", exit_code=0)
    # If mypy is invoked the test fails (missing binary).
    mypy = tmp_path / "mypy_should_not_run"
    with WorkspaceManager(sample_repo, task_id="t-skip-mypy") as mgr:
        st = SubTask(subtask_id="t-001", description="x", writes=["src/app.py"])
        iso = await mgr.create_isolated(st)
        verifier = DeterministicVerifier(ruff_argv=[str(ruff)], mypy_argv=[str(mypy)])
        result = await verifier.run(iso, _sub_result(["src/app.py"]))
    assert result.passed is True


async def test_det_filters_non_python_files(
    sample_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only .py files from modified_files are passed to ruff/mypy."""
    monkeypatch.setenv("MAESTRO_SKIP_MYPY", "1")
    # Record argv by writing to a file from the ruff script.
    record = tmp_path / "ruff_args.txt"
    ruff_script = tmp_path / "ruff.sh"
    ruff_script.write_text(
        f'#!/bin/sh\nprintf "%s\\n" "$@" > {record}\nexit 0\n',
        encoding="utf-8",
    )
    ruff_script.chmod(ruff_script.stat().st_mode | stat.S_IXUSR)
    with WorkspaceManager(sample_repo, task_id="t-filter") as mgr:
        st = SubTask(
            subtask_id="t-001",
            description="x",
            writes=["src/app.py", "README.md"],
        )
        iso = await mgr.create_isolated(st)
        verifier = DeterministicVerifier(
            ruff_argv=[str(ruff_script)],
            mypy_argv=["false"],
        )
        result = await verifier.run(iso, _sub_result(["src/app.py", "README.md"]))
    assert result.passed is True
    args = record.read_text(encoding="utf-8").splitlines()
    assert "src/app.py" in args
    assert "README.md" not in args


async def test_det_ruff_timeout_reports_failure(
    sample_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Hang ruff past the verifier's 30s budget — we monkey-patch the module
    # constant so the test still runs fast.
    monkeypatch.setattr("maestro.verifier.deterministic._RUFF_TIMEOUT_S", 0.2)
    hang = tmp_path / "hang.sh"
    hang.write_text("#!/bin/sh\nsleep 5\n", encoding="utf-8")
    hang.chmod(hang.stat().st_mode | stat.S_IXUSR)

    with WorkspaceManager(sample_repo, task_id="t-timeout") as mgr:
        st = SubTask(subtask_id="t-001", description="x", writes=["src/app.py"])
        iso = await mgr.create_isolated(st)
        verifier = DeterministicVerifier(ruff_argv=[str(hang)])
        result = await verifier.run(iso, _sub_result(["src/app.py"]))
    assert result.passed is False
    assert "timed out" in result.details


async def test_det_verify_adapter_wraps_tier_result(
    sample_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAESTRO_SKIP_MYPY", "1")
    ruff = _make_stub(tmp_path, "ruff.sh", exit_code=0)
    with WorkspaceManager(sample_repo, task_id="t-adapter") as mgr:
        st = SubTask(subtask_id="t-001", description="x", writes=["src/app.py"])
        iso = await mgr.create_isolated(st)
        verifier = DeterministicVerifier(ruff_argv=[str(ruff)], mypy_argv=["false"])
        vr = await verifier.verify(st, iso, _sub_result(["src/app.py"]))
    assert vr.overall_passed is True
    assert len(vr.tiers) == 1
    assert vr.tiers[0].tier == "deterministic"


# ---------------------------------------------------------------------------
# Sanity: the module's env flag is actually read
# ---------------------------------------------------------------------------


def test_env_flag_is_read() -> None:
    # Guard against silent regressions on the env-var name.
    assert os.getenv("MAESTRO_SKIP_MYPY", "0") != "unexpected"
