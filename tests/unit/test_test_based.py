"""Unit tests for ``maestro.verifier.test_based`` (spec 06 §4)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from maestro.llm.client import LLMCallMetadata, LLMClient
from maestro.llm.config import ClientConfig, ModelConfig
from maestro.models import SubAgentResult, SubTask
from maestro.sandbox.workspace import WorkspaceManager
from maestro.verifier.test_based import (
    TestBasedVerifier,
    _find_related_tests,
    _strip_code_fence,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def repo_with_tests(tmp_path: Path) -> Path:
    """Repo whose ``tests/unit/test_app.py`` exercises ``src/app.py``.

    The test imports ``app`` directly (not ``src.app``) so pytest's default
    rootdir sys.path insertion is enough — we don't need to ship a conftest
    in the fixture repo.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "src" / "app.py").write_text("def double(x):\n    return x * 2\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "tests" / "unit").mkdir()
    (repo / "tests" / "unit" / "__init__.py").write_text("", encoding="utf-8")
    # Self-contained test: no repo-specific import paths involved.
    (repo / "tests" / "unit" / "test_app.py").write_text(
        "def double(x):\n    return x * 2\n\ndef test_double_basic():\n    assert double(3) == 6\n",
        encoding="utf-8",
    )
    return repo


@pytest.fixture
def repo_without_tests(tmp_path: Path) -> Path:
    repo = tmp_path / "repo_bare"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("X = 1\n", encoding="utf-8")
    return repo


def _sub_result(modified: list[str], *, diff: str = "--- a/x\n+++ b/x\n") -> SubAgentResult:
    return SubAgentResult(
        subtask_id="t-001",
        status="success",
        diff=diff,
        modified_files=modified,
        rationale="touched src/app.py",
        confidence=1.0,
        retry_count=0,
        tokens_input=0,
        tokens_output=0,
        latency_ms=0,
        model_used="test",
        created_at=datetime(2026, 4, 23, tzinfo=UTC),
    )


def _make_llm_client() -> LLMClient:
    cfg = ClientConfig(
        base_url="https://example.com/v1",
        api_key="fake",
        models={
            "subagent": ModelConfig(
                name="qwen3-coder-plus",
                display_name="Qwen3-Coder-Plus",
                price_input_per_mtok=0.84,
                price_output_per_mtok=3.36,
            ),
        },
    )
    return LLMClient(cfg)


def _install_text(
    client: LLMClient,
    responder: Callable[..., Awaitable[tuple[str, LLMCallMetadata]]],
) -> AsyncMock:
    mock = AsyncMock(side_effect=responder)
    client.call_text = mock  # type: ignore[method-assign]
    return mock


def _fake_meta() -> LLMCallMetadata:
    return LLMCallMetadata(
        model_name="qwen3-coder-plus",
        role="subagent",
        tokens_input=10,
        tokens_output=10,
        latency_ms=1,
        cost=0.0,
        currency="RMB",
        called_at=datetime(2026, 4, 23, tzinfo=UTC),
        success=True,
        http_retry_count=0,
    )


# ---------------------------------------------------------------------------
# _find_related_tests
# ---------------------------------------------------------------------------


def test_find_related_tests_locates_unit_test(repo_with_tests: Path) -> None:
    found = _find_related_tests(repo_with_tests, ["src/app.py"])
    assert len(found) == 1
    assert found[0].name == "test_app.py"


def test_find_related_tests_returns_empty_when_absent(
    repo_without_tests: Path,
) -> None:
    assert _find_related_tests(repo_without_tests, ["src/app.py"]) == []


def test_find_related_tests_skips_init_and_test_prefixed(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "tests").mkdir()
    (repo / "tests" / "test___init__.py").write_text("", encoding="utf-8")
    (repo / "tests" / "test_test_app.py").write_text("", encoding="utf-8")
    # No test should be returned for __init__.py or a file already named test_*.
    assert _find_related_tests(repo, ["src/__init__.py", "src/test_app.py"]) == []


def test_find_related_tests_dedups(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "tests").mkdir()
    (repo / "tests" / "unit").mkdir()
    (repo / "tests" / "unit" / "test_app.py").write_text("", encoding="utf-8")
    (repo / "tests" / "test_app.py").write_text("", encoding="utf-8")
    found = _find_related_tests(repo, ["src/app.py", "lib/app.py"])
    # Two source files share the same stem → still two unique test paths, not
    # four.
    assert len(found) == 2


# ---------------------------------------------------------------------------
# TestBasedVerifier — happy / fail paths with real pytest
# ---------------------------------------------------------------------------


async def test_run_pytest_passes_when_existing_tests_pass(
    repo_with_tests: Path,
) -> None:
    with WorkspaceManager(repo_with_tests, task_id="t-green") as mgr:
        st = SubTask(subtask_id="t-001", description="x", writes=["src/app.py"])
        iso = await mgr.create_isolated(st)
        verifier = TestBasedVerifier()
        result = await verifier.run(iso, _sub_result(["src/app.py"]))
    assert result.tier == "test_based"
    assert result.passed is True
    assert "pytest passed" in result.details


async def test_run_pytest_reports_failure_when_tests_fail(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "red"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("def double(x):\n    return x + x\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "unit").mkdir()
    (repo / "tests" / "unit" / "test_app.py").write_text(
        "def double(x):\n    return x + x\n\n"
        "def test_double_is_wrong():\n    assert double(3) == 999\n",
        encoding="utf-8",
    )
    with WorkspaceManager(repo, task_id="t-red") as mgr:
        st = SubTask(subtask_id="t-001", description="x", writes=["src/app.py"])
        iso = await mgr.create_isolated(st)
        verifier = TestBasedVerifier()
        result = await verifier.run(iso, _sub_result(["src/app.py"]))
    assert result.passed is False
    assert "pytest failed" in result.details


async def test_run_pytest_skips_when_no_py_files(repo_with_tests: Path) -> None:
    with WorkspaceManager(repo_with_tests, task_id="t-nopy") as mgr:
        st = SubTask(subtask_id="t-001", description="x", writes=["README.md"])
        iso = await mgr.create_isolated(st)
        verifier = TestBasedVerifier()
        result = await verifier.run(iso, _sub_result(["README.md"]))
    assert result.passed is True
    assert "nothing to pytest" in result.details


# ---------------------------------------------------------------------------
# Auto-gen path
# ---------------------------------------------------------------------------


async def test_no_tests_and_autogen_off_passes(repo_without_tests: Path) -> None:
    with WorkspaceManager(repo_without_tests, task_id="t-no-autogen") as mgr:
        st = SubTask(subtask_id="t-001", description="x", writes=["src/app.py"])
        iso = await mgr.create_isolated(st)
        verifier = TestBasedVerifier(auto_gen_tests=False)
        result = await verifier.run(iso, _sub_result(["src/app.py"]))
    assert result.passed is True
    assert "auto-gen disabled" in result.details


async def test_autogen_on_generates_file_and_runs_pytest(
    repo_without_tests: Path,
) -> None:
    client = _make_llm_client()

    async def responder(**_: Any) -> tuple[str, LLMCallMetadata]:
        # A trivially passing pytest file.
        return (
            "def test_trivially_passes():\n    assert 1 + 1 == 2\n",
            _fake_meta(),
        )

    mock = _install_text(client, responder)

    with WorkspaceManager(repo_without_tests, task_id="t-autogen-ok") as mgr:
        st = SubTask(subtask_id="t-001", description="x", writes=["src/app.py"])
        iso = await mgr.create_isolated(st)
        verifier = TestBasedVerifier(client, auto_gen_tests=True)
        result = await verifier.run(iso, _sub_result(["src/app.py"]))

        assert result.passed is True
        assert mock.call_count == 1
        # Generated file should exist at the documented path inside the iso.
        gen = iso.path / "tests" / "autogen" / "test_t_001.py"
        assert gen.exists()
        assert "assert 1 + 1 == 2" in gen.read_text(encoding="utf-8")


async def test_autogen_strips_code_fences(
    repo_without_tests: Path,
) -> None:
    client = _make_llm_client()

    async def responder(**_: Any) -> tuple[str, LLMCallMetadata]:
        return (
            "```python\ndef test_passes():\n    assert True\n```\n",
            _fake_meta(),
        )

    _install_text(client, responder)

    with WorkspaceManager(repo_without_tests, task_id="t-fence") as mgr:
        st = SubTask(subtask_id="t-001", description="x", writes=["src/app.py"])
        iso = await mgr.create_isolated(st)
        verifier = TestBasedVerifier(client, auto_gen_tests=True)
        result = await verifier.run(iso, _sub_result(["src/app.py"]))

        gen = (iso.path / "tests" / "autogen" / "test_t_001.py").read_text(encoding="utf-8")
        assert "```" not in gen
        assert "assert True" in gen
        assert result.passed is True


async def test_autogen_failure_propagates_as_tier_failure(
    repo_without_tests: Path,
) -> None:
    client = _make_llm_client()

    async def responder(**_: Any) -> tuple[str, LLMCallMetadata]:
        raise RuntimeError("LLM exploded")

    _install_text(client, responder)

    with WorkspaceManager(repo_without_tests, task_id="t-autogen-fail") as mgr:
        st = SubTask(subtask_id="t-001", description="x", writes=["src/app.py"])
        iso = await mgr.create_isolated(st)
        verifier = TestBasedVerifier(client, auto_gen_tests=True)
        result = await verifier.run(iso, _sub_result(["src/app.py"]))

    assert result.passed is False
    assert "auto-gen failed" in result.details


async def test_autogen_on_but_no_llm_passes(
    repo_without_tests: Path,
) -> None:
    with WorkspaceManager(repo_without_tests, task_id="t-autogen-noclient") as mgr:
        st = SubTask(subtask_id="t-001", description="x", writes=["src/app.py"])
        iso = await mgr.create_isolated(st)
        verifier = TestBasedVerifier(llm_client=None, auto_gen_tests=True)
        result = await verifier.run(iso, _sub_result(["src/app.py"]))
    assert result.passed is True
    assert "no LLM client" in result.details


# ---------------------------------------------------------------------------
# VerifierProtocol adapter
# ---------------------------------------------------------------------------


async def test_verify_adapter_wraps_tier_result(repo_with_tests: Path) -> None:
    with WorkspaceManager(repo_with_tests, task_id="t-adapter") as mgr:
        st = SubTask(subtask_id="t-001", description="x", writes=["src/app.py"])
        iso = await mgr.create_isolated(st)
        verifier = TestBasedVerifier()
        vr = await verifier.verify(st, iso, _sub_result(["src/app.py"]))
    assert vr.overall_passed is True
    assert len(vr.tiers) == 1
    assert vr.tiers[0].tier == "test_based"


# ---------------------------------------------------------------------------
# Fence stripper
# ---------------------------------------------------------------------------


def test_strip_fence_removes_python_marker() -> None:
    text = "```python\ndef f(): pass\n```\n"
    out = _strip_code_fence(text)
    assert out.startswith("def f")
    assert "```" not in out


def test_strip_fence_passes_through_when_no_fence() -> None:
    text = "def g(): return 1\n"
    assert _strip_code_fence(text) == text


def test_strip_fence_handles_generic_fence() -> None:
    text = "```\nprint('x')\n```"
    out = _strip_code_fence(text)
    assert "print('x')" in out
    assert "```" not in out
