"""Unit tests for ``maestro.planner.repo_scanner``."""

from __future__ import annotations

from pathlib import Path

import pytest

from maestro.planner.repo_scanner import RepoContext, RepoScanner

_FIXTURE_REPO = Path(__file__).parent.parent / "fixtures" / "tiny_flask_app"


def test_scan_produces_context() -> None:
    scanner = RepoScanner(_FIXTURE_REPO)
    ctx = scanner.scan()
    assert isinstance(ctx, RepoContext)
    assert "tiny_flask_app/" in ctx.file_tree
    assert "src/" in ctx.file_tree
    assert "app.py" in ctx.file_tree
    assert "pyproject.toml" in ctx.key_files
    assert "README.md" in ctx.key_files
    # AST signatures for every top-level .py file not in key_files.
    assert "src/app.py" in ctx.file_signatures
    assert "def create_app" in ctx.file_signatures["src/app.py"]
    assert "def register_blueprint" in ctx.file_signatures["src/app.py"]


def test_scan_includes_init_py_as_key_files() -> None:
    scanner = RepoScanner(_FIXTURE_REPO)
    ctx = scanner.scan()
    # Every non-ignored __init__.py should be a key file.
    assert "src/__init__.py" in ctx.key_files
    assert "src/models/__init__.py" in ctx.key_files


def test_scan_ignores_vcs_and_caches(tmp_path: Path) -> None:
    repo = tmp_path / "noisy"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("def main():\n    pass\n", encoding="utf-8")
    (repo / ".git").mkdir()
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (repo / "__pycache__").mkdir()
    (repo / "__pycache__" / "a.pyc").write_bytes(b"\x00\x01")

    scanner = RepoScanner(repo)
    ctx = scanner.scan()
    assert ".git" not in ctx.file_tree
    assert "__pycache__" not in ctx.file_tree
    assert "src/app.py" in ctx.file_signatures


def test_scan_respects_target_files_hint(tmp_path: Path) -> None:
    repo = tmp_path / "hinted"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("X = 1\n", encoding="utf-8")
    (repo / "src" / "helpers.py").write_text("Y = 2\n", encoding="utf-8")

    scanner = RepoScanner(repo)
    ctx = scanner.scan(target_files_hint=["src/helpers.py"])
    assert "src/helpers.py" in ctx.key_files
    assert ctx.key_files["src/helpers.py"] == "Y = 2\n"


def test_scan_sheds_tests_when_over_budget(tmp_path: Path) -> None:
    repo = tmp_path / "budget"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "tests").mkdir()

    # Deliberately-big source file so we spend most of the budget on it.
    big = "def f_{n}():\n    return {n}\n"
    (repo / "src" / "app.py").write_text(
        "\n".join(big.format(n=n) for n in range(400)),
        encoding="utf-8",
    )
    (repo / "tests" / "test_app.py").write_text(
        "\n".join(f"def test_{n}():\n    assert True\n" for n in range(400)),
        encoding="utf-8",
    )

    # Tight budget forces shedding.
    scanner = RepoScanner(repo, max_context_tokens=1_000)
    ctx = scanner.scan()
    # Tests signatures are the first thing to go under pressure.
    assert "tests/test_app.py" not in ctx.file_signatures


def test_scan_rejects_non_dir() -> None:
    with pytest.raises(ValueError):
        RepoScanner(_FIXTURE_REPO / "does-not-exist")


def test_scan_rejects_tiny_budget() -> None:
    with pytest.raises(ValueError):
        RepoScanner(_FIXTURE_REPO, max_context_tokens=100)


def test_scan_respects_max_tree_entries(tmp_path: Path) -> None:
    repo = tmp_path / "wide"
    repo.mkdir()
    for n in range(800):
        (repo / f"file_{n:04d}.py").write_text("X = 0\n", encoding="utf-8")

    scanner = RepoScanner(repo, max_context_tokens=60_000)
    ctx = scanner.scan()
    assert "file tree truncated" in ctx.file_tree
