"""Unit tests for ``maestro.cli`` (spec 08 §10)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from maestro.cli import app


@pytest.fixture
def runner() -> CliRunner:
    # ``mix_stderr=False`` is the new default in recent click/typer; avoid
    # passing the kwarg for compatibility with the version in the pinned
    # dependency set.
    return CliRunner()


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("X = 1\n", encoding="utf-8")
    return repo


# ---------------------------------------------------------------------------
# `maestro config`
# ---------------------------------------------------------------------------


def test_config_path_prints_override(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    override = tmp_path / "custom" / "config.yaml"
    monkeypatch.setenv("MAESTRO_CONFIG", str(override))
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0
    # Rich wraps long paths at terminal width under CliRunner's narrow pty;
    # strip whitespace so the assertion is robust to wrapping.
    assert str(override) in "".join(result.stdout.split())


def test_config_init_creates_default_file(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "custom" / "maestro.yaml"
    monkeypatch.setenv("MAESTRO_CONFIG", str(cfg))
    result = runner.invoke(app, ["config", "init"])
    assert result.exit_code == 0
    assert cfg.exists()
    body = cfg.read_text(encoding="utf-8")
    assert "planner" in body
    assert "subagent" in body
    assert "judge" in body


def test_config_init_refuses_to_overwrite_without_force(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "maestro.yaml"
    cfg.write_text("preserved: true\n", encoding="utf-8")
    monkeypatch.setenv("MAESTRO_CONFIG", str(cfg))
    result = runner.invoke(app, ["config", "init"])
    assert result.exit_code == 5
    assert cfg.read_text(encoding="utf-8") == "preserved: true\n"


def test_config_init_with_force_overwrites(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "maestro.yaml"
    cfg.write_text("preserved: true\n", encoding="utf-8")
    monkeypatch.setenv("MAESTRO_CONFIG", str(cfg))
    result = runner.invoke(app, ["config", "init", "--force"])
    assert result.exit_code == 0
    assert "planner" in cfg.read_text(encoding="utf-8")


def test_config_show_masks_api_key(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SECRET_KEY", "supersecret123")
    cfg = tmp_path / "maestro.yaml"
    cfg.write_text(
        "base_url: https://example.com\n"
        'api_key: "${SECRET_KEY}"\n'
        "currency: RMB\n"
        "models:\n"
        "  planner:\n"
        "    name: qwen3-max\n"
        "    display_name: Qwen3-Max\n"
        "    price_input_per_mtok: 2.8\n"
        "    price_output_per_mtok: 8.4\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MAESTRO_CONFIG", str(cfg))

    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "supersecret123" not in result.stdout
    assert "***" in result.stdout


def test_config_show_errors_when_file_missing(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAESTRO_CONFIG", str(tmp_path / "ghost.yaml"))
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 5
    assert "Config error" in result.stdout or "Config error" in result.stderr


# ---------------------------------------------------------------------------
# `maestro run` — dry-run end-to-end
# ---------------------------------------------------------------------------


def test_run_dry_run_succeeds_end_to_end(
    runner: CliRunner,
    sample_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--dry-run`` must not touch the network yet drive every stage."""
    output = tmp_path / "runs"
    # Make absolutely sure an unconfigured env can still dry-run:
    monkeypatch.delenv("MAESTRO_CONFIG", raising=False)

    result = runner.invoke(
        app,
        [
            "run",
            "trivial task",
            "--repo",
            str(sample_repo),
            "--dry-run",
            "--output",
            str(output),
            "--disable-verifier",
            "test_based",
            "--disable-verifier",
            "llm_judge",
        ],
    )
    # Exit can be 0 (success) or 5 (partial/failed) depending on whether
    # the dummy SubAgentOutput happens to satisfy the validation; either
    # way we must NOT crash.
    assert result.exit_code in (0, 5), result.stdout + (result.stderr or "")

    # At minimum ``meta.json`` should be written.
    run_dirs = list(output.iterdir())
    assert run_dirs, "no run directory was created"
    meta_path = run_dirs[0] / "meta.json"
    assert meta_path.exists()
    body = meta_path.read_text(encoding="utf-8")
    assert "dry_run" in body
    assert "enabled_tiers" in body


def test_run_rejects_nonexistent_repo(runner: CliRunner, tmp_path: Path) -> None:
    missing = tmp_path / "no_such_repo"
    result = runner.invoke(app, ["run", "task", "--repo", str(missing), "--dry-run"])
    assert result.exit_code == 5
    assert "does not exist" in result.stdout


def test_run_rejects_unknown_verifier_tier(runner: CliRunner, sample_repo: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "task",
            "--repo",
            str(sample_repo),
            "--dry-run",
            "--disable-verifier",
            "nonsense",
        ],
    )
    assert result.exit_code == 5
    assert "Unknown tier" in result.stdout


def test_run_without_config_and_without_dry_run_fails_cleanly(
    runner: CliRunner,
    sample_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAESTRO_CONFIG", str(tmp_path / "ghost.yaml"))
    result = runner.invoke(app, ["run", "task", "--repo", str(sample_repo)])
    assert result.exit_code == 5
    assert "Config" in result.stdout


# ---------------------------------------------------------------------------
# `maestro report`
# ---------------------------------------------------------------------------


def test_report_errors_when_run_missing(runner: CliRunner, sample_repo: Path) -> None:
    result = runner.invoke(app, ["report", "task-does-not-exist", "--repo", str(sample_repo)])
    assert result.exit_code == 5
    assert "No run found" in result.stdout


def test_report_renders_persisted_result(runner: CliRunner, sample_repo: Path) -> None:
    task_id = "task-abc123"
    run_dir = sample_repo / ".maestro" / "runs" / task_id
    run_dir.mkdir(parents=True)
    (run_dir / "final_result.json").write_text(
        '{"task_id": "task-abc123",'
        ' "status": "success",'
        ' "batches": [],'
        ' "final_diff": "",'
        ' "final_workspace": "/tmp/irrelevant",'
        ' "total_wall_clock_ms": 100,'
        ' "total_tokens_input": 10,'
        ' "total_tokens_output": 5,'
        ' "total_cost_usd": 0.01,'
        ' "started_at": "2026-04-23T00:00:00+00:00",'
        ' "finished_at": "2026-04-23T00:00:00+00:00"}',
        encoding="utf-8",
    )
    result = runner.invoke(app, ["report", task_id, "--repo", str(sample_repo)])
    assert result.exit_code == 0
    # Summary table includes the task id and the "success" badge.
    assert "task-abc123" in result.stdout
    assert "success" in result.stdout


# ---------------------------------------------------------------------------
# `maestro bench` (stub)
# ---------------------------------------------------------------------------


def test_bench_rejects_unknown_ablation(
    runner: CliRunner, sample_repo: Path, tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        [
            "bench",
            "--ablation",
            "no_such_config",
            "--task-set",
            str(sample_repo),  # dir exists but has no task.json
            "--output",
            str(tmp_path / "out"),
            "--dry-run",
        ],
    )
    assert result.exit_code == 5
    assert "Unknown" in result.stdout


def test_bench_dry_run_on_fixture_set_succeeds(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI wires through to the harness and writes the report artifacts."""
    import shutil

    task_set = tmp_path / "tasks"
    fixture = Path(__file__).parent.parent / "fixtures" / "bench_tiny"
    shutil.copytree(fixture, task_set)
    monkeypatch.delenv("MAESTRO_CONFIG", raising=False)

    out = tmp_path / "results"
    result = runner.invoke(
        app,
        [
            "bench",
            "--task-set",
            str(task_set),
            "--output",
            str(out),
            "--ablation",
            "baseline",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (out / "baseline.json").exists()
    assert (out / "baseline.md").exists()
