"""Maestro CLI (spec 08).

Commands
--------

* ``maestro run <task>`` — execute one coding task through the full pipeline.
* ``maestro config show/init/path`` — inspect / create the LLM config.
* ``maestro report <task_id>`` — re-render a past run's summary.
* ``maestro bench`` — placeholder; the real harness ships in module 15.

Artifact layout (spec 08 §4)
----------------------------

Every ``run`` writes to ``<repo>/.maestro/runs/<task_id>/``::

    meta.json            TaskSpec + config + CLI flags
    batches/batch-N.json One BatchResult per batch
    final.diff           The user's diff to review / apply
    final_result.json    TaskResult (status, cost, timings)
    cost_report.md       Per-role / per-model cost markdown

Exit codes (spec 08 §8)
-----------------------

* 0 success, 2 PlanningError, 3 LLMCallError, 4 MergeConflictError,
  5 other value/type, 130 KeyboardInterrupt.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from maestro.llm.client import LLMClient
from maestro.llm.config import (
    ClientConfig,
    ModelConfig,
    load_config,
    resolve_config_path,
)
from maestro.llm.errors import LLMCallError, LLMConfigError
from maestro.models import TaskResult, TaskSpec, generate_task_id
from maestro.orchestrator import OrchestratorDeps, build_graph
from maestro.planner.planner import Planner, PlanningError
from maestro.sandbox.workspace import MergeConflictError, WorkspaceManager
from maestro.subagent.subagent import SubAgentFactory
from maestro.utils.logging import configure_logging
from maestro.verifier import Verifier

app = typer.Typer(
    help="Maestro: parallelized coding agent framework with verification.",
    no_args_is_help=True,
)
config_app = typer.Typer(help="Manage Maestro config.", no_args_is_help=True)
app.add_typer(config_app, name="config")

console = Console()


# ---------------------------------------------------------------------------
# `maestro run`
# ---------------------------------------------------------------------------


@app.command()
def run(
    task: str = typer.Argument(..., help="High-level task description"),
    repo: Path = typer.Option(Path.cwd(), "--repo", "-r", help="Path to target repo"),
    max_parallel: int = typer.Option(4, "--max-parallel", "-p", help="Max concurrent sub-agents"),
    max_retries: int = typer.Option(
        2, "--max-retries", help="Max retries per subtask after verify failure"
    ),
    judge_samples: int = typer.Option(3, "--judge-samples", "-k", help="K samples for LLM judge"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Skip real LLM calls (pipeline smoke test)"
    ),
    output_dir: Path | None = typer.Option(
        None, "--output", "-o", help="Override .maestro/runs output directory"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    disable_verifier: list[str] = typer.Option(
        [],
        "--disable-verifier",
        help="Disable tier(s): deterministic / test_based / llm_judge",
    ),
    auto_gen_tests: bool = typer.Option(
        False,
        "--auto-gen-tests",
        help="Let the verifier auto-generate tests when none are found",
    ),
) -> None:
    """Run a coding task through Maestro."""
    _configure_logging(verbose)
    repo = _require_repo(repo)

    enabled_tiers = _compute_enabled_tiers(disable_verifier)

    try:
        cfg = _load_config_for_run(dry_run)
    except LLMConfigError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(code=5) from exc

    spec = _build_task_spec(
        task=task,
        repo=repo,
        max_parallel=max_parallel,
        max_retries=max_retries,
        judge_samples=judge_samples,
        auto_gen_tests=auto_gen_tests,
    )

    run_dir = _resolve_run_dir(output_dir, repo, spec.task_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    exit_code = asyncio.run(
        _execute_run(
            spec=spec,
            cfg=cfg,
            run_dir=run_dir,
            enabled_tiers=enabled_tiers,
            dry_run=dry_run,
        )
    )
    raise typer.Exit(code=exit_code)


async def _execute_run(
    *,
    spec: TaskSpec,
    cfg: ClientConfig,
    run_dir: Path,
    enabled_tiers: set[str],
    dry_run: bool,
) -> int:
    client = LLMClient(cfg, dry_run=dry_run)
    planner = Planner(client)
    factory = SubAgentFactory()

    _write_json(run_dir / "meta.json", _meta_payload(spec, cfg, enabled_tiers, dry_run))

    try:
        with WorkspaceManager(spec.repo_path, spec.task_id) as workspace:
            verifier = Verifier(
                client,
                spec,
                workspace_manager=workspace,
                enabled_tiers=enabled_tiers or None,  # type: ignore[arg-type]
            )
            deps = OrchestratorDeps(
                planner=planner,
                verifier=verifier,
                workspace=workspace,
                llm_client=client,
                subagent_factory=factory,
            )
            graph = build_graph(deps)
            final_state = await graph.ainvoke({"task_spec": spec})
    except PlanningError as exc:
        console.print(f"[red]Planner failed:[/red] {exc}")
        return 2
    except LLMCallError as exc:
        console.print(f"[red]LLM call failed:[/red] {exc}")
        return 3
    except MergeConflictError as exc:
        console.print(f"[red]Merge conflict:[/red] {exc}")
        return 4
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted[/yellow]")
        return 130

    if error := final_state.get("error"):
        console.print(f"[red]Task failed:[/red] {error}")
        _write_json(run_dir / "final_result.json", {"error": error})
        return 5

    result: TaskResult | None = final_state.get("final_result")
    if result is None:
        console.print("[red]Task produced no final result[/red]")
        return 5

    await _persist_run_artifacts(run_dir, result, client)
    _render_summary(result, run_dir)
    return 0 if result.status == "success" else 5


# ---------------------------------------------------------------------------
# `maestro config`
# ---------------------------------------------------------------------------


@config_app.command("show")
def config_show() -> None:
    """Print the currently-resolved config (with secrets masked)."""
    try:
        cfg = load_config()
    except LLMConfigError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(code=5) from exc
    masked = _mask_secrets(cfg)
    console.print_json(data=masked)


@config_app.command("init")
def config_init(
    force: bool = typer.Option(False, "--force", help="Overwrite existing config"),
) -> None:
    """Write a default config file to the resolved path."""
    path = resolve_config_path()
    if path.exists() and not force:
        console.print(
            f"[yellow]Config already exists at {path}. Use --force to overwrite.[/yellow]"
        )
        raise typer.Exit(code=5)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_default_config_yaml(), encoding="utf-8")
    console.print(f"Wrote default config to {path}")


@config_app.command("path")
def config_path() -> None:
    """Print the resolved config file path."""
    console.print(str(resolve_config_path()))


# ---------------------------------------------------------------------------
# `maestro report`
# ---------------------------------------------------------------------------


@app.command()
def report(
    task_id: str = typer.Argument(..., help="Task ID to render"),
    repo: Path = typer.Option(Path.cwd(), "--repo", "-r"),
) -> None:
    """Re-render the summary for a past run."""
    run_dir = repo / ".maestro" / "runs" / task_id
    final_path = run_dir / "final_result.json"
    if not final_path.exists():
        console.print(f"[red]No run found at {run_dir}[/red]")
        raise typer.Exit(code=5)

    try:
        result = TaskResult.model_validate_json(final_path.read_text(encoding="utf-8"))
    except Exception as exc:
        console.print(f"[red]Could not parse {final_path}:[/red] {exc}")
        raise typer.Exit(code=5) from exc

    _render_summary(result, run_dir)


# ---------------------------------------------------------------------------
# `maestro bench` (stub — module 15 owns the harness)
# ---------------------------------------------------------------------------


@app.command()
def bench(
    task_set: Path = typer.Option(Path("benchmark/tasks/"), "--task-set", help="Task set dir"),
    output: Path = typer.Option(
        Path("benchmark/results/"), "--output", help="Where to write results"
    ),
    ablation: str = typer.Option("full", "--ablation"),
    parallel_tasks: int = typer.Option(1, "--parallel-tasks"),
    limit: int | None = typer.Option(None, "--limit"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Run the benchmark suite (wiring lives in module 15)."""
    del task_set, output, ablation, parallel_tasks, limit, dry_run
    console.print(
        "[yellow]`maestro bench` dispatches to the module-15 harness, "
        "which is not yet implemented.[/yellow]"
    )
    raise typer.Exit(code=5)


# ---------------------------------------------------------------------------
# Internals — config / spec / artifacts / rendering
# ---------------------------------------------------------------------------


def _configure_logging(verbose: bool) -> None:
    if verbose and not os.getenv("MAESTRO_LOG_LEVEL"):
        os.environ["MAESTRO_LOG_LEVEL"] = "DEBUG"
    configure_logging(force=True)


def _require_repo(repo: Path) -> Path:
    repo = repo.resolve()
    if not repo.exists() or not repo.is_dir():
        console.print(f"[red]Repo path does not exist:[/red] {repo}")
        raise typer.Exit(code=5)
    return repo


def _compute_enabled_tiers(disable: list[str]) -> set[str]:
    all_tiers = {"deterministic", "test_based", "llm_judge"}
    unknown = set(disable) - all_tiers
    if unknown:
        console.print(f"[red]Unknown tier(s) in --disable-verifier:[/red] {sorted(unknown)}")
        raise typer.Exit(code=5)
    return all_tiers - set(disable)


def _load_config_for_run(dry_run: bool) -> ClientConfig:
    """Load the client config, with a sensible fallback under dry-run."""
    path = resolve_config_path()
    if path.exists():
        return load_config(path)
    if dry_run:
        # Dry-run pipelines don't actually hit the provider; synthesise a
        # minimal config so users can smoke-test without a config file.
        return ClientConfig(
            base_url="https://example.com/v1",
            api_key="dry-run",
            models={
                "planner": _dummy_model("qwen3-max"),
                "subagent": _dummy_model("qwen3-coder-plus"),
                "judge": _dummy_model("deepseek-v3"),
            },
        )
    raise LLMConfigError(f"Config file not found at {path}. Run `maestro config init` first.")


def _dummy_model(name: str) -> ModelConfig:
    return ModelConfig(
        name=name,
        display_name=name,
        price_input_per_mtok=0.0,
        price_output_per_mtok=0.0,
    )


def _build_task_spec(
    *,
    task: str,
    repo: Path,
    max_parallel: int,
    max_retries: int,
    judge_samples: int,
    auto_gen_tests: bool,
) -> TaskSpec:
    return TaskSpec(
        task_id=generate_task_id(),
        description=task,
        repo_path=repo,
        max_parallel=max_parallel,
        max_retries_per_subtask=max_retries,
        judge_samples=judge_samples,
        auto_gen_tests=auto_gen_tests,
    )


def _resolve_run_dir(output: Path | None, repo: Path, task_id: str) -> Path:
    base = output if output is not None else repo / ".maestro" / "runs"
    return (base / task_id).resolve()


async def _persist_run_artifacts(run_dir: Path, result: TaskResult, client: LLMClient) -> None:
    """Write the post-run artifacts that ``maestro report`` consumes.

    Note: plan.json was in an earlier spec draft but is redundant —
    BatchResult entries already carry the per-subtask TaskDAG information.
    If future debugging needs the raw DAG, add it here alongside
    final_result.json.
    """
    _write_text(run_dir / "final.diff", result.final_diff)
    _write_text(
        run_dir / "final_result.json",
        result.model_dump_json(indent=2),
    )

    batches_dir = run_dir / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    for batch in result.batches:
        _write_text(
            batches_dir / f"batch-{batch.batch_index}.json",
            batch.model_dump_json(indent=2),
        )

    report_md = (await client.get_cost_report()).to_markdown()
    _write_text(run_dir / "cost_report.md", report_md)


def _meta_payload(
    spec: TaskSpec,
    cfg: ClientConfig,
    enabled_tiers: set[str],
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "task_spec": json.loads(spec.model_dump_json()),
        "currency": cfg.currency,
        "enabled_tiers": sorted(enabled_tiers),
        "dry_run": dry_run,
    }


def _mask_secrets(cfg: ClientConfig) -> dict[str, Any]:
    raw: dict[str, Any] = json.loads(cfg.model_dump_json())
    if raw.get("api_key"):
        raw["api_key"] = _mask(raw["api_key"])
    return raw


def _mask(value: str) -> str:
    if len(value) <= 4:
        return "***"
    return value[:2] + "***" + value[-2:]


def _default_config_yaml() -> str:
    return (
        'base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"\n'
        'api_key: "${DASHSCOPE_API_KEY}"\n'
        'currency: "RMB"\n'
        "global_semaphore_limit: 10\n"
        "models:\n"
        "  planner:\n"
        "    name: qwen3-max\n"
        "    display_name: Qwen3-Max\n"
        "    price_input_per_mtok: 2.8\n"
        "    price_output_per_mtok: 8.4\n"
        "  subagent:\n"
        "    name: qwen3-coder-plus\n"
        "    display_name: Qwen3-Coder-Plus\n"
        "    price_input_per_mtok: 0.84\n"
        "    price_output_per_mtok: 3.36\n"
        "  judge:\n"
        "    name: deepseek-v3\n"
        "    display_name: DeepSeek-V3\n"
        "    price_input_per_mtok: 0.28\n"
        "    price_output_per_mtok: 1.12\n"
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, default=str) + "\n")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_summary(result: TaskResult, run_dir: Path) -> None:
    merged = sum(len(b.merged_patches) for b in result.batches)
    failed = sum(len(b.failed_patches) for b in result.batches)
    total = sum(len(b.subtask_results) for b in result.batches)
    judge_uncertain = sum(
        1
        for b in result.batches
        for v in b.verification_results
        if v.judge_detail is not None and v.judge_detail.is_uncertain
    )

    table = Table(show_header=False, box=None)
    table.add_row("Status", _status_badge(result.status))
    table.add_row("Resolved", f"{merged}/{total} subtasks")
    if failed:
        table.add_row("Failed", str(failed))
    table.add_row("Wall-clock", _format_ms(result.total_wall_clock_ms))
    table.add_row(
        "Tokens",
        f"{result.total_tokens_input} in, {result.total_tokens_output} out",
    )
    table.add_row("Cost", f"{result.total_cost_usd:.4f}")
    table.add_row("Judge uncertainty", str(judge_uncertain))
    table.add_row("Run dir", str(run_dir))

    console.print(Panel(table, title=f"Task {result.task_id}", expand=False))


def _status_badge(status: str) -> str:
    colour = {"success": "green", "partial": "yellow", "failed": "red"}.get(status, "white")
    return f"[{colour}]{status}[/{colour}]"


def _format_ms(ms: int) -> str:
    if ms < 1000:
        return f"{ms} ms"
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f} s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted[/yellow]")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["app", "main"]
