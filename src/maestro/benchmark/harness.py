"""Benchmark harness (spec 09 §3).

``BenchmarkHarness`` is the single entry point the CLI and scripts both
use. Given a task set directory and a :class:`BenchConfig`, it:

1. Loads every ``<task_set>/<task_id>/task.json``.
2. For each task, materialises a ``TaskSpec``, runs Maestro end-to-end
   via :func:`maestro.orchestrator.build_graph`, and hands the
   resulting ``final_diff`` to :func:`benchmark.evaluator.evaluate`.
3. Assembles a :class:`BenchmarkReport` with per-task + aggregate metrics.
4. Writes the report JSON plus a human-readable Markdown digest.

Tasks run serially inside a config (`parallel_tasks=1` is the default
per spec 09 §3.1). Cross-task parallelism is left to spec 09 §6 follow-up.
"""

from __future__ import annotations

import shutil
import statistics
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from maestro.llm.client import LLMClient
from maestro.llm.config import ClientConfig, ModelConfig
from maestro.models import TaskResult, TaskSpec
from maestro.orchestrator import OrchestratorDeps, build_graph
from maestro.planner.planner import Planner
from maestro.sandbox.workspace import WorkspaceManager
from maestro.subagent.subagent import SubAgentFactory
from maestro.utils.logging import get_logger
from maestro.utils.time import utcnow
from maestro.verifier import Verifier

from .configs import get_config
from .evaluator import evaluate
from .models import (
    BenchmarkReport,
    BenchmarkTask,
    BenchStatus,
    TaskBenchmarkResult,
)

_logger = get_logger("benchmark.harness")


class BenchmarkHarness:
    """Run one :class:`BenchConfig` across a task set and emit a report."""

    def __init__(
        self,
        *,
        task_set_dir: Path,
        output_dir: Path,
        config_name: str,
        llm_config: ClientConfig | None = None,
        dry_run: bool = False,
    ) -> None:
        if not task_set_dir.exists() or not task_set_dir.is_dir():
            raise ValueError(f"Task set dir does not exist: {task_set_dir}")

        self._task_set = task_set_dir
        self._output = output_dir
        self._config_name = config_name
        self._bench_config = get_config(config_name)
        self._dry_run = dry_run
        self._llm_config = llm_config or _default_llm_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_all(self, limit: int | None = None) -> BenchmarkReport:
        """Run every task (or the first ``limit``). Returns an aggregate report."""
        tasks = self._load_tasks(limit=limit)
        _logger.info(
            "bench_run_start",
            config=self._config_name,
            task_count=len(tasks),
            dry_run=self._dry_run,
        )

        started = utcnow()
        results: list[TaskBenchmarkResult] = []
        for task in tasks:
            result = await self.run_single(task)
            results.append(result)
        finished = utcnow()

        report = _build_report(
            config_name=self._config_name,
            results=results,
            started=started,
            finished=finished,
        )
        self._persist_report(report)
        return report

    async def run_single(self, task: BenchmarkTask) -> TaskBenchmarkResult:
        """Run one benchmark task through Maestro, evaluate, return a record."""
        before = self._task_set / task.task_id / "before"
        after = self._task_set / task.task_id / "after"
        if not before.exists():
            return _error_result(
                task.task_id,
                self._config_name,
                "missing `before/` directory",
            )

        start = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix=f"bench-{task.task_id}-") as tmpdir:
            repo = Path(tmpdir) / "repo"
            shutil.copytree(before, repo)
            spec = self._build_task_spec(task, repo)
            try:
                final_state = await self._run_orchestrator(spec)
            except Exception as exc:
                _logger.error(
                    "bench_run_crash",
                    task_id=task.task_id,
                    config=self._config_name,
                    error=exc.__class__.__name__,
                    message=str(exc),
                )
                wall_ms = int((time.perf_counter() - start) * 1000)
                return TaskBenchmarkResult(
                    task_id=task.task_id,
                    config_name=self._config_name,
                    status="error",
                    resolved=False,
                    wall_clock_ms=wall_ms,
                    total_cost=0.0,
                    total_tokens_input=0,
                    total_tokens_output=0,
                    error=str(exc),
                )

            if error := final_state.get("error"):
                wall_ms = int((time.perf_counter() - start) * 1000)
                return TaskBenchmarkResult(
                    task_id=task.task_id,
                    config_name=self._config_name,
                    status="error",
                    resolved=False,
                    wall_clock_ms=wall_ms,
                    total_cost=0.0,
                    total_tokens_input=0,
                    total_tokens_output=0,
                    error=str(error),
                )

            task_result: TaskResult | None = final_state.get("final_result")
            if task_result is None:
                return _error_result(
                    task.task_id,
                    self._config_name,
                    "no final_result from orchestrator",
                )

            eval_result = await evaluate(
                task, before_dir=before, after_dir=after, final_diff=task_result.final_diff
            )

        wall_ms = int((time.perf_counter() - start) * 1000)
        status: BenchStatus = _status_from(eval_result.resolved, eval_result.error)
        return TaskBenchmarkResult(
            task_id=task.task_id,
            config_name=self._config_name,
            status=status,
            resolved=eval_result.resolved,
            wall_clock_ms=wall_ms,
            total_cost=task_result.total_cost_usd,
            total_tokens_input=task_result.total_tokens_input,
            total_tokens_output=task_result.total_tokens_output,
            patch_similarity=eval_result.patch_similarity,
            files_modified_match=eval_result.files_modified_match,
            extra_files_modified=eval_result.extra_files_modified,
            error=eval_result.error,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_tasks(self, *, limit: int | None) -> list[BenchmarkTask]:
        tasks: list[BenchmarkTask] = []
        for child in sorted(self._task_set.iterdir()):
            if not child.is_dir():
                continue
            task_json = child / "task.json"
            if not task_json.exists():
                continue
            tasks.append(BenchmarkTask.model_validate_json(task_json.read_text(encoding="utf-8")))
            if limit is not None and len(tasks) >= limit:
                break
        return tasks

    def _build_task_spec(self, task: BenchmarkTask, repo: Path) -> TaskSpec:
        return TaskSpec(
            task_id=f"bench-{task.task_id}-{uuid.uuid4().hex[:6]}",
            description=task.natural_language_prompt,
            repo_path=repo,
            target_files_hint=list(task.files_hint) or None,
            max_parallel=self._bench_config.max_parallel,
            max_retries_per_subtask=self._bench_config.max_retries,
            judge_samples=self._bench_config.judge.samples,
            judge_disagreement_threshold=self._bench_config.judge.disagreement_threshold,
            # Benchmark always uses ground-truth tests; never auto-gen.
            auto_gen_tests=False,
        )

    async def _run_orchestrator(self, spec: TaskSpec) -> dict[str, Any]:
        client = LLMClient(self._llm_config, dry_run=self._dry_run)
        planner = Planner(client)
        factory = SubAgentFactory()
        with WorkspaceManager(spec.repo_path, spec.task_id) as workspace:
            verifier = Verifier(
                client,
                spec,
                workspace_manager=workspace,
                # NOTE: pass the set as-is. Do NOT collapse ``set() or None`` —
                # Verifier treats ``None`` as "use default (all 3 tiers)" and
                # an empty set as "explicitly disable all tiers". The
                # ``baseline`` and ``parallel_only`` configs depend on the
                # latter, and collapsing them silently re-enables verify.
                enabled_tiers=set(self._bench_config.enabled_tiers),  # type: ignore[arg-type]
            )
            deps = OrchestratorDeps(
                planner=planner,
                verifier=verifier,
                workspace=workspace,
                llm_client=client,
                subagent_factory=factory,
            )
            graph = build_graph(deps)
            result: dict[str, Any] = await graph.ainvoke({"task_spec": spec})
            return result

    def _persist_report(self, report: BenchmarkReport) -> None:
        self._output.mkdir(parents=True, exist_ok=True)
        json_path = self._output / f"{self._config_name}.json"
        json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        md_path = self._output / f"{self._config_name}.md"
        md_path.write_text(_render_report_markdown(report), encoding="utf-8")
        _logger.info(
            "bench_run_done",
            config=self._config_name,
            resolve_rate=round(report.resolve_rate, 3),
            json=str(json_path),
        )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _error_result(task_id: str, config_name: str, error: str) -> TaskBenchmarkResult:
    return TaskBenchmarkResult(
        task_id=task_id,
        config_name=config_name,
        status="error",
        resolved=False,
        wall_clock_ms=0,
        total_cost=0.0,
        total_tokens_input=0,
        total_tokens_output=0,
        error=error,
    )


def _status_from(resolved: bool, err: str | None) -> BenchStatus:
    if resolved:
        return "resolved"
    if err and "diff apply failed" in err:
        return "apply_failed"
    return "unresolved"


def _build_report(
    *,
    config_name: str,
    results: list[TaskBenchmarkResult],
    started: datetime,
    finished: datetime,
) -> BenchmarkReport:
    n = len(results)
    resolved = sum(1 for r in results if r.resolved)
    return BenchmarkReport(
        run_id=f"bench-{int(started.timestamp())}-{uuid.uuid4().hex[:6]}",
        config_name=config_name,
        task_count=n,
        resolve_rate=(resolved / n) if n else 0.0,
        avg_wall_clock_ms=_mean([r.wall_clock_ms for r in results]),
        avg_cost=_mean([r.total_cost for r in results]),
        avg_tokens_input=_mean([r.total_tokens_input for r in results]),
        avg_tokens_output=_mean([r.total_tokens_output for r in results]),
        per_task_results=results,
        started_at=started,
        finished_at=finished,
    )


def _mean(values: list[float] | list[int]) -> float:
    if not values:
        return 0.0
    return float(statistics.mean(values))


def _render_report_markdown(report: BenchmarkReport) -> str:
    lines = [
        f"# Benchmark run: `{report.config_name}`",
        "",
        f"- Run ID: `{report.run_id}`",
        f"- Tasks: **{report.task_count}**",
        f"- Resolve rate: **{report.resolve_rate:.1%}**",
        f"- Avg wall-clock: **{report.avg_wall_clock_ms / 1000:.1f} s**",
        f"- Avg cost: **{report.avg_cost:.4f}**",
        f"- Avg tokens: {report.avg_tokens_input:.0f} in / {report.avg_tokens_output:.0f} out",
        "",
        "## Per-task",
        "",
        "| Task | Status | Resolved | Cost | Tokens in/out | Wall-clock (s) | Similarity |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in report.per_task_results:
        lines.append(
            f"| {r.task_id} | {r.status} | {'✔' if r.resolved else '✘'} | "
            f"{r.total_cost:.4f} | {r.total_tokens_input}/{r.total_tokens_output} | "
            f"{r.wall_clock_ms / 1000:.1f} | {r.patch_similarity:.2f} |"
        )
    return "\n".join(lines) + "\n"


def _default_llm_config() -> ClientConfig:
    """A zero-cost stub config suitable for ``--dry-run`` harness smoke tests."""
    return ClientConfig(
        base_url="https://example.com/v1",
        api_key="dry-run",
        models={
            "planner": ModelConfig(
                name="qwen3-max",
                display_name="Qwen3-Max",
                price_input_per_mtok=0.0,
                price_output_per_mtok=0.0,
            ),
            "subagent": ModelConfig(
                name="qwen3-coder-plus",
                display_name="Qwen3-Coder-Plus",
                price_input_per_mtok=0.0,
                price_output_per_mtok=0.0,
            ),
            "judge": ModelConfig(
                name="deepseek-v3",
                display_name="DeepSeek-V3",
                price_input_per_mtok=0.0,
                price_output_per_mtok=0.0,
            ),
        },
    )


__all__ = ["BenchmarkHarness"]
