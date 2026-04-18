# Spec 08 — CLI

> **位置**：`src/maestro/cli.py`
> **依赖**：Typer, Rich, Orchestrator
> **被依赖**：用户入口

## 1. 命令设计

Maestro 提供以下命令：

```bash
# 主命令：执行一个任务
maestro run "Add user authentication" --repo ./my-project

# Dry run：不调 LLM，只验证 pipeline
maestro run "..." --repo ./my-project --dry-run

# Benchmark：跑 benchmark 任务集
maestro bench --task-set benchmark/tasks/ --output benchmark/results/run-001.json

# Benchmark with ablation
maestro bench --task-set benchmark/tasks/ --ablation parallel_only
maestro bench --task-set benchmark/tasks/ --ablation verify_none

# Show config
maestro config show

# Show cost report for a previous run
maestro report <run_id>
```

## 2. 主入口

```python
# src/maestro/cli.py

app = typer.Typer(help="Maestro: parallelized coding agent framework with verification.")


@app.command()
def run(
    task: str = typer.Argument(..., help="High-level task description"),
    repo: Path = typer.Option(Path.cwd(), "--repo", "-r", help="Path to target repo"),
    max_parallel: int = typer.Option(4, "--max-parallel", "-p", help="Max concurrent sub-agents"),
    max_retries: int = typer.Option(2, "--max-retries", help="Max retries per subtask"),
    judge_samples: int = typer.Option(3, "--judge-samples", "-k", help="K samples for LLM judge"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Skip actual LLM calls"),
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="Where to save artifacts"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    disable_verifier: list[str] = typer.Option([], "--disable-verifier", help="Disable tiers: deterministic / test_based / llm_judge"),
):
    """Run a coding task through Maestro."""
    ...
```

## 3. 交互 UI（Rich）

### 3.1 进度可视化

CLI 运行时展示分阶段进度：

```
🎼 Maestro
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 Task: Add user authentication with signup, login, reset
📁 Repo: ./my-project

[1/5] Planning…  ✅  Decomposed into 4 subtasks across 2 batches
[2/5] Scheduling…  ✅  Batch 0: [signup, login, reset] parallel

┌─ Batch 0 ─────────────────────────────────────────────┐
│  ⏳ signup      [explore] reading src/models/user.py │
│  ✅ login       tier1 pass, tier2 pass, judge 0.82    │
│  🔄 reset       retry 1/2 (tier1 failed: E501)        │
└───────────────────────────────────────────────────────┘

[3/5] Merging patches…  ✅
[4/5] Running batch 1…
[5/5] Report…

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✨ Summary
  Resolved: 4/4 subtasks
  Wall-clock: 3m 24s
  Tokens: 142k in, 18k out
  Cost: $0.31
  Judge uncertainty: 0 subtasks

📄 Diff saved to: .maestro/runs/<task_id>/final.diff
```

使用 Rich 的 `Live`、`Progress`、`Panel` 组件实现。

### 3.2 Verbose mode

`-v` 开启后，每个 sub-agent 的 explore/plan/write 步骤都 log 出来。

### 3.3 Non-TTY 模式

检测 `sys.stdout.isatty()`，非 TTY 时输出纯文本进度（用于 CI、benchmark）。

## 4. 输出 artifact

每次 `run` 命令产生一个 run 目录：

```
.maestro/runs/<task_id>/
├── meta.json              # TaskSpec + config used
├── plan.json              # Planner output (TaskDAG serialized)
├── batches/
│   ├── batch-0.json       # BatchResult
│   └── batch-1.json
├── transcripts/           # Full LLM transcripts (gzipped)
│   ├── planner.jsonl.gz
│   ├── subagent-001.jsonl.gz
│   └── ...
├── final.diff             # Final unified diff
├── final_result.json      # TaskResult
└── cost_report.md         # Human-readable cost summary
```

**目录位置**：默认 `<repo>/.maestro/runs/<task_id>/`。通过 `--output` 覆盖。

## 5. Config 命令

```python
@app.command("config")
def config_cmd(
    action: str = typer.Argument(..., help="show / init / path"),
):
    """Manage Maestro config."""
    if action == "show":
        cfg = load_config()
        console.print(cfg.model_dump_json(indent=2))
    elif action == "init":
        _create_default_config()
        console.print(f"Wrote default config to {DEFAULT_CONFIG_PATH}")
    elif action == "path":
        console.print(str(DEFAULT_CONFIG_PATH))
```

## 6. Report 命令

```python
@app.command()
def report(
    run_id: str = typer.Argument(..., help="Task ID"),
    repo: Path = typer.Option(Path.cwd(), "--repo", "-r"),
):
    """Show report for a previous run."""
    run_dir = repo / ".maestro" / "runs" / run_id
    result = TaskResult.model_validate_json((run_dir / "final_result.json").read_text())
    _render_report(result)
```

## 7. Benchmark 命令

详见 `specs/09-benchmark.md`，CLI 入口：

```python
@app.command("bench")
def bench_cmd(
    task_set: Path = typer.Option(Path("benchmark/tasks/"), help="Task set dir"),
    output: Path = typer.Option(Path("benchmark/results/"), help="Where to write results"),
    ablation: str = typer.Option("full", help="full / parallel_only / verify_none / no_judge"),
    parallel_tasks: int = typer.Option(1, "--parallel-tasks", help="How many benchmark tasks to run in parallel"),
    limit: int | None = typer.Option(None, "--limit", help="Run only first N tasks"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Run benchmark suite."""
    ...
```

## 8. 错误处理

统一的异常到 exit code 映射：

| 异常 | Exit code | 输出 |
|---|---|---|
| PlanningError | 2 | stderr + suggestion |
| LLMCallError | 3 | stderr + retry hint |
| MergeConflictError | 4 | stderr + conflicted files |
| ValueError / TypeError | 5 | stderr + traceback (verbose only) |
| KeyboardInterrupt | 130 | 清理临时目录后退出 |
| Normal completion | 0 | 正常总结 |

## 9. 环境变量

| 变量 | 作用 |
|---|---|
| `DASHSCOPE_API_KEY` | 阿里云百炼 API key（必须） |
| `MAESTRO_CONFIG` | 覆盖默认 config 路径 |
| `MAESTRO_LOG_LEVEL` | DEBUG / INFO / WARNING / ERROR (default INFO) |
| `MAESTRO_LOG_LLM_CONTENT` | 1 = log full prompts/outputs (debug only) |
| `MAESTRO_SKIP_MYPY` | 1 = skip mypy in Tier 1 verifier |

## 10. 测试要求

`tests/unit/test_cli.py`（使用 typer.testing.CliRunner）：
- `maestro run` 参数解析
- `maestro config show` 输出
- `maestro bench` 参数解析
- Exit code 映射

`tests/integration/test_cli_e2e.py`：
- `maestro run --dry-run` 端到端（不调 API）

## 11. 面试 talking points

1. **Artifact 分层存储**：transcripts 单独 gzip、plan/batch 单独 JSON、final.diff 纯文本——每种消费者（用户、benchmark harness、调试）都能高效取用
2. **Rich + Live + Non-TTY 降级**：同一套逻辑适配交互和 CI 场景
3. **Config init 命令**：降低上手门槛，是开源项目该有的体贴
