# Spec 09 — Benchmark

> **位置**：`benchmark/`
> **依赖**：`src/maestro/` 所有模块
> **被依赖**：无（独立可执行）

**Benchmark 是简历数字的来源**。必须认真设计。

## 1. 目标

在一个**可复现、真实、小规模**的 Python bug-fix benchmark 上，对比：

- Maestro 各种配置（full / parallel_only / verify_none / no_judge）
- Single-agent baseline（串行、无验证）

产出硬数字：
- **Resolve rate**：patch 通过 ground-truth 测试的比例
- **Wall-clock speedup**：Maestro vs baseline 的时间比
- **Token cost**：每任务平均 input/output token 和美元成本
- **Pareto frontier**：cost vs resolve rate 的 scatter plot

## 2. Benchmark 数据设计

### 2.1 数据源

从以下 5 个 Python 项目挖**真实 bug-fix PR**：

| Repo | 选择理由 |
|---|---|
| `psf/requests` | 经典 HTTP 库，活跃维护，测试充分 |
| `pallets/click` | CLI 框架，中等规模，易隔离 |
| `pallets/flask` | Web 框架，经典 |
| `encode/httpx` | 现代 HTTP 客户端 |
| `tiangolo/typer` | CLI 框架（依赖 click），简单清晰 |

### 2.2 PR 挑选标准

**每个 PR 必须满足**：

1. 是 bug-fix（有 issue 链接，PR 标题包含 "fix"）
2. 修改文件数 ≤ 3
3. 改动行数 ≤ 50
4. 有关联的 test（failing test 变 passing 的模式）
5. 不涉及外部系统（网络、文件系统副作用较少）
6. merge 已久（至少 6 个月）避免数据污染

### 2.3 数据格式

每个 benchmark task 是一个目录：

```
benchmark/tasks/
├── requests-7654/
│   ├── task.json            # metadata
│   ├── before/              # repo snapshot before fix
│   ├── after/               # repo snapshot after fix (ground truth)
│   ├── failing_test.json    # which test was failing before
│   └── expected_files.json  # which files should be modified
├── click-2103/
└── ...
```

`task.json`:
```json
{
  "task_id": "requests-7654",
  "repo": "psf/requests",
  "pr_url": "https://github.com/psf/requests/pull/7654",
  "issue_url": "https://github.com/psf/requests/issues/7650",
  "description": "Fix: Session.send drops custom adapters when following redirects",
  "natural_language_prompt": "When a Session follows a redirect, it should preserve the custom adapter for the new URL. Currently it falls back to the default adapter.",
  "failing_tests": ["tests/test_session.py::test_adapter_preserved_on_redirect"],
  "expected_modified_files": ["src/requests/sessions.py"],
  "files_hint": ["src/requests/sessions.py", "src/requests/adapters.py"],
  "difficulty": "medium",
  "source_commit": "abc123..."
}
```

### 2.4 目标规模

**30-40 个 task**：

| Repo | 目标数量 |
|---|---|
| requests | 8-10 |
| click | 6-8 |
| flask | 6-8 |
| httpx | 6-8 |
| typer | 4-6 |

### 2.5 数据构建流程

构建脚本 `benchmark/build_tasks.py`：

```bash
# For each repo:
python benchmark/build_tasks.py \
    --repo psf/requests \
    --output benchmark/tasks/ \
    --target-count 10 \
    --since 2023-01-01 --until 2024-12-31
```

脚本逻辑：
1. 用 GitHub API 列出时间范围内的 merged PR
2. 过滤：标题含 "fix"、修改文件 ≤ 3、改动 ≤ 50 行
3. 对每个 PR：clone base commit、clone fix commit、提取 failing test
4. 运行 base commit 的 failing test 确认它真的 fail
5. 运行 fix commit 的 failing test 确认它 pass
6. 保存为标准格式

**人工验证**：脚本自动筛出候选后，**人工 review 最终选入 benchmark 的 PR**，确保质量。

### 2.6 Natural language prompt 生成

`task.natural_language_prompt` 不直接用 PR 标题（可能泄漏答案）。构造规则：

1. 用 Qwen3-Max 读 failing test + bug description（不读 fix），生成用户视角的 prompt
2. 人工审核 prompt，确保描述的是"需要做什么"而不是"怎么做"
3. 比如不说"修改 sessions.py 的 resolve_redirects 方法"，而说"用户反馈在 redirect 场景下自定义 adapter 不生效"

这个 prompt 生成过程本身要在文档中明确，避免 benchmark 数据泄漏争议。

## 3. Harness

### 3.1 接口

```python
# benchmark/harness.py

class BenchmarkHarness:
    """Runs Maestro (or baseline) on a set of benchmark tasks."""

    def __init__(
        self,
        task_set_dir: Path,
        output_dir: Path,
        config_name: str,  # "full" / "parallel_only" / "verify_none" / "baseline"
        parallel_tasks: int = 1,
    ):
        ...

    async def run_all(self, limit: int | None = None) -> BenchmarkReport:
        """Run all tasks (or first `limit`), return aggregate report."""
        ...

    async def run_single(self, task_dir: Path) -> TaskBenchmarkResult:
        """Run one task, evaluate, return result."""
        ...
```

### 3.2 单任务执行流程

```python
async def run_single(self, task_dir):
    meta = json.loads((task_dir / "task.json").read_text())

    # 1. Prepare fresh copy of `before/` as the target repo
    work_dir = self._output_dir / f"run-{meta['task_id']}"
    shutil.copytree(task_dir / "before", work_dir / "repo")

    # 2. Construct TaskSpec from meta
    task_spec = TaskSpec(
        task_id=f"bench-{meta['task_id']}",
        description=meta["natural_language_prompt"],
        repo_path=work_dir / "repo",
        target_files_hint=meta.get("files_hint"),
        ...
    )

    # 3. Run Maestro (or baseline)
    orchestrator = build_orchestrator_for_config(self._config_name, task_spec)
    start = time.perf_counter()
    result = await orchestrator.run(task_spec)
    wall_clock_ms = int((time.perf_counter() - start) * 1000)

    # 4. Evaluate: apply final_diff to before/, run failing_tests
    evaluation = await self._evaluate(
        final_diff=result.final_diff,
        task_dir=task_dir,
        failing_tests=meta["failing_tests"],
    )

    return TaskBenchmarkResult(
        task_id=meta["task_id"],
        resolved=evaluation.resolved,
        wall_clock_ms=wall_clock_ms,
        total_cost_usd=result.total_cost_usd,
        total_tokens_input=result.total_tokens_input,
        total_tokens_output=result.total_tokens_output,
        patch_similarity=evaluation.patch_similarity,
        error=evaluation.error,
    )
```

### 3.3 评估函数

```python
async def _evaluate(self, final_diff, task_dir, failing_tests):
    # Apply Maestro's diff to a fresh before/ copy
    eval_dir = self._output_dir / "eval" / uuid.uuid4().hex
    shutil.copytree(task_dir / "before", eval_dir)
    apply_ok, _ = _apply_unified_diff(final_diff, eval_dir)

    if not apply_ok:
        return EvalResult(resolved=False, patch_similarity=0.0, error="diff apply failed")

    # Run failing tests — they should now pass
    ok, output = await _run_pytest(failing_tests, cwd=eval_dir)

    # Compute patch similarity vs ground truth (after/)
    similarity = _compute_patch_similarity(eval_dir, task_dir / "after")

    return EvalResult(resolved=ok, patch_similarity=similarity, error=None if ok else output[-500:])
```

`patch_similarity`: normalized Levenshtein of the diff lines, or AST-based similarity. Simpler: line-level Jaccard similarity of changed lines.

## 4. 配置（Config）

Benchmark 运行时用 `--ablation` 指定配置：

| Config name | 描述 |
|---|---|
| `baseline` | 单 agent 串行执行，无 verifier。对比基准 |
| `parallel_only` | DAG 并行调度，无 verifier |
| `verify_none` | 串行，无 verifier（去掉 parallel 和 verify，只保留 planner+subagent） |
| `verify_t1` | 串行 + Tier 1 verifier |
| `verify_t12` | 串行 + Tier 1 + Tier 2 |
| `verify_t123` | 串行 + 全 verifier（只开 verify） |
| `parallel_verify_t1` | 并行 + Tier 1 |
| `parallel_verify_t12` | 并行 + Tier 1 + Tier 2 |
| `full` | 并行 + 全 verifier（Maestro 完整配置） |

每个 config 构造函数：

```python
# benchmark/configs.py

def build_orchestrator_for_config(config_name, task_spec):
    if config_name == "baseline":
        return BaselineOrchestrator(...)  # single agent, no verify
    if config_name == "parallel_only":
        return MaestroOrchestrator(enabled_tiers=set())
    if config_name == "verify_t12":
        return MaestroOrchestrator(
            max_parallel=1,
            enabled_tiers={"deterministic", "test_based"},
        )
    # ... etc
```

## 5. 报告

### 5.1 单 run 报告

每次 `maestro bench` 产生一份 `BenchmarkReport`：

```python
class BenchmarkReport(BaseModel):
    run_id: str
    config_name: str
    task_count: int
    resolve_rate: float            # resolved / total
    avg_wall_clock_ms: float
    avg_cost_usd: float
    avg_tokens_input: float
    avg_tokens_output: float
    per_task_results: list[TaskBenchmarkResult]
    started_at: datetime
    finished_at: datetime
```

### 5.2 跨 run 对比报告

手动运行 4-6 个 config，每个产生一份 report。然后：

```bash
python benchmark/compare.py \
    --runs benchmark/results/*.json \
    --output benchmark/results/COMPARISON.md
```

`compare.py` 输出：

1. **Resolve rate 柱状图**
2. **Wall-clock 对比柱状图**
3. **Cost vs Resolve rate 散点图（Pareto frontier）**
4. **每 task 级别的胜负矩阵**（哪些 task 只有 full config 能过）
5. **Markdown table 汇总**

图用 matplotlib，PNG 存到 `benchmark/results/figures/`。

## 6. 消融实验计划

预算：200 RMB ≈ $28。按每任务平均 $0.3-0.5 估算，能跑：

| Config | 任务数 × 次数 | 预估成本 |
|---|---|---|
| baseline | 30 × 1 | $3 |
| verify_t12 | 30 × 1 | $4 |
| verify_t123 | 30 × 1 | $7 |
| parallel_only | 30 × 1 | $3 |
| parallel_verify_t12 | 30 × 1 | $5 |
| full | 30 × 1 | $8 |
| **合计** | | **~$30** |

Buffer 留给失败重跑和调试。

## 7. 实施步骤

**Week 5（5.3-5.9）**：

1. **Day 1-2**: 写 `build_tasks.py`，构造 10 个 task（先 requests 5 + click 5 验证 pipeline）
2. **Day 3**: 扩展到 30 task（补齐 flask/httpx/typer）
3. **Day 4**: 写 harness 和 eval 函数
4. **Day 5**: 跑 `baseline` + `full` 两个 config 做 smoke test
5. **Day 6-7**: 跑完整 6 个 config，生成初版报告

**Week 6（5.10-5.16）**：

1. **Day 1-2**: 分析结果、画图、写 COMPARISON.md
2. **Day 3-4**: 失败 case 分析（human review 10% 失败 task，判断是 Maestro 错还是 benchmark 有问题）
3. **Day 5-7**: 迭代 Maestro 修 bug、调 prompt，重跑部分 config，更新最终数字

## 8. 潜在风险与应对

| 风险 | 应对 |
|---|---|
| PR 里 failing test 依赖真实 HTTP | 建 task 时过滤掉 network-heavy test |
| 构建 benchmark 本身很花时间 | Week 5 前半专注构建，允许降级为 20 task 而非 30 |
| 阿里云 API 超预算 | 跑 config 时先 dry-run 估 token，超预算先砍 config 数量 |
| 失败 case 很多是 "diff apply fail" 而非真错 | 在 eval 时区分"方案正确但 diff format 错" vs "方案错" |
| Planner 对简单 task 过度分解 | Planner prompt 里加入 "for tasks under 50 LOC, prefer single subtask" |

## 9. 面试 talking points

1. **自建 benchmark 本身是亮点**：不是套 SWE-bench，而是从 5 个真实 Python repo 挖 30 个 bug-fix PR，人工筛选 + 生成去敏感化的 natural language prompt
2. **Ablation matrix 完整**：6+ 个 config 覆盖 parallel 维度 × verify 维度，能清楚回答"提升来自哪"
3. **成本分层的实证**：Pareto frontier 图直接展示 cost-quality trade-off
4. **失败分析**：10% 失败 case 的 human review 数据是加分项——展示科学态度
5. **这是最终产出的核心**：简历三个数字全部从 BenchmarkReport 里取，直接 copy COMPARISON.md 的最后一张表
