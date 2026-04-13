# Spec 04 — Scheduler（DAG 调度器）

> **位置**：`src/maestro/scheduler/`
> **依赖**：`models.py`、`subagent/`、`verifier/`、`sandbox/`、`utils/priority_queue.py`
> **被依赖**：Orchestrator

## 1. 职责

Scheduler 是 Maestro 的核心工程模块。给定一个 `TaskDAG`，它需要：

1. 把 DAG 拆解为按拓扑顺序的 **batch**（同 batch 内任务可并行）
2. 对每个 batch 内的任务并行发给 sub-agent 执行
3. 运行时检测 **写冲突**（即使 Planner 没发现）
4. 管理全局并发数（不能一次发 50 个 agent）
5. 根据 verification 结果决定 commit / retry / abort
6. 整合每个 batch 的 patch 到主 workspace

## 2. 核心抽象

### 2.1 DAG 拆解

```python
# src/maestro/scheduler/dag.py

class DAGError(Exception):
    """Raised for any DAG structural error."""


def topological_batches(dag: TaskDAG) -> list[list[SubTask]]:
    """Group subtasks into parallel-execution batches.

    Algorithm:
    - Compute in-degree for each subtask
    - Round 0: all subtasks with in_degree == 0 form batch 0
    - Round N: subtasks whose all deps are in batches 0..N-1 form batch N
    - Within a batch, sort by priority DESC then by subtask_id

    Raises DAGError if cycle detected or references missing ids.
    """
    ...


def detect_write_conflicts(batch: list[SubTask]) -> list[tuple[str, str]]:
    """Find pairs of subtasks in the same batch that write to the same file.

    Returns list of (subtask_id_a, subtask_id_b) conflicts.
    Empty list means no conflicts.
    """
    ...
```

### 2.2 冲突处理策略

Planner 出的 DAG 理论上不应同 batch 写冲突，但 LLM 不完美。Scheduler 作为最后防线：

1. 执行前 `detect_write_conflicts`
2. 如果检测到冲突，**将冲突对中优先级较低的 subtask 延迟到下一 batch**（等价于插入一条隐式依赖）
3. 记录 `conflict_adjustments` 到 `BatchResult.conflicts_detected`，benchmark 分析用

## 3. 主 Scheduler 类

### 3.1 接口

```python
# src/maestro/scheduler/scheduler.py

class Scheduler:
    """DAG scheduler for parallel sub-agent execution."""

    def __init__(
        self,
        dag: TaskDAG,
        task_spec: TaskSpec,
        subagent_factory: SubAgentFactory,
        verifier: Verifier,
        workspace_manager: WorkspaceManager,
        llm_client: LLMClient,
    ):
        self._dag = dag
        self._spec = task_spec
        self._subagent_factory = subagent_factory
        self._verifier = verifier
        self._workspace = workspace_manager
        self._llm = llm_client

        self._semaphore = asyncio.Semaphore(task_spec.max_parallel)
        self._priority_queue: AsyncPriorityQueue = AsyncPriorityQueue()
        self._retry_counts: dict[str, int] = {}  # subtask_id -> count

    async def execute(self) -> list[BatchResult]:
        """Main entry point. Execute all batches in order, return per-batch results."""
        ...

    async def _execute_batch(self, batch_index: int, subtasks: list[SubTask]) -> BatchResult:
        """Execute one batch in parallel."""
        ...

    async def _run_subtask_with_verify(self, subtask: SubTask) -> tuple[SubAgentResult, VerificationResult]:
        """Run a single subtask: spawn sub-agent → verify → return."""
        ...

    async def _handle_verify_failure(
        self, subtask: SubTask, sub_result: SubAgentResult, verify_result: VerificationResult
    ) -> tuple[SubAgentResult, VerificationResult] | None:
        """Retry logic. Returns new result or None if retries exhausted."""
        ...
```

### 3.2 执行流程（伪代码）

```python
async def execute(self) -> list[BatchResult]:
    batches = topological_batches(self._dag)
    results: list[BatchResult] = []

    for batch_index, batch in enumerate(batches):
        # 1. Detect & resolve write conflicts
        conflicts = detect_write_conflicts(batch)
        if conflicts:
            batch, deferred = _defer_lower_priority(batch, conflicts)
            # Deferred subtasks are inserted into next batch
            if batch_index + 1 < len(batches):
                batches[batch_index + 1].extend(deferred)
            else:
                batches.append(deferred)

        # 2. Execute batch in parallel
        result = await self._execute_batch(batch_index, batch)
        results.append(result)

        # 3. Merge successful patches to main workspace
        await self._workspace.merge_patches(
            [r for r in result.subtask_results if r.subtask_id in result.merged_patches]
        )

        # 4. If everything in batch failed, abort
        if not result.merged_patches and result.failed_patches:
            break

    return results


async def _execute_batch(self, batch_index, subtasks):
    async with asyncio.TaskGroup() as tg:
        tasks = {
            subtask.subtask_id: tg.create_task(self._run_subtask_with_verify(subtask))
            for subtask in subtasks
        }
    # After TaskGroup exits, all tasks complete or raised
    ...
```

### 3.3 `_run_subtask_with_verify` 伪代码

```python
async def _run_subtask_with_verify(self, subtask):
    async with self._semaphore:
        # 1. Create isolated workspace for this subtask
        workspace = await self._workspace.create_isolated(subtask)

        # 2. Spawn sub-agent
        subagent = self._subagent_factory.create(subtask, workspace, self._llm)
        sub_result = await subagent.run(self._dag.global_context)

        # 3. If sub-agent failed, no verification
        if sub_result.status != "success":
            return sub_result, _empty_verification(subtask.subtask_id)

        # 4. Verify
        verify_result = await self._verifier.verify(subtask, workspace, sub_result)

        # 5. If failed and retries available, recurse
        if not verify_result.overall_passed:
            retry_result = await self._handle_verify_failure(subtask, sub_result, verify_result)
            if retry_result is not None:
                return retry_result

        return sub_result, verify_result
```

### 3.4 重试逻辑

```python
async def _handle_verify_failure(self, subtask, sub_result, verify_result):
    current_retry = self._retry_counts.get(subtask.subtask_id, 0)
    if current_retry >= self._spec.max_retries_per_subtask:
        return None  # Exhausted

    self._retry_counts[subtask.subtask_id] = current_retry + 1

    # Re-run sub-agent with verify failure feedback
    workspace = await self._workspace.create_isolated(subtask)
    subagent = self._subagent_factory.create(
        subtask,
        workspace,
        self._llm,
        prior_attempt=sub_result,
        prior_failure=verify_result,
    )
    new_sub_result = await subagent.run(self._dag.global_context)

    if new_sub_result.status != "success":
        return new_sub_result, _empty_verification(subtask.subtask_id)

    new_verify_result = await self._verifier.verify(subtask, workspace, new_sub_result)

    if not new_verify_result.overall_passed:
        # Recurse
        return await self._handle_verify_failure(subtask, new_sub_result, new_verify_result)

    return new_sub_result, new_verify_result
```

**重试 feedback 注入**：见 `specs/05-subagent.md` 中 prior_attempt/prior_failure 的处理。

## 4. 优先级队列（自定义）

### 4.1 为什么需要

`asyncio.Semaphore` 只能限制并发数，不能保证高优先级任务先拿到 slot。Maestro 需要：

- 关键路径 task（blocking 后续 batch）优先级高 → 先执行
- 低优先级 task（如可选的 doc update）后执行

### 4.2 实现

```python
# src/maestro/utils/priority_queue.py

class AsyncPriorityQueue:
    """Priority queue for asyncio tasks.

    Higher priority number = executed first.
    """

    def __init__(self):
        self._heap: list[tuple[int, int, asyncio.Event]] = []  # (neg_priority, seq, event)
        self._lock = asyncio.Lock()
        self._seq = 0

    async def acquire_slot(self, priority: int) -> None:
        """Wait until it's our turn based on priority."""
        ...

    def release_slot(self) -> None:
        """Signal next in queue."""
        ...


class PrioritySemaphore:
    """Semaphore that respects task priority."""

    def __init__(self, max_concurrent: int):
        self._max = max_concurrent
        self._in_flight = 0
        self._pq = AsyncPriorityQueue()

    @asynccontextmanager
    async def acquire(self, priority: int):
        await self._pq.acquire_slot(priority)
        # Also respect hard concurrency limit
        ...
        try:
            yield
        finally:
            self._in_flight -= 1
            self._pq.release_slot()
```

### 4.3 使用方式

Scheduler 把原先的 `asyncio.Semaphore` 替换为 `PrioritySemaphore`：

```python
async def _run_subtask_with_verify(self, subtask):
    async with self._prio_semaphore.acquire(priority=subtask.priority):
        ...
```

**注**：如果 Week 3 时间紧，可以先用普通 `Semaphore`，Week 4 再升级为 `PrioritySemaphore`。作为"可选优化"在 Week 4 兜底。**这是面试 talking point，必须做出来**。

## 5. LangGraph 集成

Scheduler 和 Orchestrator 的关系：

- **Orchestrator** 是 LangGraph 的高层 state machine：Plan → Schedule → Merge → Report
- **Scheduler** 是 Schedule 阶段的内部实现，不使用 LangGraph（因为这一层是循环+并行，用 asyncio 更自然）

LangGraph state 定义：

```python
# src/maestro/orchestrator.py

class OrchestratorState(TypedDict):
    task_spec: TaskSpec
    dag: TaskDAG | None
    batch_results: list[BatchResult]
    final_result: TaskResult | None
    error: str | None
```

图结构：

```
START → plan_node → schedule_node → merge_node → report_node → END
                 ↘ error_node ↗
```

- `plan_node`: 调 Planner.plan()，填充 `dag`
- `schedule_node`: 调 Scheduler.execute()，填充 `batch_results`
- `merge_node`: 生成 final_diff，填充 `final_result`
- `report_node`: 写入日志、输出 CLI 可见的摘要
- `error_node`: 任何一步失败转到这里，记录 error

## 6. 数据流示例

给定 `TaskDAG` 4 subtasks（signup、login、reset 并行，register 依赖前 3），`max_parallel=3`：

```
Batch 0: [signup, login, reset]
  ↓ 3 sub-agents 并行启动，每个进入独立 workspace
  ↓ sub-agent 输出 diff
  ↓ 每个 diff 进三层 verifier
  ↓ signup pass、login pass、reset fail-tier1-retry-pass
  ↓ 三个 patch 合并到 main workspace
Batch 1: [register]
  ↓ 1 sub-agent，读取已合并的 signup/login/reset
  ↓ verify pass
  ↓ 合并到 main workspace
Report: 最终 diff
```

## 7. 测试要求

`tests/unit/test_dag.py`：
- 空 DAG、单节点、线性链、分叉合并、菱形、完全并行
- 有环 DAG 抛 DAGError
- detect_write_conflicts 的单元测试

`tests/unit/test_priority_queue.py`：
- 高优先级先执行
- FIFO 同优先级
- 并发上限

`tests/unit/test_scheduler.py`（mock sub-agent 和 verifier）：
- 全部成功
- 部分 sub-agent 失败
- verify 重试
- 写冲突延迟

`tests/integration/test_scheduler_e2e.py`：
- 用真实小 DAG + fake sub-agents 测完整流程

## 8. 关键实现注意

1. **`asyncio.TaskGroup`（3.11+）** 比 `asyncio.gather` 更好：任一 task 抛异常会 cancel 其他 task，避免资源泄漏
2. **Workspace 隔离是原子操作**：每个 sub-agent 的 workspace 是独立临时目录，不会互相污染
3. **Batch 失败不一定全量 abort**：如果某 batch 有 1/3 成功，仍允许下一 batch 执行（只要它们不依赖失败的 subtask）
4. **Retry count 全局共享**：同一 subtask 在不同调用栈上的重试都算总数

## 9. 面试 talking points

1. **DAG 调度 vs 线性 agent loop**：现代 coding agent 大多是 linear loop（planner → executor → critic，串行），Maestro 的 DAG 并行是真正的调度层抽象
2. **运行时冲突检测 + 自动降级**：Planner 可能出错，Scheduler 作为最后防线在运行时检测并降级，体现 defensive engineering
3. **自定义 PrioritySemaphore**：标准 asyncio 原语不够用时自己写，展示对并发原语的深入理解
4. **asyncio.TaskGroup 取代 gather**：体现对 Python 3.11+ 新特性的了解
5. **LangGraph + asyncio 混用**：外层 state machine 用 LangGraph（天然适合条件分支），内层并行用 asyncio（天然适合 I/O 并发）——工具选型的 trade-off
