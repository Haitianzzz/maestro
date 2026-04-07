# Spec 01 — 数据模型（Data Models）

> **位置**：`src/maestro/models.py`
> **依赖**：Pydantic v2
> **被依赖**：几乎所有模块

## 1. 设计原则

- 所有跨模块传递的数据必须是 Pydantic model，不允许裸 dict
- 所有 LLM 的 structured output 必须有对应 Pydantic model 约束
- 所有 model 必须有 `model_config = ConfigDict(frozen=True)`（不可变），避免并发修改
- 时间戳统一用 `datetime.datetime`（UTC），不允许裸 timestamp int

## 2. 核心数据模型

### 2.1 TaskSpec

用户输入的顶层任务描述。

```python
class TaskSpec(BaseModel):
    """Top-level task input from user."""
    model_config = ConfigDict(frozen=True)

    task_id: str                    # UUID, 由系统生成
    description: str                # 用户自然语言描述
    repo_path: Path                 # 目标 repo 绝对路径
    target_files_hint: list[str] | None = None  # 用户可选提示（如果指定文件）
    max_parallel: int = 4           # 并行 sub-agent 上限
    max_retries_per_subtask: int = 2  # 每个 subtask 重试上限
    judge_samples: int = 3          # LLM-Judge 采样次数 K
    judge_disagreement_threshold: float = 0.3  # 分歧度阈值
    created_at: datetime
```

### 2.2 SubTask

Planner 输出的单个子任务。

```python
class SubTask(BaseModel):
    """One unit of work produced by Planner."""
    model_config = ConfigDict(frozen=True)

    subtask_id: str                 # 格式: "{task_id}-{index}"
    description: str                # 子任务自然语言描述
    reads: list[str]                # 声明要读的文件（相对 repo 根）
    writes: list[str]               # 声明要写的文件（相对 repo 根）
    depends_on: list[str]           # 依赖的其他 subtask_id
    priority: int = 0               # 调度优先级，0 最低，越大越优先
    estimated_difficulty: Literal["easy", "medium", "hard"] = "medium"
```

**关键约束**：
- `writes` 是真实意图，Sub-agent 只能修改 `writes` 里声明的文件
- 运行时若 Sub-agent 尝试修改 `writes` 外的文件，直接 reject 并重试
- `reads` 仅用于给 Sub-agent 注入 context，不做强制检查

### 2.3 TaskDAG

Planner 输出的完整 DAG。

```python
class TaskDAG(BaseModel):
    """Full DAG of subtasks."""
    model_config = ConfigDict(frozen=True)

    task_id: str
    subtasks: list[SubTask]
    global_context: str              # 对整个任务的高层说明，传给每个 sub-agent

    def validate_dag(self) -> None:
        """Check: no cycle, all depends_on refer to existing subtask_ids."""
        # Implementation in scheduler/dag.py
        ...
```

### 2.4 SubAgentResult

Sub-agent 执行后返回的结构化结果。

```python
class SubAgentResult(BaseModel):
    """Structured output from sub-agent. Enforced via structured output."""
    model_config = ConfigDict(frozen=True)

    subtask_id: str
    status: Literal["success", "failed", "rejected"]
    # success: sub-agent 认为自己完成了
    # failed: sub-agent 明确表达无法完成
    # rejected: 违反 writes 约束、超时等系统级拒绝

    diff: str                       # unified diff 格式
    modified_files: list[str]       # 实际修改的文件列表
    rationale: str                  # sub-agent 的推理摘要（不是 full transcript）
    confidence: float               # 0-1，sub-agent 自评的信心分数
    retry_count: int                # 本 subtask 已经重试的次数

    tokens_input: int               # 用于成本统计
    tokens_output: int
    latency_ms: int
    model_used: str                 # 记录实际用的模型

    created_at: datetime
```

### 2.5 VerificationResult

一次验证的完整结果（三层合一）。

```python
class TierResult(BaseModel):
    """Result of a single verifier tier."""
    model_config = ConfigDict(frozen=True)

    tier: Literal["deterministic", "test_based", "llm_judge"]
    passed: bool
    details: str                    # 错误信息或成功摘要
    latency_ms: int
    cost_usd: float = 0.0           # Tier 1/2 零成本


class LLMJudgeDetail(BaseModel):
    """Extra detail from LLM-Judge with multi-sampling."""
    model_config = ConfigDict(frozen=True)

    samples: list[float]            # K 次采样的分数
    mean_score: float
    disagreement: float             # 标准差 or 其他分歧度量
    is_uncertain: bool              # disagreement > threshold
    judge_model: str


class VerificationResult(BaseModel):
    """Full verification result for one patch."""
    model_config = ConfigDict(frozen=True)

    subtask_id: str
    overall_passed: bool            # 所有已执行 tier 都 pass 且 judge 不 uncertain
    tiers: list[TierResult]
    judge_detail: LLMJudgeDetail | None = None  # 仅当 Tier 3 执行时有
    total_latency_ms: int
    total_cost_usd: float
```

### 2.6 BatchResult

Scheduler 一轮 batch 执行后的聚合。

```python
class BatchResult(BaseModel):
    """Result of executing one parallel batch."""
    model_config = ConfigDict(frozen=True)

    batch_index: int
    subtask_results: list[SubAgentResult]
    verification_results: list[VerificationResult]
    merged_patches: list[str]       # 通过 verification 的 subtask_id 列表
    retried_patches: list[str]      # 触发重试的 subtask_id 列表
    failed_patches: list[str]       # 重试耗尽仍失败的 subtask_id 列表
    conflicts_detected: list[tuple[str, str]]  # 运行时检测到的 (id1, id2) 冲突对
```

### 2.7 TaskResult

整个 Task 执行完的最终结果。

```python
class TaskResult(BaseModel):
    """Final result of a complete task execution."""
    model_config = ConfigDict(frozen=True)

    task_id: str
    status: Literal["success", "partial", "failed"]
    batches: list[BatchResult]
    final_diff: str                 # 合并后的完整 unified diff
    final_workspace: Path            # 最终 workspace 路径（调用方可取 artifact）

    total_wall_clock_ms: int
    total_tokens_input: int
    total_tokens_output: int
    total_cost_usd: float

    started_at: datetime
    finished_at: datetime
```

## 3. LLM I/O Schema（structured output）

### 3.1 Planner 输出 schema

Planner 调用 LLM 时，强制 structured output 为以下 schema（Pydantic model 自动转 JSON Schema）：

```python
class PlannerOutput(BaseModel):
    """Structured output from Planner LLM."""
    model_config = ConfigDict(frozen=True)

    subtasks: list[SubTask]
    global_context: str
    planning_rationale: str         # Planner 自述的规划思路，用于 debug
```

### 3.2 Sub-agent 输出 schema

Sub-agent 同样强制 structured output：

```python
class SubAgentOutput(BaseModel):
    """Structured output from Sub-agent LLM."""
    model_config = ConfigDict(frozen=True)

    status: Literal["success", "failed"]
    diff: str                       # unified diff
    modified_files: list[str]
    rationale: str
    confidence: float
```

注意：`SubAgentOutput` 是 LLM 直出的结果，`SubAgentResult`（见 2.4）是框架包装后的结果（加 token、latency、retry_count 等元数据）。两者分离。

### 3.3 LLM-Judge 输出 schema

```python
class JudgeOutput(BaseModel):
    """Structured output from LLM-Judge per sample."""
    model_config = ConfigDict(frozen=True)

    score: float                    # 0-1
    passes_requirements: bool
    reasoning: str                  # 短 rationale
    detected_issues: list[str]      # 如果有问题，列出来
```

## 4. 实现要求

### 4.1 必须实现的方法

```python
# models.py

def generate_task_id() -> str:
    """Generate UUID4 task id."""

def generate_subtask_id(task_id: str, index: int) -> str:
    """Generate subtask_id in format {task_id}-{index:03d}."""
```

### 4.2 序列化

所有 model 必须可序列化为 JSON，用于日志和 benchmark 结果存储：

```python
# 使用方式示例
result_json = task_result.model_dump_json(indent=2)
```

### 4.3 测试要求

`tests/unit/test_models.py` 必须包含：
- 每个 model 的合法构造测试
- 每个 model 的非法输入测试（字段缺失、类型错误）
- `TaskDAG.validate_dag()` 的测试：空 DAG、单节点、线性链、有环、跨 subtask 引用不存在的 id
- Pydantic structured output schema 的序列化测试

## 5. 示例

### 5.1 一个完整 TaskDAG 示例

```python
task_id = "task-abc123"

dag = TaskDAG(
    task_id=task_id,
    global_context="Add user authentication with signup, login, password-reset endpoints. Use existing SQLAlchemy User model.",
    subtasks=[
        SubTask(
            subtask_id=f"{task_id}-001",
            description="Implement /auth/signup endpoint",
            reads=["src/models/user.py", "src/db.py"],
            writes=["src/auth/signup.py", "src/auth/__init__.py"],
            depends_on=[],
            priority=2,
        ),
        SubTask(
            subtask_id=f"{task_id}-002",
            description="Implement /auth/login endpoint",
            reads=["src/models/user.py", "src/db.py"],
            writes=["src/auth/login.py"],
            depends_on=[],
            priority=2,
        ),
        SubTask(
            subtask_id=f"{task_id}-003",
            description="Implement /auth/reset-password endpoint",
            reads=["src/models/user.py", "src/db.py"],
            writes=["src/auth/reset.py"],
            depends_on=[],
            priority=1,
        ),
        SubTask(
            subtask_id=f"{task_id}-004",
            description="Register auth routes in main app",
            reads=["src/app.py"],
            writes=["src/app.py"],
            depends_on=[f"{task_id}-001", f"{task_id}-002", f"{task_id}-003"],
            priority=3,
        ),
    ],
)
```

**Scheduler 对这个 DAG 的处理**：
- Batch 0: `[001, 002, 003]` 并行（无依赖、writes 不冲突）
- Batch 1: `[004]` 串行（依赖 batch 0 全部完成）

## 6. 非目标

以下不在本 spec 范围内，后续版本可能引入：

- Task 取消（中途 abort）—— 本版本不支持
- Task 恢复（resume from checkpoint）—— 本版本不支持
- 跨语言任务 —— 本版本只支持 Python
- 远程 repo 操作 —— 本版本只支持本地 repo
