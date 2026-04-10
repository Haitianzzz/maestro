# Spec 03 — Planner

> **位置**：`src/maestro/planner/`
> **依赖**：`LLMClient`、`models.py`（TaskSpec、TaskDAG、SubTask）
> **被依赖**：Orchestrator

## 1. 职责

Planner 是 Maestro 的第一阶段。输入用户的自然语言任务描述 + repo 结构，输出一个带依赖关系的 `TaskDAG`。

**核心要求**：
- 子任务的 `reads` / `writes` 声明必须尽量准确（后续 Scheduler 据此做冲突检测）
- 依赖关系要合理（真正需要前置的 task 才声明 `depends_on`）
- 子任务粒度适中：太粗 → 并行度低；太细 → overhead 高。默认粒度"一个 endpoint / 一个独立功能单元"

## 2. 接口

```python
class Planner:
    """Decomposes a high-level task into a DAG of subtasks."""

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    async def plan(self, task_spec: TaskSpec) -> TaskDAG:
        """Main entry point.

        Steps:
        1. Scan repo to produce a repo summary (file tree + key files content)
        2. Build planner prompt with task description + repo summary
        3. Call LLM with structured output (PlannerOutput schema)
        4. Validate output: DAG acyclicity, subtask IDs unique, writes/reads paths valid
        5. Return TaskDAG
        """
        ...
```

## 3. Repo 扫描（Context Assembly）

Planner 给 LLM 的 context 必须包含足够的 repo 信息，但又不能爆炸。策略：

### 3.1 必须包含的信息

- **文件树**：最多 500 个 entry，超过则省略测试、vendor、build 目录
- **关键文件的完整内容**：
  - `pyproject.toml` / `setup.py` / `setup.cfg` 之一（看 import 路径约定）
  - `README.md` 前 100 行
  - `src/**/__init__.py` 所有（通常小，揭示模块结构）
  - 如果 `TaskSpec.target_files_hint` 指定了文件，这些文件完整读入
- **关键文件的摘要**（如果 repo 大）：
  - 每个 `.py` 文件的 top-level class/function 签名（用 AST 提取）

### 3.2 Context 大小预算

Planner 调用 Qwen3-Max，给 context 预算：

- 输入 token 上限：**60k tokens**（约 150k 字符）
- 如果 repo 超出预算，按以下优先级裁剪：
  1. 删除测试文件内容
  2. 删除 vendor / third-party 目录
  3. 只保留文件签名不保留实现
  4. 文件树折叠到顶层 2 层

### 3.3 实现要求

```python
class RepoScanner:
    """Scans a repo and produces LLM-friendly context."""

    def __init__(self, repo_path: Path, max_context_tokens: int = 60_000):
        self._repo = repo_path
        self._budget = max_context_tokens

    def scan(self, target_files_hint: list[str] | None = None) -> RepoContext:
        """Scan and return structured context."""
        ...


class RepoContext(BaseModel):
    """Structured repo context passed to planner prompt."""
    file_tree: str                  # text-formatted tree
    key_files: dict[str, str]       # filename -> content
    file_signatures: dict[str, str] # filename -> AST-extracted signatures
    total_tokens_estimated: int
```

**Token 估算**：使用 `tiktoken` 或简单近似（1 token ≈ 4 chars for Chinese, 1 token ≈ 4 chars for English）。

## 4. Prompt 设计

### 4.1 System prompt（固定）

```
You are Maestro Planner, a coding task decomposition agent.

Your job is to decompose a user's high-level programming task into a DAG of subtasks that can be executed by independent sub-agents.

HARD RULES:
1. Each subtask MUST declare `reads` and `writes` as file paths relative to repo root.
2. `writes` MUST be precise — sub-agents are only allowed to modify files in `writes`. If you leave a file out, the sub-agent cannot modify it.
3. Use `depends_on` ONLY for true dependencies (e.g., file A is imported by file B; A must be written first).
4. Subtasks with no file-write conflict AND no dependency SHOULD be parallelizable.
5. Keep subtask granularity at "one logical feature unit" level — not per-function, not per-file unless a file is a whole feature.
6. Output MUST conform to the provided JSON schema strictly.

Produce a clear `planning_rationale` describing your decomposition strategy.
```

### 4.2 User prompt 模板

```
# Task
{task_description}

# Repo structure
{file_tree}

# Key files
{key_files_formatted}

# File signatures (for context)
{signatures_formatted}

# Instructions
Decompose this task into 2-8 subtasks. Prioritize parallelism when possible.
Return JSON matching PlannerOutput schema.
```

实现时放在 `src/maestro/planner/prompts.py`，用 Jinja2 或 f-string。

## 5. 后处理与验证

LLM 输出 `PlannerOutput` 后，Planner 必须做以下验证：

### 5.1 结构性验证

- 每个 `subtask_id` 全 DAG 内唯一
- 所有 `depends_on` 引用的 id 都存在
- DAG 无环（用拓扑排序算法检测）

### 5.2 路径验证

- 所有 `reads` 路径对应的文件**必须存在**于 repo（除非 writes 也包含该路径，因为 write-only 可以创建新文件）
- 所有 `writes` 路径必须在 repo 根目录下，不允许 `..` 或绝对路径
- 不允许写入 `.git/`、`.venv/`、`node_modules/` 等特殊目录

### 5.3 失败处理

如果验证失败，Planner 有**最多 2 次重试**机会，每次把错误信息回传给 LLM 要求修正：

```
Your previous plan had the following issues:
- subtask task-xxx-002 reads "src/missing.py" which does not exist
- DAG has a cycle: task-xxx-001 -> task-xxx-002 -> task-xxx-001

Please fix and re-output.
```

2 次仍失败则抛 `PlanningError`，上层决定是否降级为单 subtask 串行执行。

## 6. 边界情况处理

| 情况 | 处理 |
|---|---|
| 任务太简单，Planner 只出 1 个 subtask | 允许。单 subtask DAG 合法 |
| LLM 不生成任何 subtask | PlanningError |
| writes 冲突（两个 subtask 写同一文件且互不依赖） | Planner 侧不检测（Scheduler 负责）；但 `planning_rationale` 应解释为什么这么拆 |
| reads 包含 writes 路径（循环依赖） | 允许，sub-agent 可先读旧内容再写新内容 |
| target_files_hint 指定了不存在的文件 | 视为"要新建这些文件"，传给 LLM 时加说明 |

## 7. 测试要求

`tests/unit/test_planner.py`（用 mock LLMClient）：
- 单 subtask 输出
- 多 subtask 有依赖
- 多 subtask 可并行
- 检测有环 DAG 并触发重试
- 检测路径不存在并触发重试
- 2 次重试耗尽抛 PlanningError

`tests/integration/test_planner_e2e.py`（可选，live）：
- 对真实小 repo（`tests/fixtures/tiny_flask_app/`）跑 planner，验证输出合理

## 8. 关键示例

**输入**：
```python
TaskSpec(
    description="Add user authentication with signup, login, and password reset",
    repo_path=Path("/tmp/demo-app"),
    target_files_hint=None,
)
```

**期望输出**（DAG 结构）：

```
signup (writes: auth/signup.py)      \
login (writes: auth/login.py)         → route_registration (writes: app.py)
reset (writes: auth/reset.py)        /
```

即 Planner 应产出 4 个 subtask，前 3 个并行、最后一个依赖前 3 个。

**planning_rationale 示例**：

```
The task naturally decomposes into three endpoint implementations
(signup, login, reset) that are independent — each writes a separate
file and reads only shared models. A final subtask registers these
routes in the main app, depending on all three being done.
```

## 9. 面试 talking points

- **带依赖的 DAG vs 平铺 list**：多数 agent 框架让 LLM 输出任务列表，Maestro 强制输出 DAG，这是工程上更严格的约束，也是调度器工作的前提
- **Writes 强制声明 + 运行时 enforce**：防止 sub-agent 越界改文件，这是 harness engineering 的真实约束机制
- **Context assembly 的 budget 管理**：Planner context 60k token 的预算管理，展示对 context engineering 的理解
- **Retry with error feedback**：LLM 输出错了不是直接 fail，而是带着错误信息再问一次，这是生产级 agent 的常见 pattern
