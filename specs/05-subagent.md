# Spec 05 — Sub-agent

> **位置**：`src/maestro/subagent/`
> **依赖**：`LLMClient`、`models.py`、`sandbox/`
> **被依赖**：Scheduler

## 1. 职责

Sub-agent 是实际执行代码修改的 LLM agent。每个 SubTask 对应一个 Sub-agent 实例。

**核心要求**：
- 在独立 workspace 内操作，不影响其他 Sub-agent
- 只能修改 `SubTask.writes` 声明的文件
- 输出严格符合 `SubAgentOutput` schema
- 支持"带失败反馈的重试"：Scheduler 可以把上一次的失败信息喂给它

## 2. 架构决策

### 2.1 Sub-agent 是一次性的还是多步的？

**决策**：采用**受控的多步模式**。

- 单次 LLM 调用直接输出 diff 的方式太粗糙（复杂 subtask 经常写错）
- 完全自由的 ReAct loop 又不可控
- 折中：固定 3 步——**Explore**（读文件）→ **Plan**（生成计划）→ **Write**（输出 diff）

**Explore 步允许 LLM 使用 tool call**：`read_file(path)`。每次调用是一轮 LLM call，最多 5 轮。

**Plan 步**：LLM 产出一份内部计划文本。

**Write 步**：LLM 产出 `SubAgentOutput`（structured）。

### 2.2 为什么不用 full tool-use loop

- Tool loop 复杂，debug 难，token 消耗不稳定
- 固定 3 步让每个 subtask 的 LLM 调用次数可预测（3-8 次），成本可控
- 面试讲 harness engineering 时，"有约束的 loop" 比"自由 loop" 更体现工程判断

## 3. 接口

```python
# src/maestro/subagent/subagent.py

class SubAgent:
    """Executes one SubTask in an isolated workspace."""

    def __init__(
        self,
        subtask: SubTask,
        workspace: IsolatedWorkspace,
        llm_client: LLMClient,
        prior_attempt: SubAgentResult | None = None,
        prior_failure: VerificationResult | None = None,
    ):
        self._subtask = subtask
        self._workspace = workspace
        self._llm = llm_client
        self._prior = prior_attempt
        self._prior_failure = prior_failure

    async def run(self, global_context: str) -> SubAgentResult:
        """Run the 3-phase loop. Returns final result."""
        ...


class SubAgentFactory:
    """Factory wrapping SubAgent construction."""

    def create(
        self,
        subtask: SubTask,
        workspace: IsolatedWorkspace,
        llm_client: LLMClient,
        prior_attempt: SubAgentResult | None = None,
        prior_failure: VerificationResult | None = None,
    ) -> SubAgent:
        return SubAgent(subtask, workspace, llm_client, prior_attempt, prior_failure)
```

## 4. 执行流程

### 4.1 Phase 1: Explore（受限 tool loop）

```python
async def _explore(self, global_context: str) -> ExploreOutput:
    """Up to 5 rounds of tool calls to read files."""
    messages = [
        {"role": "system", "content": EXPLORE_SYSTEM_PROMPT},
        {"role": "user", "content": self._build_explore_prompt(global_context)},
    ]

    tools = [READ_FILE_TOOL_SCHEMA]
    files_read: dict[str, str] = {}  # path -> content
    max_rounds = 5

    for round_num in range(max_rounds):
        response = await self._llm.call_with_tools(
            role="subagent",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        if not response.tool_calls:
            # Agent decided to stop exploring
            break

        for tc in response.tool_calls:
            if tc.name == "read_file":
                path = tc.arguments["path"]
                content = self._workspace.read_file(path)  # may read reads[] + writes[] existing files
                files_read[path] = content
                messages.append(_tool_result_message(tc, content))

    return ExploreOutput(files_read=files_read, messages_so_far=messages)
```

**读取权限**：Sub-agent 可以读取任何 `reads` 或 `writes` 声明过的文件；读取其他文件会返回 permission denied。

### 4.2 Phase 2: Plan

Explore 结束后，一次 LLM 调用产生内部 plan（不是 structured output，是自由文本），用于 chain-of-thought 质量提升。

```python
async def _plan(self, explore_output: ExploreOutput) -> str:
    messages = explore_output.messages_so_far + [
        {"role": "user", "content": PLAN_PROMPT_TEMPLATE.format(subtask=self._subtask)},
    ]
    plan_text, _ = await self._llm.call_text(role="subagent", messages=messages, temperature=0.3)
    return plan_text
```

### 4.3 Phase 3: Write

```python
async def _write(self, plan_text: str, messages_history: list[dict]) -> SubAgentOutput:
    messages = messages_history + [
        {"role": "assistant", "content": plan_text},
        {"role": "user", "content": WRITE_PROMPT_TEMPLATE.format(subtask=self._subtask)},
    ]
    output, _ = await self._llm.call_structured(
        role="subagent",
        messages=messages,
        output_schema=SubAgentOutput,
        temperature=0.2,
    )
    return output
```

### 4.4 整体 run

```python
async def run(self, global_context: str) -> SubAgentResult:
    start = time.perf_counter()
    total_tokens_in = 0
    total_tokens_out = 0

    try:
        explore = await self._explore(global_context)
        plan = await self._plan(explore)
        output = await self._write(plan, explore.messages_so_far)
    except Exception as e:
        return _build_failed_result(self._subtask, reason=str(e))

    # Validate: did the sub-agent write files outside its writes?
    illegal = _find_illegal_writes(output.diff, self._subtask.writes)
    if illegal:
        return _build_rejected_result(
            self._subtask,
            reason=f"Sub-agent attempted to modify files outside writes: {illegal}",
        )

    # Apply diff to isolated workspace
    apply_ok, apply_err = self._workspace.apply_diff(output.diff)
    if not apply_ok:
        return _build_rejected_result(self._subtask, reason=f"Diff apply failed: {apply_err}")

    latency_ms = int((time.perf_counter() - start) * 1000)

    return SubAgentResult(
        subtask_id=self._subtask.subtask_id,
        status="success" if output.status == "success" else "failed",
        diff=output.diff,
        modified_files=output.modified_files,
        rationale=output.rationale,
        confidence=output.confidence,
        retry_count=_determine_retry_count(self._prior),
        tokens_input=total_tokens_in,
        tokens_output=total_tokens_out,
        latency_ms=latency_ms,
        model_used=self._llm.config.models["subagent"].name,
        created_at=datetime.utcnow(),
    )
```

## 5. Prompt 设计

### 5.1 Explore system prompt

```
You are Maestro Sub-Agent, a focused coding agent working on one specific subtask.

You are in EXPLORE phase. Your goal is to read relevant files to understand the codebase before making changes.

Rules:
- Use the `read_file` tool to read files listed in `reads` and `writes` of your subtask.
- Do not attempt to read files outside your permission — they will return "permission denied".
- Stop exploring as soon as you have enough context. Do not over-read.
- You have at most 5 read_file calls.
```

### 5.2 Plan prompt

```
Now you have enough context. Before writing code, draft your plan:

Subtask: {subtask.description}
Files you must modify: {subtask.writes}
Files you may reference: {subtask.reads}

Write a short plan (max 200 words) covering:
1. What changes you will make to each file in `writes`
2. Any new imports or dependencies
3. Edge cases you will handle
```

### 5.3 Write prompt

```
Execute the plan. Produce:

1. A unified diff (format: `diff --git ...` style) modifying ONLY files in writes.
2. A short rationale.
3. Your confidence [0-1] — honest self-assessment.

You MUST respond as JSON matching SubAgentOutput schema.

HARD RULES:
- Do not include diffs for files not in `writes`.
- If creating a new file, the diff should show it as a new file.
- Use 3-line context markers in unified diff.
```

### 5.4 重试时的 prior_failure 注入

如果 `self._prior_failure` 存在，在 Write prompt 前追加：

```
## Prior attempt failed

Your previous attempt at this subtask failed the following checks:

{formatted_failure_details}

Previous diff:
```diff
{prior.diff}
```

Analyze what went wrong and produce a corrected version.
```

## 6. 工具：read_file

```python
READ_FILE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from the repo. Path must be relative to repo root.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
        },
    },
}
```

实现：调 `IsolatedWorkspace.read_file(path)`，做 permission check。

## 7. 错误处理

| 情况 | 处理 |
|---|---|
| LLM call 抛 LLMCallError | 返回 `status="failed"`，reason 记录在 rationale |
| Diff 无法解析 | 返回 `status="rejected"` |
| Diff 包含 writes 外的文件 | 返回 `status="rejected"` |
| Diff 应用后文件编码错误 | 返回 `status="rejected"` |
| 探索时 read_file 越权 | 返回 `"permission denied"`，不计入 illegal writes |

## 8. 测试要求

`tests/unit/test_subagent.py`（mock LLMClient、WorkspaceManager）：
- 成功路径：explore 读 2 个文件、plan、write diff、合法、success
- Explore 直接 skip（LLM 不 call tool）
- Explore 5 轮上限
- Write diff 越权被 reject
- Diff 无法解析被 reject
- 带 prior_failure 的重试 prompt 正确注入

## 9. 面试 talking points

1. **受控的 3 阶段 loop**：不是自由 ReAct，不是单次 call，是 Explore → Plan → Write 的结构化 loop，每阶段职责清晰、成本可预测
2. **Writes enforce 的运行时检查**：不是相信 LLM 遵守 prompt，而是 diff 应用前做机器可读的 illegal write 检测
3. **Prior failure feedback loop**：重试时注入失败信息，不是简单重跑，展示对 agent failure recovery 的工程设计
4. **Permission-based file access**：sub-agent 的读权限受 `reads + writes` 约束，这是 harness 层面的安全约束
