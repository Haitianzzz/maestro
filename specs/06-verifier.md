# Spec 06 — Verifier（分层验证系统）

> **位置**：`src/maestro/verifier/`
> **依赖**：`LLMClient`、`models.py`、`sandbox/`
> **被依赖**：Scheduler

**Maestro 最核心的差异化模块**。面试 30 分钟应重点讲这一模块。

## 1. 职责

Verifier 对 Sub-agent 输出的 patch 做独立验证。三层架构：

| Tier | 名称 | 成本 | 作用 |
|---|---|---|---|
| 1 | Deterministic | 零 API 成本 | linter + 类型检查，过滤语法层面错误 |
| 2 | Test-based | 零 API 成本 | 跑 pytest，过滤行为层面错误 |
| 3 | LLM-as-Judge | 有 API 成本 | 语义层面审查，使用**多采样 + 分歧检测** |

**核心原则**：前层失败直接 reject，不进入下一层。便宜的先跑，贵的后跑。

## 2. 主 Verifier 类

```python
# src/maestro/verifier/__init__.py

class Verifier:
    """Three-tier verification orchestrator."""

    def __init__(
        self,
        llm_client: LLMClient,
        task_spec: TaskSpec,
        enabled_tiers: set[Literal["deterministic", "test_based", "llm_judge"]] | None = None,
    ):
        self._llm = llm_client
        self._spec = task_spec
        self._enabled = enabled_tiers or {"deterministic", "test_based", "llm_judge"}

        self._tier1 = DeterministicVerifier()
        self._tier2 = TestBasedVerifier()
        self._tier3 = LLMJudgeVerifier(llm_client, task_spec)

    async def verify(
        self,
        subtask: SubTask,
        workspace: IsolatedWorkspace,
        sub_result: SubAgentResult,
    ) -> VerificationResult:
        """Run enabled tiers in order, short-circuit on failure."""
        ...
```

### 2.1 短路流程

```python
async def verify(self, subtask, workspace, sub_result):
    start = time.perf_counter()
    tier_results: list[TierResult] = []
    judge_detail: LLMJudgeDetail | None = None
    total_cost = 0.0

    # Tier 1
    if "deterministic" in self._enabled:
        t1 = await self._tier1.run(workspace, sub_result)
        tier_results.append(t1)
        if not t1.passed:
            return _assemble(tier_results, judge_detail, start, total_cost, overall=False)

    # Tier 2
    if "test_based" in self._enabled:
        t2 = await self._tier2.run(workspace, sub_result)
        tier_results.append(t2)
        if not t2.passed:
            return _assemble(tier_results, judge_detail, start, total_cost, overall=False)

    # Tier 3
    if "llm_judge" in self._enabled:
        t3, judge_detail = await self._tier3.run(subtask, workspace, sub_result)
        tier_results.append(t3)
        total_cost += t3.cost_usd
        if not t3.passed:
            return _assemble(tier_results, judge_detail, start, total_cost, overall=False)
        if judge_detail.is_uncertain:
            # Pass but mark uncertain
            return _assemble(tier_results, judge_detail, start, total_cost, overall=False)

    return _assemble(tier_results, judge_detail, start, total_cost, overall=True)
```

### 2.2 消融开关

`enabled_tiers` 参数支持消融实验：

- `{"deterministic"}`: 只跑 Tier 1
- `{"deterministic", "test_based"}`: 跑前两层，跳过 judge
- `{"llm_judge"}`: 只跑 judge（测试 judge 的 standalone 可靠性）
- `None` / 默认: 全开

Benchmark harness 在消融实验时控制此参数。

## 3. Tier 1: DeterministicVerifier

### 3.1 接口

```python
# src/maestro/verifier/deterministic.py

class DeterministicVerifier:
    """Runs ruff + mypy on the isolated workspace."""

    async def run(self, workspace: IsolatedWorkspace, sub_result: SubAgentResult) -> TierResult:
        """Run ruff then mypy. Any error = fail."""
        ...
```

### 3.2 实现

```python
async def run(self, workspace, sub_result):
    start = time.perf_counter()

    # 1. Run ruff on modified files
    ruff_ok, ruff_output = await _run_subprocess(
        ["ruff", "check", *sub_result.modified_files],
        cwd=workspace.path,
        timeout=30,
    )

    if not ruff_ok:
        return TierResult(
            tier="deterministic",
            passed=False,
            details=f"ruff failed:\n{ruff_output}",
            latency_ms=_elapsed_ms(start),
            cost_usd=0.0,
        )

    # 2. Run mypy (optional, can be disabled via env var)
    if os.getenv("MAESTRO_SKIP_MYPY", "0") != "1":
        mypy_ok, mypy_output = await _run_subprocess(
            ["mypy", "--ignore-missing-imports", *sub_result.modified_files],
            cwd=workspace.path,
            timeout=60,
        )
        if not mypy_ok:
            return TierResult(
                tier="deterministic",
                passed=False,
                details=f"mypy failed:\n{mypy_output}",
                latency_ms=_elapsed_ms(start),
                cost_usd=0.0,
            )

    return TierResult(
        tier="deterministic",
        passed=True,
        details="ruff and mypy passed",
        latency_ms=_elapsed_ms(start),
        cost_usd=0.0,
    )
```

### 3.3 配置

- ruff 配置遵循 repo 原有的 `ruff.toml` / `pyproject.toml`，如无则用 Maestro 默认（精简配置）
- mypy 默认用 `--ignore-missing-imports`（benchmark 的 repo 不一定有完整类型标注）

## 4. Tier 2: TestBasedVerifier

### 4.1 策略

有两种情况：

**情况 A：subtask 有关联测试文件**
- 关联规则：如果 subtask 的 `writes` 包含 `src/foo.py`，则尝试运行 `tests/test_foo.py`、`tests/unit/test_foo.py` 等路径
- 或者 repo 有 CI 配置指定测试目录，跑该目录下所有测试

**情况 B：subtask 无关联测试**
- **让 LLM 生成 test case** 再跑
- 这是 Maestro 的一个亮点特性：当 repo 测试覆盖率低时，自动生成测试

### 4.2 接口

```python
# src/maestro/verifier/test_based.py

class TestBasedVerifier:
    """Runs pytest, auto-generating tests if none exist."""

    def __init__(self, llm_client: LLMClient | None = None):
        # llm_client optional — only needed for auto test generation
        self._llm = llm_client

    async def run(self, workspace: IsolatedWorkspace, sub_result: SubAgentResult) -> TierResult:
        existing_tests = self._find_related_tests(workspace, sub_result.modified_files)

        if existing_tests:
            return await self._run_pytest(workspace, existing_tests)

        if self._llm is None:
            # Auto-gen disabled, treat as pass (can't verify behaviorally)
            return TierResult(
                tier="test_based",
                passed=True,
                details="No related tests found; auto-gen disabled",
                latency_ms=0,
                cost_usd=0.0,
            )

        # Auto-generate tests
        generated_test_file = await self._generate_tests(workspace, sub_result)
        return await self._run_pytest(workspace, [generated_test_file])

    def _find_related_tests(self, workspace, modified_files) -> list[Path]:
        """Find test files related to modified source files."""
        ...

    async def _run_pytest(self, workspace, test_files) -> TierResult:
        ok, output = await _run_subprocess(
            ["pytest", "-x", "--tb=short", *[str(t) for t in test_files]],
            cwd=workspace.path,
            timeout=120,
        )
        return TierResult(
            tier="test_based",
            passed=ok,
            details=output[-2000:] if output else "",  # 截尾避免日志爆炸
            latency_ms=...,
            cost_usd=0.0,
        )

    async def _generate_tests(self, workspace, sub_result) -> Path:
        """Ask LLM to generate tests for the modified files."""
        ...
```

### 4.3 Auto-gen test 的 prompt

```
You are a test-writing assistant. Given the following code change, write a pytest test file that verifies the new/changed behavior.

Modified files:
{diff}

Rationale of the change:
{sub_result.rationale}

Rules:
- Only write tests. Do not modify implementation.
- Use pytest style (no unittest.TestCase).
- Include at least 3 test cases: happy path, edge case, error case.
- Tests should pass if the implementation is correct.

Output: the complete test file content (python code, no markdown fences).
```

Auto-gen 使用 **Qwen3-Coder-Plus**（便宜的 coding 专精模型），不用 judge 模型。

### 4.4 重要：auto-gen 的 benchmark 限制

Auto-gen test 在 benchmark 上有 bias 风险：LLM 生成的测试可能偏向它自己理解的正确性，而不是真实正确性。因此：

- **Benchmark 上默认关闭 auto-gen**，用 benchmark 提供的 ground-truth test
- Production CLI 使用时开启 auto-gen

Claude Code 实施时用 `workspace.config.auto_gen_tests: bool` 控制。

## 5. Tier 3: LLMJudgeVerifier（核心亮点）

### 5.1 为什么多采样 + 分歧检测

直接引用你研究经验：**LLM-as-Judge 在 verifiable-answer 类任务中存在 silent correctness leakage**（EMNLP 2026 in progress）。

工程化应对：
- 对同一 patch 采样 **K=3 次** judge 评分（每次不同 temperature 或 prompt 轻微扰动）
- 计算**分歧度**（标准差或 K 次结果的 disagreement metric）
- 分歧度高 → 标记 `is_uncertain=True`，视为"不通过"（不能因为 mean 过了就放行）
- 分歧度低且 mean 过阈值 → 通过

### 5.2 接口

```python
# src/maestro/verifier/llm_judge.py

class LLMJudgeVerifier:
    """Multi-sample LLM-as-Judge with disagreement detection."""

    def __init__(self, llm_client: LLMClient, task_spec: TaskSpec):
        self._llm = llm_client
        self._spec = task_spec
        self._K = task_spec.judge_samples             # default 3
        self._threshold = task_spec.judge_disagreement_threshold  # default 0.3

    async def run(
        self,
        subtask: SubTask,
        workspace: IsolatedWorkspace,
        sub_result: SubAgentResult,
    ) -> tuple[TierResult, LLMJudgeDetail]:
        """Run K judge samples, aggregate, detect uncertainty."""
        ...
```

### 5.3 主流程

```python
async def run(self, subtask, workspace, sub_result):
    start = time.perf_counter()

    # Build shared prompt context
    prompt_context = self._build_context(subtask, sub_result, workspace)

    # K parallel samples with varied temperature
    temperatures = [0.1, 0.5, 0.9][:self._K]  # 或其他扰动策略
    async with asyncio.TaskGroup() as tg:
        sample_tasks = [
            tg.create_task(self._single_judge_call(prompt_context, temp))
            for temp in temperatures
        ]

    samples = [t.result() for t in sample_tasks]
    # samples: list[tuple[JudgeOutput, LLMCallMetadata]]

    scores = [s[0].score for s in samples]
    passes = [s[0].passes_requirements for s in samples]

    mean_score = sum(scores) / len(scores)
    disagreement = _compute_disagreement(scores, passes)
    is_uncertain = disagreement > self._threshold

    total_cost = sum(s[1].cost_usd for s in samples)

    # Pass criteria: mean > 0.6 AND majority passes AND not uncertain
    passed = (
        mean_score > 0.6
        and sum(passes) > self._K / 2
        and not is_uncertain
    )

    tier_result = TierResult(
        tier="llm_judge",
        passed=passed,
        details=_format_judge_details(samples, mean_score, disagreement),
        latency_ms=_elapsed_ms(start),
        cost_usd=total_cost,
    )

    judge_detail = LLMJudgeDetail(
        samples=scores,
        mean_score=mean_score,
        disagreement=disagreement,
        is_uncertain=is_uncertain,
        judge_model=self._llm.config.models["judge"].name,
    )

    return tier_result, judge_detail


async def _single_judge_call(self, prompt_context, temperature) -> tuple[JudgeOutput, LLMCallMetadata]:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt_context},
    ]
    return await self._llm.call_structured(
        role="judge",
        messages=messages,
        output_schema=JudgeOutput,
        temperature=temperature,
    )
```

### 5.4 分歧度计算

```python
def _compute_disagreement(scores: list[float], passes: list[bool]) -> float:
    """Disagreement metric.

    Two signals combined:
    - Score standard deviation (continuous)
    - Pass/fail disagreement (binary)
    """
    if len(scores) < 2:
        return 0.0

    # Normalized std
    std = statistics.stdev(scores)

    # Binary disagreement: 0 if all agree, 1 if 50/50
    pass_rate = sum(passes) / len(passes)
    binary_disagreement = 1 - abs(pass_rate - 0.5) * 2

    # Weighted combination
    return 0.5 * std + 0.5 * binary_disagreement
```

`threshold` 默认 0.3，benchmark 可调。

### 5.5 Judge system prompt

```
You are Maestro Judge, an independent code reviewer.

You are given:
1. A subtask description (what was supposed to be done)
2. A diff produced by a coding agent
3. The original files before the diff was applied

Your job: evaluate if the diff correctly implements the subtask.

Score 0.0-1.0:
- 1.0: Perfectly implements the subtask, no bugs, good style
- 0.8: Correct implementation with minor issues (style, naming)
- 0.6: Mostly correct but has a notable issue (missing edge case, suboptimal)
- 0.4: Partially correct, significant issue
- 0.2: Largely incorrect
- 0.0: Completely wrong or does nothing

passes_requirements = true iff score >= 0.6 AND you are confident the diff doesn't introduce bugs.

Output JSON matching JudgeOutput schema. Be concise in reasoning (under 200 words).
```

### 5.6 Judge user prompt 模板

```
# Subtask
ID: {subtask.subtask_id}
Description: {subtask.description}
Files to modify: {subtask.writes}

# Agent's rationale
{sub_result.rationale}

# Agent's diff
```diff
{sub_result.diff}
```

# Original files (before diff)
{formatted_original_files}

# Instructions
Evaluate correctness. Return JSON.
```

### 5.7 成本估算

K=3，每次 judge 调用大约 2-4k input + 500-1000 output tokens。

以 DeepSeek-V3 $0.28/$1.12 per MTok 计算：
- 单次 patch 的 judge 成本 ≈ 3 × (3k × $0.28/M + 700 × $1.12/M) ≈ $0.005
- 30 题 benchmark × 平均 3 patches × $0.005 ≈ $0.45

可控。

## 6. 测试要求

`tests/unit/test_deterministic.py`：
- ruff 成功/失败
- mypy 成功/失败
- 多文件同时检查

`tests/unit/test_test_based.py`：
- pytest 成功
- pytest 失败（详细 output 截尾）
- auto-gen fallback

`tests/unit/test_llm_judge.py`（mock LLMClient）：
- K=3 samples 全部通过
- K=3 samples 全部失败
- K=3 samples 分歧大（1 pass 2 fail）→ is_uncertain=True
- Cost 累加正确

`tests/integration/test_verifier_e2e.py`：
- 真实 workspace + fake LLM judge，测完整 short-circuit

## 7. 面试 talking points（这个模块要详细讲）

**这是项目的最大技术亮点，请准备 10 分钟的详细讲述**：

1. **分层架构的工程哲学**：
   - 便宜的确定性检查放前面（零成本），贵的 LLM judge 放最后
   - 短路逻辑：前层失败绝不进入后层
   - 面试官常问：「为什么不直接让 LLM 一次性 review？」答：「因为 90% 的 patch 错在编译层和测试层，不需要 LLM 参与。让 LLM 只处理剩下的 10% 是成本最优解。」

2. **LLM Judge 的多采样与分歧检测**：
   - 引用自己论文的发现：LLM-as-Judge 在 verifiable 任务上有 silent correctness leakage
   - 工程化应对：K 次采样 + 分歧度 + uncertainty flag
   - 面试官如问：「为什么不用更强的 judge 模型？」答：「因为单次强模型判断仍然有信号泄漏，多次采样检测分歧能捕捉到模型犹豫的 case，这是本质上的方法改进而非单纯提升算力。」

3. **自动生成测试的 fallback**：
   - 多数 coding agent 框架不做这层
   - 体现"现实世界 repo 测试覆盖率常不完整"这个真实工程观察
   - 在 benchmark 上关闭以避免 bias，在 production 上开启——体现 benchmark-production 设计权衡

4. **消融实验设计**：
   - 支持 tier-level 开关，benchmark 上对比 {单 T1, T1+T2, T1+T2+T3} 的 resolve rate 和 cost 曲线
   - 简历数字来源
