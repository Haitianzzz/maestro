# CLAUDE.md — Claude Code 协作指南

> 本文件是给 Claude Code 的工作指南。开工前请完整阅读本文档，以及 `DESIGN.md` 和 `specs/` 下的全部规格书。

## 1. 项目一句话

Maestro 是面向成本受限场景的并行化 Python coding agent 框架，通过 DAG 调度 + 分层 verification 追求"又快又准又便宜"。详细背景见 `DESIGN.md`。

## 2. 阅读顺序（必读）

开工第一件事：按顺序阅读以下文档，建立完整心智模型后再动代码。

1. `DESIGN.md`
2. `specs/01-data-models.md`
3. `specs/02-llm-client.md`
4. `specs/03-planner.md`
5. `specs/04-scheduler.md`
6. `specs/05-subagent.md`
7. `specs/06-verifier.md`
8. `specs/07-sandbox.md`
9. `specs/08-cli.md`
10. `specs/09-benchmark.md`
11. `specs/10-experiments.md`
12. `specs/11-prompt-evolution.md`（**可选 stretch goal**，Week 6 条件满足才读）

**不要跳读**。Maestro 的模块之间耦合紧密，理解一个模块需要了解其他模块的接口。

## 3. 实施顺序（与 Week 对齐）

本项目总时长 6 周，起始日 2026-04-05，结束 2026-05-16。截至今天（4.23），Week 1-2 已完成，Week 3 进行中。

### Week 1（已完成）
- [x] `pyproject.toml`、基础 lint/test CI
- [x] `src/maestro/models.py`（spec 01）
- [x] `src/maestro/llm/client.py`（spec 02）
- [x] 基础 CLI 骨架（typer）
- [x] 基础 logging（structlog）

### Week 2（已完成）
- [x] `src/maestro/sandbox/workspace.py`（spec 07）
- [x] `src/maestro/planner/planner.py`（spec 03）
- [x] `src/maestro/scheduler/dag.py` 拓扑排序与冲突检测（spec 04）
- [x] 集成测试：planner 对 tiny fixture repo 产出合法 DAG

### Week 3（进行中，截止 4.25）
- [ ] `src/maestro/subagent/subagent.py` 3 阶段 loop（spec 05）
- [ ] `src/maestro/scheduler/scheduler.py` 主循环（spec 04）
- [ ] `src/maestro/orchestrator.py` LangGraph state machine（spec 04 §5）
- [ ] `src/maestro/utils/priority_queue.py` AsyncPriorityQueue + PrioritySemaphore
- [ ] 端到端 smoke test：对 tiny fixture repo 完整走一遍（verify 用 stub）

### Week 4（4.26-5.2）
- [ ] `src/maestro/verifier/deterministic.py`（spec 06 §3）
- [ ] `src/maestro/verifier/test_based.py`（spec 06 §4）
- [ ] `src/maestro/verifier/llm_judge.py` 多采样 + 分歧检测（spec 06 §5）
- [ ] 重试逻辑完整实现（spec 04 §3.4）
- [ ] 单元测试覆盖率 ≥ 75%

### Week 5（5.3-5.9）
- [ ] `benchmark/build_tasks.py`（spec 09 §2）
- [ ] 构造 30 个 benchmark task（手工审核）
- [ ] `benchmark/harness.py`（spec 09 §3）
- [ ] `benchmark/configs.py`（spec 09 §4）
- [ ] Smoke test：跑 baseline + full，各 5 task

### Week 6（5.10-5.16）
- [ ] 完整实验矩阵（spec 10 §2）
- [ ] `benchmark/analysis/analyze.py`（spec 10 §5）
- [ ] `benchmark/results/REPORT.md`（spec 10 §7）
- [ ] `benchmark/results/FAILURE_ANALYSIS.md`
- [ ] README 完善、demo GIF
- [ ] **(Stretch goal) GEPA prompt evolution**（spec 11）——仅在 E1-E6 ablation 于 Day 3 完成时启动。严格遵守 spec 11 §0 门槛和 §10 禁止事项

## 4. 工作方式约定

### 4.1 每个任务开始时

1. 读对应 spec
2. 写一个简短的"我的实施计划"（内部思考，不需要输出长篇）
3. 开始实现

### 4.2 实现过程中

1. **不得偏离 spec**：如果 spec 不够用或有错，提出修改建议（TODO 注释），但不要擅自扩展接口
2. **优先写类型**：先定义 Pydantic model 和接口签名，再写实现
3. **先写测试再写实现**：关键模块（Planner、Scheduler、Verifier）采用 TDD
4. **小步 commit**：一个功能点一个 commit，commit message 英文规范

### 4.3 Commit Message 规范

```
<type>(<scope>): <subject>

<body optional>
```

Type 用：`feat`、`fix`、`refactor`、`test`、`docs`、`chore`
Scope 用模块名：`planner`、`scheduler`、`subagent`、`verifier`、`sandbox`、`llm`、`cli`、`benchmark`

例子：
```
feat(scheduler): add topological batching with write-conflict detection

Implements topological_batches() and detect_write_conflicts() per spec 04 §2.1.
Subtasks with matching writes within a batch are now automatically deferred
to the next batch based on priority.

Tests: 12 new unit tests in test_dag.py covering cycles, conflicts, priorities.
```

### 4.4 代码风格

- Python 3.11+，类型注解强制
- `ruff check` 和 `ruff format` pass
- `mypy --strict` 在 `src/maestro/` 下 pass（`benchmark/` 下允许宽松）
- 所有 public 函数/类有 docstring（英文，Google style）
- 不使用裸 dict 传递跨模块数据，必须 Pydantic model

### 4.5 依赖管理

- 使用 `uv` 或 `poetry`（Claude Code 选其一）
- 主要依赖：`openai`、`pydantic>=2`、`langgraph`、`typer`、`rich`、`structlog`、`unidiff`、`aiofiles`
- 开发依赖：`pytest`、`pytest-asyncio`、`pytest-cov`、`ruff`、`mypy`、`pre-commit`

## 5. 测试策略

### 5.1 单元测试

`tests/unit/` 下，每个模块对应一个 `test_<module>.py`。目标覆盖率 ≥ 75%。

Mock LLM 调用：用 pytest fixture 构造 mock `LLMClient`。

### 5.2 集成测试

`tests/integration/` 下，用 tiny fixture repo（`tests/fixtures/tiny_flask_app/`）。

- `test_planner_e2e.py`：真实 repo → planner → DAG
- `test_scheduler_e2e.py`：fake sub-agents + 真实 scheduler
- `test_verifier_e2e.py`：真实 verifier（除 LLM judge 用 stub）
- `test_full_e2e.py`：完整流程（LLM 部分用 stub）

### 5.3 Live 测试

`tests/live/` 下，需要环境变量 `MAESTRO_LIVE_TEST=1` + 真实 API key 才跑。CI 默认不跑。

## 6. 禁止事项

1. **不要在业务模块直接调 OpenAI SDK**，必须通过 `LLMClient`
2. **不要在 spec 之外扩展接口**，如需扩展先更新 spec（在对应 `specs/*.md` 文件改动，然后 commit 里说明）
3. **不要 print**，用 `structlog` logger
4. **不要硬编码 API key**、模型名、价格——走 config
5. **不要 swallow exception**，要么 raise 要么 log 错误
6. **不要在主代码里用 `async def main()` 的 hack**，CLI 用 typer 的标准方式
7. **不要生成超过 100 行的单个函数**，拆分

## 7. 遇到歧义时

如果 spec 在某点有歧义或缺失，按以下顺序处理：

1. 先查 `DESIGN.md` §3（关键设计决策）
2. 如仍不明确，选择**更保守的方案**（更少功能、更严格约束）
3. 在代码中加 `# TODO(haitian): clarify with spec - <issue>`
4. commit message 里标注

## 8. 面试视角（指导 Claude Code 实施时的品味）

本项目将用于研究生实习/秋招面试。Claude Code 实施时，在每个技术决策点要考虑"这会如何被面试官评价"：

- 代码要**让面试官读懂**：清晰的命名、有意义的注释
- 关键算法要有**伪代码式注释**（便于面试 whiteboard 讲解）
- 重要决策点在代码里留下 `# DESIGN: ...` 注释，引用 DESIGN.md 相关段落
- Log 要**信息密度高**：不是 "starting task" 而是 "task=abc-123 subtasks=4 batches=2 max_parallel=4"

## 9. 当前状态（2026-04-23）

- 当前分支: `main`
- Week 3 Day 3
- 下一步：开始实施 `src/maestro/subagent/subagent.py`
- 已有代码：models、llm client、planner、sandbox、scheduler/dag

开工：
```bash
# 进入项目目录
cd /path/to/maestro

# 安装依赖
uv sync  # or `poetry install`

# 跑已有测试确认环境 OK
pytest

# 开始本周任务
# 先读 spec 05，写 subagent 实现 + 测试
```

## 10. 提交前检查清单

在每次完成一个模块/阶段后：

- [ ] `ruff check .` pass
- [ ] `ruff format .` applied
- [ ] `mypy src/maestro/` pass
- [ ] `pytest tests/unit/` pass
- [ ] 相关 spec 的"测试要求" 对应的测试全部存在
- [ ] 新增的 public 函数有 docstring
- [ ] 当前模块的 README section（如果 spec 要求）已更新
- [ ] Commit message 符合规范
