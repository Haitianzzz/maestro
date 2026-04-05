# Maestro — 开发设计文档

> **项目定位**：面向成本受限场景的并行化 coding agent 框架。通过 DAG 调度实现任务自动并行、通过分层 verification 保证输出可靠性、通过分层选模优化成本，在自建真实 PR benchmark 上追求"又快、又准、又便宜"的工业可落地方案。

---

## 0. 文档阅读指南

本文档面向 Claude Code 和开发者。Claude Code 实施本项目时，**严格按以下顺序阅读**：

1. 本文档（`DESIGN.md`）——建立架构全景
2. `specs/01-data-models.md`——熟悉核心数据结构
3. `specs/02-llm-client.md`——熟悉 LLM 调用抽象（所有模块都依赖）
4. `specs/03-planner.md`——任务分解与 DAG 生成
5. `specs/04-scheduler.md`——DAG 调度与并行执行
6. `specs/05-subagent.md`——子代理执行逻辑
7. `specs/06-verifier.md`——分层验证
8. `specs/07-sandbox.md`——文件系统隔离与合并
9. `specs/08-cli.md`——用户入口
10. `specs/09-benchmark.md`——评测体系与自动化 harness
11. `specs/10-experiments.md`——消融实验规格
12. `specs/11-prompt-evolution.md`——（可选）GEPA prompt 演化，Week 6 stretch goal

**关键原则**：任何模块的实现不得超出对应 spec 的接口定义。接口不够用时，修改 spec，不要在代码里扩展。

---

## 1. 项目背景与定位

### 1.1 要解决的真实问题

当前主流 coding agent（Claude Code、Cursor Agent、OpenHands、oh-my-claudecode 等）在工业落地时普遍存在三个工程问题：

**问题一：串行执行，任务慢**

大部分 coding agent 采用单 agent 串行模式，一次 coding 任务 wall-clock 时间普遍在数十分钟到小时级别。许多任务（如多文件独立修改、多组件并行开发）天然可并行，现有框架未系统性利用。

**问题二：无强制验证，不可靠**

Agent 产出代码常存在"自信地写错"现象：编译不过、测试不过、逻辑错误但表面合理。现有框架最多在 prompt 中要求 agent 自检，缺乏强制性、独立性的验证环节。

**问题三：成本高，难以规模化**

全链路使用顶级模型（Claude Opus/Sonnet、GPT-4o）单次任务成本数美元。真实工业场景不可能全用最贵模型，但现有框架缺乏清晰的分层选模方案。

### 1.2 Maestro 的方案

针对上述三个问题，Maestro 提出三个对应的核心机制：

| 问题 | Maestro 的机制 | 预期指标 |
|---|---|---|
| 串行慢 | **DAG 调度 + 文件级冲突检测**：Planner 输出带依赖关系的 DAG，Scheduler 自动识别可并行分支并行发 sub-agent | wall-clock 相比 baseline 提升 2-3× |
| 不可靠 | **三层 Verification**：确定性检查（linter、类型） → 测试（pytest） → LLM-as-Judge（多采样 + 分歧检测） | resolve rate 相比 baseline 提升 10-20 个百分点 |
| 贵 | **三档模型分层**：Planner 用强模型（单次调用但高价值） + Sub-agent 用性价比模型（并发多但单价低） + 确定性检查本地零成本 | Pareto 曲线上找到 cost-quality 最优点 |

### 1.3 与同类项目的差异化

| 对比项目 | 它们做的 | Maestro 不同之处 |
|---|---|---|
| Claude Code / Cursor Agent | 单 agent 交互式 coding | 自动并行调度、强制 verification |
| oh-my-claudecode | tmux 模板 + prompt 编排 | 真正的 DAG 调度、冲突检测、分层 verify |
| OpenHands / SWE-agent | 单 agent + sandbox | 并行 + verification 两个独立维度的工程化 |
| Hermes Agent (Nous Research) | 通用 agent 平台、skills 系统 + 多 gateway + 持久 memory | Maestro 是垂直 coding framework，可作为 Hermes 的一个 coding skill 被调用，但本体定位在调度与验证层 |
| LangGraph / CrewAI | 通用多 agent 框架 | 专注 coding 场景、带 verification loop、成本优化 |

**Maestro 不追求通用性**，专精 Python 代码修改场景（bug fix、feature addition、refactor），在此垂直场景上把架构做深。

---

## 2. 架构总览

### 2.1 系统分层

```
┌─────────────────────────────────────────────┐
│              CLI Layer (Typer)              │  ← 用户入口
├─────────────────────────────────────────────┤
│          Orchestrator (LangGraph)           │  ← 状态机主循环
├──────────────┬──────────────┬───────────────┤
│   Planner    │  Scheduler   │    Verifier   │  ← 核心三大件
│              │   + DAG      │   (3 tiers)   │
├──────────────┴──────────────┴───────────────┤
│         Sub-agent Pool (asyncio)            │  ← 并发执行单元
├─────────────────────────────────────────────┤
│            Sandbox (File-level)             │  ← 隔离与合并
├─────────────────────────────────────────────┤
│      LLM Client (OpenAI-compatible)         │  ← 模型调用抽象
└─────────────────────────────────────────────┘
```

### 2.2 核心流程

一次完整的 Maestro 调用流程：

```
用户 CLI 输入任务描述
         │
         ▼
[Phase 1] Planner：读取 repo 结构 → 输出 Task DAG（含文件 read/write 声明）
         │
         ▼
[Phase 2] Scheduler：拓扑排序 → 识别可并行 batch
         │
         ▼
[Phase 3] Sub-agent Pool：并行执行同一 batch 内的 task（每个 task 独立 sandbox workspace）
         │
         ▼
[Phase 4] Verifier（三层）：
         ├─ Tier 1 Deterministic: ruff + mypy（本地零成本）
         ├─ Tier 2 Test-based: pytest（本地零成本）
         └─ Tier 3 LLM-as-Judge: 多采样 + 分歧检测（API 成本）
         │
         ▼
[Phase 5] 根据 Verification 结果：
         ├─ 全通过 → 合并到主 workspace → 进入下一 batch
         ├─ Tier 1/2 失败 → sub-agent 重试（最多 N 次）
         └─ Tier 3 分歧大 → 标记 uncertainty，返回上层决策
         │
         ▼
所有 batch 完成 → 最终 diff 呈现给用户
```

### 2.3 LangGraph 状态机

使用 LangGraph 实现上述流程的主要考虑是：

1. **天然支持条件分支和循环**：verify 失败后回到 sub-agent 重试，天然适合 graph 表达
2. **状态可视化**：LangGraph 的 trace 机制配合你自己的 structured logging，对 debug 多 agent 系统极其有用
3. **你（Zhong Haitian）已有经验**：Deep Research Agent 项目已用 LangGraph 实现五阶段状态机，本项目可直接迁移经验

详细状态图见 `specs/04-scheduler.md`。

---

## 3. 关键设计决策与权衡

本节记录开发中所有"为什么这样、为什么不那样"的决策。**面试时这些是核心 talking points**，Claude Code 实施时必须遵守。

### 3.1 为什么用文件级冲突检测而不是函数级

**决策**：Task 的读写声明只到文件粒度（如 `writes: ["src/auth/login.py"]`）。

**理由**：
- 函数级冲突检测需要 AST 分析、跨文件调用图，工程复杂度跃升
- 真实 coding 任务中，两个并行 task 改同一文件的概率远低于改不同文件的概率
- 文件级粗粒度足以覆盖 80% 场景，剩余 20% 的细粒度冲突通过运行时重试兜底

**什么时候重新考虑**：如果 benchmark 显示冲突重试率超过 30%，再引入函数级。当前版本不做。

### 3.2 为什么三层 Verifier 而不是单层 LLM Judge

**决策**：Verification 强制按 Deterministic → Test → LLM-Judge 顺序，前层失败直接 reject，不进入下一层。

**理由**：
- **成本**：Deterministic 和 Test 是本地执行零 API 成本，LLM-Judge 每次 verify 要花 token。把便宜的放前面，失败早退出
- **可靠性**：Linter 和 pytest 的结果是**客观事实**（编译过就是过），LLM-Judge 是**概率判断**。事实先于判断
- **你的研究经验直接用上**：LLM-as-Judge 存在 silent correctness leakage（你论文的结论），所以 Judge 的结论要用多采样 + 分歧检测而非单次打分

详细实现见 `specs/06-verifier.md`。

### 3.3 为什么 LLM Judge 用多采样而不是单次打分

**决策**：Tier 3 LLM-as-Judge 对每个 patch 采样 K=3 次（不同温度或不同 prompt），计算分歧度；分歧度低于阈值视为"确信"，高于阈值标记"不确定"打回人工或更高阶模型。

**理由**：
- 直接引用你论文（EMNLP 2026 in progress）的结论：在 verifiable-answer 类任务中，LLM-Judge 存在 silent correctness leakage，单次判断不可靠
- 多采样 + 分歧检测是对这个问题的工程化应对
- 这是 Maestro 相比其他框架最独特的技术点之一，面试时是硬核 talking point

**成本控制**：K=3 而不是 K=5。前者成本 3×，可控；后者成本 5× 边际收益递减。

### 3.4 为什么 Sub-agent 输出必须结构化

**决策**：Sub-agent 不能直接返回自由文本，必须返回 `SubAgentResult` 结构（见 `specs/01-data-models.md`）。

**理由**：
- Main agent 的 context 只保留结构化摘要，full transcript 落盘，避免 context 爆炸
- 便于 Verifier 机器可读（patch + rationale + confidence 分开处理）
- 便于后续数据分析（每个 task 的 retry count、confidence 分布都可统计）

### 3.5 为什么三档分层选模

**决策**：Planner 用 Qwen3-Max、Sub-agent 用 Qwen3-Coder-Plus、Verifier Tier 3 用 DeepSeek-V3 或 Qwen3-Coder（更便宜一档）。

**理由**：
- Planner 调用次数少（每任务 1 次）但质量敏感（规划错后续全错），用最强
- Sub-agent 并发多（每 batch 多个）且 coding-specific，用 coding 专精性价比模型
- Verifier Tier 3 调用次数最多（每个 patch 调 K 次 × 多个 patch），用最便宜且"够用"的模型
- 这是真实工业场景的做法。面试被问到成本时，这是硬核答案

### 3.6 为什么用 asyncio 而不是 multiprocessing / Ray

**决策**：并行基础设施用 `asyncio` + `asyncio.Semaphore` + 自定义优先级队列。

**理由**：
- Maestro 的并行本质是"同时等多个 LLM API 返回"——纯 I/O 等待，asyncio 最匹配
- multiprocessing / Ray 是 CPU 密集或分布式场景，Maestro 单机且非 CPU 密集，用它们是过度工程
- 引入 `Semaphore` 限制 API 并发数（避免阿里云 rate limit）
- 引入**优先级队列**：关键路径 task（blocking 后续 batch）优先级高，这是 asyncio 原生 `gather` 做不到的，是 Maestro 的独特技术点

**与 Deep Research Agent 的区分**：
- Deep Research Agent 的 asyncio：子问题级 `asyncio.gather` 并行检索（基础用法）
- Maestro 的 asyncio：基于 DAG 的动态调度器 + 信号量限流 + 优先级队列（框架设计）

两个项目都用 asyncio 但在**不同层次的技术决策**——前者是"用 async 做并行请求"，后者是"基于 async 构建任务调度器"。

### 3.7 为什么文件级 Sandbox 而不是 Docker 容器

**决策**：每个 sub-agent 在独立的临时目录（`tempfile.mkdtemp()` 得到的路径）下操作，执行完后通过 diff 合并回主 workspace。不使用 Docker。

**理由**：
- Docker 启动开销大（秒级），会显著拖慢并行收益
- Python 项目修改本身不需要真正的进程隔离，文件系统隔离足够
- 本地跑 pytest / ruff / mypy 直接在 Python 环境里，不用容器
- 安全性：不执行不可信代码，Sub-agent 生成的 patch 都由用户控制的 repo 约束

**何时升级**：若未来支持不可信任务或跨语言，再引入 Docker。本版本不做。

---

## 4. 技术栈

| 组件 | 选型 | 版本 | 备注 |
|---|---|---|---|
| 语言 | Python | 3.11+ | 使用 `asyncio.TaskGroup`（3.11+ 特性） |
| Agent 编排 | LangGraph | latest | 状态机主循环 |
| 并发 | asyncio | stdlib | + Semaphore + 自定义优先级队列 |
| LLM SDK | openai | latest | 阿里云百炼兼容 OpenAI API |
| CLI | Typer | latest | + Rich 做终端可视化 |
| 数据模型 | Pydantic | v2 | 所有 LLM I/O 用 structured output |
| 测试 | pytest | latest | 同时是 benchmark 执行引擎 |
| Linter | ruff | latest | Tier 1 Verifier |
| 类型检查 | mypy | latest | Tier 1 Verifier（可选） |
| Benchmark | 自建 | — | 见 `specs/09-benchmark.md` |
| 日志 | structlog | latest | JSON 结构化日志 |

**依赖原则**：能用 stdlib 的不引第三方库；能用 Pydantic 的不手写 dataclass；能用 Typer 的不手写 argparse。

---

## 5. 目录结构

```
maestro/
├── DESIGN.md                    # 本文档
├── README.md                    # 项目介绍（英文）
├── CLAUDE.md                    # Claude Code 协作指南
├── pyproject.toml               # 包管理
├── specs/                       # 模块规格书
│   ├── 01-data-models.md
│   ├── 02-llm-client.md
│   ├── 03-planner.md
│   ├── 04-scheduler.md
│   ├── 05-subagent.md
│   ├── 06-verifier.md
│   ├── 07-sandbox.md
│   ├── 08-cli.md
│   ├── 09-benchmark.md
│   └── 10-experiments.md
├── src/
│   └── maestro/
│       ├── __init__.py
│       ├── cli.py              # Typer 入口
│       ├── orchestrator.py     # LangGraph 主循环
│       ├── models.py           # Pydantic 数据模型
│       ├── planner/
│       │   ├── __init__.py
│       │   └── planner.py
│       ├── scheduler/
│       │   ├── __init__.py
│       │   ├── dag.py
│       │   └── scheduler.py
│       ├── subagent/
│       │   ├── __init__.py
│       │   └── subagent.py
│       ├── verifier/
│       │   ├── __init__.py
│       │   ├── deterministic.py
│       │   ├── test_based.py
│       │   └── llm_judge.py
│       ├── sandbox/
│       │   ├── __init__.py
│       │   └── workspace.py
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── client.py
│       │   └── config.py
│       └── utils/
│           ├── __init__.py
│           ├── logging.py
│           └── priority_queue.py
├── tests/
│   ├── unit/
│   └── integration/
├── benchmark/
│   ├── tasks/                  # 30-40 个 bug-fix PR 数据
│   ├── harness.py              # 评测执行器
│   └── results/                # 实验结果
└── examples/
    └── ...                     # 给用户的使用示例
```

---

## 6. 开发时间线

**起始日期**：2026-04-05
**结束日期**：2026-05-16（6 周）
**当前位置**：2026-04-23，Week 3 进行中

| 周 | 日期 | 核心产出 | 状态 |
|---|---|---|---|
| 1 | 4.5 - 4.11 | 项目骨架、CLI 框架、LLM Client、数据模型 | ✅ 已完成 |
| 2 | 4.12 - 4.18 | Planner、DAG、Scheduler 主循环、Sandbox | ✅ 已完成 |
| 3 | 4.19 - 4.25 | Sub-agent 执行、LangGraph 主编排 | 🚧 进行中 |
| 4 | 4.26 - 5.2 | 三层 Verifier 完整实现、重试逻辑 | ⏳ 未开始 |
| 5 | 5.3 - 5.9 | Benchmark 构建（30 题 PR 集）+ harness | ⏳ 未开始 |
| 6 | 5.10 - 5.16 | 消融实验、数据分析、README 打磨 | ⏳ 未开始 |

**Git commit 节奏**：每个模块完成独立 commit，不攒大 commit。在 `README.md` 完善前保持 feature 开发，最后一周打磨文档。

---

## 7. 预算与成本约束

**总 API 预算**：350 RMB（阿里云百炼账户余额）

**分配**：
- 开发调试期：100 RMB（Week 3-4 实现期间小规模试跑）
- 正式 Benchmark：200 RMB（Week 5-6 完整消融实验）
- Buffer：50 RMB

**模型价格参考**（以阿里云百炼 2026 年 Q1 定价为准，实际执行时以当前价格为准）：

| 模型 | 角色 | 预计调用 |
|---|---|---|
| Qwen3-Max | Planner | 每任务 1 次，30-40 任务 |
| Qwen3-Coder-Plus | Sub-agent | 每任务 3-5 次（含重试），30-40 任务 |
| DeepSeek-V3 或 Qwen3-Coder | Verifier Tier 3 | 每 patch 3 次采样 × 多 patch |

**成本控制策略**：
- 所有 LLM 调用强制走 `LLMClient` 抽象，内置 token 统计
- 每次实验 run 自动生成 cost report（每模型、每角色的 token 和费用）
- Benchmark harness 带 `--dry-run` 模式，不调 API 只验证 pipeline
- 所有 prompt 模板集中管理，便于压缩 token

详见 `specs/02-llm-client.md` 和 `specs/09-benchmark.md`。

---

## 8. 成功标准

项目完成度验收（用于面试 talking points 和 README 数字）：

### 8.1 功能性

- [ ] CLI `maestro run <task>` 能端到端执行
- [ ] 支持至少 3 个并行 sub-agent
- [ ] 三层 Verifier 全部实现，每层可单独开关（用于消融）
- [ ] Benchmark 集至少 30 题，全部可自动化评测
- [ ] 生成结构化实验报告（JSON + Markdown）

### 8.2 性能指标（在 30 题 benchmark 上）

| 指标 | Baseline | Maestro 目标 |
|---|---|---|
| Wall-clock（任务平均） | 基准 | 2-3× 加速 |
| Resolve rate（pytest 通过率） | 基准 | +10-20 pp |
| Token cost（任务平均） | 基准 | 生成 Pareto 曲线，找到 cost-quality 最优点 |

### 8.3 可交付

- [ ] GitHub 公开仓库，README 完整
- [ ] 至少一个演示视频或 GIF
- [ ] Benchmark 结果 JSON 原始数据 + 分析报告 Markdown
- [ ] 简历 3 个硬数字（取自实验结果）

---

## 9. 面试叙事（项目完成后的 30 秒电梯演讲）

> "我做了一个叫 Maestro 的 coding agent 框架，专门解决现在 coding agent 三个工业落地的痛点：慢、不可靠、贵。
>
> 架构上我做了三个核心决策：第一，planner 输出带文件读写声明的 DAG，scheduler 自动识别可并行分支——我在自建的 30 题真实 PR benchmark 上做到 X× 加速。第二，强制三层 verification——确定性检查、测试、以及做了多采样分歧检测的 LLM-as-Judge——resolve rate 提升 Y 个百分点，LLM-Judge 的多采样这块直接用到了我 EMNLP 论文里对 silent correctness leakage 的发现。第三，分层选模——planner 用最强的、sub-agent 用性价比的、verifier 用便宜的——在 cost 和 quality 之间画了一条 Pareto 曲线，每任务成本降低 Z%。
>
> 和 oh-my-claudecode、OpenHands 这些项目相比，它们主要在 prompt 和 tmux 编排层做工作，我的项目在执行层和验证层做了 DAG 调度、冲突检测、分层验证这些更底层的工程。"

---

## 10. 附录：术语表

| 术语 | 含义 |
|---|---|
| Task | 用户输入的高层编程任务（如"加一个登录功能"） |
| SubTask | Planner 拆解后的子任务（如"写注册 endpoint"） |
| Batch | Scheduler 拓扑排序后同一层可并行执行的 SubTask 集合 |
| Patch | Sub-agent 对 workspace 产生的 diff |
| Verifier | 三层验证系统的统称 |
| Tier 1/2/3 | 分别指 Deterministic / Test-based / LLM-as-Judge |
| Workspace | Sandbox 内 sub-agent 操作的临时目录 |
| Main Workspace | 合并所有 patch 后的最终目录 |
| DAG | Directed Acyclic Graph，Task 依赖图 |
| Resolve Rate | Benchmark 上最终 pytest 全部通过的任务比例 |
| Silent Correctness Leakage | 见作者论文——LLM-as-Judge 在 verifiable 任务上表面评分合理但实际泄漏 correctness 信号的现象 |
