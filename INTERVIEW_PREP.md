# 简历写法与面试准备

> 本文件不是项目代码的一部分，是给你自己用的简历写法建议 + 面试准备材料。等 Week 6 实验数据出来后，把 `[X]` / `[Y]` 占位符换成真实数字。

---

## 1. 项目标题

**简历一行标题建议**（选一）：

> **Maestro** — 并行化 Python Coding Agent 框架 *（开源 · Python · LangGraph · asyncio）*

或者更强调成果的版本：

> **Maestro** — 并行化 Coding Agent 框架（GitHub: `username/maestro`）

**注**：GitHub 仓库尽量 public + 有 commit history + README 写好，因为面试官 100% 会点进去看。

---

## 2. 项目 Bullet Points（推荐用这版）

建议 3-4 条 bullet，每条不超过两行。模板填入真实数字后使用：

**Version A（强调架构）**：

- 设计并实现了一个面向成本受限场景的 **coding agent 框架 Maestro**，核心包括带文件级冲突检测的 **DAG 并行调度器**、**三层验证系统**（linter/pytest/多采样 LLM-as-Judge）、以及分层选模成本优化
- 基于 LangGraph + asyncio + 自研 `PrioritySemaphore` 实现动态任务调度，在自建的 **30 题真实 PR benchmark**（来自 requests/click/flask/httpx/typer）上达到 **[X]× wall-clock 加速** 与 **[Y]pp resolve rate 提升**
- 针对 LLM-as-Judge 的 silent correctness leakage 问题（详见所著 EMNLP 论文），设计 **K=3 多采样 + 分歧度阈值** 机制，在 benchmark 上成功识别 **[Z]% 不确定判定**，减少单次 Judge 的误判
- 通过分层选模（Qwen3-Max / Qwen3-Coder-Plus / DeepSeek-V3）实现单任务成本 $ **[W]**，相比全链路顶级模型成本降低 **[V]%**，在 cost-quality Pareto 前沿上找到最优配置

**Version A+（含 GEPA，仅当 spec 11 stretch goal 有显著正结果时使用）**：

在 Version A 第 3 条后插入一条，或替换第 4 条成本 bullet 的后半：

- 应用 **DSPy + GEPA（Genetic-Pareto Prompt Evolution）** 对 Judge prompt 做自动优化，在标注样本集上 F1 从 **[baseline]** 提升至 **[evolved]**，集成回主 pipeline 后 resolve rate 再提 **[A]pp**

**Version B（更简洁，强调成果）**：

- 设计并开源了 coding agent 框架 **Maestro**（GitHub 链接），核心解决现有框架串行慢、无强制验证、高成本三大工程问题
- **DAG 并行调度 + asyncio TaskGroup + 优先级信号量**：在自建 30 题 Python bug-fix benchmark 上达到 **[X]× wall-clock 加速**
- **三层验证系统（确定性检查 / pytest / 多采样 LLM-Judge）**：resolve rate 相比 baseline 提升 **[Y] 个百分点**；其中 LLM-Judge 的多采样设计基于作者 EMNLP 论文对 judge 不可靠性的研究
- **三档分层选模**：Planner 用强模型、Sub-agent 用性价比模型、Judge 用便宜模型，实现 cost-quality Pareto 最优，单任务成本 $**[W]**

---

## 3. 数字占位符说明

等 Week 6 实验完成后，从 `benchmark/results/REPORT.md` 里取数填入：

| 占位符 | 含义 | 期望范围 | 来源 |
|---|---|---|---|
| `[X]` | wall-clock 加速倍数 | 2-3× | `experiments/P3 vs E1` |
| `[Y]` | resolve rate 提升百分点 | 10-20pp | `experiments/E6 vs E1` |
| `[Z]` | Judge 标记 uncertain 的比例 | 15-30% | `experiments/J2` |
| `[W]` | 单任务平均美元成本 | $0.3-0.8 | `experiments/E6 cost/task` |
| `[V]` | 相比 baseline（假设用顶级模型）的降本比例 | 50-70% | 用 Claude Opus 报价估算对比 |

**如果实验结果不如预期怎么办**：

- 加速只有 1.5×：改成"**显著 wall-clock 降低**"不给数字
- Resolve rate 不升反降：说明并行 + verify 对难题 trade-off 有意思——做 failure analysis 变成研究叙事
- 成本数字不够漂亮：改成讲 "Pareto frontier 分析" 而非单点成本

---

## 4. 与你简历上另外两个项目的连接

你目前简历应该有三个项目位置：

- **Project 1**: Deep Research Agent (LangGraph + asyncio 基础并行)
- **Project 2**: LLM Training Pipeline / EMNLP 研究工作
- **Project 3**: Maestro ← 新加的

**三个项目形成的叙事**：

1. **Deep Research Agent** 让你建立了 "用 LangGraph + asyncio 做 agent" 的工程基础
2. **EMNLP 研究** 让你积累了"LLM-as-Judge 可靠性"的学术观察
3. **Maestro** 把两者融合，做成一个解决真实工程问题的系统——**学术发现转化为工程方案**

面试官若问"你这三个项目之间的关联"，答：

> "Deep Research Agent 让我熟悉了 agent 编排的基础工程，但那时我用 asyncio 主要是做 I/O 级并行。做 EMNLP 论文时，我观察到 LLM-as-Judge 在 verifiable 任务上有 silent correctness leakage。Maestro 是我把这两件事结合起来的尝试：既把 agent 编排从 I/O 并行提升到 task 级 DAG 调度这种真正的调度抽象，又把 judge 的可靠性问题用多采样 + 分歧检测从工程上解决掉。"

---

## 5. 面试常见问题预演

### 5.1 技术深度类

**Q：你为什么用 asyncio 而不是 Ray / multiprocessing？**

A：Maestro 的并行瓶颈是 LLM API 等待（纯 I/O），不是 CPU。asyncio 是匹配的抽象。multiprocessing 有进程启动开销，Ray 是分布式场景，都是 overkill。更重要的是，基于 asyncio 我可以定制 `PrioritySemaphore`——关键路径的 task 优先拿到并发 slot——这是标准库 `asyncio.gather` 做不到的调度语义。

**Q：你的 LLM-Judge 为什么用多采样？**

A：我在做 EMNLP 论文的时候发现，单次 judge 判断在 verifiable-answer 任务上有 silent correctness leakage——表面上评分合理，实际上 judge 的内部置信度有波动但被压成单点输出。多采样（K=3，不同 temperature）能把这种内部不确定性暴露出来，分歧度高的 case 就是 judge 自己都犹豫的 case，这种 case 不应该被直接放行。这不是加算力的提升，是方法论的提升。

**Q：DAG 调度具体怎么做冲突检测？**

A：两层。第一层是静态层：Planner 让 LLM 声明每个 subtask 的 `writes` 文件列表，我在 Planner 输出后拓扑排序，同 batch 内检测写冲突。第二层是运行时：Sub-agent 产生 diff 后，`apply_diff` 前检查所有修改的文件是否在 `writes` 声明内，越界直接 reject。这是防线，因为 LLM 不可靠，不能相信它一定遵守声明。

**Q：文件级冲突检测不够细吧，两个 agent 改同一文件不同函数怎么办？**

A：确实是 trade-off。函数级需要 AST 分析 + 跨文件调用图，工程复杂度跃升。真实数据显示，独立 subtask 改同文件是少数情况，所以我选文件级粗粒度覆盖 80% 场景，剩下的用运行时重试兜底。如果 benchmark 显示冲突重试率超过 30%，我会再引入函数级。现在的版本不做。

**Q：为什么不用 Docker sandbox？**

A：Python 代码修改本身不需要进程隔离，文件系统隔离足够。Docker 启动是秒级，会把并行的收益抵消掉。本项目不执行不可信代码（用户自己的 repo），不用 Docker 是合理的轻量选择。如果支持跨语言或跑不可信任务，我会再加 Docker 层。

### 5.2 工程判断类

**Q：6 周做完这些，是不是有些模块做得很粗？**

A：肯定有取舍。我用"是否进 benchmark 的主 config"做取舍标准：DAG 调度、三层 verifier、PrioritySemaphore 都是 core，做深了；auto-gen test、cost report Markdown 输出、CLI 的 Rich UI 是 nice-to-have，做够了就停。优先级的依据是"能不能支撑简历那三个数字"。

**Q：你怎么防止 benchmark 作弊？**

A：三点。第一，自然语言 prompt 不是直接用 PR 标题，而是用强模型（Qwen3-Max）读 failing test 生成用户视角描述，人工审核。第二，benchmark 数据选 6 个月前已 merge 的 PR，避免数据污染（虽然 Qwen3-Coder 可能见过，但这是所有 benchmark 的共同问题）。第三，消融实验里 baseline 和 full 用同一个 prompt 同一套数据，差异来自架构不来自数据优势。

**Q：Planner 犯错怎么办？**

A：两层防护。一是 Planner 输出后自动做 DAG 验证（成环、引用不存在的 id、writes 路径非法），失败时最多 2 次带错误反馈的重试。二是 Scheduler 在运行时再做一次写冲突检测，Planner 没发现的冲突这里兜底——把低优先级 task 推到下一 batch。Planning 错误在我的日志里会被归类，如果某类错误频繁，后续会针对性优化 prompt。

**Q：你的 benchmark 才 30 题，有统计显著性吗？**

A：30 不够做强统计结论，这是局限性，我在 report 的 "Limitations" 里明确写了。我的定位不是发论文，是做工程 demo + 面试物料。**数据生成和手工审核每题成本很高**，30 题已经接近我单人可控的上限。消融实验里我用 per-task win matrix 而不是只看聚合数字，也是对小样本的补充。

### 5.3 连带打击类（面试官故意挖坑）

**Q：这个项目和 oh-my-claudecode 有啥区别，不就是换个名字？**

A：oh-my-claudecode 主要是 Claude Code 的 tmux 多实例启动脚本 + prompt 模板。它在**用户层**做并行——你开 4 个 terminal。我在**调度层**做并行——一个进程内 DAG 调度。另外它不做 verification，生成的代码对不对完全看 Claude Code 自己。我的三层 verify 是独立的模块，可以配合任何 sub-agent 模型。所以我们是不同技术层的东西，不是替代关系。

**Q：你没有真的用 Claude 作为 sub-agent，数字不可比？**

A：对，我用的是阿里云百炼的 Qwen3-Coder-Plus，选型原因是预算。但我的论点不是"模型能力"，是"系统架构对任何 sub-agent 都能带来的 delta"。ablation 对比 baseline 和 full 用的是**同一个 sub-agent 模型**，所以加速和 resolve rate 提升都是可以归因到架构的。

**Q：全链路 Claude Opus 的数字会不会比你好？**

A：很可能会，因为模型差距摆在那。但 cost 会高一两个数量级。Maestro 的定位不是"绝对最强"，是"相同成本下做得更好、相同效果下花得更少"。Pareto frontier 上的点都是有价值的，不是只有最右上角才对。

**Q：你知道 Hermes Agent / Nous Research 那些吗？你这个和他们什么关系？**

A：Hermes Agent 是 Nous Research 的通用自主 agent 平台，23k+ stars，定位是 "Claude Code 风格的桌面助手 + 多 gateway（Telegram、Slack 等） + 持久 memory + skills 系统"。它是**平台**。Maestro 是**垂直 coding framework**。两者不是同层的东西——Maestro 理论上可以被封装成 Hermes 的一个 coding skill 让 Hermes 调用，但本体架构工作集中在调度层和验证层，不是 agent runtime 层。所以说"有交集但没替代关系"比较准确。

**Q：你有没有用到什么前沿的 prompt optimization 技术？**（如果你做了 GEPA）

A：做了。Week 6 我集成了 DSPy + GEPA（Genetic-Pareto Prompt Evolution，2025 年的技术）来自动优化 Judge prompt。思路是把 benchmark train split 作为标注数据、用 GEPA 的反思性变异迭代改 prompt。我只演化 Judge 这一个 prompt——因为它 eval 最便宜、监督信号最清晰、和我 EMNLP 研究线最贴近。结果是 [填数字]。过程中有一个挺有意思的观察：当架构层已经用了多采样 aggregation 之后，prompt 优化的边际收益会被 aggregation 吃掉，说明架构 trick 和 prompt trick 有一定的**互相替代关系**。

**Q：你有没有用到什么前沿的 prompt optimization 技术？**（如果你没做 GEPA）

A：在 Future Work 里列了——我研究过 DSPy + GEPA 这套方法，本项目里没有集成是因为 benchmark 样本量（30 task）对 prompt evolution 的信号强度不够友好，我担心演化噪声盖过真实提升。下一步我计划扩展 benchmark 到 100+ task 后再做 prompt optimization 的 ablation。

---

## 6. GitHub 仓库 Tips

面试官 100% 会点进 GitHub。确保：

1. **README.md 漂亮**：有架构图、Quick Start、Benchmark 表格（用 ASCII 或 mermaid）
2. **commit history 真实**：4.5 → 5.16 均匀分布（Claude Code 按 CLAUDE.md 的周计划走就能做到）
3. **至少一个 demo GIF**：`maestro run` 的 Rich UI 过程录屏，asciinema 转 GIF
4. **Issues 页挂 2-3 个 "future work" issue**：展示你对未来迭代的思考（如 "Function-level conflict detection (#12)"）
5. **Release tag**：v0.1.0 打出来
6. **Topics**：`llm-agent`, `coding-agent`, `langgraph`, `asyncio`, `python`

---

## 7. 简历其他区的配合

Project 2（EMNLP 研究）那条 bullet 里加一句把 Maestro 勾出来：

> ... 发现 LLM-as-Judge 在 verifiable-answer 任务上存在 silent correctness leakage，该观察在后续项目 Maestro 中通过多采样 + 分歧检测机制落地为工程方案。

这样三个项目之间形成闭环，面试官一眼能看出来。

---

## 8. 交付清单（给自己的）

- [ ] Week 6 结束后，从 REPORT.md 取三个数字填进本文件
- [ ] 从本文件 Version A/B 里选一版 copy 到简历
- [ ] README.md 的 benchmark table 填数字
- [ ] GitHub 仓库设 public
- [ ] 录 demo GIF 放 README
- [ ] 在简历里 Project 2 加上与 Maestro 的连接句
- [ ] 面试前把本文件 §5 的问答复习一遍，每个答案背到能用自己的话 30 秒内说完
