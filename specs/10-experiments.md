# Spec 10 — Experiments（消融实验）

> **位置**：`benchmark/experiments/`
> **依赖**：Benchmark Harness
> **被依赖**：最终报告

## 1. 总览

本 spec 定义 Week 6 的完整消融实验矩阵，用于产出简历数字和对外报告。

所有实验在**同一 benchmark 数据集**（Week 5 构建的 30 task）上运行，唯一变量是 Maestro 的配置。

## 2. 实验矩阵

### 2.1 主对比：Full vs Baseline vs Ablations

| 实验 ID | Config | Parallel | Verifier Tiers | 目的 |
|---|---|---|---|---|
| E1 | baseline | No (max_parallel=1) | None | 对比基准 |
| E2 | parallel_only | Yes | None | 隔离 parallel 贡献 |
| E3 | verify_t1 | No | T1 | 隔离 T1 贡献 |
| E4 | verify_t12 | No | T1+T2 | 隔离 T1+T2 贡献 |
| E5 | verify_all | No | T1+T2+T3 | 隔离完整 verify 贡献 |
| E6 | full | Yes | T1+T2+T3 | Maestro 完整配置 |

### 2.2 Judge 参数敏感性

单独针对 LLM Judge 做小规模实验（仅 10 task）：

| 实验 ID | K samples | Threshold | 目的 |
|---|---|---|---|
| J1 | K=1 | n/a | 单 judge baseline |
| J2 | K=3, threshold=0.3 | — | 默认配置 |
| J3 | K=3, threshold=0.5 | — | 宽松阈值 |
| J4 | K=5, threshold=0.3 | — | 更多 sample |

J4 成本 5x J1，只在 10 task 上跑，看是否值得。

### 2.3 并行度敏感性（可选）

| 实验 ID | max_parallel | 目的 |
|---|---|---|
| P1 | 1 | 等价 baseline |
| P2 | 2 | |
| P3 | 4 (default) | |
| P4 | 8 | 测试 rate limit 极限 |

## 3. 指标

每个实验产出以下指标（每 task 一条记录 + aggregate）：

### 3.1 主指标

| 指标 | 定义 |
|---|---|
| **Resolve Rate** | resolved_tasks / total_tasks |
| **Wall-clock (mean)** | 平均单任务 wall-clock 时间 (ms) |
| **Speedup** | baseline_wall_clock / experiment_wall_clock |
| **Cost (mean)** | 平均单任务美元成本 |
| **Tokens (mean)** | 平均单任务 input + output token |

### 3.2 诊断指标（帮助分析）

| 指标 | 定义 |
|---|---|
| Retry Rate | avg retries per subtask |
| Judge Uncertainty Rate | fraction of judge calls flagged uncertain |
| Tier 1 Fail Rate | fraction of patches failing T1 |
| Tier 2 Fail Rate | fraction of patches failing T2 |
| Tier 3 Fail Rate | fraction of patches failing T3 |
| Write Conflict Rate | fraction of batches with runtime conflicts |

### 3.3 Patch 质量

| 指标 | 定义 |
|---|---|
| Patch Similarity | mean similarity vs ground truth |
| Files Modified Match | fraction where Maestro touched exactly the expected files |
| Extra Files Modified | avg # of "extra" files Maestro touched beyond expected |

## 4. 执行计划

### 4.1 时间分配（Week 6）

| Day | 任务 |
|---|---|
| 1 (5.10) | 跑 E1, E2 (便宜的) + 验证 harness |
| 2 (5.11) | 跑 E3, E4 |
| 3 (5.12) | 跑 E5, E6 (贵的) |
| 4 (5.13) | 跑 J1-J4 |
| 5 (5.14) | 分析 + 画图 + Failure case review |
| 6 (5.15) | 重跑任何需要补的 config |
| 7 (5.16) | 写 final report + README 更新 |

### 4.2 预算分配

总 API 预算 200 RMB ≈ $28 用于最终实验（另 150 RMB 留 Week 3-5 调试）：

| 实验组 | 估算成本 |
|---|---|
| E1 (baseline) | $2-3 |
| E2 (parallel_only) | $2-3 |
| E3-E5 (verify ablations) | $10-15 |
| E6 (full) | $5-8 |
| J1-J4 (judge ablation, 10 task each) | $3-5 |
| Buffer | $3 |

## 5. 数据分析

### 5.1 自动化分析脚本

`benchmark/analysis/analyze.py`：

```python
class ExperimentAnalyzer:
    def __init__(self, reports: dict[str, BenchmarkReport]):
        self._reports = reports

    def compute_speedups(self) -> dict[str, float]:
        """Speedup of each config relative to baseline."""

    def compute_pareto(self) -> list[tuple[str, float, float]]:
        """Return (config, cost, resolve_rate) for Pareto plot."""

    def task_win_matrix(self) -> pd.DataFrame:
        """For each task, which configs solved it."""

    def render_report(self, output_path: Path):
        """Emit COMPARISON.md with tables and embedded figures."""
```

### 5.2 关键图表

1. **Figure 1: Resolve Rate by Config**（柱状图）
2. **Figure 2: Wall-clock by Config**（柱状图）
3. **Figure 3: Cost vs Resolve Rate Pareto**（散点图，Maestro full 应在 frontier 上）
4. **Figure 4: Tier Failure Breakdown**（堆叠条形图，每 config 各 tier 失败占比）
5. **Figure 5: Per-task Win Matrix**（heatmap，task × config，绿=resolve 红=fail）

## 6. Failure Analysis

**Week 6 Day 5** 做 failure case review：

1. 取 E6 (full) 未 resolve 的所有 task
2. 人工打开每个 task 的 `final_result.json` 和 `final.diff`
3. 分类失败原因：
   - `planner_wrong`: DAG 分解错了（比如拆成了错的 writes）
   - `subagent_wrong`: 实现错了
   - `verify_too_strict`: verify 判错（本来对的被标为 fail）
   - `verify_too_lax`: verify 没抓到（本来错的被标为 pass，ground truth test 失败）
   - `diff_apply_error`: 技术性错误（format 错）
   - `benchmark_bug`: benchmark task 本身有问题

4. 输出 `benchmark/results/FAILURE_ANALYSIS.md`

这份分析是**科学态度**的体现，面试讲起来加分。

## 7. Final Report 结构

`benchmark/results/REPORT.md` 最终结构：

```
# Maestro Benchmark Report

## 1. Summary
- 30 tasks from 5 Python repos
- Best config (Maestro full): X% resolve rate, Y× speedup vs baseline, $Z per task
- Key finding: ...

## 2. Benchmark Construction
- Data sources
- PR selection criteria
- Natural language prompt generation

## 3. Experimental Setup
- 6 main configs + 4 judge configs
- Total $X spent on API calls

## 4. Main Results
- Resolve rate table
- Speedup table
- Pareto plot

## 5. Ablation Analysis
- Parallel's marginal contribution
- Each verifier tier's marginal contribution
- Judge multi-sampling's effect

## 6. Failure Analysis
- Categorized failure causes
- Lessons learned

## 7. Limitations
- Small benchmark size (30)
- Python-only
- ...

## 8. Reproduction
- Commands to reproduce each experiment
```

## 8. 简历数字的选择

Report 写完后，选 3 个最强的数字放简历。**候选**：

- "**X% resolve rate** (vs baseline Y%) on a 30-task self-constructed Python bug-fix benchmark"
- "**Z× wall-clock speedup** via DAG-based parallel scheduling"
- "**$W per task** ({W/baseline}% cost reduction via tiered model selection)"
- "**N percentage points** resolve rate improvement attributable to layered verification (T1+T2+T3 vs no verify)"
- "LLM-Judge multi-sampling flagged **K% of patches** as uncertain, reducing silent false positives observed in single-sample judge"

最终选哪三个取决于实验结果。如果实验结果理想，建议选：

1. Resolve rate 提升
2. Speedup
3. LLM-Judge uncertainty flagging 效果

第三个是**最独特**的数字，和你论文的联动最强。

## 9. 面试 talking points

1. **完整 ablation matrix**：不是"我做了个 agent 跑了个数"，而是"我做了 6 个 config 的 ablation 对比，能分离出 parallel 和 verify 各自的贡献"
2. **Failure analysis**：对未 resolve 的 task 做了分类归因，不是"做完就算"
3. **Pareto frontier**：不是单一指标最优，而是展示 cost-quality trade-off
4. **Judge ablation 独立做**：验证多采样是否真的有效，与论文结论对齐
