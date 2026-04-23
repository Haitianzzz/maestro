# Spec 11 — Prompt Evolution via GEPA（可选，Week 6 stretch goal）

> **状态**：**可选**。仅在 Week 6 进度允许时执行。不是 Maestro 的核心模块。

## 0. 执行前置条件（硬门槛）

只有以下条件**全部满足**才启动本 spec 的实现：

1. Week 5 的 benchmark 构建已完成（30 task 全部就绪、human review 通过）
2. Week 6 Day 3（5.12）前，主 ablation 实验 E1-E6 已全部跑完并有数据
3. API 预算剩余 ≥ $8
4. Maestro 核心功能（planner/scheduler/verifier）稳定，无 blocking bug

**任一条件不满足，跳过本 spec。** 在 `REPORT.md` 的 Future Work 段落中讨论 GEPA 作为下一步方向即可。

## 1. 动机

Maestro 的 Judge prompt（见 `specs/06-verifier.md` §5.5）是手工编写的。Judge 判断的质量显著依赖 prompt 措辞。

本模块验证一个假设：**用自动化 prompt 优化是否能提升 Judge 对 ground truth 的一致性。**

**为什么只优化 Judge（不优化 Planner / Sub-agent）**：

1. **Judge 评估便宜**：单次 LLM call 就是一次 judge，不需要跑完整 Maestro pipeline
2. **Judge 有清晰监督信号**：patch 是否真的通过 ground truth pytest
3. **和你的研究叙事对齐**：EMNLP 论文研究的就是 LLM-as-Judge 可靠性
4. **scope 可控**：只改一个模块，不碰主 pipeline

## 2. 技术路线：DSPy + GEPA

**GEPA**（Genetic-Pareto Prompt Evolution，2025 年新技术）结合：

- 遗传算法结构（population、mutation、selection）
- 反思性变异（LLM 分析失败案例、提出 prompt 改进）
- Pareto 前沿选择（保留多个非支配候选）

**DSPy** 是 Stanford NLP 的开源 prompt optimization 框架，原生支持 GEPA。

## 3. 评估信号

### 3.1 数据集构造

在 benchmark 30 task 中划分 **train 20 / test 10**（固定 split，保存 `benchmark/gepa_split.json`）。

对 train 20 task 的每个 task：
1. 用 baseline Maestro（只 planner + subagent，不开 verify）生成 2-4 个 patch 候选——可通过多次运行、不同温度得到多样 patch
2. 对每个 patch，运行真实 ground truth pytest，得到 pass/fail 标签
3. 产出 `(subtask_description, diff, ground_truth_pass)` triple

**目标样本量**：50-80 个 labeled (task, patch) pair。

**成本**：$2-3（主要是 subagent 生成 patch；pytest 本地跑不花钱）

### 3.2 Judge F1 作为 primary metric

```
precision = judge_pass ∩ ground_truth_pass / judge_pass
recall    = judge_pass ∩ ground_truth_pass / ground_truth_pass
F1        = 2 * precision * recall / (precision + recall)
```

次要 metric：
- **Calibration**：Judge 的 score（0-1）与 ground truth 的相关性（Spearman）
- **Uncertainty rate**：多采样分歧度超阈值的比例（和 hand-crafted 对比看是否降低）

## 4. 实现

### 4.1 目录结构

```
benchmark/
├── experiments/
│   ├── gepa/
│   │   ├── prepare_data.py       # 3.1 数据构造
│   │   ├── evolve.py             # GEPA 主循环
│   │   ├── evaluate.py           # baseline vs evolved 对比
│   │   ├── eval_data.jsonl       # 标注数据（生成产物）
│   │   ├── evolved_prompt.txt    # 演化后的 prompt（最佳候选）
│   │   └── evolution_log.jsonl   # 每代每候选的得分日志
│   └── gepa_split.json           # train/test task id split
```

### 4.2 Prepare data

```python
# benchmark/experiments/gepa/prepare_data.py

async def prepare_gepa_data(task_set_dir, split_file, output):
    """Generate labeled (task, patch, ground_truth) dataset."""
    split = json.loads(split_file.read_text())
    train_task_ids = split["train"]  # 20 task ids
    
    examples = []
    for task_id in train_task_ids:
        task_dir = task_set_dir / task_id
        meta = json.loads((task_dir / "task.json").read_text())
        
        # Generate 3 patches by running baseline Maestro 3 times with different temperatures
        patches = []
        for temp in [0.2, 0.5, 0.8]:
            patch = await generate_baseline_patch(task_dir, temperature=temp)
            patches.append(patch)
        
        # Dedup
        unique_patches = dedup_by_content(patches)
        
        # Evaluate each patch against ground truth
        for patch in unique_patches:
            eval_result = await run_ground_truth_tests(task_dir, patch)
            examples.append({
                "task_id": task_id,
                "subtask_description": meta["natural_language_prompt"],
                "diff": patch.diff,
                "ground_truth_pass": eval_result.passed,
            })
    
    _write_jsonl(output, examples)
    return examples
```

### 4.3 Evolution

```python
# benchmark/experiments/gepa/evolve.py

import dspy
from dspy.teleprompt import GEPA

class JudgePredictor(dspy.Signature):
    """Evaluate whether a code patch correctly implements the subtask."""
    subtask_description: str = dspy.InputField()
    diff: str = dspy.InputField()
    passes: bool = dspy.OutputField(desc="Does the patch correctly implement the subtask?")
    score: float = dspy.OutputField(desc="Confidence 0-1")
    reasoning: str = dspy.OutputField(desc="Brief rationale")


def f1_metric(example, pred, trace=None):
    """Evaluator: F1 score of judge verdict vs ground truth."""
    if pred.passes == example.ground_truth_pass:
        return 1.0
    return 0.0
    # Note: For per-example metric we return 0/1; F1 is computed over full set.


async def evolve():
    # Load labeled data
    examples = _load_jsonl("benchmark/experiments/gepa/eval_data.jsonl")
    trainset = [dspy.Example(**e).with_inputs("subtask_description", "diff") for e in examples]
    
    # Configure DSPy to use the judge model
    lm = dspy.LM(
        model=f"openai/{get_config().models['judge'].name}",
        api_base=get_config().base_url,
        api_key=get_config().api_key,
    )
    dspy.configure(lm=lm)
    
    baseline_judge = dspy.Predict(JudgePredictor)
    
    # Configure GEPA with tight budget
    teleprompter = GEPA(
        metric=f1_metric,
        max_iterations=5,
        population_size=4,
        # Other GEPA hyperparams per DSPy docs
    )
    
    evolved_judge = teleprompter.compile(baseline_judge, trainset=trainset)
    
    # Save evolved prompt
    evolved_text = _extract_prompt_text(evolved_judge)
    Path("benchmark/experiments/gepa/evolved_prompt.txt").write_text(evolved_text)
    
    return evolved_judge
```

### 4.4 Evaluate

```python
# benchmark/experiments/gepa/evaluate.py

async def evaluate_evolution():
    """Compare baseline vs evolved judge on test split."""
    examples = _load_test_data()
    
    baseline_preds = [baseline_judge(subtask_description=e.desc, diff=e.diff) for e in examples]
    evolved_preds  = [evolved_judge(subtask_description=e.desc, diff=e.diff) for e in examples]
    
    baseline_f1 = _compute_f1(baseline_preds, examples)
    evolved_f1 = _compute_f1(evolved_preds, examples)
    
    _write_comparison_report(baseline_f1, evolved_f1)
```

## 5. 集成回主 Maestro（仅当 evolved 显示改进）

**改进判定**：evolved_f1 - baseline_f1 ≥ **0.03**（3 个百分点）。

满足判定时：

1. 将 `evolved_prompt.txt` 内容**作为新的 Judge system prompt** 注册到 `src/maestro/verifier/prompts.py`
2. 添加 CLI flag `--evolved-judge`（默认 False）
3. 重跑 E6 实验：full config + evolved judge
4. 在 `REPORT.md` 记录 evolved judge 对最终 resolve rate 的边际贡献

**不满足改进判定时**：

1. 不集成回主 Maestro
2. 在 `REPORT.md` 写 negative finding 段（见 §8 Fallback）

## 6. 预算

| 阶段 | 成本估计 |
|---|---|
| Prepare data（生成 60-80 patches） | $2-3 |
| GEPA 演化（5 iter × 4 pop × ~60 eval ≈ 1200 calls） | $2-3 |
| Test set evaluation | $1 |
| 集成后 E6 重跑（仅当集成） | $2 |
| **Total** | **$7-9** |

**硬上限**：$8。超过立即停止。evolve.py 里用 `LLMClient.get_cost_report()` 每 iter 打印累计花费，超 $7 自动 abort。

## 7. 时间线（Week 6 sub-schedule）

仅在 Day 3 已完成 E1-E6 ablation 才启动：

| Day | 任务 |
|---|---|
| 5.10 Day 1 | E1-E3 + 为 GEPA 准备 train/test split |
| 5.11 Day 2 | E4-E5 |
| 5.12 Day 3 | E6（full config）|
| 5.13 Day 4 | ✅ GATE: 评估条件满足后启动 → GEPA data prep + evolve |
| 5.14 Day 5 | GEPA test evaluation + 决定是否集成 |
| 5.15 Day 6 | 如集成：重跑 E6 with evolved judge；否则：写 negative finding 段 |
| 5.16 Day 7 | Report 写作 + 打磨 |

## 8. Fallback：负结果处理

如果 evolved judge 没有显著改进（Δ F1 < 0.03），在 `REPORT.md` 添加一段如下（示例文本）：

> ### 6.3 Prompt Evolution: Negative Result
>
> We applied DSPy + GEPA to automatically optimize the Judge prompt, using a labeled dataset of 60 (task, patch, ground_truth) triples from the train split. After 5 evolution iterations with population size 4, the best evolved prompt achieved F1=X on the 10-task test split, compared to F1=Y for the hand-crafted baseline (Δ = Z).
>
> The marginal improvement suggests that for this task distribution and the multi-sample (K=3) aggregation already in place, the hand-crafted judge prompt captures most of the achievable reliability. Prompt optimization may yield larger gains on (a) larger datasets that afford richer evolutionary signal, or (b) Judge prompts for tasks without strong supervised signal where the baseline prompt may be more arbitrarily chosen.

**这是加分，不是减分**——面试官看到你做了实验、得到负结果、诚实汇报，比吹牛强得多。

## 9. 面试 talking points（视 GEPA 结果调整）

### 如果 GEPA 有显著改进

> "我在 Maestro 里集成了 DSPy + GEPA（Genetic-Pareto Prompt Evolution）做 Judge prompt 的自动优化。用 benchmark 的 train split 构造 60 个标注样本，演化 5 代，evolved judge 在 test 上 F1 从 X 提升到 Y。这个改进来源是 GEPA 的反思性变异——它能看着 judge 判错的 case 自动提出 prompt 修改建议，有点像自动化的 prompt engineering。最终我在完整 benchmark 上对比了 hand-crafted judge 和 evolved judge，evolved judge 让 Maestro 整体 resolve rate 再提升 Z 个百分点。"

### 如果 GEPA 没有改进

> "我试过用 DSPy + GEPA 自动优化 Judge prompt，构造了 60 个标注样本做演化。结果演化后的 prompt 对比 hand-crafted 只差 2pp 以内，不显著。我的解释是：当你已经用了 K=3 多采样 + 分歧检测，单次 judge 质量的边际空间已经被压缩了——prompt 优化的收益被 aggregation 层吃掉了。这其实挺有意思，说明架构层的 trick（多采样）和 prompt 层的 trick（自动优化）会互相替代。下一步我会在更大的 benchmark 上验证这个观察。"

两种情况都加分。**这是本模块的隐藏价值——把风险转化成叙事**。

## 10. 禁止事项

- ❌ 不得同时演化 Planner / Sub-agent / Judge 三个 prompt
- ❌ 不得超过 $8 预算
- ❌ 不得在 E1-E6 ablation 未完成时启动
- ❌ 不得在 evolve 未完成时修改主 `src/maestro/verifier/llm_judge.py`
- ❌ evolved prompt 作为 **附加选项**（CLI flag），不是替换——baseline judge 必须保留可用
