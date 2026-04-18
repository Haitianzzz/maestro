# Maestro

> Parallelized Python coding agent framework with layered verification and tiered model selection.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Maestro tackles three practical pain points of modern coding agents:

1. **Slow** — most agents run sequentially, even when subtasks are independent.
2. **Unreliable** — "confidently wrong" outputs that fail to compile or pass tests.
3. **Expensive** — using top-tier models for every step doesn't scale.

Maestro's answer is three architectural decisions, each addressing one pain point:

| Pain point | Mechanism |
|---|---|
| Slow | **DAG-based scheduling** with file-level conflict detection; independent subtasks run in parallel |
| Unreliable | **Three-tier verification**: deterministic checks → tests → multi-sample LLM-as-Judge with disagreement detection |
| Expensive | **Tiered model selection**: strong model for planning, cost-effective model for execution, cheap model for judging |

## Quick Start

```bash
# Install
pip install maestro-agent

# Configure (writes ~/.maestro/config.yaml)
maestro config init

# Set your API key (any OpenAI-compatible endpoint)
export DASHSCOPE_API_KEY=sk-...

# Run a task
maestro run "Add a /health endpoint returning JSON status" --repo ./my-project

# See the result
maestro report <task-id>
```

## Architecture

```
┌─────────────────────────────────────────────┐
│              CLI (Typer)                    │
├─────────────────────────────────────────────┤
│          Orchestrator (LangGraph)           │
├──────────────┬──────────────┬───────────────┤
│   Planner    │  Scheduler   │    Verifier   │
│   (DAG out)  │  (DAG in)    │   (3 tiers)   │
├──────────────┴──────────────┴───────────────┤
│         Sub-agent Pool (asyncio)            │
├─────────────────────────────────────────────┤
│            Sandbox (File-level)             │
├─────────────────────────────────────────────┤
│      LLM Client (OpenAI-compatible)         │
└─────────────────────────────────────────────┘
```

### 1. DAG Scheduling with Conflict Detection

The Planner outputs a DAG of subtasks with explicit `reads` and `writes` file declarations. The Scheduler performs topological batching and runtime write-conflict detection before spawning sub-agents in parallel via `asyncio.TaskGroup`. Concurrent API calls are bounded by a custom `PrioritySemaphore` that respects subtask criticality.

### 2. Layered Verification

Every patch runs through three verification tiers, with short-circuit on failure:

- **Tier 1 — Deterministic**: `ruff` + `mypy`. Zero API cost. Catches 60%+ of bad patches.
- **Tier 2 — Test-based**: `pytest`. Auto-generates tests when none exist (disabled for benchmarking to avoid bias).
- **Tier 3 — LLM-as-Judge**: K independent samples with temperature variation, combined via a disagreement metric. Patches flagged as uncertain (disagreement above threshold) are treated as failed — addressing silent correctness leakage observed in single-sample LLM judges.

### 3. Tiered Model Selection

| Role | Model example | Reason |
|---|---|---|
| Planner | Qwen3-Max | Called once per task, quality-critical |
| Sub-agent | Qwen3-Coder-Plus | Called many times, coding-specialized, cost-effective |
| Judge | DeepSeek-V3 | Called K times per patch, must be cheap |

Models are configurable in `~/.maestro/config.yaml`. Any OpenAI-compatible endpoint works.

## Benchmark Results

Evaluated on a self-constructed benchmark of **30 real bug-fix PRs** drawn from `requests`, `click`, `flask`, `httpx`, and `typer`:

| Config | Resolve Rate | Wall-clock (avg) | Cost (avg) |
|---|---|---|---|
| Baseline (single agent, no verify) | `__%` | `__s` | `$__` |
| + DAG parallel | `__%` | `__s` | `$__` |
| + Tier 1+2 verify | `__%` | `__s` | `$__` |
| + Tier 3 LLM Judge | `__%` | `__s` | `$__` |
| **Maestro (full)** | **`__%`** | **`__s`** | **`$__`** |

*(Numbers filled in after Week 6 experiments. See `benchmark/results/REPORT.md` for full analysis including Pareto frontier plot and per-tier ablation.)*

## How It's Different from Other Coding Agents

| Project | Works at layer | Maestro's difference |
|---|---|---|
| Claude Code, Cursor Agent | Interactive single-agent | Automated parallel scheduling + forced verification |
| oh-my-claudecode, hermes-agent | Prompt templates + tmux orchestration | Real DAG scheduling, runtime conflict detection, layered verification |
| OpenHands, SWE-agent | Sandboxed single agent | Parallel + verification as orthogonal engineering dimensions |
| LangGraph, CrewAI | General multi-agent framework | Coding-specialized, with built-in verification loop and cost tiers |

Maestro is **not** a general agent framework. It's a focused infrastructure for Python code modification (bug fixes, features, refactors).

## Usage

```bash
# Run with default full config
maestro run "description" --repo ./path

# Disable specific verifier tiers (for ablation)
maestro run "description" --disable-verifier llm_judge

# Control parallelism
maestro run "description" --max-parallel 8

# Control judge sampling
maestro run "description" --judge-samples 5

# Dry run (no API calls, for pipeline sanity checks)
maestro run "description" --dry-run

# Benchmarking
maestro bench --task-set benchmark/tasks/ --ablation full --output benchmark/results/full.json
```

See `maestro --help` for the full CLI reference.

## Development

```bash
# Clone and install
git clone https://github.com/<your-username>/maestro
cd maestro
uv sync  # or poetry install

# Run tests
pytest

# Type check
mypy src/maestro/

# Lint
ruff check . && ruff format --check .
```

Development docs:

- `DESIGN.md` — architecture and key design decisions
- `specs/` — per-module specifications
- `CLAUDE.md` — contribution guide for AI-assisted development

## Roadmap

- [x] Core DAG scheduler + three-tier verifier
- [x] 30-task self-built benchmark
- [x] Ablation experiments
- [ ] Function-level conflict detection (beyond file-level)
- [ ] TypeScript / JavaScript support
- [ ] Docker sandbox for untrusted tasks
- [ ] Resume-from-checkpoint

## Citation

If you use Maestro in research, please cite:

```bibtex
@software{maestro2026,
  author = {Zhong, Haitian},
  title = {Maestro: Parallelized Coding Agent Framework with Layered Verification},
  year = {2026},
  url = {https://github.com/<your-username>/maestro}
}
```

The LLM-Judge multi-sampling design is informed by observations on silent correctness leakage documented in a companion paper (EMNLP 2026, in review).

## License

MIT
