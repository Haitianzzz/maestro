# Spec 07 — Sandbox / Workspace Manager

> **位置**：`src/maestro/sandbox/`
> **依赖**：`models.py`
> **被依赖**：Scheduler、Sub-agent、Verifier

## 1. 职责

管理 sub-agent 执行环境的文件系统隔离，以及最终的 diff 合并。

**决策回顾**（见 DESIGN.md §3.7）：用**文件系统级**隔离（临时目录复制），不用 Docker。

## 2. 抽象

### 2.1 WorkspaceManager（顶层）

```python
# src/maestro/sandbox/workspace.py

class WorkspaceManager:
    """Manages main workspace and isolated sub-agent workspaces."""

    def __init__(self, repo_path: Path, task_id: str):
        self._source_repo = repo_path
        self._task_id = task_id
        self._root = Path(tempfile.mkdtemp(prefix=f"maestro-{task_id}-"))
        self._main_workspace = self._root / "main"
        self._isolated_workspaces: dict[str, IsolatedWorkspace] = {}

        # Copy source repo to main workspace
        self._initialize_main()

    async def create_isolated(self, subtask: SubTask) -> "IsolatedWorkspace":
        """Create an isolated workspace for a subtask, copied from current main."""
        ...

    async def merge_patches(self, results: list[SubAgentResult]) -> MergeReport:
        """Apply successful patches to main workspace. Returns merge report."""
        ...

    def get_final_diff(self) -> str:
        """Diff between source_repo and main_workspace."""
        ...

    def cleanup(self) -> None:
        """Remove all workspaces."""
        ...
```

### 2.2 IsolatedWorkspace

```python
class IsolatedWorkspace:
    """One sub-agent's isolated view of the repo."""

    def __init__(self, path: Path, subtask: SubTask):
        self._path = path
        self._subtask = subtask

    @property
    def path(self) -> Path:
        return self._path

    def read_file(self, relative_path: str) -> str:
        """Read a file. Enforces read permission based on subtask.reads + subtask.writes."""
        ...

    def apply_diff(self, diff_text: str) -> tuple[bool, str | None]:
        """Apply a unified diff. Returns (success, error_message)."""
        ...

    def get_diff_from_base(self) -> str:
        """Generate diff vs the state when this workspace was created."""
        ...
```

## 3. 初始化与隔离

### 3.1 主 workspace 的创建

```python
def _initialize_main(self):
    # Copy source repo to main workspace (fast path: cp -r or shutil.copytree)
    shutil.copytree(
        self._source_repo,
        self._main_workspace,
        ignore=shutil.ignore_patterns(
            ".git", ".venv", "__pycache__", "*.pyc", "node_modules", ".tox", "dist", "build",
        ),
    )

    # Take a snapshot for later diff
    self._base_snapshot = _take_snapshot(self._main_workspace)
```

### 3.2 隔离 workspace 的创建

```python
async def create_isolated(self, subtask: SubTask) -> IsolatedWorkspace:
    iso_path = self._root / "iso" / subtask.subtask_id
    iso_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy from current main state (not source!)
    # This ensures sub-agents in batch N see merged patches from batches 0..N-1
    shutil.copytree(self._main_workspace, iso_path)

    ws = IsolatedWorkspace(iso_path, subtask)
    self._isolated_workspaces[subtask.subtask_id] = ws
    return ws
```

**关键**：isolated workspace 复制自 `main_workspace`（当前状态），不是 source_repo。这样 batch N 的 sub-agent 能看到 batch 0..N-1 的成果。

## 4. Read Permission

```python
def read_file(self, relative_path: str) -> str:
    allowed = set(self._subtask.reads) | set(self._subtask.writes)
    if relative_path not in allowed:
        raise PermissionError(
            f"Sub-agent for {self._subtask.subtask_id} attempted to read "
            f"'{relative_path}' which is not in reads or writes."
        )

    abs_path = self._path / relative_path
    if not abs_path.exists():
        raise FileNotFoundError(f"{relative_path}")

    if not _is_safe_file(abs_path):
        raise PermissionError(f"{relative_path} is not a safe file type")

    return abs_path.read_text(encoding="utf-8")
```

`_is_safe_file`: 只允许 `.py, .md, .txt, .cfg, .toml, .ini, .yml, .yaml, .json` 扩展名，且文件大小 < 500KB。

## 5. Diff 应用

### 5.1 Unified diff parser

使用标准 library 中的 `unidiff`（需要添加依赖）或自己实现轻量 parser。

```python
def apply_diff(self, diff_text: str) -> tuple[bool, str | None]:
    try:
        patch_set = PatchSet.from_string(diff_text)
    except Exception as e:
        return False, f"Failed to parse diff: {e}"

    # Validate: all target files are in subtask.writes
    writes_set = set(self._subtask.writes)
    for patched_file in patch_set:
        target = patched_file.target_file.lstrip("a/").lstrip("b/")
        if target not in writes_set:
            return False, f"Diff targets '{target}' which is not in writes {writes_set}"

    # Apply each file
    for patched_file in patch_set:
        target = patched_file.target_file.lstrip("b/")
        abs_target = self._path / target

        if patched_file.is_added_file:
            # Create new file
            abs_target.parent.mkdir(parents=True, exist_ok=True)
            content = _reconstruct_from_hunks(patched_file)
            abs_target.write_text(content, encoding="utf-8")
        elif patched_file.is_removed_file:
            abs_target.unlink(missing_ok=True)
        else:
            # Modify existing file
            if not abs_target.exists():
                return False, f"Target file '{target}' does not exist"
            original = abs_target.read_text(encoding="utf-8")
            modified = _apply_hunks(original, patched_file)
            abs_target.write_text(modified, encoding="utf-8")

    return True, None
```

### 5.2 冲突处理

Sub-agent 生成的 diff 可能 apply 失败（hunk 不匹配）。此时：
- 返回 `(False, "Hunk mismatch at file X line Y")`
- Sub-agent 的 result 被标记为 `status="rejected"`
- Scheduler 触发重试（带 prior_failure 反馈，告诉 LLM 上次 diff apply 失败）

## 6. Patch 合并到 main

```python
async def merge_patches(self, results: list[SubAgentResult]) -> MergeReport:
    merged: list[str] = []
    conflicts: list[MergeConflict] = []

    for result in results:
        if result.status != "success":
            continue

        iso_ws = self._isolated_workspaces[result.subtask_id]

        # Apply iso workspace's delta to main
        try:
            self._apply_workspace_delta(iso_ws, self._main_workspace, result.modified_files)
            merged.append(result.subtask_id)
        except MergeConflictError as e:
            conflicts.append(MergeConflict(
                subtask_id=result.subtask_id,
                file=e.file,
                reason=e.reason,
            ))

    return MergeReport(merged=merged, conflicts=conflicts)


def _apply_workspace_delta(self, iso_ws, main_ws, modified_files):
    """Copy modified files from iso to main, detecting conflicts."""
    for rel_path in modified_files:
        iso_file = iso_ws.path / rel_path
        main_file = main_ws / rel_path

        if not iso_file.exists():
            # File was deleted in iso → delete in main
            main_file.unlink(missing_ok=True)
            continue

        iso_content = iso_file.read_text(encoding="utf-8")

        if main_file.exists():
            main_content = main_file.read_text(encoding="utf-8")
            # Check conflict: did someone else modify this file between iso creation and now?
            if _file_changed_since_snapshot(main_content, rel_path):
                raise MergeConflictError(file=rel_path, reason="File modified by another batch")

        main_file.parent.mkdir(parents=True, exist_ok=True)
        main_file.write_text(iso_content, encoding="utf-8")
```

**冲突处理策略**：
- 同 batch 内，Scheduler 已经通过 `detect_write_conflicts` 避免同文件冲突
- 跨 batch，后一 batch 基于前一 batch 合并后的 main 创建 iso，理论上不会冲突
- 实际冲突发生时（边界情况），标记为 `conflicts`，上层决定如何处理（最简单策略：reject 该 subtask，记录在 BatchResult）

## 7. 最终 diff

```python
def get_final_diff(self) -> str:
    """Produce unified diff between source_repo and main_workspace."""
    # Use `git diff --no-index` for consistent format
    result = subprocess.run(
        ["git", "diff", "--no-index", "--", str(self._source_repo), str(self._main_workspace)],
        capture_output=True,
        text=True,
    )
    # git returns 1 when diff is non-empty, not an error
    return result.stdout
```

## 8. Cleanup

```python
def cleanup(self) -> None:
    """Remove the root tempdir."""
    if self._root.exists():
        shutil.rmtree(self._root, ignore_errors=True)
```

`WorkspaceManager` 实现 context manager：

```python
def __enter__(self):
    return self

def __exit__(self, *exc):
    self.cleanup()
```

Orchestrator 使用时：

```python
with WorkspaceManager(repo_path, task_id) as ws_mgr:
    ... # all task execution
# auto cleanup
```

**benchmark 模式下可保留 workspace 供调试**：`WorkspaceManager(..., keep_on_exit=True)`

## 9. 测试要求

`tests/unit/test_workspace.py`：
- 创建 main workspace 时 ignore patterns 生效
- 创建 iso workspace 独立于 main
- read_file 权限检查
- apply_diff 成功路径
- apply_diff 违反 writes 约束
- apply_diff hunk 不匹配
- merge_patches 正常
- merge_patches 冲突
- cleanup 删除所有临时目录

`tests/integration/test_workspace_e2e.py`：
- 用真实小 repo，完整走一遍 create → iso → diff → merge → cleanup

## 10. 面试 talking points

1. **文件系统级隔离而非 Docker**：trade-off 分析——Docker 启动开销秒级会拖慢并行；Python 代码修改不需要进程隔离。工程上选轻量方案
2. **Read permission 运行时 enforce**：sub-agent 读文件受 reads+writes 约束，不是"相信 LLM"，是 harness 层面的安全约束
3. **Iso workspace 基于 current main 而非 source**：batch N 能看到 batch 0..N-1 的成果，这是顺序 DAG 执行的语义正确保证
4. **Diff apply 前 validate writes**：不合法直接 reject，不会污染 workspace
5. **Context manager 自动 cleanup**：防止临时目录泄漏
