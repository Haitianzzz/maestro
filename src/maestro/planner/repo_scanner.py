"""Repo scanner — assembles LLM-friendly context for the planner (spec 03 §3).

The scanner walks a repo and produces a :class:`RepoContext` containing:

* A text-formatted file tree (bounded at ~500 entries)
* Full contents of small "key" files (``pyproject.toml``, ``README.md``,
  top-level ``__init__.py`` files, plus any ``target_files_hint`` entries)
* AST-extracted signatures (top-level ``class`` / ``def``) for every ``.py``
  file that is not a key file — gives the planner coverage without the cost
  of reading every line

Context budget is driven by ``max_context_tokens``. We estimate 1 token ≈
4 characters (cheap and roughly right for Python source; tiktoken would be
more accurate but adds a heavy dependency to the hot path). If the initial
scan exceeds the budget we progressively shed content in the order laid
out by spec 03 §3.2.

Divergences from spec 03 §3.2
-----------------------------

The spec lists four shedding steps; our implementation applies three. The
two omitted steps are handled differently and don't need their own round:

* **"Drop vendor / third-party"** — vendor and third-party directories
  (``.venv``, ``venv``, ``node_modules``, ``.tox``, ``build``, ``dist``,
  etc.) are already excluded by ``_IGNORE_DIRS`` during the initial walk,
  so by the time we measure the context budget there is nothing vendor-y
  left to shed. Running it as a separate step would be a no-op.
* **"Keep signatures only, not implementations"** — we never include raw
  source implementations in the context in the first place. ``key_files``
  is a fixed allow-list (``pyproject.toml``, ``README.md``,
  ``__init__.py`` files, explicit hints); everything else enters the
  context only as AST-extracted signatures. The equivalent knob in our
  pipeline is "drop signatures entirely", which is step 3 below.

Shedding order actually applied, in sequence, while still over budget:

1. Drop AST signatures for files under ``tests/`` / ``test/`` directories.
2. Drop README content from ``key_files`` (keep ``pyproject.toml`` because
   it reveals the import layout).
3. Drop all remaining AST signatures — tree + key files only.
4. Re-render the file tree capped at depth 2.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from maestro.utils.logging import get_logger

_logger = get_logger("maestro.planner.scanner")

# Directories we never descend into (vendor, VCS, caches).
_IGNORE_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".tox",
        "dist",
        "build",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".maestro",
        ".idea",
        ".vscode",
    }
)

# Tests directory names we shed first when over-budget.
_TEST_DIR_NAMES = frozenset({"tests", "test"})

# Files whose full content is always valuable to the planner if present.
_ALWAYS_KEY_FILES = (
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "README.md",
    "README.rst",
)

# Files the planner reads for signatures (via AST) and also for whole-file
# content if inside a hint list.
_SOURCE_SUFFIX = ".py"

# Rough chars-per-token heuristic; see module docstring.
_CHARS_PER_TOKEN = 4

_MAX_TREE_ENTRIES = 500


@dataclass(frozen=True)
class RepoContext:
    """Structured repo context passed to the planner prompt."""

    file_tree: str
    key_files: dict[str, str] = field(default_factory=dict)
    file_signatures: dict[str, str] = field(default_factory=dict)
    total_tokens_estimated: int = 0


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _context_tokens(tree: str, key_files: dict[str, str], signatures: dict[str, str]) -> int:
    total = _estimate_tokens(tree)
    for content in key_files.values():
        total += _estimate_tokens(content)
    for sig in signatures.values():
        total += _estimate_tokens(sig)
    return total


class RepoScanner:
    """Scan a repo and produce planner-friendly context."""

    def __init__(self, repo_path: Path, max_context_tokens: int = 60_000) -> None:
        if not repo_path.exists() or not repo_path.is_dir():
            raise ValueError(f"Repo path {repo_path} is not a directory")
        if max_context_tokens < 1_000:
            raise ValueError("max_context_tokens must be >= 1000")
        self._repo = repo_path
        self._budget = max_context_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self, target_files_hint: list[str] | None = None) -> RepoContext:
        """Produce a :class:`RepoContext` for the planner prompt."""
        hint = set(target_files_hint or [])

        all_py_files = sorted(self._walk_py_files())
        key_files = self._collect_key_files(hint)
        file_tree = self._render_tree(max_entries=_MAX_TREE_ENTRIES)
        signatures = self._extract_signatures(all_py_files, skip=set(key_files))

        # Progressive shedding if we are over budget.
        total = _context_tokens(file_tree, key_files, signatures)
        if total > self._budget:
            signatures = self._drop_test_signatures(signatures)
            total = _context_tokens(file_tree, key_files, signatures)
        if total > self._budget:
            # Drop README content (still keep pyproject since it reveals the
            # import layout).
            key_files = {k: v for k, v in key_files.items() if "README" not in k}
            total = _context_tokens(file_tree, key_files, signatures)
        if total > self._budget:
            # Give up on signatures entirely — tree + key files only.
            signatures = {}
            total = _context_tokens(file_tree, key_files, signatures)
        if total > self._budget:
            # Collapse the file tree to two levels.
            file_tree = self._render_tree(max_entries=_MAX_TREE_ENTRIES, max_depth=2)
            total = _context_tokens(file_tree, key_files, signatures)

        _logger.info(
            "repo_scanned",
            repo=str(self._repo),
            key_files=len(key_files),
            signatures=len(signatures),
            tokens_estimated=total,
            budget=self._budget,
        )
        return RepoContext(
            file_tree=file_tree,
            key_files=key_files,
            file_signatures=signatures,
            total_tokens_estimated=total,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _walk_py_files(self) -> list[str]:
        out: list[str] = []
        for child in self._walk_entries():
            if child.is_file() and child.suffix == _SOURCE_SUFFIX:
                out.append(str(child.relative_to(self._repo)))
        return out

    def _walk_entries(self) -> list[Path]:
        """Breadth-first walk yielding every non-ignored entry."""
        out: list[Path] = []
        stack: list[Path] = [self._repo]
        while stack:
            here = stack.pop(0)
            try:
                children = sorted(here.iterdir())
            except OSError:
                continue
            for child in children:
                if child.name in _IGNORE_DIRS:
                    continue
                out.append(child)
                if child.is_dir():
                    stack.append(child)
        return out

    def _render_tree(self, *, max_entries: int, max_depth: int | None = None) -> str:
        lines: list[str] = [f"{self._repo.name}/"]
        overflow = False
        count = 0
        for entry in self._walk_entries():
            rel = entry.relative_to(self._repo)
            depth = len(rel.parts)
            if max_depth is not None and depth > max_depth:
                continue
            indent = "  " * depth
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{indent}{rel.parts[-1]}{suffix}")
            count += 1
            if count >= max_entries:
                overflow = True
                break
        if overflow:
            lines.append("  ... (file tree truncated)")
        return "\n".join(lines)

    def _collect_key_files(self, hint: set[str]) -> dict[str, str]:
        key_files: dict[str, str] = {}

        for name in _ALWAYS_KEY_FILES:
            candidate = self._repo / name
            if candidate.exists() and candidate.is_file():
                key_files[name] = self._read_capped(candidate, "README" in name)

        # All top-level ``__init__.py`` under src/ and direct child packages.
        for init_path in self._repo.rglob("__init__.py"):
            if any(part in _IGNORE_DIRS for part in init_path.parts):
                continue
            rel = init_path.relative_to(self._repo).as_posix()
            key_files[rel] = self._read_capped(init_path, readme=False)

        # Explicit target-file hints: include even if they don't yet exist
        # (sub-agent may create them) — just skip missing ones silently.
        for hinted in hint:
            abs_path = self._repo / hinted
            if abs_path.exists() and abs_path.is_file():
                key_files[hinted] = self._read_capped(abs_path, readme=False)

        return key_files

    @staticmethod
    def _read_capped(path: Path, readme: bool) -> str:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        if readme:
            # README 前 100 行 per spec 03 §3.1
            lines = text.splitlines()
            text = "\n".join(lines[:100])
        return text

    def _extract_signatures(self, py_files: list[str], skip: set[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for rel in py_files:
            if rel in skip:
                continue
            abs_path = self._repo / rel
            try:
                source = abs_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            sigs = _extract_top_level_signatures(source)
            if sigs:
                out[rel] = sigs
        return out

    @staticmethod
    def _drop_test_signatures(signatures: dict[str, str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for rel, sig in signatures.items():
            parts = Path(rel).parts
            if any(part in _TEST_DIR_NAMES for part in parts):
                continue
            out[rel] = sig
        return out


def _extract_top_level_signatures(source: str) -> str:
    """Return a newline-joined string of top-level class/function signatures."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    lines: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            lines.append(f"class {node.name}:")
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            args = ast.unparse(node.args)
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            lines.append(f"{prefix} {node.name}({args})")
    return "\n".join(lines)


__all__ = ["RepoContext", "RepoScanner"]
