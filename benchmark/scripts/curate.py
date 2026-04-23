#!/usr/bin/env python3
"""Curate benchmark tasks from a human-picked PR shortlist.

Reads ``pr_shortlist.json`` (schema documented in the sibling README).
For each entry:

1. Parse ``<owner>/<repo>#<N>`` from ``pr_url``.
2. Shallow-clone (``--depth 50``) the repo into a reusable cache dir.
3. Fetch ``pull/<N>/head`` and resolve ``(head_sha, parent_sha, title)``.
4. ``git archive`` the parent tree to ``<task_id>/before/`` and the head
   tree to ``<task_id>/after/``.
5. Write ``task.json`` matching :class:`~maestro.benchmark.models.BenchmarkTask`.
6. Set up ``uv venv`` + install into ``.venv-before`` / ``.venv-after`` and
   run the declared ``failing_tests`` in each. Before must go RED, after
   must go GREEN — otherwise the task is flagged BROKEN in the manifest.

Any single failure is caught: it is logged, recorded as BROKEN in the
manifest, and the loop moves on. The script exits 0 if at least 5 tasks
end OK, else 1.

Prerequisites (checked at runtime):

* ``git`` on PATH
* ``uv`` on PATH
* network access (for ``git clone`` and ``uv pip install``)

PR selection is a separate, human, curation pass — this script only
deterministically materialises what the curator has already chosen.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

# Import maestro.benchmark.models without requiring the package to be
# installed — we're shipping inside the same repo.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from maestro.benchmark.models import BenchmarkTask  # noqa: E402

# ---------------------------------------------------------------------------
# Input / output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ShortlistEntry:
    task_id: str
    pr_url: str
    failing_tests: list[str]
    expected_modified_files: list[str]
    difficulty: str = "medium"
    description_override: str | None = None
    files_hint: list[str] = field(default_factory=list)


@dataclass
class CurateResult:
    task_id: str
    pr_url: str
    difficulty: str
    status: str  # "OK" | "BROKEN"
    reason: str
    expected_modified_files: list[str]
    changed_lines: int | None
    failing_tests: list[str]
    description: str


# ---------------------------------------------------------------------------
# PR URL parsing
# ---------------------------------------------------------------------------


_PR_URL_RE = re.compile(r"github\.com/([^/]+)/([^/]+?)(?:\.git)?/pull/(\d+)")


def parse_pr_url(url: str) -> tuple[str, str, int]:
    match = _PR_URL_RE.search(url)
    if not match:
        raise ValueError(f"not a github pull URL: {url}")
    return match.group(1), match.group(2), int(match.group(3))


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=check,
        timeout=timeout,
    )


def _require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"required tool `{name}` not found on PATH")


# ---------------------------------------------------------------------------
# Git operations
# ---------------------------------------------------------------------------


def clone_shallow(owner: str, repo: str, workdir: Path) -> Path:
    """Shallow-clone ``owner/repo`` into ``workdir``, idempotent."""
    dest = workdir / f"{owner}__{repo}"
    if dest.exists():
        return dest
    url = f"https://github.com/{owner}/{repo}.git"
    _run(["git", "clone", "--depth", "50", url, str(dest)], timeout=300)
    return dest


def fetch_pr(repo_dir: Path, pr_number: int) -> None:
    """Fetch the PR's head into FETCH_HEAD."""
    _run(
        ["git", "fetch", "--depth=50", "origin", f"pull/{pr_number}/head"],
        cwd=repo_dir,
        timeout=180,
    )


def resolve_commits(repo_dir: Path) -> tuple[str, str, str]:
    """Return ``(head_sha, parent_sha, commit_title)``.

    The PR-head commit is the "after" state; its parent is "before". Titles
    come from ``git log -1 --format=%s`` — fine for PR-as-single-commit
    fixes, which is what the shortlist targets. A ``description_override``
    in the shortlist is preferred when it's provided.
    """
    head = _run(["git", "rev-parse", "FETCH_HEAD"], cwd=repo_dir).stdout.strip()
    parent = _run(["git", "rev-parse", f"{head}^"], cwd=repo_dir).stdout.strip()
    title = _run(["git", "log", "-1", "--format=%s", head], cwd=repo_dir).stdout.strip()
    return head, parent, title


def archive_commit(repo_dir: Path, sha: str, dest: Path) -> None:
    """Extract the tree at ``sha`` into ``dest`` via ``git archive | tar``."""
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.Popen(
        ["git", "archive", sha, "--format=tar"],
        cwd=repo_dir,
        stdout=subprocess.PIPE,
    )
    untar = subprocess.Popen(
        ["tar", "-x", "-C", str(dest)],
        stdin=archive.stdout,
    )
    assert archive.stdout is not None
    archive.stdout.close()
    untar_rc = untar.wait()
    archive_rc = archive.wait()
    if archive_rc != 0 or untar_rc != 0:
        raise RuntimeError(
            f"git archive | tar failed (archive_rc={archive_rc}, untar_rc={untar_rc}) for {sha}"
        )


def count_changed_lines(repo_dir: Path, parent: str, head: str) -> int:
    proc = _run(["git", "diff", "--numstat", parent, head], cwd=repo_dir)
    total = 0
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            with contextlib.suppress(ValueError):
                total += int(parts[0]) + int(parts[1])
    return total


# ---------------------------------------------------------------------------
# Natural-language prompt synthesis
# ---------------------------------------------------------------------------


_PREFIX_RE = re.compile(r"^(fix(es|ed)?|bug(fix)?|patch)[\s:!\-]+", re.IGNORECASE)


def make_prompt(title: str, override: str | None) -> str:
    """Strip ``Fix:`` / ``Bug:`` prefixes so the prompt reads as user intent.

    The goal is that the planner gets a description of the bug's symptom
    without the "PR title" framing. ``description_override`` from the
    shortlist always wins — use that when the raw title would leak which
    file or function to touch.
    """
    if override:
        return override.strip()
    cleaned = _PREFIX_RE.sub("", title).strip()
    if cleaned and cleaned[0].isupper() and len(cleaned) > 1 and cleaned[1].islower():
        cleaned = cleaned[0].lower() + cleaned[1:]
    cleaned = cleaned.rstrip(".")
    if not cleaned:
        cleaned = title
    return f"{cleaned}. Please fix it."


# ---------------------------------------------------------------------------
# task.json
# ---------------------------------------------------------------------------


def write_task_json(
    task_dir: Path,
    entry: ShortlistEntry,
    owner: str,
    repo: str,
    title: str,
    head_sha: str,
) -> BenchmarkTask:
    task = BenchmarkTask(
        task_id=entry.task_id,
        repo=f"{owner}/{repo}",
        pr_url=entry.pr_url,
        description=title,
        natural_language_prompt=make_prompt(title, entry.description_override),
        failing_tests=entry.failing_tests,
        expected_modified_files=entry.expected_modified_files,
        files_hint=entry.files_hint or entry.expected_modified_files,
        difficulty=entry.difficulty,
        source_commit=head_sha,
    )
    (task_dir / "task.json").write_text(task.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return task


# ---------------------------------------------------------------------------
# Per-task venv + pytest
# ---------------------------------------------------------------------------


def _create_venv_and_install(repo_dir: Path, venv_path: Path) -> tuple[bool, str]:
    """Create a uv venv, install repo deps + pytest into it."""
    if venv_path.exists():
        shutil.rmtree(venv_path)
    try:
        _run(["uv", "venv", str(venv_path)], timeout=120)
    except subprocess.CalledProcessError as exc:
        return False, f"uv venv failed: {(exc.stderr or '')[-300:]}"

    python = venv_path / "bin" / "python"
    pip_base = [
        "uv",
        "pip",
        "install",
        "--python",
        str(python),
    ]

    installable = (repo_dir / "pyproject.toml").exists() or (repo_dir / "setup.py").exists()
    has_req = (repo_dir / "requirements.txt").exists()

    try:
        if installable:
            _run([*pip_base, "-e", str(repo_dir)], timeout=600)
        if has_req:
            _run([*pip_base, "-r", str(repo_dir / "requirements.txt")], timeout=600)
        # pytest is required even if deps already pulled it in — belt and
        # braces, and a no-op when already present.
        _run([*pip_base, "pytest"], timeout=180)
    except subprocess.CalledProcessError as exc:
        return False, f"install failed: {((exc.stderr or exc.stdout) or '')[-300:]}"
    except subprocess.TimeoutExpired:
        return False, "install timed out"

    return True, "installed"


def _run_pytest_in_venv(
    repo_dir: Path, venv_path: Path, failing_tests: list[str], *, timeout: float = 180.0
) -> tuple[int, str]:
    pytest_bin = venv_path / "bin" / "pytest"
    if not pytest_bin.exists():
        return 127, f"pytest not found in venv: {pytest_bin}"
    cmd = [str(pytest_bin), "--tb=no", "-q", *failing_tests]
    try:
        proc = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return -1, "pytest timed out"
    return proc.returncode, (proc.stdout + proc.stderr)[-1200:]


def validate_task(task_dir: Path, failing_tests: list[str]) -> tuple[bool, str]:
    """Verify: before/ red, after/ green.

    Mirrors the contract of :func:`maestro.benchmark.build_tasks.validate_task_dir`,
    with the extra layer of per-directory uv venvs so the target repo's
    own dependencies are installed and pytest runs under them. Returns
    ``(ok, message)`` where the message is suitable for the manifest.
    """
    before = task_dir / "before"
    after = task_dir / "after"

    ok, msg = _create_venv_and_install(before, task_dir / ".venv-before")
    if not ok:
        return False, f"before env: {msg}"
    rc_before, out_before = _run_pytest_in_venv(before, task_dir / ".venv-before", failing_tests)
    if rc_before == 0:
        return False, (
            f"before/ pytest unexpectedly passed (rc=0); want RED.\ntail: {out_before[-400:]}"
        )

    ok, msg = _create_venv_and_install(after, task_dir / ".venv-after")
    if not ok:
        return False, f"after env: {msg}"
    rc_after, out_after = _run_pytest_in_venv(after, task_dir / ".venv-after", failing_tests)
    if rc_after != 0:
        return False, (
            f"after/ pytest did not pass (rc={rc_after}); want GREEN.\ntail: {out_after[-400:]}"
        )

    return True, f"before rc={rc_before}, after rc=0"


# ---------------------------------------------------------------------------
# Per-entry driver
# ---------------------------------------------------------------------------


def process_entry(entry: ShortlistEntry, tasks_dir: Path, clone_cache: Path) -> CurateResult:
    def _broken(reason: str, *, title: str = "") -> CurateResult:
        return CurateResult(
            task_id=entry.task_id,
            pr_url=entry.pr_url,
            difficulty=entry.difficulty,
            status="BROKEN",
            reason=reason,
            expected_modified_files=entry.expected_modified_files,
            changed_lines=None,
            failing_tests=entry.failing_tests,
            description=title,
        )

    try:
        owner, repo, pr_number = parse_pr_url(entry.pr_url)
    except ValueError as exc:
        return _broken(str(exc))

    task_dir = tasks_dir / entry.task_id
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True)

    try:
        repo_dir = clone_shallow(owner, repo, clone_cache)
        fetch_pr(repo_dir, pr_number)
        head_sha, parent_sha, title = resolve_commits(repo_dir)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        tail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            tail = (exc.stderr or exc.stdout or "")[-300:]
        return _broken(f"git clone/fetch failed: {tail or exc}")

    try:
        archive_commit(repo_dir, parent_sha, task_dir / "before")
        archive_commit(repo_dir, head_sha, task_dir / "after")
    except Exception as exc:
        return _broken(f"archive failed: {exc}", title=title)

    try:
        changed = count_changed_lines(repo_dir, parent_sha, head_sha)
    except subprocess.CalledProcessError:
        changed = 0

    write_task_json(task_dir, entry, owner, repo, title, head_sha)

    ok, msg = validate_task(task_dir, entry.failing_tests)
    return CurateResult(
        task_id=entry.task_id,
        pr_url=entry.pr_url,
        difficulty=entry.difficulty,
        status="OK" if ok else "BROKEN",
        reason=msg,
        expected_modified_files=entry.expected_modified_files,
        changed_lines=changed,
        failing_tests=entry.failing_tests,
        description=title,
    )


# ---------------------------------------------------------------------------
# Manifest rendering
# ---------------------------------------------------------------------------


def render_manifest(results: list[CurateResult]) -> str:
    generated = datetime.now(UTC).isoformat(timespec="seconds")
    lines = [
        "# Benchmark task manifest",
        "",
        f"Generated: {generated}",
        "",
        "| status | task_id | difficulty | Δ lines | failing_tests | PR | description |",
        "|---|---|---|---:|---|---|---|",
    ]
    for r in results:
        tests = ", ".join(f"`{t}`" for t in r.failing_tests) or "-"
        delta = "?" if r.changed_lines is None else str(r.changed_lines)
        desc = (r.description or "-").replace("|", r"\|")
        lines.append(
            f"| {r.status} | `{r.task_id}` | {r.difficulty} | {delta} | "
            f"{tests} | [{r.pr_url}]({r.pr_url}) | {desc} |"
        )

    broken = [r for r in results if r.status == "BROKEN"]
    if broken:
        lines += ["", "## Broken tasks — diagnostic tails", ""]
        for r in broken:
            lines.append(f"### `{r.task_id}`")
            lines.append(f"- PR: {r.pr_url}")
            reason = r.reason.replace("`", "'")
            lines.append(f"- Reason:\n\n```\n{reason[:800]}\n```")
            lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Shortlist loader + main
# ---------------------------------------------------------------------------


def load_shortlist(path: Path) -> list[ShortlistEntry]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"shortlist root must be a JSON array, got {type(raw).__name__}")
    out: list[ShortlistEntry] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"shortlist[{i}] must be an object")
        out.append(
            ShortlistEntry(
                task_id=item["task_id"],
                pr_url=item["pr_url"],
                failing_tests=list(item["failing_tests"]),
                expected_modified_files=list(item.get("expected_modified_files", [])),
                difficulty=item.get("difficulty", "medium"),
                description_override=item.get("description_override"),
                files_hint=list(item.get("files_hint", [])),
            )
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shortlist",
        type=Path,
        default=Path(__file__).parent / "pr_shortlist.json",
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=_PROJECT_ROOT / "benchmark" / "tasks",
    )
    parser.add_argument(
        "--clone-cache",
        type=Path,
        default=None,
        help="Directory to cache git clones across runs. Default: ephemeral tempdir.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_PROJECT_ROOT / "benchmark" / "tasks" / "MANIFEST.md",
    )
    args = parser.parse_args()

    try:
        _require_tool("git")
        _require_tool("uv")
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not args.shortlist.exists():
        print(f"shortlist not found: {args.shortlist}", file=sys.stderr)
        return 1

    entries = load_shortlist(args.shortlist)
    args.tasks_dir.mkdir(parents=True, exist_ok=True)

    tmp_holder: tempfile.TemporaryDirectory[str] | None = None
    if args.clone_cache is None:
        tmp_holder = tempfile.TemporaryDirectory(prefix="maestro-curate-")
        clone_cache = Path(tmp_holder.name)
    else:
        args.clone_cache.mkdir(parents=True, exist_ok=True)
        clone_cache = args.clone_cache

    results: list[CurateResult] = []
    try:
        for entry in entries:
            print(f"[curate] {entry.task_id}: {entry.pr_url}")
            result = process_entry(entry, args.tasks_dir, clone_cache)
            print(f"[curate]   -> {result.status}: {result.reason[:200]}")
            results.append(result)
    finally:
        if tmp_holder is not None:
            tmp_holder.cleanup()

    manifest = render_manifest(results)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(manifest, encoding="utf-8")

    ok_tasks = [r for r in results if r.status == "OK"]
    broken_tasks = [r for r in results if r.status == "BROKEN"]
    deltas = [r.changed_lines for r in ok_tasks if r.changed_lines is not None]
    avg_delta = (sum(deltas) / len(deltas)) if deltas else 0.0
    diff_dist: dict[str, int] = {}
    for r in ok_tasks:
        diff_dist[r.difficulty] = diff_dist.get(r.difficulty, 0) + 1

    print()
    print(f"Summary: {len(ok_tasks)} OK, {len(broken_tasks)} broken")
    print(f"Avg Δ lines (OK): {avg_delta:.1f}")
    print(f"Difficulty distribution (OK): {diff_dist}")
    print(f"Manifest: {args.manifest}")

    return 0 if len(ok_tasks) >= 5 else 1


if __name__ == "__main__":
    sys.exit(main())
