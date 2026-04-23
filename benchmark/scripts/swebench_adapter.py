#!/usr/bin/env python3
"""SWE-bench Lite → Maestro benchmark tasks (spec 09 §2.5 variant).

Downloads ``princeton-nlp/SWE-bench_Lite`` (test split), filters to a
whitelist of laptop-friendly repos with small patches, diversifies
selection across repos, and materialises each pick as
``benchmark/tasks/<instance_id>/{task.json,before/,after/}``.

For every picked row:

1. Clone ``{repo}`` with ``--filter=blob:none --no-checkout`` into a
   shared cache dir (full commit graph, blobs lazy-fetched). This is
   the SWE-bench way of handling old ``base_commit`` SHAs.
2. ``git archive base_commit | tar -x`` → ``before/``.
3. Copy ``before/`` → ``after/`` and apply ``row['patch']`` +
   ``row['test_patch']`` via ``git apply``.
4. Write ``task.json`` using the fields the dataset already provides
   (``problem_statement`` is user-voice, not a PR title — no prompt
   rewriting needed).
5. Validate with :func:`curate.validate_task` (per-dir uv venv + install +
   pytest-red-before / pytest-green-after). Failures are recorded in
   the manifest with a categorised reason; the loop never raises.

Exit status: 0 iff at least 5 tasks end OK, else 1.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from curate import _require_tool  # noqa: E402

from maestro.benchmark.models import BenchmarkTask  # noqa: E402

# ---------------------------------------------------------------------------
# Selection config
# ---------------------------------------------------------------------------

SMALL_REPOS: set[str] = {
    "psf/requests",
    "pytest-dev/pytest",
    "mwaskom/seaborn",
    "pylint-dev/pylint",
    "sphinx-doc/sphinx",
}

TARGET_COUNT = 8
MAX_PATCH_LINES = 50


# ---------------------------------------------------------------------------
# Per-repo install recipes — SWE-bench's official harness ships one per
# instance; we ship a coarser per-repo fallback that covers the whitelist.
# Each recipe hand-picks a Python version that the PR era compiles against
# plus the dev/test extras + environment variables the build step needs.
#
# Each entry is informed by the failure categories we observed on the first
# run (install_failed / after_not_green due to missing plugins / setuptools
# quirks).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InstallRecipe:
    python_version: str = "3.11"
    extras: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    extra_packages: tuple[str, ...] = ()
    # When True, install the repo with ``--no-build-isolation``. Needed when
    # the repo's ``[build-system] requires`` pins an old setuptools that
    # lacks ``build_editable`` (PEP 660). Combined with ``extra_packages``
    # pre-installing setuptools>=64 in the venv so the build can see it.
    no_build_isolation: bool = False


RECIPES: dict[str, InstallRecipe] = {
    "psf/requests": InstallRecipe(
        # Old (2014-2015) requests uses ``from collections import Mapping``
        # which was removed in Python 3.10. 3.9 is the newest that still has it.
        python_version="3.9",
        extra_packages=("pytest", "pytest-httpbin", "pytest-mock"),
    ),
    "pytest-dev/pytest": InstallRecipe(
        # setuptools-scm wants either a git tag or SETUPTOOLS_SCM_PRETEND_VERSION.
        python_version="3.11",
        extras=("testing",),
        env=(("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYTEST", "7.0.0"),),
    ),
    "mwaskom/seaborn": InstallRecipe(
        # seaborn's [dev] extra pulls pytest + pytest-xdist, but not
        # pytest-asyncio which its conftest references. Install it explicitly.
        python_version="3.11",
        extras=("dev", "stats"),
        extra_packages=("pytest-asyncio",),
    ),
    "pylint-dev/pylint": InstallRecipe(
        # Newer pylint's pyproject pins ``setuptools~=62.6`` which predates
        # PEP-660 editable support. Disable build isolation so the editable
        # install uses the setuptools we pre-install below instead of the
        # build-time one from the isolated env.
        # ``py`` is the legacy package tests/test_self.py imports via
        # ``from py._path.local import LocalPath``.
        python_version="3.11",
        no_build_isolation=True,
        extra_packages=(
            "setuptools>=64",
            "wheel",
            "pytest",
            "pytest-benchmark",
            "pytest-xdist",
            "pytest-timeout",
            "py",
        ),
    ),
    "sphinx-doc/sphinx": InstallRecipe(
        # Older sphinx imports ``pkg_resources`` which recent setuptools (80+)
        # dropped, uses ``jinja2.environmentfilter`` which jinja2 3.1 renamed,
        # and pulls the external ``roman`` package for LaTeX roman numerals.
        python_version="3.9",
        extra_packages=(
            "setuptools<60",
            "jinja2<3.1",
            "roman",
            "pytest",
            "html5lib",
        ),
    ),
}


# ---------------------------------------------------------------------------
# Recipe-driven validator (replaces curate.validate_task for SWE-bench tasks)
# ---------------------------------------------------------------------------


def _run_with_env(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: float | None = None,
    env_extra: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    env = None
    if env_extra:
        env = {**os.environ, **env_extra}
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=check,
        timeout=timeout,
        env=env,
    )


def _setup_venv_with_recipe(
    repo_dir: Path, venv_path: Path, recipe: InstallRecipe
) -> tuple[bool, str]:
    """Create a pinned-Python uv venv and install repo + extras per recipe."""
    if venv_path.exists():
        shutil.rmtree(venv_path)
    try:
        _run_with_env(
            ["uv", "venv", "--python", recipe.python_version, str(venv_path)],
            timeout=300,
        )
    except subprocess.CalledProcessError as exc:
        return False, f"uv venv failed: {(exc.stderr or '')[-400:]}"
    except subprocess.TimeoutExpired:
        return False, "uv venv timed out"

    python = venv_path / "bin" / "python"
    env_extra = dict(recipe.env)
    pip_base = ["uv", "pip", "install", "--python", str(python)]

    # When build isolation is disabled, extras must already be in the venv
    # before the editable install runs (so the build picks up setuptools>=64,
    # etc). So we install extra_packages FIRST in that mode; otherwise we do
    # the repo first (normal order).
    def _install_extras() -> tuple[bool, str]:
        if not recipe.extra_packages:
            return True, ""
        try:
            _run_with_env(
                [*pip_base, *recipe.extra_packages],
                timeout=300,
                env_extra=env_extra,
            )
        except subprocess.CalledProcessError as exc:
            return False, f"extra install failed: {((exc.stderr or exc.stdout) or '')[-400:]}"
        except subprocess.TimeoutExpired:
            return False, "extra install timed out"
        return True, ""

    def _install_repo() -> tuple[bool, str]:
        target_spec = str(repo_dir)
        if recipe.extras:
            target_spec = f"{repo_dir}[{','.join(recipe.extras)}]"
        cmd = [*pip_base, "-e", target_spec]
        if recipe.no_build_isolation:
            cmd.insert(3, "--no-build-isolation")
        try:
            _run_with_env(cmd, timeout=900, env_extra=env_extra)
        except subprocess.CalledProcessError as exc:
            return False, f"install failed: {((exc.stderr or exc.stdout) or '')[-400:]}"
        except subprocess.TimeoutExpired:
            return False, "install timed out"
        return True, ""

    if recipe.no_build_isolation:
        ok, msg = _install_extras()
        if not ok:
            return False, msg
        ok, msg = _install_repo()
        if not ok:
            return False, msg
    else:
        ok, msg = _install_repo()
        if not ok:
            return False, msg
        ok, msg = _install_extras()
        if not ok:
            return False, msg

    # If the repo ships ``requirements_test.txt`` (pylint + friends do), install
    # it so test-time imports resolve.
    test_req = repo_dir / "requirements_test.txt"
    if test_req.exists():
        try:
            _run_with_env(
                [*pip_base, "-r", str(test_req)],
                timeout=300,
                env_extra=env_extra,
            )
        except subprocess.CalledProcessError as exc:
            # Non-fatal: test requirements may pin versions that conflict with
            # what the recipe already installed. Log and continue.
            print(
                f"note: requirements_test.txt install failed in {repo_dir}: "
                f"{((exc.stderr or exc.stdout) or '')[-200:]}",
                file=sys.stderr,
            )
        except subprocess.TimeoutExpired:
            print(f"note: requirements_test.txt install timed out in {repo_dir}", file=sys.stderr)

    return True, "ok"


def _choose_pytest_config(repo_dir: Path) -> Path:
    """Pick a config file to pin pytest to, so it does not walk up and find
    Maestro's own ``pyproject.toml`` (which sets ``asyncio_mode``).

    Preference: the repo's existing pytest config if present; otherwise write a
    blank stub at ``<repo>/._swebench_pytest.ini``.
    """
    for name in ("pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini"):
        candidate = repo_dir / name
        if candidate.exists():
            return candidate
    fallback = repo_dir / "._swebench_pytest.ini"
    if not fallback.exists():
        fallback.write_text("[pytest]\n", encoding="utf-8")
    return fallback


def _run_pytest_with_recipe(
    repo_dir: Path,
    venv_path: Path,
    failing_tests: list[str],
    recipe: InstallRecipe,
    *,
    timeout: float = 240.0,
) -> tuple[int, str]:
    pytest_bin = venv_path / "bin" / "pytest"
    if not pytest_bin.exists():
        return 127, f"pytest not found in venv: {pytest_bin}"
    # Pin pytest's config to the repo's own (or a blank stub) so it does NOT
    # walk up to Maestro's pyproject.toml and inherit ``asyncio_mode = "auto"``.
    config = _choose_pytest_config(repo_dir)
    cmd = [
        str(pytest_bin),
        "-c",
        str(config),
        "--rootdir",
        str(repo_dir),
        "--tb=no",
        "-q",
        "-p",
        "no:cacheprovider",
        *failing_tests,
    ]
    env_extra = dict(recipe.env)
    try:
        proc = _run_with_env(cmd, cwd=repo_dir, timeout=timeout, env_extra=env_extra, check=False)
    except subprocess.TimeoutExpired:
        return -1, "pytest timed out"
    return proc.returncode, (proc.stdout + proc.stderr)[-1500:]


def validate_task_with_recipe(
    task_dir: Path, failing_tests: list[str], recipe: InstallRecipe
) -> tuple[bool, str]:
    """before/ RED, after/ GREEN — with per-repo python + extras + env vars."""
    before = task_dir / "before"
    after = task_dir / "after"

    ok, msg = _setup_venv_with_recipe(before, task_dir / ".venv-before", recipe)
    if not ok:
        return False, f"before env: {msg}"
    rc_b, out_b = _run_pytest_with_recipe(before, task_dir / ".venv-before", failing_tests, recipe)
    if rc_b == 0:
        return False, (
            f"before/ pytest unexpectedly passed (rc=0); want RED.\ntail: {out_b[-500:]}"
        )

    ok, msg = _setup_venv_with_recipe(after, task_dir / ".venv-after", recipe)
    if not ok:
        return False, f"after env: {msg}"
    rc_a, out_a = _run_pytest_with_recipe(after, task_dir / ".venv-after", failing_tests, recipe)
    if rc_a != 0:
        return False, (f"after/ pytest did not pass (rc={rc_a}); want GREEN.\ntail: {out_a[-500:]}")

    return True, (
        f"before rc={rc_b}, after rc=0 "
        f"[py={recipe.python_version} extras={','.join(recipe.extras) or '-'}]"
    )


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
        cmd, cwd=cwd, capture_output=True, text=True, check=check, timeout=timeout
    )


# ---------------------------------------------------------------------------
# Patch inspection
# ---------------------------------------------------------------------------


def patch_line_count(patch: str) -> int:
    added = 0
    removed = 0
    for line in patch.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return added + removed


_TEST_PATH_RE = re.compile(r"(^|/)tests?/|(^|/)test_[^/]+\.py$|(^|/)[^/]+_test\.py$")


def parse_modified_source_files(patch: str) -> list[str]:
    """Source files touched by ``patch`` (tests excluded), de-duped, stable order."""
    files: list[str] = []
    seen: set[str] = set()
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) < 4:
                continue
            target = parts[3]
            if target.startswith("b/"):
                target = target[2:]
            if not target or target == "/dev/null":
                continue
            if _TEST_PATH_RE.search(target):
                continue
            if target not in seen:
                seen.add(target)
                files.append(target)
    return files


def classify_difficulty(lines: int) -> str:
    if lines <= 10:
        return "easy"
    if lines <= 25:
        return "medium"
    return "hard"


# ---------------------------------------------------------------------------
# Git clone + archive + apply
# ---------------------------------------------------------------------------


def clone_partial(owner: str, repo: str, workdir: Path) -> Path:
    """Partial clone (commit graph only, blobs lazy). Idempotent."""
    dest = workdir / f"{owner}__{repo}"
    if dest.exists():
        return dest
    url = f"https://github.com/{owner}/{repo}.git"
    _run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", url, str(dest)],
        timeout=900,
    )
    return dest


def ensure_commit_present(repo_dir: Path, sha: str) -> None:
    """If ``sha`` is unreachable (unlikely with partial clone but possible
    for force-pushed SHAs), try fetching it directly."""
    try:
        _run(["git", "cat-file", "-e", sha], cwd=repo_dir, timeout=30)
    except subprocess.CalledProcessError:
        _run(["git", "fetch", "origin", sha], cwd=repo_dir, timeout=300)


def archive_sha(repo_dir: Path, sha: str, dest: Path) -> None:
    """Extract the tree at ``sha`` via ``git archive | tar``."""
    dest.mkdir(parents=True, exist_ok=True)
    proc_archive = subprocess.Popen(
        ["git", "archive", sha, "--format=tar"],
        cwd=repo_dir,
        stdout=subprocess.PIPE,
    )
    proc_tar = subprocess.Popen(["tar", "-x", "-C", str(dest)], stdin=proc_archive.stdout)
    assert proc_archive.stdout is not None
    proc_archive.stdout.close()
    tar_rc = proc_tar.wait()
    archive_rc = proc_archive.wait()
    if archive_rc != 0 or tar_rc != 0:
        raise RuntimeError(
            f"git archive | tar failed (archive_rc={archive_rc}, tar_rc={tar_rc}) for {sha}"
        )


def apply_patch_in(target_dir: Path, patch: str, label: str) -> None:
    """Apply a unified diff in-place inside ``target_dir`` via ``patch -p1``.

    We use :program:`patch(1)` (not ``git apply``) because ``target_dir`` is
    usually inside Maestro's own git worktree. ``git apply`` walks up to find
    the enclosing ``.git`` and then gets confused about which repo's state it
    should diff against, silently "skipping" patches it considers
    already-applied. ``patch(1)`` is context-free and reliably rewrites the
    files the diff names.
    """
    if not patch.strip():
        return
    patch_file = target_dir / f"._swebench_{label}.patch"
    patch_file.write_text(patch, encoding="utf-8")
    try:
        _run(
            [
                "patch",
                "-p1",
                "--forward",
                "--no-backup-if-mismatch",
                "-i",
                patch_file.name,
            ],
            cwd=target_dir,
            timeout=120,
        )
    finally:
        if patch_file.exists():
            patch_file.unlink()


# ---------------------------------------------------------------------------
# Row → task
# ---------------------------------------------------------------------------


def problem_statement_oneline(text: str, cap: int = 200) -> str:
    if not text:
        return ""
    first = text.splitlines()[0].strip()
    return first[:cap] + ("..." if len(first) > cap else "")


def categorise_reason(reason: str) -> str:
    low = reason.lower()
    if "install failed" in low or low.startswith("before env:") or low.startswith("after env:"):
        return "install_failed"
    if "before/ pytest unexpectedly passed" in reason:
        return "before_unexpected_green"
    if "after/ pytest did not pass" in reason:
        return "after_not_green"
    if "timed out" in low:
        return "timeout"
    return "other"


def process_row(row: dict[str, Any], tasks_dir: Path, clone_cache: Path) -> dict[str, Any]:
    instance_id = row["instance_id"]
    repo = row["repo"]
    patch = row["patch"]
    lines_changed = patch_line_count(patch)
    failing_tests_raw = row["FAIL_TO_PASS"]

    base_meta: dict[str, Any] = {
        "task_id": instance_id,
        "repo": repo,
        "patch_lines": lines_changed,
        "failing_tests_count": len(json.loads(failing_tests_raw)) if failing_tests_raw else 0,
        "description": problem_statement_oneline(row["problem_statement"]),
        "status": "BROKEN",
        "reason": "",
        "category": "other",
    }

    try:
        owner, name = repo.split("/", 1)
    except ValueError:
        base_meta["reason"] = f"unexpected repo format: {repo}"
        return base_meta

    task_dir = tasks_dir / instance_id
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True)

    # --- clone + archive ---

    try:
        repo_dir = clone_partial(owner, name, clone_cache)
        ensure_commit_present(repo_dir, row["base_commit"])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        tail = (getattr(exc, "stderr", None) or getattr(exc, "stdout", None) or str(exc))[-300:]
        base_meta["reason"] = f"clone/fetch failed: {tail}"
        base_meta["category"] = "other"
        return base_meta

    try:
        archive_sha(repo_dir, row["base_commit"], task_dir / "before")
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        base_meta["reason"] = f"archive before failed: {exc}"
        base_meta["category"] = "other"
        return base_meta

    try:
        shutil.copytree(task_dir / "before", task_dir / "after")
    except OSError as exc:
        base_meta["reason"] = f"copy before→after failed: {exc}"
        base_meta["category"] = "other"
        return base_meta

    try:
        apply_patch_in(task_dir / "after", patch, "src")
        apply_patch_in(task_dir / "after", row["test_patch"], "test")
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or exc.stdout or "")[-300:]
        base_meta["reason"] = f"git apply failed: {tail}"
        base_meta["category"] = "other"
        return base_meta

    # --- task.json ---

    try:
        failing_tests = json.loads(failing_tests_raw)
    except json.JSONDecodeError as exc:
        base_meta["reason"] = f"FAIL_TO_PASS parse failed: {exc}"
        return base_meta

    expected_files = parse_modified_source_files(patch)
    task = BenchmarkTask(
        task_id=instance_id,
        repo=repo,
        pr_url="",
        description=problem_statement_oneline(row["problem_statement"]),
        natural_language_prompt=row["problem_statement"],
        failing_tests=failing_tests,
        expected_modified_files=expected_files,
        files_hint=expected_files,
        difficulty=classify_difficulty(lines_changed),
        source_commit=row["base_commit"],
    )
    (task_dir / "task.json").write_text(task.model_dump_json(indent=2) + "\n", encoding="utf-8")

    # --- validate (uv venv + install + pytest, per recipe) ---

    recipe = RECIPES.get(repo, InstallRecipe())
    try:
        ok, msg = validate_task_with_recipe(task_dir, failing_tests, recipe)
    except Exception as exc:
        base_meta["reason"] = f"validator crashed: {exc}"
        base_meta["category"] = "other"
        return base_meta

    base_meta["reason"] = msg
    base_meta["category"] = categorise_reason(msg) if not ok else "ok"
    base_meta["status"] = "OK" if ok else "BROKEN"
    return base_meta


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def filter_candidates(ds: Any) -> dict[str, list[dict[str, Any]]]:
    """Return ``{repo: [rows]}`` of rows matching our whitelist + size cap,
    with each bucket sorted ascending by patch size."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ds:
        if row["repo"] not in SMALL_REPOS:
            continue
        lines = patch_line_count(row["patch"])
        if lines == 0 or lines > MAX_PATCH_LINES:
            continue
        buckets[row["repo"]].append(dict(row))
    for rows in buckets.values():
        rows.sort(key=lambda r: patch_line_count(r["patch"]))
    return buckets


def pick_diverse(buckets: dict[str, list[dict[str, Any]]], n: int) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    repos = sorted(buckets.keys())
    while len(picked) < n:
        advanced = False
        for repo in repos:
            if buckets[repo]:
                picked.append(buckets[repo].pop(0))
                advanced = True
                if len(picked) == n:
                    break
        if not advanced:
            break
    return picked


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def render_manifest(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Benchmark task manifest (SWE-bench Lite)",
        "",
        f"Generated: {datetime.now(UTC).isoformat(timespec='seconds')}",
        "",
        "| status | instance_id | repo | patch Δ | failing_tests | one-line problem |",
        "|---|---|---|---:|---:|---|",
    ]
    for r in results:
        desc = (r["description"] or "-").replace("|", r"\|")[:120]
        lines.append(
            f"| {r['status']} | `{r['task_id']}` | {r['repo']} | "
            f"{r['patch_lines']} | {r['failing_tests_count']} | {desc} |"
        )
    broken = [r for r in results if r["status"] == "BROKEN"]
    if broken:
        lines += ["", "## Broken tasks — diagnostic tails", ""]
        for r in broken:
            reason = (r.get("reason") or "")[:800].replace("`", "'")
            lines += [
                f"### `{r['task_id']}`",
                f"- Category: `{r.get('category', '?')}`",
                "- Detail:",
                "",
                "```",
                reason,
                "```",
                "",
            ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=_PROJECT_ROOT / "benchmark" / "tasks",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_PROJECT_ROOT / "benchmark" / "tasks" / "MANIFEST.md",
    )
    parser.add_argument(
        "--clone-cache",
        type=Path,
        default=None,
        help="Directory to cache git clones. Default: ephemeral tempdir.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=TARGET_COUNT,
        help=f"Maximum tasks to pick (default {TARGET_COUNT}).",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Emit task directories without running the venv/pytest validator.",
    )
    args = parser.parse_args()

    try:
        _require_tool("git")
        _require_tool("tar")
        _require_tool("patch")
        if not args.skip_validate:
            _require_tool("uv")
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not args.skip_validate:
        # Prewarm the set of Python versions the recipes need so the first
        # ``uv venv --python X.Y`` call inside process_row doesn't stall while
        # downloading the interpreter.
        pyvers = sorted({r.python_version for r in RECIPES.values()})
        for v in pyvers:
            print(f"[adapter] ensuring python {v} is available ...")
            try:
                subprocess.run(
                    ["uv", "python", "install", v],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
            except subprocess.CalledProcessError as exc:
                print(
                    f"warning: uv python install {v} failed: {(exc.stderr or '')[-200:]}",
                    file=sys.stderr,
                )
            except subprocess.TimeoutExpired:
                print(f"warning: uv python install {v} timed out", file=sys.stderr)

    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    buckets = filter_candidates(ds)
    print("[adapter] candidates per repo (after whitelist + size cap):")
    for repo in sorted(buckets):
        print(f"  {repo}: {len(buckets[repo])}")

    picked = pick_diverse(buckets, args.limit)
    print(f"[adapter] picked {len(picked)} tasks:")
    for r in picked:
        print(f"  {r['instance_id']:<40} {r['repo']:<25} Δ={patch_line_count(r['patch'])}")

    args.tasks_dir.mkdir(parents=True, exist_ok=True)

    tmp_holder: tempfile.TemporaryDirectory[str] | None = None
    if args.clone_cache is None:
        tmp_holder = tempfile.TemporaryDirectory(prefix="maestro-swebench-")
        clone_cache = Path(tmp_holder.name)
    else:
        args.clone_cache.mkdir(parents=True, exist_ok=True)
        clone_cache = args.clone_cache

    results: list[dict[str, Any]] = []
    try:
        for row in picked:
            print(f"[adapter] processing {row['instance_id']} ({row['repo']}) ...")
            if args.skip_validate:
                # Produce directories but short-circuit validation.
                r = process_row_skip_validate(row, args.tasks_dir, clone_cache)
            else:
                r = process_row(row, args.tasks_dir, clone_cache)
            print(f"[adapter]   → {r['status']} ({r['category']}): {r['reason'][:200]}")
            results.append(r)
    finally:
        if tmp_holder is not None:
            tmp_holder.cleanup()

    manifest = render_manifest(results)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(manifest, encoding="utf-8")

    ok = [r for r in results if r["status"] == "OK"]
    broken = [r for r in results if r["status"] == "BROKEN"]
    by_cat: dict[str, int] = defaultdict(int)
    for r in broken:
        by_cat[r["category"]] += 1

    print()
    print(f"Attempted: {len(results)}")
    print(f"OK: {len(ok)}")
    print(f"BROKEN: {len(broken)}")
    if broken:
        print("  by category:")
        for cat, n in sorted(by_cat.items()):
            print(f"    {cat}: {n}")
    print(f"Manifest: {args.manifest}")

    return 0 if len(ok) >= 5 else 1


def process_row_skip_validate(
    row: dict[str, Any], tasks_dir: Path, clone_cache: Path
) -> dict[str, Any]:
    """Build the task directory without running the validator (cheap smoke run)."""
    import swebench_adapter as self_mod

    original = self_mod.validate_task_with_recipe

    def _noop(_task_dir: Path, _failing: list[str], _recipe: InstallRecipe) -> tuple[bool, str]:
        return True, "validation skipped (--skip-validate)"

    self_mod.validate_task_with_recipe = _noop  # type: ignore[assignment]
    try:
        return process_row(row, tasks_dir, clone_cache)
    finally:
        self_mod.validate_task_with_recipe = original  # type: ignore[assignment]


if __name__ == "__main__":
    sys.exit(main())
