# Benchmark task curation

`curate.py` turns a human-picked shortlist of GitHub PRs into the
`benchmark/tasks/<task_id>/{task.json, before/, after/}` layout Maestro's
harness consumes. **Picking the PRs is a manual step** — the script only
does the deterministic tail (clone, snapshot, validate).

## Prerequisites

- `git` on `PATH` (network access to `github.com`)
- `uv` on `PATH` (`pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Write access to `benchmark/tasks/` in this repo

## 1. Prepare the shortlist

Copy `pr_shortlist.json.example` to `pr_shortlist.json` and edit. Each entry:

| field | required | notes |
|---|---|---|
| `task_id` | yes | e.g. `"task-001"`. Used as the directory name under `benchmark/tasks/`. |
| `pr_url` | yes | Full PR URL: `https://github.com/<owner>/<repo>/pull/<N>`. |
| `failing_tests` | yes | List of pytest selectors. Accepts `path/to/test.py` or `path/to/test.py::test_name`. These must fail on `before/` and pass on `after/`. |
| `expected_modified_files` | yes | Source files Maestro is expected to touch. Populates `BenchmarkTask.expected_modified_files` for `files_modified_match` accounting. |
| `files_hint` | no | Optional planner starting point. Defaults to `expected_modified_files`. |
| `difficulty` | no | `"easy" / "medium" / "hard"`. Default `"medium"`. Target distribution per spec 09 §2.4: 4 easy / 3 medium / 1 hard. |
| `description_override` | no | Plain-English bug symptom. **Strongly recommended** — PR titles often leak which file or function to touch, which poisons the planner's context. Write a user-voice prompt: *"Requests' Session drops custom adapters on redirect, breaking callers that route per-host."* |

### Picking PRs — selection checklist

For each PR, verify (the script does **not** re-check these):

1. Merged, not closed/open.
2. Bug fix only — no features, refactors, or docs.
3. 1-3 `.py` files changed, 3-50 total lines.
4. The PR either adds a failing test or modifies one from green to covering-the-bug. Note the `path::name` selector for `failing_tests`.
5. Tests don't require network, a live DB, or a GPU.
6. Python 3.10+ compatible.
7. Merged within the last ~2 years (repo structure stable enough that checkouts still install).

## 2. Run

```bash
uv run python benchmark/scripts/curate.py \
  --shortlist benchmark/scripts/pr_shortlist.json \
  --tasks-dir benchmark/tasks \
  --clone-cache /tmp/maestro-repo-cache  # optional; reused across runs
```

Per-entry flow:
- Clone `<owner>/<repo>` (cached if already present).
- `git fetch origin pull/<N>/head` → `(head_sha, parent_sha, title)`.
- `git archive parent | tar -x → task-<id>/before/`, same for head → `after/`.
- Write `task.json`.
- `uv venv task-<id>/.venv-before`, install the repo (pyproject or requirements.txt) + pytest, run the declared tests. Must go red.
- Same for `.venv-after`. Must go green.

The script **never raises** for a single-task failure — it records the reason in the manifest and moves on. Exit code is `0` iff at least 5 tasks end OK.

## 3. Read `MANIFEST.md`

Generated at `benchmark/tasks/MANIFEST.md`. One row per task:

```
| status | task_id | difficulty | Δ lines | failing_tests | PR | description |
```

- `status == "OK"` — `before/` was red, `after/` was green. Safe to use.
- `status == "BROKEN"` — something failed; a diagnostic tail is appended below the table. Common causes:
  - `before env: install failed: ...` — the repo's dependencies don't install cleanly. Usually a build-time Rust/C extension or a package pinned to a Python version you don't have.
  - `before/ pytest unexpectedly passed` — the failing test you listed wasn't actually red. Check you used the right `path::name` selector and the right PR parent.
  - `after/ pytest did not pass` — the PR didn't fix the test you listed, or install pulled in a different version of a dep than the PR expected. Occasionally fixed by pinning versions.

## 4. Replace a broken task

Curation is idempotent per `task_id`:

1. Pick a replacement PR from the same repo (to amortise the clone cache).
2. Edit `pr_shortlist.json` — change that entry's `pr_url` / `failing_tests` / `expected_modified_files`.
3. Re-run `curate.py`. The old `task-<id>/` directory is wiped and rebuilt from scratch; stale `.venv-before`/`.venv-after` are recreated.

If a whole repo turns out to be install-hostile (e.g. every PR from it fails `uv pip install -e .`), drop it entirely and rebalance toward the other four candidate repos.

## 5. What the script is NOT

- It doesn't call the GitHub API or `gh` CLI. PR titles come from commit messages (`git log -1 --format=%s`). Use `description_override` when the commit title is poor.
- It doesn't pin dependency versions. If a repo's tests are flaky against head-of-dep-tree, rerun — or curate around it.
- It doesn't run the Maestro pipeline. That's `maestro bench` (spec 09 §3 / module 15). Curation only assembles the inputs.

## 6. SWE-bench Lite adapter (sibling script)

`swebench_adapter.py` takes a different, fully-automated path: it pulls
`princeton-nlp/SWE-bench_Lite` from HuggingFace, filters to a whitelist of
laptop-friendly repos (see `SMALL_REPOS` in the script), picks patches
≤ 50 lines diversified across repos, and materialises `task.json` +
`before/` + `after/` via the same workspace machinery. Per-repo install
recipes (`RECIPES` in the script) pin Python version, extras, env vars
(e.g. `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYTEST`), and opt-in
`--no-build-isolation` so PEP-660 editable installs work against old
build-system pins.

### Known limitations

**Not all SWE-bench Lite tasks reproduce outside the official Docker
harness.** The upstream harness ships one Conda environment recipe per
instance with exact dependency versions; our per-repo recipes are a
coarser approximation. On the 8-task pilot run we observed two recurring
failure classes that no amount of per-repo tuning fixed:

1. **`before_unexpected_green`** — the failing test passes on
   `base_commit` in our env (e.g. `psf__requests-2674`, `sphinx-doc__sphinx-7738`
   on some configs). The bug depends on a dependency version different
   from the one our recipe resolves to, so `before/` is not actually red.
2. **`after_not_green`** — the gold patch installs cleanly but the
   `FAIL_TO_PASS` test still fails in our env (e.g.
   `pylint-dev__pylint-7080`, `mwaskom__seaborn-3407`). Usually a
   runtime dep version mismatch or hard-coded path assumption.

The 5 excluded tasks hit one of these. Full diagnostic tails are
preserved in the repo's historical `MANIFEST.md` git history (look at
the commit that introduced `benchmark/tasks/MANIFEST.md`).

Running against the official Docker harness is future work — it would
close the repro gap but pulls a heavy Docker-in-CI dependency into
Maestro. For now, the 3-task subset in `benchmark/tasks/MANIFEST.md` is
what we benchmark against.
