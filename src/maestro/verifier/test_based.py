"""Tier 2 verifier: pytest-based behavioural check (spec 06 §4).

Runs pytest against tests that cover the sub-agent's modified source files.
Two paths:

1. **Related tests exist** — find ``tests/test_<modname>.py`` /
   ``tests/unit/test_<modname>.py`` (or any other path that names the
   module) and invoke pytest on those files.
2. **No related tests** — optionally ask an LLM to generate a test file
   for the diff, then run pytest on it. This is a benchmark-versus-
   production trade-off (see below) and is controlled by
   ``TaskSpec.auto_gen_tests``.

Why the auto-gen fallback is OFF by default
-------------------------------------------

The auto-generated test reflects the LLM's *own* understanding of the
change, so using it as ground truth on a benchmark risks a feedback loop:
the sub-agent misreads the task, the test-writer reproduces the same
misreading, and the "verifier" rubber-stamps the bug. Benchmark always
ships ground-truth tests; only ``maestro run`` in production sets
``auto_gen_tests=True`` where the user is the ultimate ground truth.

Additionally, even when ``auto_gen_tests=True`` the resulting pytest is
best understood as a smoke test / sanity check — it verifies that the
diff exhibits the behaviour the agent claims in its rationale, NOT that
the diff correctly solves the user's task. The user remains the ground
truth in production.

Subprocess plumbing reuses :func:`~maestro.verifier.deterministic.run_subprocess`
from module 11 (spec 06 §4.2).
"""

from __future__ import annotations

import time
from pathlib import Path

from maestro.llm.client import LLMClient
from maestro.models import SubAgentResult, SubTask, TierResult, VerificationResult
from maestro.sandbox.workspace import IsolatedWorkspace
from maestro.utils.logging import get_logger

from .deterministic import run_subprocess

_logger = get_logger("maestro.verifier.test_based")

_PYTEST_TIMEOUT_S = 120.0
_OUTPUT_TAIL_BYTES = 2000

# Where we'll look for tests that cover ``src/foo.py``: any path matching
# one of these relative templates inside the workspace.
_TEST_LOOKUP_TEMPLATES = (
    "tests/test_{module}.py",
    "tests/unit/test_{module}.py",
    "tests/integration/test_{module}.py",
    "test/test_{module}.py",
    "test_{module}.py",
)

# Prompt for the auto-generated tests (spec 06 §4.3). Kept here so the
# file is the single home for Tier 2 plumbing.
_AUTOGEN_SYSTEM_PROMPT = """\
You are a test-writing assistant. Given a code change, write a pytest test \
file that verifies the new or changed behaviour.
"""

_AUTOGEN_USER_PROMPT_TEMPLATE = """\
You are writing a SMOKE TEST for a code change an agent just produced.
This is not ground-truth verification — it is a sanity check that the
change exhibits the behaviour the agent claims.

# Agent's stated intent
{rationale}

# The change
```diff
{diff}
```

# Rules
- Only write tests. Do not modify the implementation.
- Use pytest style (no unittest.TestCase).
- Include at least 3 test cases covering the behaviour the agent claims.
- Output ONLY the complete Python test file content, no markdown fences,
  no prose.
"""


class TestBasedVerifier:
    """Tier 2 pytest verifier."""

    # pytest collects classes whose names start with "Test"; opt out so our
    # own test suite doesn't try to instantiate this one.
    __test__ = False

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        *,
        auto_gen_tests: bool = False,
        pytest_argv: list[str] | None = None,
    ) -> None:
        self._llm = llm_client
        self._auto_gen = auto_gen_tests
        # Default invocation: stop at first failure, short tracebacks to keep
        # log volume down.
        self._pytest_argv = pytest_argv or ["pytest", "-x", "--tb=short"]

    # ------------------------------------------------------------------
    # Scheduler-compatible entry point
    # ------------------------------------------------------------------

    async def run(self, workspace: IsolatedWorkspace, sub_result: SubAgentResult) -> TierResult:
        start = time.perf_counter()

        source_files = [f for f in sub_result.modified_files if f.endswith(".py")]
        if not source_files:
            return _pass_tier("no .py files modified; nothing to pytest", start)

        existing = _find_related_tests(workspace.path, source_files)
        if existing:
            return await self._run_pytest(workspace, existing, start)

        if not self._auto_gen:
            # Benchmark / strict mode: no tests, no behavioural evidence —
            # we pass because we cannot reject (spec 06 §4.2 path B off).
            return _pass_tier(
                "no related tests found; auto-gen disabled (benchmark mode)",
                start,
            )

        if self._llm is None:
            return _pass_tier(
                "no related tests found; auto-gen enabled but no LLM client",
                start,
            )

        try:
            generated = await self._generate_tests(workspace, sub_result)
        except Exception as exc:
            _logger.warning(
                "test_based_autogen_failed",
                subtask_id=sub_result.subtask_id,
                error=exc.__class__.__name__,
                message=str(exc),
            )
            return _fail_tier(f"auto-gen failed: {exc}", start)
        return await self._run_pytest(workspace, [generated], start)

    # ------------------------------------------------------------------
    # VerifierProtocol adapter (mirrors DeterministicVerifier)
    # ------------------------------------------------------------------

    async def verify(
        self,
        subtask: SubTask,
        workspace: IsolatedWorkspace,
        sub_result: SubAgentResult,
    ) -> VerificationResult:
        del subtask
        tier = await self.run(workspace, sub_result)
        return VerificationResult(
            subtask_id=sub_result.subtask_id,
            overall_passed=tier.passed,
            tiers=[tier],
            judge_detail=None,
            total_latency_ms=tier.latency_ms,
            total_cost_usd=tier.cost_usd,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _run_pytest(
        self,
        workspace: IsolatedWorkspace,
        test_files: list[Path],
        start: float,
    ) -> TierResult:
        targets = [str(p.relative_to(workspace.path)) for p in test_files]
        result = await run_subprocess(
            [*self._pytest_argv, *targets],
            cwd=workspace.path,
            timeout=_PYTEST_TIMEOUT_S,
        )
        if result.timed_out:
            return _fail_tier("pytest timed out", start)
        if not result.ok:
            details = _tail(f"pytest failed (rc={result.returncode}):\n{result.combined}")
            _logger.info(
                "test_based_pytest_failed",
                targets=targets,
                returncode=result.returncode,
            )
            return _fail_tier(details, start)

        _logger.debug("test_based_passed", targets=targets)
        return _pass_tier(f"pytest passed on {len(targets)} file(s)", start)

    async def _generate_tests(
        self, workspace: IsolatedWorkspace, sub_result: SubAgentResult
    ) -> Path:
        """Ask the LLM for a test file and write it into the iso workspace.

        The path is ``tests/autogen/test_<subtask_id>.py`` so generated
        artifacts don't collide with any real tests the repo already owns.

        CONCURRENCY + WRITES INVARIANT: This method writes to the iso
        workspace *outside* of ``subtask.writes``. This is safe today
        because:

        - ``merge_patches`` only copies files listed in
          ``SubAgentResult.modified_files`` back to ``main/`` (see
          ``WorkspaceManager._apply_workspace_delta``).
        - ``get_final_diff`` compares ``baseline/`` vs ``main/``, so the
          iso-only file never reaches the user's diff.
        - Each retry attempt creates a fresh iso, so stale autogen files
          never leak between attempts.

        If any of those three assumptions change, this method must be
        updated to either (a) declare the autogen file in a synthetic
        writes list or (b) write to a sibling tempdir outside the
        workspace.
        """
        assert self._llm is not None  # caller guards this
        prompt = _AUTOGEN_USER_PROMPT_TEMPLATE.format(
            diff=sub_result.diff,
            rationale=sub_result.rationale or "(no rationale)",
        )
        text, _meta = await self._llm.call_text(
            role="subagent",  # uses the cheaper coding model for auto-gen
            messages=[
                {"role": "system", "content": _AUTOGEN_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        # Strip accidental code fences the model sometimes emits despite
        # the "no markdown" instruction.
        content = _strip_code_fence(text)

        out_dir = workspace.path / "tests" / "autogen"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"test_{_slugify(sub_result.subtask_id)}.py"
        out_path.write_text(content, encoding="utf-8")
        return out_path


# ---------------------------------------------------------------------------
# Helpers (pure, exposed for testing)
# ---------------------------------------------------------------------------


def _find_related_tests(repo_root: Path, source_files: list[str]) -> list[Path]:
    """Return test files that plausibly cover any of ``source_files``.

    Looks for each source file's stem under the well-known template paths.
    Duplicates are de-duped while preserving first-seen order.
    """
    found: list[Path] = []
    seen: set[Path] = set()
    for rel in source_files:
        module = Path(rel).stem
        if module.startswith("test_") or module == "__init__":
            continue
        for template in _TEST_LOOKUP_TEMPLATES:
            candidate = repo_root / template.format(module=module)
            if candidate.exists() and candidate not in seen:
                found.append(candidate)
                seen.add(candidate)
    return found


def _pass_tier(details: str, start: float) -> TierResult:
    return TierResult(
        tier="test_based",
        passed=True,
        details=details,
        latency_ms=_elapsed_ms(start),
        cost_usd=0.0,
    )


def _fail_tier(details: str, start: float) -> TierResult:
    return TierResult(
        tier="test_based",
        passed=False,
        details=details,
        latency_ms=_elapsed_ms(start),
        cost_usd=0.0,
    )


def _elapsed_ms(start: float) -> int:
    return max(0, int((time.perf_counter() - start) * 1000))


def _tail(text: str) -> str:
    if len(text) <= _OUTPUT_TAIL_BYTES:
        return text
    return "... (truncated)\n" + text[-_OUTPUT_TAIL_BYTES:]


def _strip_code_fence(text: str) -> str:
    """Strip ```python ... ``` fences an LLM may emit despite instructions."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop opening fence (possibly ```python).
        first_newline = stripped.find("\n")
        if first_newline == -1:
            return stripped
        body = stripped[first_newline + 1 :]
        if body.rstrip().endswith("```"):
            body = body.rstrip()
            body = body[: body.rfind("```")].rstrip() + "\n"
        return body
    return text


def _slugify(subtask_id: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in subtask_id)


__all__ = ["TestBasedVerifier"]
