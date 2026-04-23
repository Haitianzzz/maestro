"""Tier 1 verifier: deterministic static checks (spec 06 §3).

Runs ``ruff`` and (optionally) ``mypy`` on the files the sub-agent
modified. Any error → tier fails, short-circuiting the rest of the
verifier chain. Zero API cost.

Why deterministic first?
------------------------

Linter / type-checker catch 80% of patch bugs (syntax errors, undefined
names, obvious type mismatches) at zero token cost. Running them before
any API-billed tier is the cost-cheapest way to reject bad patches
(spec 06 §7).

Environment knobs:

* ``MAESTRO_SKIP_MYPY=1`` — disable the mypy step entirely (spec 06 §3.2).
  Benchmark targets may not have complete type annotations, so this
  escape hatch exists to avoid spurious failures there.

This module is deliberately a single public ``DeterministicVerifier`` class
plus a tiny :func:`run_subprocess` helper. The helper is exported so
modules 12 and 13 can reuse the same timeout / capture shape.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from dataclasses import dataclass

from maestro.models import SubAgentResult, SubTask, TierResult, VerificationResult
from maestro.sandbox.workspace import IsolatedWorkspace
from maestro.utils.logging import get_logger

_logger = get_logger("maestro.verifier.deterministic")

_RUFF_TIMEOUT_S = 30.0
_MYPY_TIMEOUT_S = 60.0
_OUTPUT_TAIL_BYTES = 2000


@dataclass(frozen=True)
class SubprocessResult:
    """Outcome of a bounded subprocess run."""

    ok: bool
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def combined(self) -> str:
        if self.stdout and self.stderr:
            return f"{self.stdout}\n--- stderr ---\n{self.stderr}"
        return self.stdout or self.stderr or ""


async def run_subprocess(
    argv: list[str],
    *,
    cwd: str | os.PathLike[str],
    timeout: float,
    env: dict[str, str] | None = None,
) -> SubprocessResult:
    """Run ``argv`` with a hard timeout, capturing stdout/stderr.

    Returns a :class:`SubprocessResult`. Only a :class:`FileNotFoundError`
    (the command itself is not on PATH) propagates — anything else, including
    a non-zero exit, is encoded into the returned record.
    """
    merged_env = {**os.environ, **(env or {})}
    _logger.debug("subprocess_start", argv=argv, cwd=str(cwd))
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=os.fspath(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=merged_env,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        _logger.warning(
            "subprocess_timeout",
            argv=argv,
            timeout_s=timeout,
        )
        return SubprocessResult(ok=False, returncode=-1, stdout="", stderr="", timed_out=True)

    stdout = stdout_b.decode("utf-8", errors="replace")
    stderr = stderr_b.decode("utf-8", errors="replace")
    ok = proc.returncode == 0
    return SubprocessResult(
        ok=ok,
        returncode=int(proc.returncode or 0),
        stdout=stdout,
        stderr=stderr,
    )


class DeterministicVerifier:
    """Runs ruff (always) + mypy (unless ``MAESTRO_SKIP_MYPY=1``)."""

    def __init__(
        self,
        *,
        ruff_argv: list[str] | None = None,
        mypy_argv: list[str] | None = None,
    ) -> None:
        # Overridable for tests. Defaults match spec 06 §3.2.
        self._ruff_argv = ruff_argv or ["ruff", "check"]
        self._mypy_argv = mypy_argv or ["mypy", "--ignore-missing-imports"]

    # ------------------------------------------------------------------
    # Scheduler-compatible entry point
    # ------------------------------------------------------------------

    async def run(self, workspace: IsolatedWorkspace, sub_result: SubAgentResult) -> TierResult:
        """Run ruff then (optionally) mypy. First failure short-circuits."""
        start = time.perf_counter()

        target_files = [f for f in sub_result.modified_files if f.endswith(".py")]
        if not target_files:
            # Nothing Python to check (e.g. docs-only patch). Pass.
            return TierResult(
                tier="deterministic",
                passed=True,
                details="no .py files modified; nothing to check",
                latency_ms=_elapsed_ms(start),
                cost_usd=0.0,
            )

        ruff = await run_subprocess(
            [*self._ruff_argv, *target_files],
            cwd=workspace.path,
            timeout=_RUFF_TIMEOUT_S,
        )
        if ruff.timed_out:
            return _fail("ruff timed out", start)
        if not ruff.ok:
            details = _tail(f"ruff failed (rc={ruff.returncode}):\n{ruff.combined}")
            _logger.info("det_ruff_failed", subtask_id=sub_result.subtask_id)
            return _fail(details, start)

        if os.getenv("MAESTRO_SKIP_MYPY", "0") != "1":
            mypy = await run_subprocess(
                [*self._mypy_argv, *target_files],
                cwd=workspace.path,
                timeout=_MYPY_TIMEOUT_S,
            )
            if mypy.timed_out:
                return _fail("mypy timed out", start)
            if not mypy.ok:
                details = _tail(f"mypy failed (rc={mypy.returncode}):\n{mypy.combined}")
                _logger.info("det_mypy_failed", subtask_id=sub_result.subtask_id)
                return _fail(details, start)

        _logger.debug("det_passed", subtask_id=sub_result.subtask_id)
        return TierResult(
            tier="deterministic",
            passed=True,
            details="ruff and mypy passed",
            latency_ms=_elapsed_ms(start),
            cost_usd=0.0,
        )

    # ------------------------------------------------------------------
    # VerifierProtocol: scheduler calls .verify(); we delegate to run().
    # ------------------------------------------------------------------

    async def verify(
        self,
        subtask: SubTask,
        workspace: IsolatedWorkspace,
        sub_result: SubAgentResult,
    ) -> VerificationResult:
        """Tier-only adapter so this class satisfies ``VerifierProtocol``.

        Wraps the single ``TierResult`` into a :class:`VerificationResult`
        so ad-hoc users (and integration tests) can plug a
        ``DeterministicVerifier`` straight into the scheduler before the
        full three-tier :class:`Verifier` lands in module 13.
        """
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _elapsed_ms(start: float) -> int:
    return max(0, int((time.perf_counter() - start) * 1000))


def _tail(text: str) -> str:
    if len(text) <= _OUTPUT_TAIL_BYTES:
        return text
    return "... (truncated)\n" + text[-_OUTPUT_TAIL_BYTES:]


def _fail(details: str, start: float) -> TierResult:
    return TierResult(
        tier="deterministic",
        passed=False,
        details=details,
        latency_ms=_elapsed_ms(start),
        cost_usd=0.0,
    )


__all__ = ["DeterministicVerifier", "SubprocessResult", "run_subprocess"]
